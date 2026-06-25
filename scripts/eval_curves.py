"""Gate 2 — conditional quality + diversity as CURVES (README §8).

NOT ROUGE/BLEU: quality = semantic similarity to reference(s) via an independent
embedder; diversity = multi-reference COVERAGE (primary) guarded by per-sample
VALIDITY. Two modes, auto-detected from the checkpoint:

  * flow ckpt  (key "predictor") -> sweep guidance_scale (the original behavior).
  * hybrid ckpt (key "hybrid")   -> sweep the sampling TEMPERATURE s, the thesis
    test: does coverage rise with s while validity holds? (the residual-buys-
    valid-diversity question). Run this on a MULTI-REFERENCE task (mscoco).

    # flow (single-ref map read, gigaword):
    python -m Sync.scripts.eval_curves --task gigaword --ckpt runs/predictor_best.pt

    # hybrid coverage/validity sweep (multi-ref, mscoco):
    python -m Sync.scripts.eval_curves --task mscoco --ckpt runs/hybrid_best.pt \
        --k 8 --temps 0 0.5 1.0 1.5 --guidance 1.0 --show 10

Decision rule (hybrid sweep): coverage ↑ with s AND validity high -> residual buys
valid diversity (writeup-worthy); coverage ↑ but validity collapses -> fake/
incoherent spread; coverage ~flat -> hybrid ≈ regressor + noise.
"""

from __future__ import annotations

import argparse
import json

from ..codec import SonarCodecAdapter
from ..config import get_preset
from ..data import Chunker, load_task_pairs
from ..generate import TextGenerator
from ..metrics import SemanticScorer, guidance_curve, temperature_curve
from ..predictor import CountHead, FlowMatchingPredictor, HybridPredictor
from ..training import EmaModel, load_ckpt, load_whitening


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gigaword")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-type", choices=["auto", "flow", "hybrid"], default="auto",
                    help="auto-detect from checkpoint keys (predictor=flow, hybrid=hybrid)")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--k", type=int, default=8, help="samples per prompt")
    ap.add_argument("--guidance", type=float, nargs="+", default=None,
                    help="flow: guidance sweep (default 1.0 1.5 2.0 3.0). "
                         "hybrid: single guidance for decoding (default 1.0)")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5],
                    help="hybrid only: sampling temperature s sweep")
    ap.add_argument("--steps", type=int, default=None, help="ODE steps override")
    ap.add_argument("--heldout-cache-dir", default=None,
                    help="accepted for CLI compatibility; NOT needed — references are "
                         "read as text from the task loader and whitening loads from "
                         "the TRAIN cache (coverage is scored on decoded text).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--show", type=int, default=0,
                    help="print this many (prompt, references, K-samples) examples")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = get_preset(args.task)

    ck = load_ckpt(args.ckpt, map_location="cpu")
    model_type = args.model_type
    if model_type == "auto":
        model_type = "hybrid" if "hybrid" in ck else "flow"
    if model_type == "hybrid":
        cfg.predictor.hybrid = True
    cfg.validate()
    device = args.device
    print(f"checkpoint {args.ckpt} -> model_type={model_type}")
    if args.heldout_cache_dir is not None:
        print("  note: --heldout-cache-dir is unused (coverage is text-scored; "
              "refs come from the task loader, whitening from the train cache)")

    if args.smoke:
        args.limit = 16
        args.k = 4
        args.temps = [0.0, 1.0]
        if args.guidance is None:
            args.guidance = [1.0, 2.0]

    codec = SonarCodecAdapter(cfg.codec, cfg.tokenizer, device=device)
    chunker = Chunker(cfg.chunk)
    count_head = CountHead(cfg.predictor)
    count_head.load_state_dict(ck["count_head"])
    whitening = load_whitening(cfg, device)

    if model_type == "hybrid":
        hybrid = HybridPredictor(cfg.predictor)
        hybrid.load_state_dict(ck["hybrid"])
        if not args.no_ema and "ema" in ck:
            ema = EmaModel(hybrid, ck["ema"]["decay"])
            ema.load_state_dict(ck["ema"])
            ema.copy_to(hybrid)
            print("loaded EMA weights into hybrid")
        gen = TextGenerator(
            cfg, codec, None, count_head, chunker, device=device,
            whitening=whitening, hybrid=hybrid,
        )
    else:
        predictor = FlowMatchingPredictor(cfg.predictor)
        predictor.load_state_dict(ck["predictor"])
        if not args.no_ema and "ema" in ck:
            ema = EmaModel(predictor, ck["ema"]["decay"])
            ema.load_state_dict(ck["ema"])
            ema.copy_to(predictor)
            print("loaded EMA weights into predictor")
        gen = TextGenerator(
            cfg, codec, predictor, count_head, chunker, device=device, whitening=whitening,
        )

    prompts, refs = [], []
    for p, _r, rf in load_task_pairs(args.task, split=args.split, limit=args.limit):
        prompts.append(p)
        refs.append(rf)
    n_ref = sum(len(r) for r in refs) / max(1, len(refs))
    print(f"loaded {len(prompts)} prompts; mean refs/input = {n_ref:.2f} "
          f"({'MULTI-ref' if n_ref > 1.01 else 'SINGLE-ref — coverage is degenerate'})")

    scorer = SemanticScorer()
    guidance_dec = (args.guidance[0] if args.guidance else 1.0)

    if model_type == "hybrid":
        def sample_fn(prompt: str, s: float) -> list[str]:
            return gen.sample_many(
                prompt, k=args.k, guidance_scale=guidance_dec, steps=args.steps,
                temperature=s, seed=0,
            )

        curves = temperature_curve(prompts, refs, sample_fn, args.temps, scorer)
        print("\n" + "=" * 78)
        print(f"HYBRID COVERAGE/VALIDITY SWEEP over temperature s "
              f"(k={args.k}, decode guidance={guidance_dec})")
        print("=" * 78)
        print(json.dumps(curves, indent=2))
        print("\n  s     coverage  validity  quality  distinct  distinct2")
        for s in args.temps:
            c = curves[s]
            print(f"  {s:<5g} {c['coverage']:8.4f}  {c['validity']:8.4f}  "
                  f"{c['quality']:7.4f}  {c['distinctness']:8.4f}  {c['distinct2']:8.4f}")
        cov0 = curves[args.temps[0]]["coverage"]
        covT = curves[args.temps[-1]]["coverage"]
        valT = curves[args.temps[-1]]["validity"]
        val0 = curves[args.temps[0]]["validity"]
        print("\n" + "-" * 78)
        if covT - cov0 >= 0.02 and valT >= val0 - 0.05:
            verdict = ("COVERAGE RISES, VALIDITY HOLDS -> the residual buys VALID "
                       "diversity. Accuracy (s=0) + valid spread (s=1) the hybrid "
                       "achieves and pure flow/regression cannot. Writeup-worthy.")
        elif covT - cov0 >= 0.02:
            verdict = ("COVERAGE RISES but VALIDITY COLLAPSES -> incoherent spread "
                       "(fake diversity). Investigate the residual flow "
                       "(off-manifold / over-scaled) before any writeup.")
        else:
            verdict = ("COVERAGE ~FLAT -> the residual adds no valid diversity; "
                       "hybrid ≈ regressor + noise. Doesn't earn its complexity; "
                       "stop and reassess scope.")
        print("VERDICT:", verdict)
        print("-" * 78)
        sweep_vals = args.temps
    else:
        guidance_vals = args.guidance or [1.0, 1.5, 2.0, 3.0]

        def sample_fn(prompt: str, g: float) -> list[str]:
            return gen.sample_many(prompt, k=args.k, guidance_scale=g, steps=args.steps)

        curves = guidance_curve(prompts, refs, sample_fn, guidance_vals, scorer)
        print(json.dumps(curves, indent=2))
        sweep_vals = guidance_vals

    if args.show > 0:
        n_show = min(args.show, len(prompts))
        knob = "s" if model_type == "hybrid" else "guidance"
        show_at = args.temps[-1] if model_type == "hybrid" else (sweep_vals[0])
        print("\n" + "=" * 78)
        print(f"[examples] {knob}={show_at}  (showing {n_show})")
        print("=" * 78)
        for i in range(n_show):
            samples = sample_fn(prompts[i], show_at)
            print(f"\n--- example {i} ---")
            print("  input    :", prompts[i][:200])
            print("  refs     :", " | ".join(refs[i])[:300] if refs[i] else "(none)")
            for j, s in enumerate(samples):
                print(f"  sample {j} :", s[:200])


if __name__ == "__main__":
    main()
