"""Gate 2 — conditional quality + diversity as CURVES vs guidance (README §8).

NOT ROUGE/BLEU: quality = semantic similarity to reference(s) via an independent
embedder; diversity = multi-reference coverage (primary) + distinctness. Run the
diversity read on a one-to-many task (mscoco); the map read on gigaword.

    python -m Sync.scripts.eval_curves --task gigaword --ckpt runs/predictor_final.pt
"""

from __future__ import annotations

import argparse
import json

from ..codec import SonarCodecAdapter
from ..config import get_preset
from ..data import Chunker, load_task_pairs
from ..generate import TextGenerator
from ..metrics import SemanticScorer, guidance_curve
from ..predictor import CountHead, FlowMatchingPredictor
from ..training import EmaModel, load_ckpt, load_whitening


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gigaword")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--k", type=int, default=5, help="samples per prompt")
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--steps", type=int, default=None, help="ODE steps override")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--show", type=int, default=0,
                    help="print this many (prompt, reference, samples) examples per guidance")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.limit = 16
        args.k = 3
        args.guidance = [1.0, 2.0]

    cfg = get_preset(args.task)
    cfg.validate()
    device = args.device

    codec = SonarCodecAdapter(cfg.codec, cfg.tokenizer, device=device)
    chunker = Chunker(cfg.chunk)
    predictor = FlowMatchingPredictor(cfg.predictor)
    count_head = CountHead(cfg.predictor)

    ck = load_ckpt(args.ckpt, map_location="cpu")
    predictor.load_state_dict(ck["predictor"])
    count_head.load_state_dict(ck["count_head"])
    if not args.no_ema and "ema" in ck:
        ema = EmaModel(predictor, ck["ema"]["decay"])
        ema.load_state_dict(ck["ema"])
        ema.copy_to(predictor)
        print("loaded EMA weights into predictor")

    whitening = load_whitening(cfg, device)
    gen = TextGenerator(
        cfg, codec, predictor, count_head, chunker, device=device, whitening=whitening
    )

    prompts, refs = [], []
    for p, _r, rf in load_task_pairs(args.task, split=args.split, limit=args.limit):
        prompts.append(p)
        refs.append(rf)

    scorer = SemanticScorer()

    def sample_fn(prompt: str, g: float) -> list[str]:
        return gen.sample_many(prompt, k=args.k, guidance_scale=g, steps=args.steps)

    curves = guidance_curve(prompts, refs, sample_fn, args.guidance, scorer)
    print(json.dumps(curves, indent=2))

    if args.show > 0:
        n_show = min(args.show, len(prompts))
        for g in args.guidance:
            print("\n" + "=" * 78)
            print(f"[examples] guidance={g}  (showing {n_show})")
            print("=" * 78)
            for i in range(n_show):
                samples = sample_fn(prompts[i], g)
                print(f"\n--- example {i} ---")
                print("  prompt   :", prompts[i])
                print("  ref      :", " | ".join(refs[i]) if refs[i] else "(none)")
                for j, s in enumerate(samples):
                    print(f"  sample {j} :", s)


if __name__ == "__main__":
    main()
