"""Whitening round-trip + decoder-tolerance break-test (predictor-free).

Adjudicates the latent->text path after the decoder-free latent metric showed the
predictor GENERALIZES (held-out cos >> marginal). If predicted latents are
directionally right yet text is garbage, the failure is in the path. This probes:

  PART A — whitening inverse sanity (pure math, no decode):
      invert(apply(x)) must == x. If not, W_inv@W != I and we decode from the
      wrong space — a bug that alone explains good-loss / garbage-text.

  PART B — round-trip decode:
      decode(encode(ref))            (the ceiling path)   vs
      decode(invert(apply(encode(ref))))  (whiten round-trip)
      must produce the same text. If round-trip text degrades, whitening breaks
      decode even on REAL latents.

  PART C — noise-tolerance break-test (the calibration that matters):
      add noise of scale sigma in whitened space, un-whiten, decode. Report, per
      sigma, the raw-space cosine to the clean latent AND the decode quality /
      degeneracy. This maps "latent cosine -> decodability", so we can read off
      whether the predictor's held-out cosine (~0.49) is anywhere near the
      fidelity SONAR's decoder needs.

    python -m Sync.scripts.gate_whitening --task gigaword --limit 50
"""

from __future__ import annotations

import argparse

import torch

from ..codec import SonarCodecAdapter
from ..config import get_preset
from ..data import Chunker, load_task_pairs
from ..metrics import SemanticScorer
from ..training import load_whitening


def _degeneracy(texts: list[str]) -> tuple[float, float]:
    """(repetition_ratio, placeholder_frac) averaged. Higher = worse.
    repetition_ratio = 1 - unique_tokens/total_tokens; placeholder_frac = share of
    tokens made only of '#'/'%' (the gigaword digit placeholder floods)."""
    reps, holds = [], []
    for t in texts:
        toks = t.split()
        if not toks:
            reps.append(1.0)
            holds.append(1.0)
            continue
        reps.append(1.0 - len(set(toks)) / len(toks))
        holds.append(sum(1 for w in toks if set(w) <= set("#%")) / len(toks))
    n = max(1, len(reps))
    return sum(reps) / n, sum(holds) / n


def _mean_cos_raw(a: torch.Tensor, b: torch.Tensor) -> float:
    """mean per-row cosine over [N, ..., d] tensors (flatten trailing dims)."""
    a2 = a.reshape(a.shape[0], -1)
    b2 = b.reshape(b.shape[0], -1)
    return float(torch.nn.functional.cosine_similarity(a2, b2, dim=1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gigaword")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--noise", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=4)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    cfg = get_preset(args.task)
    cfg.validate()

    codec = SonarCodecAdapter(cfg.codec, cfg.tokenizer, device=device)
    chunker = Chunker(cfg.chunk)
    whitening = load_whitening(cfg, device)
    scorer = SemanticScorer()

    # collect reference headlines (single sentences on gigaword)
    refs: list[str] = []
    for _p, response, rf in load_task_pairs(args.task, split=args.split, limit=args.limit):
        text = (rf[0] if rf else response)
        c = chunker.chunk(text)
        if c:
            refs.append(c[0])
    if not refs:
        raise RuntimeError("no references collected")
    print(f"collected {len(refs)} reference sentences\n")

    Z = codec.encode_texts(refs).to(device)              # [N, 1, d] raw
    Zw = whitening.apply(Z)                                # whitened
    Zrt = whitening.invert(Zw)                             # round-trip back to raw

    # ---- PART A: inverse sanity (math only) --------------------------------
    abs_err = float((Z - Zrt).abs().max())
    rel_err = float((Z - Zrt).norm() / Z.norm().clamp(min=1e-12))
    cos_rt = _mean_cos_raw(Z, Zrt)
    print("=" * 70)
    print("PART A — whitening inverse sanity:  invert(apply(x)) ?= x")
    print("=" * 70)
    print(f"  max abs error : {abs_err:.3e}")
    print(f"  rel L2 error  : {rel_err:.3e}")
    print(f"  mean cosine   : {cos_rt:.6f}")
    print("  -> expect error ~1e-5 and cosine ~1.0 (W_inv@W == I). "
          "If not, whitening is the bug.")

    # ---- PART B: round-trip decode -----------------------------------------
    text_raw = codec.decode_latents(Z)                    # encode->decode (ceiling)
    text_rt = codec.decode_latents(Zrt)                   # encode->whiten->unwhiten->decode
    sim_raw = float(scorer.cos(text_raw, refs).mean())
    sim_rt = float(scorer.cos(text_rt, refs).mean())
    sim_raw_vs_rt = float(scorer.cos(text_raw, text_rt).mean())
    print("\n" + "=" * 70)
    print("PART B — round-trip decode (real latents)")
    print("=" * 70)
    print(f"  sem-sim  decode(encode)            vs ref : {sim_raw:.4f}  (ceiling)")
    print(f"  sem-sim  decode(roundtrip)         vs ref : {sim_rt:.4f}")
    print(f"  sem-sim  decode(encode) vs decode(roundtrip): {sim_raw_vs_rt:.4f}  "
          "(should be ~1.0)")

    # ---- PART C: noise-tolerance break-test --------------------------------
    print("\n" + "=" * 70)
    print("PART C — noise-tolerance break-test (whitened-space noise -> decode)")
    print("=" * 70)
    print(f"  {'sigma':>6} {'raw-cos':>9} {'sem-sim':>9} {'rep-ratio':>10} {'#-frac':>8}")
    rows = []
    for sigma in args.noise:
        if sigma == 0.0:
            Zn = Zrt
        else:
            noise = torch.randn(Zw.shape, generator=torch.Generator(device=device).manual_seed(args.seed), device=device)
            Zn = whitening.invert(Zw + sigma * noise)
        raw_cos = _mean_cos_raw(Z, Zn)
        texts = codec.decode_latents(Zn)
        sem = float(scorer.cos(texts, refs).mean())
        rep, hold = _degeneracy(texts)
        rows.append((sigma, raw_cos, sem, rep, hold, texts))
        print(f"  {sigma:6.2f} {raw_cos:9.4f} {sem:9.4f} {rep:10.4f} {hold:8.4f}")

    print("\n  Interpretation: find the raw-cos at which sem-sim collapses / "
          "rep-ratio & #-frac spike. That is the latent fidelity SONAR decode "
          "REQUIRES. Compare to the predictor's held-out cos (~0.49).")

    # show a few decodes at each noise level for eyeballing
    if args.show > 0:
        for sigma, raw_cos, sem, rep, hold, texts in rows:
            print(f"\n--- sigma={sigma} (raw-cos={raw_cos:.3f}) ---")
            for t in texts[:args.show]:
                print("   ", t[:160])


if __name__ == "__main__":
    main()
