"""Decoder-free held-out latent metric — memorization vs generalization.

The decoder is NEVER called here. We measure the predictor purely in SONAR latent
space: for a prompt, sample a target latent zhat, and compare it (cosine) to the
"correct" latent z* = SONAR.encode(reference). We report this on TRAIN prompts vs
HELD-OUT prompts; the gap is the memorization signal. Two baselines (marginal,
identity), both on the held-out set in raw SONAR space, calibrate the raw cosine.

    python -m Sync.scripts.gate_latent --task gigaword --ckpt runs/predictor_final.pt \
        --train-split train --heldout-split validation --limit 300 --k 1

Decision (see the brief):
  - held-out ~= marginal           -> MEMORIZATION  (scale data)
  - held-out clearly > marginal,
    and >= identity                -> GENERALIZES   (architecture validated)
  - held-out > marginal but ~= identity -> WEAK conditional value (SONAR substrate?)

Correctness invariants enforced below:
  * zhat is un-whitened (invert) before comparing; z* is raw encode() — BOTH raw.
  * decoder is never imported/called.
  * train/held-out splits are disjoint (train slice vs validation).
  * matched sampling settings (guidance/steps/seed/count) across both sets.
  * per-example cosines on L2-normalized vectors, then averaged.
"""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F

from ..codec import SonarCodecAdapter
from ..components import expand_chunk_mask
from ..config import get_preset
from ..data import Chunker, load_task_pairs
from ..predictor import CountHead, FlowMatchingPredictor, FlowSampler
from ..training import EmaModel, load_ckpt, load_whitening


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gigaword")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--heldout-split", default="validation")
    ap.add_argument("--limit", type=int, default=300, help="examples per split (matched)")
    ap.add_argument("--k", type=int, default=1, help="samples per prompt (avg cosine)")
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=50, help="ODE steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    cfg = get_preset(args.task)
    cfg.validate()
    pcfg = cfg.predictor
    N_ctx, M, q, d = pcfg.n_ctx_chunks, pcfg.n_tgt_chunks, pcfg.latents_per_chunk, pcfg.latent_dim

    codec = SonarCodecAdapter(cfg.codec, cfg.tokenizer, device=device)
    chunker = Chunker(cfg.chunk)
    predictor = FlowMatchingPredictor(pcfg)
    count_head = CountHead(pcfg)

    ck = load_ckpt(args.ckpt, map_location="cpu")
    predictor.load_state_dict(ck["predictor"])
    count_head.load_state_dict(ck["count_head"])
    if not args.no_ema and "ema" in ck:
        ema = EmaModel(predictor, ck["ema"]["decay"])
        ema.load_state_dict(ck["ema"])
        ema.copy_to(predictor)
        print("loaded EMA weights into predictor")
    predictor.to(device).eval()
    count_head.to(device).eval()

    whitening = load_whitening(cfg, device)
    sampler = FlowSampler(predictor, count_head, pcfg)

    @torch.no_grad()
    def encode_chunk0(text: str):
        """raw SONAR latent of the FIRST chunk of `text` -> [q*d] or None."""
        chunks = chunker.chunk(text)[:M]
        if not chunks:
            return None
        z = codec.encode_texts(chunks).to(device)        # [c, q, d] raw
        return z[0].reshape(-1)                            # chunk 0 -> [q*d]

    @torch.no_grad()
    def predict_latents(prompt: str, base_seed: int):
        """k predicted target latents (chunk 0), un-whitened to raw SONAR space."""
        chunks = chunker.chunk(prompt)[:N_ctx]
        if not chunks:
            return []
        C_un = codec.encode_texts(chunks).to(device)      # [n, q, d] raw
        C_w = whitening.apply(C_un)                        # -> whitened (training space)
        n = C_w.shape[0]
        C = torch.zeros(1, N_ctx, q, d, device=device)
        C[0, :n] = C_w
        ctx_mask = torch.zeros(1, N_ctx, dtype=torch.bool, device=device)
        ctx_mask[0, :n] = True
        ctx_tok_mask = expand_chunk_mask(ctx_mask, q)
        C_flat = C.reshape(1, N_ctx * q, d)

        outs = []
        for j in range(args.k):
            gen = torch.Generator(device=device).manual_seed(base_seed + j)
            Z_w, _m = sampler.sample(
                C_flat, ctx_tok_mask, steps=args.steps,
                guidance_scale=args.guidance, generator=gen,
            )
            Z_raw = whitening.invert(Z_w).reshape(M, q, d)  # back to raw SONAR space
            outs.append(Z_raw[0].reshape(-1))               # chunk 0 -> [q*d]
        return outs

    def conditional_cosines(pairs, label):
        cond, refs_lat = [], []
        for i, (prompt, response, refs) in enumerate(pairs):
            ref_text = (refs[0] if refs else response)
            zstar = encode_chunk0(ref_text)
            if zstar is None:
                continue
            refs_lat.append(zstar)
            zk = predict_latents(prompt, base_seed=args.seed + i * max(1, args.k))
            if not zk:
                continue
            cond.append(sum(_cos(z, zstar) for z in zk) / len(zk))
        print(f"  [{label}] usable={len(cond)}")
        return cond, refs_lat

    print(f"\nsettings: guidance={args.guidance} steps={args.steps} k={args.k} "
          f"seed={args.seed} limit={args.limit}  (decoder bypassed)")

    train_pairs = list(load_task_pairs(args.task, split=args.train_split, limit=args.limit))
    heldout_pairs = list(load_task_pairs(args.task, split=args.heldout_split, limit=args.limit))

    print("computing TRAIN conditional ...")
    train_cond, train_ref_lat = conditional_cosines(train_pairs, "train")
    print("computing HELD-OUT conditional + identity ...")
    heldout_cond, heldout_ref_lat = [], []
    identity = []
    for i, (prompt, response, refs) in enumerate(heldout_pairs):
        ref_text = (refs[0] if refs else response)
        zstar = encode_chunk0(ref_text)
        if zstar is None:
            continue
        heldout_ref_lat.append(zstar)
        zk = predict_latents(prompt, base_seed=args.seed + i * max(1, args.k))
        if zk:
            heldout_cond.append(sum(_cos(z, zstar) for z in zk) / len(zk))
        zident = encode_chunk0(prompt)                     # encode(first sentence of prompt)
        if zident is not None:
            identity.append(_cos(zident, zstar))

    # marginal baseline: random TRAIN target latent vs held-out z*, raw space
    rng = random.Random(args.seed)
    marginal = []
    if train_ref_lat:
        for zstar in heldout_ref_lat:
            zr = train_ref_lat[rng.randrange(len(train_ref_lat))]
            marginal.append(_cos(zr, zstar))

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    tc, hc, mg, idn = mean(train_cond), mean(heldout_cond), mean(marginal), mean(identity)

    print("\n" + "=" * 70)
    print("DECODER-FREE LATENT METRIC (raw SONAR space, mean cosine)")
    print("=" * 70)
    print(f"  train   conditional cos : {tc:.4f}  (n={len(train_cond)})")
    print(f"  heldout conditional cos : {hc:.4f}  (n={len(heldout_cond)})")
    print(f"  marginal baseline       : {mg:.4f}  (random train target vs heldout z*)")
    print(f"  identity baseline       : {idn:.4f}  (prompt lede vs heldout z*)")
    print(f"  --- train - heldout gap : {tc - hc:+.4f}")
    print(f"  --- heldout - marginal  : {hc - mg:+.4f}")
    print(f"  --- heldout - identity  : {hc - idn:+.4f}")

    # decision rule
    eps = 0.03
    if (hc - mg) < eps:
        verdict = ("MEMORIZATION — held-out no better than ignoring the prompt. "
                   "Next: scale data (500k-1M pairs, epochs 5-30, keep LR floor).")
    elif (hc - idn) > eps:
        verdict = ("GENERALIZES — held-out beats both marginal and identity. "
                   "Architecture validated at this scale; proceed to the scaled run.")
    else:
        verdict = ("WEAK conditional value — beats marginal but ~= identity (echoes "
                   "the input). First evidence SONAR's q=1 space may be a poor "
                   "substrate -> consider the custom VAE. Not a quit signal.")
    print("\nVERDICT:", verdict)
    print("=" * 70)


if __name__ == "__main__":
    main()
