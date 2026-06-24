"""Reconciliation gate — in-loop (cache-based) latent metric vs standalone
SONAR-live computation on ONE checkpoint. Run BEFORE the long training run.

The in-loop metric reads z*/identity from the cache; the original gate_latent
computed them by calling SONAR live. They must agree, or the cache path (dtype,
whitening space, chunk-0 identity, split alignment) is subtly wrong and every
[eval step ...] line in the long run would be untrustworthy. This asserts the two
agree within tolerance and exits non-zero if not.

    # build a small held-out cache first (raw latents; reuses train whitening):
    #   python -m Sync.scripts.precompute_latents --task gigaword --split validation \
    #       --limit 300 --cache-dir ./cache/gigaword_val --no-whiten
    python -m Sync.scripts.reconcile_latent --task gigaword \
        --ckpt runs/predictor_40000.pt --heldout-cache-dir ./cache/gigaword_val --limit 300

Note: this is a MEAN-level reconciliation over `limit` examples. The conditional
cosine carries sampler noise (per-example seeding differs between the batched
cache path and the per-example live path), but the mean over a few hundred
examples agrees tightly. The deterministic `identity` baseline is the strongest
faithfulness signal — it should match to ~1e-2.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import torch
import torch.nn.functional as F

from ..codec import SonarCodecAdapter
from ..components import expand_chunk_mask
from ..config import get_preset
from ..data import Chunker, PairLatentDataset, load_task_pairs
from ..predictor import CountHead, FlowMatchingPredictor, FlowSampler
from ..training import EmaModel, _latent_metric_from_cache, load_ckpt, load_whitening


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gigaword")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heldout-cache-dir", required=True)
    ap.add_argument("--heldout-split", default="validation")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.03)
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

    # ---- cache side (the in-loop metric, verbatim) -------------------------
    train_ds = PairLatentDataset(cfg.data, pcfg)
    heldout_ds = PairLatentDataset(replace(cfg.data, latent_cache_dir=args.heldout_cache_dir), pcfg)
    cache_met = _latent_metric_from_cache(
        predictor, count_head, whitening, train_ds, heldout_ds,
        pcfg, args.limit, args.guidance, args.steps, args.seed, device,
    )

    # ---- live side (SONAR), same first `limit` validation prompts ----------
    sampler = FlowSampler(predictor, count_head, pcfg)

    @torch.no_grad()
    def enc0(text: str):
        ch = chunker.chunk(text)[:M]
        if not ch:
            return None
        return codec.encode_texts(ch).to(device)[0].reshape(-1)

    cond, ident = [], []
    for prompt, response, refs in load_task_pairs(args.task, split=args.heldout_split, limit=args.limit):
        zstar = enc0(refs[0] if refs else response)
        if zstar is None:
            continue
        chunks = chunker.chunk(prompt)[:N_ctx]
        if not chunks:
            continue
        C_un = codec.encode_texts(chunks).to(device)
        C_w = whitening.apply(C_un)
        n = C_w.shape[0]
        C = torch.zeros(1, N_ctx, q, d, device=device)
        C[0, :n] = C_w
        cmask = torch.zeros(1, N_ctx, dtype=torch.bool, device=device)
        cmask[0, :n] = True
        ctm = expand_chunk_mask(cmask, q)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        Zw, _m = sampler.sample(
            C.reshape(1, N_ctx * q, d), ctm,
            steps=args.steps, guidance_scale=args.guidance, generator=gen,
        )
        zhat = whitening.invert(Zw).reshape(M, q, d)[0].reshape(-1)
        cond.append(float(F.cosine_similarity(zhat, zstar, dim=0)))
        zident = enc0(prompt)
        if zident is not None:
            ident.append(float(F.cosine_similarity(zident, zstar, dim=0)))

    live_cond = sum(cond) / max(1, len(cond))
    live_ident = sum(ident) / max(1, len(ident))

    print("\n                 cache      live     |diff|")
    def row(name, a, b):
        print(f"  {name:14s} {a:8.4f} {b:8.4f} {abs(a - b):8.4f}")
    row("heldout_cos", cache_met["heldout_cos"], live_cond)
    row("identity", cache_met["identity"], live_ident)
    print(f"  marginal(cache) {cache_met['marginal']:.4f}   train_cos(cache) {cache_met['train_cos']:.4f}")

    d_cond = abs(cache_met["heldout_cos"] - live_cond)
    d_ident = abs(cache_met["identity"] - live_ident)
    ok = (d_cond <= args.tol) and (d_ident <= args.tol)
    print(f"\n{'PASS' if ok else 'FAIL'}: heldout_cos diff {d_cond:.4f}, identity diff {d_ident:.4f} "
          f"(tol {args.tol})")
    if not ok:
        print("  -> cache path diverges from SONAR-live; DO NOT launch the long run. "
              "Check cache dtype / whitening space / chunk-0 / split alignment.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
