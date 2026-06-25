"""Probe 3 — decode the predictor's own held-out latents (pre-VAE battery).

We have never run the predictor's HELD-OUT latents through the proven-robust SONAR
decoder at the LOWEST guidance (least off-manifold) with decode hygiene — only
earlier degenerate samples at mixed/high guidance. This reads what the established
~0.459 held-out cosine actually sounds like as text.

Method (matches the brief): load predictor_best.pt (EMA weights), encode each
held-out PROMPT live, sample target latents at guidance 1.0 / steps 50, un-whiten,
decode chunk-0 via SonarCodecAdapter.decode_latents with no_repeat_ngram_size=3
and repetition_penalty=1.5, and dump (prompt, reference, decoded) triples. Also
reports the per-example latent cosine (predicted chunk-0 vs SONAR.encode(reference))
so the text is anchored to the same number the gate tracks.

    python -m Sync.scripts.probe_decode --task gigaword \
        --ckpt runs/predictor_best.pt --limit 20

Interpretation:
  * coherent, on-topic, loosely-correct headlines -> 0.459 is a weak-but-WORKING
    summarizer, not garbage -> the VAE becomes an *improvement*, not a rescue.
  * still token-salad at guidance 1.0 WITH the guards -> 0.459 is genuinely too
    low for usable output -> better latents needed regardless.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from ..codec import SonarCodecAdapter
from ..components import expand_chunk_mask
from ..config import get_preset
from ..data import Chunker, load_task_pairs
from ..metrics import SemanticScorer
from ..predictor import CountHead, FlowMatchingPredictor, FlowSampler
from ..training import EmaModel, load_ckpt, load_whitening
from .probe_regress import DirectRegressor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="gigaword")
    ap.add_argument("--ckpt", default="runs/predictor_best.pt")
    ap.add_argument("--model-type", choices=["auto", "flow", "regress"], default="auto",
                    help="auto-detect from checkpoint keys (predictor=flow, model=regress)")
    ap.add_argument("--space", choices=["whitened", "raw"], default="whitened",
                    help="regressor only: geometry it was TRAINED in (must match)")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=20, help="prompts to decode + dump")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="flow only: lowest = least off-manifold")
    ap.add_argument("--steps", type=int, default=50, help="flow only: ODE steps")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=3)
    ap.add_argument("--repetition-penalty", type=float, default=1.5)
    ap.add_argument("--max-seq-len", type=int, default=64)
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
    whitening = load_whitening(cfg, device)

    ck = load_ckpt(args.ckpt, map_location="cpu")
    model_type = args.model_type
    if model_type == "auto":
        model_type = "regress" if "model" in ck and "predictor" not in ck else "flow"
    print(f"checkpoint {args.ckpt} -> model_type={model_type}")

    if model_type == "flow":
        predictor = FlowMatchingPredictor(pcfg)
        count_head = CountHead(pcfg)
        predictor.load_state_dict(ck["predictor"])
        count_head.load_state_dict(ck["count_head"])
        if not args.no_ema and "ema" in ck:
            ema = EmaModel(predictor, ck["ema"]["decay"])
            ema.load_state_dict(ck["ema"])
            ema.copy_to(predictor)
            print("loaded EMA weights into predictor")
        predictor.to(device).eval()
        count_head.to(device).eval()
        sampler = FlowSampler(predictor, count_head, pcfg)
    else:
        regressor = DirectRegressor(pcfg)
        regressor.load_state_dict(ck["model"])
        if not args.no_ema and "ema" in ck:
            ema = EmaModel(regressor, ck["ema"]["decay"])
            ema.load_state_dict(ck["ema"])
            ema.copy_to(regressor)
            print("loaded EMA weights into regressor")
        regressor.to(device).eval()

    @torch.no_grad()
    def encode_chunk0(text: str):
        chunks = chunker.chunk(text)[:M]
        if not chunks:
            return None
        z = codec.encode_texts(chunks).to(device)      # [c, q, d] raw
        return z[0].reshape(-1)                          # [q*d]

    @torch.no_grad()
    def _encode_context(prompt: str):
        """prompt -> (whitened-or-raw context canvas [1,N_ctx*q,d], ctx token mask) or None."""
        chunks = chunker.chunk(prompt)[:N_ctx]
        if not chunks:
            return None
        C_un = codec.encode_texts(chunks).to(device)    # [n,q,d] raw
        n = C_un.shape[0]
        C = torch.zeros(1, N_ctx, q, d, device=device)
        C[0, :n] = C_un if (model_type == "regress" and args.space == "raw") else whitening.apply(C_un)
        ctx_mask = torch.zeros(1, N_ctx, dtype=torch.bool, device=device)
        ctx_mask[0, :n] = True
        ctm = expand_chunk_mask(ctx_mask, q)
        return C.reshape(1, N_ctx * q, d), ctm

    @torch.no_grad()
    def predict_chunk0(prompt: str, seed: int):
        """raw SONAR chunk-0 latent predicted for `prompt`, or None."""
        enc = _encode_context(prompt)
        if enc is None:
            return None
        C_flat, ctm = enc
        if model_type == "flow":
            gen = torch.Generator(device=device).manual_seed(seed)
            Zw, _m = sampler.sample(
                C_flat, ctm, steps=args.steps,
                guidance_scale=args.guidance, generator=gen,
            )
            Zraw = whitening.invert(Zw).reshape(M, q, d)
        else:
            ttm = torch.ones(1, M * q, dtype=torch.bool, device=device)
            pred = regressor(C_flat, ctm, ttm)          # [1, M*q, d]
            pred = pred if args.space == "raw" else whitening.invert(pred)
            Zraw = pred.reshape(M, q, d)
        return Zraw[0]                                   # [q, d]

    if model_type == "flow":
        print(f"\nsettings: flow guidance={args.guidance} steps={args.steps} "
              f"no_repeat_ngram={args.no_repeat_ngram_size} rep_penalty={args.repetition_penalty}")
    else:
        print(f"\nsettings: regressor (deterministic mean) space={args.space} "
              f"no_repeat_ngram={args.no_repeat_ngram_size} rep_penalty={args.repetition_penalty}")

    pairs = list(load_task_pairs(args.task, split=args.split, limit=args.limit))
    prompts, refs, zhat_list, ref_lat = [], [], [], []
    for i, (prompt, response, rf) in enumerate(pairs):
        z0 = predict_chunk0(prompt, args.seed + i)
        if z0 is None:
            continue
        ref_text = rf[0] if rf else response
        zstar = encode_chunk0(ref_text)
        if zstar is None:
            continue
        prompts.append(prompt)
        refs.append(rf if rf else [response])
        zhat_list.append(z0)
        ref_lat.append(zstar)

    if not zhat_list:
        print("no decodable prompts.")
        return

    # decode predicted chunk-0 latents (batched) with the hygiene guards
    Z = torch.stack(zhat_list, dim=0)                   # [B, q, d] raw
    decoded = codec.decode_latents(
        Z, batch_size=64, max_seq_len=args.max_seq_len,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
    )

    # latent cosine (predicted chunk-0 vs encode(reference)) — ties text to 0.459
    zhat = torch.stack([z.reshape(-1) for z in zhat_list], dim=0)
    zstar = torch.stack(ref_lat, dim=0)
    lat_cos = F.cosine_similarity(zhat, zstar, dim=1)

    # independent semantic similarity of decoded text to references
    scorer = SemanticScorer()
    first_ref = [r[0] for r in refs]
    sem = scorer.cos(decoded, first_ref)

    expected = 0.578 if model_type == "regress" else 0.459
    head = ("DECODED REGRESSOR MEAN LATENTS" if model_type == "regress"
            else "DECODED FLOW PREDICTOR LATENTS")
    print("\n" + "=" * 78)
    print(f"PROBE 3 — {head} (held-out, guarded)")
    print("=" * 78)
    for i in range(len(decoded)):
        print(f"\n--- example {i} ---")
        print("  prompt   :", prompts[i][:200])
        print("  ref      :", " | ".join(refs[i])[:200])
        print("  decoded  :", decoded[i][:200])
        print(f"  latent_cos {float(lat_cos[i]):+.3f}   sem_sim {float(sem[i]):+.3f}")

    print("\n" + "=" * 78)
    print(f"  mean latent cosine (vs {expected:.3f} expected) : {float(lat_cos.mean()):.4f}  "
          f"(n={len(decoded)})")
    print(f"  mean decoded sem-sim to reference      : {float(sem.mean()):.4f}")
    print("=" * 78)
    if model_type == "regress":
        print("Read (the blurry-mean check): clean, on-topic (if slightly generic) "
              "headlines clearly better than flow's salad -> mean is a sound base for "
              "the hybrid. Bland/hedged DESPITE higher cosine -> averaging landed in a "
              "blurry region; the residual flow is needed for coherence, not just "
              "diversity (weight s accordingly).")
    print("Read: coherent/on-topic -> 0.459 is weak-but-working (VAE = improvement). "
          "Token-salad even here -> 0.459 too low for usable output (need better latents).")


if __name__ == "__main__":
    main()
