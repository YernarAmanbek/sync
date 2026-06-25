"""Probe 1 — information-presence oracles (pre-VAE diagnostic battery).

The capacity test, decoupled from any learned model. We ask: how well can the
held-out TARGET headline latent be predicted from the ARTICLE's latents by
non-parametric (k-NN) / linear (ridge) means? That bounds what is recoverable on
SONAR's q=1 substrate, independent of the flow predictor.

Everything is scored in RAW SONAR space, mean of per-example cosines, on the same
held-out set the in-loop metric / gate_latent uses — so the numbers are directly
comparable to the established held-out cosine (~0.459), marginal (~0.19), and
identity (~0.39).

    python -m Sync.scripts.probe_oracle --train-cache ./cache/gigaword \
        --heldout-cache ./cache/gigaword_val --heldout-limit 300

Reads the existing caches only (no SONAR, no whitening — raw latents throughout).

Interpretation (see the brief's decision matrix):
  * both oracles cap near 0.459 (within ~0.03)  -> info isn't recoverable from
    q=1 article latents -> CAPACITY CEILING confirmed -> build the custom VAE.
  * either oracle clearly beats 0.459 (>= ~0.55) -> the info IS present and the
    flow predictor is failing to extract it -> NOT capacity. A *linear* map
    beating the big nonlinear predictor is especially damning.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Cache -> raw representations (no whitening; q is averaged away)
# --------------------------------------------------------------------------- #
def load_reps(cache_dir: str, limit, device, chunk_rows: int = 100_000,
              want_lede: bool = True):
    """Build, from a Phase-2 latent cache:
      ctx_pool [num, d] : mean over q AND over the valid context chunks (article)
      ctx_lede [num, d] : chunk-0 of the context (article lede) — cross-check
      tgt0     [num, td]: chunk-0 of the target (headline), flattened over q
    All RAW SONAR latents. td == q*d (== d for SONAR q=1).
    """
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    num, N, M, q, d = meta["num"], meta["N_ctx"], meta["M"], meta["q"], meta["d"]
    n_arr = np.load(os.path.join(cache_dir, "n.npy"))
    ctx = np.memmap(os.path.join(cache_dir, "context.f32"), dtype="float32",
                    mode="r", shape=(num, N, q, d))
    tgt = np.memmap(os.path.join(cache_dir, "target.f32"), dtype="float32",
                    mode="r", shape=(num, M, q, d))
    use = num if limit is None else min(int(limit), num)

    ctx_pool = torch.empty(use, d, dtype=torch.float32)
    ctx_lede = torch.empty(use, d, dtype=torch.float32) if want_lede else None
    tgt0 = torch.empty(use, q * d, dtype=torch.float32)

    ar = torch.arange(N)
    for s in range(0, use, chunk_rows):
        e = min(s + chunk_rows, use)
        c = torch.from_numpy(np.array(ctx[s:e])).float()        # [b,N,q,d]
        cm = c.mean(dim=2)                                       # over q -> [b,N,d]
        nb = torch.from_numpy(n_arr[s:e].astype("int64"))       # [b]
        mask = (ar[None, :] < nb[:, None]).float()              # [b,N]
        denom = mask.sum(1, keepdim=True).clamp(min=1.0)
        ctx_pool[s:e] = (cm * mask[..., None]).sum(1) / denom
        if want_lede:
            ctx_lede[s:e] = cm[:, 0, :]
        t = torch.from_numpy(np.array(tgt[s:e, 0])).float()     # [b,q,d]
        tgt0[s:e] = t.reshape(e - s, -1)

    return {"ctx_pool": ctx_pool, "ctx_lede": ctx_lede, "tgt0": tgt0, "meta": meta}


def _mean_cos(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    return float(F.cosine_similarity(pred, tgt, dim=1).mean())


# --------------------------------------------------------------------------- #
# 1a — k-NN oracle (similarity-weighted target retrieval)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def knn_oracle(train_ctx, train_tgt, held_ctx, held_tgt, ks, device,
               chunk_rows: int = 100_000):
    """For each held-out article, retrieve its k nearest TRAIN articles by cosine
    over the context rep, predict the headline latent as the similarity-weighted
    mean of those train headline latents, score cosine to the true headline.
    Returns {k: mean_cosine}. Full 300xT search via chunked GPU matmul."""
    H = held_ctx.shape[0]
    T = train_ctx.shape[0]
    kmax = max(ks)

    htn = F.normalize(held_ctx.to(device), dim=1)               # [H,d]
    sims = torch.empty(H, T, device=device, dtype=torch.float32)
    for s in range(0, T, chunk_rows):
        e = min(s + chunk_rows, T)
        tn = F.normalize(train_ctx[s:e].to(device), dim=1)      # [c,d]
        sims[:, s:e] = htn @ tn.T

    vals, idx = sims.topk(kmax, dim=1)                          # [H,kmax]
    idx_cpu = idx.cpu()
    held_tgt_d = held_tgt.to(device)

    out = {}
    for k in ks:
        vk = vals[:, :k].clamp(min=0.0)                        # [H,k] cosine weights
        ik = idx_cpu[:, :k]
        tg = train_tgt[ik].to(device)                          # [H,k,td]
        wsum = vk.sum(1, keepdim=True)                         # [H,1]
        pred = (vk[..., None] * tg).sum(1) / wsum.clamp(min=1e-6)
        bad = wsum.squeeze(1) < 1e-6
        if bad.any():
            pred[bad] = tg[bad].mean(1)
        out[k] = _mean_cos(pred, held_tgt_d)
    return out


# --------------------------------------------------------------------------- #
# 1b — ridge (closed-form linear probe)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ridge_oracle(train_ctx, train_tgt, held_ctx, held_tgt, lambdas, device,
                 fit_n: int):
    """Closed-form ridge from context rep -> target latent, centered. Sweep the
    regularizer, return {lambda: mean_cosine} on held-out."""
    n = min(fit_n, train_ctx.shape[0])
    X = train_ctx[:n].to(device).double()                      # [n,d]
    Y = train_tgt[:n].to(device).double()                      # [n,td]
    xm = X.mean(0, keepdim=True)
    ym = Y.mean(0, keepdim=True)
    Xc, Yc = X - xm, Y - ym
    d = Xc.shape[1]
    XtX = Xc.T @ Xc                                            # [d,d]
    XtY = Xc.T @ Yc                                            # [d,td]
    I = torch.eye(d, device=device, dtype=torch.float64)
    Hx = (held_ctx.to(device).double() - xm)
    Ht = held_tgt.to(device).double()

    out = {}
    for lam in lambdas:
        W = torch.linalg.solve(XtX + lam * I, XtY)            # [d,td]
        pred = Hx @ W + ym
        out[lam] = float(F.cosine_similarity(pred.float(), Ht.float(), dim=1).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-cache", default="./cache/gigaword")
    ap.add_argument("--heldout-cache", default="./cache/gigaword_val")
    ap.add_argument("--heldout-limit", type=int, default=300,
                    help="held-out examples (match the gate's eval-n for comparability)")
    ap.add_argument("--knn-train-limit", type=int, default=None,
                    help="cap the k-NN train pool (default: full cache)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20, 100])
    ap.add_argument("--ridge-n", type=int, default=100_000)
    ap.add_argument("--ridge-lambdas", type=float, nargs="+",
                    default=[1.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--lede", action="store_true",
                    help="also run the chunk-0/lede context-rep cross-check (extra RAM)")
    ap.add_argument("--chunk-rows", type=int, default=100_000)
    ap.add_argument("--baseline", type=float, default=0.459,
                    help="established held-out flow cosine, for the verdict")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"loading held-out reps from {args.heldout_cache} (limit={args.heldout_limit}) ...")
    held = load_reps(args.heldout_cache, args.heldout_limit, device,
                     args.chunk_rows, want_lede=True)
    print(f"loading train reps from {args.train_cache} "
          f"(limit={args.knn_train_limit or 'all'}) ... this reads the memmap")
    train = load_reps(args.train_cache, args.knn_train_limit, device,
                      args.chunk_rows, want_lede=args.lede)

    Htgt = held["tgt0"]
    Ttgt = train["tgt0"]
    print(f"  train pool: {Ttgt.shape[0]}  held-out: {Htgt.shape[0]}  "
          f"dim(ctx)={held['ctx_pool'].shape[1]} dim(tgt)={Htgt.shape[1]}")

    # context-rep baselines (raw space), to anchor the oracle numbers
    identity = _mean_cos(held["ctx_lede"], Htgt)               # article lede vs headline
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(Ttgt.shape[0], generator=g)[:Htgt.shape[0]]
    marginal = _mean_cos(Ttgt[perm], Htgt)                     # random train headline

    print("\nrunning k-NN oracle (pooled context rep) ...")
    knn_pool = knn_oracle(train["ctx_pool"], Ttgt, held["ctx_pool"], Htgt,
                          args.ks, device, args.chunk_rows)
    knn_lede = None
    if args.lede and train["ctx_lede"] is not None:
        print("running k-NN oracle (lede context rep, cross-check) ...")
        knn_lede = knn_oracle(train["ctx_lede"], Ttgt, held["ctx_lede"], Htgt,
                              args.ks, device, args.chunk_rows)

    print("running ridge oracle (pooled context rep) ...")
    ridge_pool = ridge_oracle(train["ctx_pool"], Ttgt, held["ctx_pool"], Htgt,
                              args.ridge_lambdas, device, args.ridge_n)

    best_knn = max(knn_pool.values())
    best_ridge = max(ridge_pool.values())
    best_oracle = max(best_knn, best_ridge)

    print("\n" + "=" * 70)
    print("PROBE 1 — INFORMATION-PRESENCE ORACLES (raw SONAR space, mean cosine)")
    print("=" * 70)
    print(f"  baselines:  marginal {marginal:.4f}   identity {identity:.4f}   "
          f"flow-predictor (established) {args.baseline:.4f}")
    print("\n  1a  k-NN oracle (pooled context):")
    for k in args.ks:
        print(f"        k={k:<4d} -> {knn_pool[k]:.4f}")
    if knn_lede is not None:
        print("      k-NN oracle (lede context, cross-check):")
        for k in args.ks:
            print(f"        k={k:<4d} -> {knn_lede[k]:.4f}")
    print("\n  1b  ridge oracle (pooled context):")
    for lam in args.ridge_lambdas:
        print(f"        lambda={lam:<8g} -> {ridge_pool[lam]:.4f}")

    print("\n  best k-NN   :", f"{best_knn:.4f}")
    print("  best ridge  :", f"{best_ridge:.4f}")
    print("  best oracle :", f"{best_oracle:.4f}")

    eps = 0.03
    delta = best_oracle - args.baseline
    print("=" * 70)
    if best_oracle >= args.baseline + 0.09:
        verdict = (
            f"INFO IS PRESENT — best oracle {best_oracle:.3f} beats the flow "
            f"predictor's {args.baseline:.3f} by {delta:+.3f}. The headline IS "
            "recoverable from SONAR's q=1 article latents; the flow predictor is "
            "leaving accuracy on the table. NOT a capacity ceiling — investigate "
            "the predictor/objective (run Probe 2 to confirm regression vs flow). "
            "A linear (ridge) win here would be especially damning."
        )
    elif delta <= eps:
        verdict = (
            f"CAPACITY-CEILING SIGNAL — best oracle {best_oracle:.3f} is within "
            f"{eps:.2f} of the flow predictor's {args.baseline:.3f}. Even a k-NN "
            "over 1M examples and a linear map can't extract a better headline "
            "from q=1 article latents. The information isn't recoverable on this "
            "substrate -> the custom q>1 VAE is justified. (Confirm with Probe 2 "
            "capping near 0.46 too.)"
        )
    else:
        verdict = (
            f"AMBIGUOUS — best oracle {best_oracle:.3f} edges the flow predictor's "
            f"{args.baseline:.3f} by {delta:+.3f}, but below the ~0.55 'clearly "
            "present' line. Run Probe 2 (direct regression) to disambiguate "
            "objective-vs-capacity before deciding on the VAE."
        )
    print("VERDICT:", verdict)
    print("=" * 70)


if __name__ == "__main__":
    main()
