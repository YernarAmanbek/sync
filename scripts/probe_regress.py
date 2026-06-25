"""Probe 2 — direct-regression baseline (pre-VAE diagnostic battery).

The OBJECTIVE test: does flow matching leave point-accuracy on the table versus
plain regression? We reuse the predictor backbone but strip the generative parts —
no time embedding, a learned target-query replaces z_t, cross-attention to the
context is kept — and train it to predict the target latent DIRECTLY with a
1-cosine (or MSE) loss on the same 1M cache, same LR floor.

Why regression is the right *diagnostic* here even though we rejected it as a
*product*: as a product, MSE/regression mean-collapses and loses diversity. As a
measurement, the conditional mean is close to the right answer on a near-
deterministic task like Gigaword, so it reads the best point-accuracy the
architecture can reach — exactly what we want to compare against flow's 0.459.

Scored with the same decoder-free held-out cosine the gate uses (raw SONAR space).

    python -m Sync.scripts.probe_regress --task gigaword \
        --heldout-cache-dir ./cache/gigaword_val \
        --max-steps 30000 --lr 1.5e-4 --min-lr-ratio 0.1 --val-every 2000

Interpretation:
  * regressor beats 0.459 meaningfully -> flow matching was underperforming ->
    switch to a regression/hybrid objective; the substrate is fine, no VAE needed
    for accuracy.
  * regressor also caps ~0.459 -> the objective isn't the bottleneck (consistent
    with the capacity ceiling).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..components import (
    ChunkAwarePositionalEmbedding,
    TransformerStack,
    expand_chunk_mask,
)
from ..config import PredictorConfig, get_preset
from ..data import PairLatentDataset, collate_pairs
from ..training import (
    EmaModel,
    load_whitening,
    lr_lambda,
    make_optimizer,
    save_ckpt,
)


# --------------------------------------------------------------------------- #
# Direct regressor — predictor backbone minus the flow machinery
# --------------------------------------------------------------------------- #
class DirectRegressor(nn.Module):
    """Learned target-query -> cross-attend to context -> predict target latent.
    No time embedding, no noise, no CFG. Same backbone width/depth as the flow
    predictor so the comparison is apples-to-apples on capacity."""

    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg
        self.q = cfg.latents_per_chunk
        self.M = cfg.n_tgt_chunks
        self.d = cfg.latent_dim
        n_query = self.M * self.q
        self.query = nn.Parameter(torch.randn(1, n_query, cfg.d_model) * 0.02)
        self.ctx_proj = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.tgt_pos = ChunkAwarePositionalEmbedding(cfg.n_tgt_chunks, cfg.latents_per_chunk, cfg.d_model)
        self.ctx_pos = ChunkAwarePositionalEmbedding(cfg.n_ctx_chunks, cfg.latents_per_chunk, cfg.d_model)
        self.backbone = TransformerStack(
            cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.ffn_mult, cfg.dropout, cross_attn=True
        )
        self.out_proj = nn.Linear(cfg.d_model, cfg.latent_dim)

    def forward(self, context, context_mask, target_mask):
        B = context.shape[0]
        h = self.tgt_pos(self.query.expand(B, -1, -1))      # [B, M*q, d_model]
        ctx = self.ctx_pos(self.ctx_proj(context))          # [B, N*q, d_model]
        h = self.backbone(h, self_mask=target_mask, context=ctx, context_mask=context_mask)
        return self.out_proj(h)                              # [B, M*q, d] (whitened space)


@torch.no_grad()
def direct_metric(model, whitening, train_ds, heldout_ds, pcfg, n, device,
                  space: str = "whitened") -> dict:
    """Decoder-free held-out cosine for the regressor (deterministic; no sampler).
    Same construction as the gate's latent metric so numbers are comparable.

    `space="raw"` skips whitening entirely (the model lives in raw SONAR space),
    matching the geometry ridge/k-NN are scored in — used to test whether the
    train/eval geometry mismatch is what caps the whitened-space model."""
    q, M, d = pcfg.latents_per_chunk, pcfg.n_tgt_chunks, pcfg.latent_dim

    def gather(ds):
        return collate_pairs([ds[i] for i in range(min(n, len(ds)))])

    def cond(batch):
        ctx = batch["context"].to(device)                   # [B,N,q,d] raw
        tgt = batch["target"].to(device)                    # [B,M,q,d] raw
        cmask = batch["context_mask"].to(device)
        B, N = ctx.shape[0], ctx.shape[1]
        ctx_in = (ctx if space == "raw" else whitening.apply(ctx)).reshape(B, N * q, d)
        ctm = expand_chunk_mask(cmask, q)
        ttm = torch.ones(B, M * q, dtype=torch.bool, device=device)  # predict full canvas
        pred = model(ctx_in, ctm, ttm)
        pred_raw = (pred if space == "raw" else whitening.invert(pred)).reshape(B, M, q, d)
        zhat0 = pred_raw[:, 0].reshape(B, -1)
        zstar0 = tgt[:, 0].reshape(B, -1)
        ident0 = ctx[:, 0].reshape(B, -1)
        c = F.cosine_similarity(zhat0, zstar0, dim=1)
        return c, zstar0, ident0

    h_cos, h_zstar, h_ident = cond(gather(heldout_ds))
    t_cos, t_zstar, _ = cond(gather(train_ds))
    identity = float(F.cosine_similarity(h_ident, h_zstar, dim=1).mean())
    k = min(h_zstar.shape[0], t_zstar.shape[0])
    perm = torch.randperm(t_zstar.shape[0], generator=torch.Generator().manual_seed(0))[:k]
    marginal = float(F.cosine_similarity(h_zstar[:k], t_zstar[perm.to(t_zstar.device)], dim=1).mean())
    return {
        "heldout_cos": float(h_cos.mean()),
        "train_cos": float(t_cos.mean()),
        "identity": identity,
        "marginal": marginal,
        "gap": float(t_cos.mean() - h_cos.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="gigaword")
    ap.add_argument("--heldout-cache-dir", default="./cache/gigaword_val")
    ap.add_argument("--max-steps", type=int, default=30_000)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--loss", choices=["cos", "mse"], default="cos")
    ap.add_argument("--space", choices=["whitened", "raw"], default="whitened",
                    help="train+score geometry. 'raw' drops whitening to match the "
                         "ridge/k-NN geometry (cosine is not affine-invariant, so the "
                         "whitened-space optimum is not the raw-space optimum).")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--out-dir", default="./runs")
    ap.add_argument("--baseline", type=float, default=0.459)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = get_preset(args.task)
    cfg.validate()
    pcfg = cfg.predictor
    q, M, d = pcfg.latents_per_chunk, pcfg.n_tgt_chunks, pcfg.latent_dim

    base = cfg.train.phase2
    ocfg = replace(
        base, lr=args.lr, min_lr_ratio=args.min_lr_ratio, max_steps=args.max_steps,
        batch_size=args.batch_size or base.batch_size,
    )
    if args.smoke:
        ocfg = replace(ocfg, max_steps=min(ocfg.max_steps, 60), warmup_steps=10,
                       batch_size=min(ocfg.batch_size, 16))
        args.val_every = min(args.val_every, 30)
        args.eval_n = min(args.eval_n, 64)

    ds = PairLatentDataset(cfg.data, pcfg)
    loader = DataLoader(ds, batch_size=ocfg.batch_size, shuffle=True,
                        collate_fn=collate_pairs, num_workers=cfg.data.num_workers,
                        drop_last=True)
    heldout_cfg = replace(cfg.data, latent_cache_dir=args.heldout_cache_dir)
    heldout_ds = PairLatentDataset(heldout_cfg, pcfg)

    whitening = load_whitening(cfg, device)
    model = DirectRegressor(pcfg).to(device)
    opt = make_optimizer(model, ocfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, ocfg))
    ema = EmaModel(model, ocfg.ema_decay)

    # raw space carries a large latent mean; bf16's ~3 sig-digits can swamp the
    # directional signal, so train raw-space in fp32 to keep the geometry honest.
    use_amp = device == "cuda" and args.space != "raw"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"PROBE 2 — direct regression: task={args.task} examples={len(ds)} "
          f"heldout={len(heldout_ds)} steps={ocfg.max_steps} bs={ocfg.batch_size} "
          f"lr={ocfg.lr} floor={ocfg.min_lr_ratio} loss={args.loss} space={args.space}")

    best = -1.0

    def run_eval(step: int) -> None:
        nonlocal best
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        model.eval()
        met = direct_metric(model, whitening, ds, heldout_ds, pcfg, args.eval_n, device,
                            space=args.space)
        hc = met["heldout_cos"]
        print(f"[eval step {step}] heldout_cos {hc:.4f} / flow {args.baseline:.3f} "
              f"(best {max(best, hc):.4f}) | train_cos {met['train_cos']:.4f} | "
              f"gap {met['gap']:+.4f} | marginal {met['marginal']:.4f} | "
              f"identity {met['identity']:.4f}")
        if hc > best:
            best = hc
            save_ckpt(os.path.join(args.out_dir, "regress_best.pt"),
                      model=model, ema=ema, step=step, heldout_cos=hc)
            print(f"  new best heldout_cos {hc:.4f} -> regress_best.pt")
        model.load_state_dict(backup)
        model.train()

    model.train()
    step = 0
    done = False
    while not done:
        for batch in loader:
            if args.space == "raw":
                context = batch["context"].to(device)
                target = batch["target"].to(device)
            else:
                context = whitening.apply(batch["context"].to(device))
                target = whitening.apply(batch["target"].to(device))
            cmask = batch["context_mask"].to(device)
            tmask = batch["target_mask"].to(device)
            B = context.shape[0]
            C = context.reshape(B, -1, d)
            Zt = target.reshape(B, -1, d)
            ctm = expand_chunk_mask(cmask, q)
            ttm = expand_chunk_mask(tmask, q)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(C, ctm, ttm)
                w = ttm.float()
                if args.loss == "cos":
                    cos = F.cosine_similarity(pred, Zt, dim=-1)      # [B,M*q]
                    loss = ((1.0 - cos) * w).sum() / w.sum().clamp(min=1.0)
                else:
                    w3 = w[:, :, None]
                    loss = ((pred - Zt) ** 2 * w3).sum() / (w3.sum().clamp(min=1.0) * d)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg.grad_clip)
            opt.step()
            sched.step()
            ema.update(model)

            step += 1
            if step % 50 == 0:
                print(f"step {step} | {args.loss}_loss {loss.item():.4f} "
                      f"| lr {sched.get_last_lr()[0]:.2e}")
            if step % args.val_every == 0:
                run_eval(step)
            if step >= ocfg.max_steps:
                done = True
                break

    run_eval(step)
    save_ckpt(os.path.join(args.out_dir, "regress_final.pt"),
              model=model, ema=ema, step=step)

    print("\n" + "=" * 70)
    print("PROBE 2 — DIRECT REGRESSION RESULT")
    print("=" * 70)
    print(f"  best held-out cosine : {best:.4f}")
    print(f"  flow predictor (est) : {args.baseline:.4f}")
    delta = best - args.baseline
    if delta >= 0.09:
        print(f"  VERDICT: regression beats flow by {delta:+.3f} -> the OBJECTIVE was "
              "the bottleneck. Switch predictor to a regression/hybrid head; no VAE "
              "needed for accuracy.")
    elif delta <= 0.03:
        print(f"  VERDICT: regression caps ~flow ({delta:+.3f}) -> objective is NOT the "
              "bottleneck. Consistent with the capacity ceiling (combine with Probe 1).")
    else:
        print(f"  VERDICT: regression edges flow by {delta:+.3f} (sub-0.09) -> weak "
              "signal; read alongside Probe 1's oracles before deciding.")
    print("=" * 70)


if __name__ == "__main__":
    main()
