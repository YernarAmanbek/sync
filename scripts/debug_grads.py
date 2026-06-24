"""Diagnostic — confirm or kill the "context not wired" hypothesis (order-of-ops 1).

Replicates ONE predictor training step exactly and reports per-module gradient
norms, so we can see whether the CONTEXT path (`ctx_proj`, `ctx_pos`, the
cross-attention modules in the backbone) actually receives gradient — versus the
target/self path. Also prints the realized cfg_dropout rate.

    python -m Sync.scripts.debug_grads --task gigaword
    python -m Sync.scripts.debug_grads --task gigaword --ckpt runs/predictor_final.pt --steps 20

Reads the same cache as training (run precompute_latents first).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..components import expand_chunk_mask
from ..config import get_preset
from ..data import PairLatentDataset, collate_pairs
from ..predictor import CountHead, FlowMatchingPredictor, flow_matching_target
from ..training import load_ckpt, load_whitening


# map a parameter name to a coarse functional group
def _bucket(name: str) -> str:
    if "ctx_proj" in name:
        return "context: ctx_proj"
    if "ctx_pos" in name:
        return "context: ctx_pos"
    if "cross_attn_mod" in name or "cross_norm" in name:
        return "context: cross_attention"
    if "null_context" in name:
        return "context: null_embed"
    if "in_proj" in name:
        return "target: in_proj"
    if "tgt_pos" in name:
        return "target: tgt_pos"
    if "self_attn" in name:
        return "target: self_attention"
    if "out_proj" in name:
        return "target: out_proj"
    if "time_embed" in name:
        return "target: time_embed"
    return "other (mlp/norm)"


def _grad_report(model: torch.nn.Module) -> dict[str, tuple[float, int, int]]:
    """group -> (grad L2 norm, #params with grad, #params total)."""
    acc: dict[str, list[float]] = {}
    for name, p in model.named_parameters():
        g = _bucket(name)
        sq, has, tot = acc.get(g, [0.0, 0, 0])
        tot += p.numel()
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
            has += p.numel()
        acc[g] = [sq, has, tot]
    return {g: (v[0] ** 0.5, int(v[1]), int(v[2])) for g, v in acc.items()}


def _step(predictor, count_head, batch, whitening, q, M, device, context_on: bool):
    context = whitening.apply(batch["context"].to(device))
    target = whitening.apply(batch["target"].to(device))
    cmask = batch["context_mask"].to(device)
    tmask = batch["target_mask"].to(device)
    m = batch["m"].to(device)
    B, d = context.shape[0], context.shape[-1]

    C = context.reshape(B, -1, d)
    Z0 = target.reshape(B, -1, d)
    ctx_tok_mask = expand_chunk_mask(cmask, q)
    tgt_tok_mask = expand_chunk_mask(tmask, q)

    t = torch.rand(B, device=device)
    eps = torch.randn_like(Z0)
    z_t, v = flow_matching_target(Z0, eps, t)

    ctx_in = C if context_on else None
    ctxmask_in = ctx_tok_mask if context_on else None

    v_hat = predictor(z_t, t, ctx_in, ctxmask_in, tgt_tok_mask)
    w = tgt_tok_mask.float()[:, :, None]
    flow_loss = ((v_hat - v) ** 2 * w).sum() / (w.sum().clamp(min=1.0) * d)
    count_logits = count_head(C, ctx_tok_mask)
    count_loss = F.cross_entropy(count_logits, m.clamp(min=0, max=M))
    return flow_loss + count_loss, flow_loss, count_loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gigaword")
    ap.add_argument("--ckpt", default=None, help="optional trained ckpt; else random init")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=10, help="conditional steps to average grad-norms over")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    cfg = get_preset(args.task)
    cfg.validate()
    q = cfg.predictor.latents_per_chunk
    M = cfg.predictor.n_tgt_chunks

    ds = PairLatentDataset(cfg.data, cfg.predictor)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_pairs, num_workers=0, drop_last=True)
    whitening = load_whitening(cfg, device)

    predictor = FlowMatchingPredictor(cfg.predictor).to(device)
    count_head = CountHead(cfg.predictor).to(device)
    if args.ckpt:
        ck = load_ckpt(args.ckpt, map_location="cpu")
        predictor.load_state_dict(ck["predictor"])
        count_head.load_state_dict(ck["count_head"])
        print(f"loaded ckpt {args.ckpt}")
    else:
        print("random init (no ckpt)")

    modules = torch.nn.ModuleList([predictor, count_head])

    # ---- Part A: gradient flow with context FORCED ON ----------------------
    print("\n" + "=" * 70)
    print(f"PART A — grad-norms, context FORCED ON (avg over {args.steps} steps)")
    print("=" * 70)
    it = iter(loader)
    agg: dict[str, list[float]] = {}
    last_losses = (0.0, 0.0, 0.0)
    for _ in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        modules.zero_grad(set_to_none=True)
        loss, fl, cl = _step(predictor, count_head, batch, whitening, q, M, device, context_on=True)
        loss.backward()
        last_losses = (float(loss), float(fl), float(cl))
        for g, (norm, has, tot) in _grad_report(predictor).items():
            agg.setdefault(g, []).append(norm)

    print(f"sample losses: total={last_losses[0]:.4f} flow={last_losses[1]:.4f} count={last_losses[2]:.4f}\n")
    rep = _grad_report(predictor)  # for param counts
    print(f"{'group':28s} {'mean grad-norm':>16s} {'params w/ grad':>16s}")
    for g in sorted(agg):
        mean_norm = sum(agg[g]) / len(agg[g])
        _, has, tot = rep[g]
        flag = "  <-- ZERO/NO GRAD" if mean_norm < 1e-12 else ""
        print(f"{g:28s} {mean_norm:16.6e} {f'{has}/{tot}':>16s}{flag}")

    # ---- Part B: one step with context OFF (sanity: ctx path must be dead) --
    print("\n" + "=" * 70)
    print("PART B — single step, context OFF (ctx path SHOULD be zero, null_embed SHOULD get grad)")
    print("=" * 70)
    modules.zero_grad(set_to_none=True)
    try:
        batch = next(it)
    except StopIteration:
        batch = next(iter(loader))
    loss, fl, cl = _step(predictor, count_head, batch, whitening, q, M, device, context_on=False)
    loss.backward()
    for g, (norm, has, tot) in sorted(_grad_report(predictor).items()):
        print(f"{g:28s} {norm:16.6e} {f'{has}/{tot}':>16s}")

    # ---- Part C: realized cfg_dropout rate ---------------------------------
    cfg_drop = cfg.predictor.cfg_dropout
    trials = 200_000
    drops = int((torch.rand(trials) < cfg_drop).sum().item())
    print("\n" + "=" * 70)
    print("PART C — cfg_dropout")
    print("=" * 70)
    print(f"configured cfg_dropout       : {cfg_drop:.4f}")
    print(f"realized (simulated, n={trials}): {drops / trials:.4f}")
    print("NOTE: dropout is per-BATCH in training.py (whole batch goes unconditional "
          "on a dropped step), not per-example.")


if __name__ == "__main__":
    main()
