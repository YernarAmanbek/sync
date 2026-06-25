"""Training loops for the three phases (README §1) and shared utilities.

Phase 1: train the codec, then FREEZE and compute the latent scale.
Phase 2: train the predictor on cached, scaled latents (frozen codec).
Phase 3: optional light joint finetune to tighten predictor<->decoder coupling.

Each loop is intentionally explicit about the failure-mode mitigations that make
or break this architecture (KL annealing/free bits for the codec; EMA + CFG
dropout for the predictor)."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os

from .codec import CodecInterface, LatentCodec
from .config import Config, OptimConfig
from .data import Whitening, collate_pairs, compute_latent_whitening
from .components import expand_chunk_mask
from .predictor import (
    CountHead,
    FlowMatchingPredictor,
    FlowSampler,
    HybridPredictor,
    HybridSampler,
    flow_matching_target,
)


# --------------------------------------------------------------------------- #
# Shared utilities
# --------------------------------------------------------------------------- #
def make_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    """AdamW with no weight decay on norms/biases/embeddings."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "embed" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)


def lr_lambda(step: int, cfg: OptimConfig) -> float:
    """Linear warmup then cosine decay over max_steps. Decays to `min_lr_ratio`
    (not 0) so the sustained LR can be held above zero in the useful regime."""
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    prog = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    floor = getattr(cfg, "min_lr_ratio", 0.0)
    return floor + (1.0 - floor) * cosine


def beta_schedule(step: int, beta_max: float, warmup_steps: int) -> float:
    """Linear KL-weight annealing 0 -> beta_max (posterior-collapse mitigation). Concrete."""
    return beta_max * min(1.0, step / max(1, warmup_steps))


class EmaModel:
    """Exponential moving average of model params. Critical for the predictor;
    sampling/inference uses EMA weights, not the raw ones.
    AGENT TASK: store shadow params, `update(model)` each step, `copy_to(model)`
    / context manager for eval, include in checkpoints."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                s.copy_(v)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        sd = model.state_dict()
        model.load_state_dict(
            {k: self.shadow[k].to(sd[k].dtype) for k in sd}, strict=True
        )

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd["decay"]
        self.shadow = {k: v.float() for k, v in sd["shadow"].items()}


def save_ckpt(path: str, **objects) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {}
    for k, v in objects.items():
        payload[k] = v.state_dict() if hasattr(v, "state_dict") else v
    torch.save(payload, path)


def load_ckpt(path: str, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location)


# --------------------------------------------------------------------------- #
# Phase 1 — codec
# --------------------------------------------------------------------------- #
def train_codec(codec: LatentCodec, loader: DataLoader, cfg: Config) -> None:
    """Self-supervised VAE training.

    Per step:
      batch -> {tokens, pad_mask, lengths}
      beta = beta_schedule(step, codec.cfg.beta_max, codec.cfg.beta_warmup_steps)
      mask_ratio ~ U(cmlm_mask_low, cmlm_mask_high]  shape [B]
      out = codec(tokens, pad_mask, lengths, beta, mask_ratio)
      backward(out["loss"]); clip; step; (optional EMA)

    Validation gate (must pass before Phase 2 — README §4):
      - reconstruction: encode->decode_latent held-out sentences, measure exact-match
        / BLEU; should be high.
      - smoothness: interpolate two latents, decode the midpoints, check plausibility.
      - watch KL: it must stay meaningfully > 0 (else posterior collapse — raise
        free_bits / slow beta annealing / shrink decoder).

    AGENT TASK: full loop with AMP (cfg.train.phase1.amp_dtype), logging of
    recon/kl/length separately, checkpointing every ckpt_every."""
    raise NotImplementedError("AGENT: train_codec")


WHITENING_FILENAME = "whitening.npz"


@torch.no_grad()
def freeze_and_scale(cfg: Config, sample_size: Optional[int] = None, mode: str = "zca") -> Whitening:
    """Fit and persist the latent **whitening** (README §8) from the precomputed
    cache. Replaces the naive per-dim scaling: SONAR's space is anisotropic and
    correlated, so we mean-center + decorrelate (ZCA/PCA) the latents.

    Reads the target memmap in `cfg.data.latent_cache_dir`, samples valid latents,
    fits `Whitening`, saves it next to the cache, and returns it.
    """
    import json

    import numpy as np

    cache = cfg.data.latent_cache_dir
    with open(os.path.join(cache, "meta.json")) as f:
        meta = json.load(f)
    num, M, q, d = meta["num"], meta["M"], meta["q"], meta["d"]
    m_arr = np.load(os.path.join(cache, "m.npy"))
    target = np.memmap(
        os.path.join(cache, "target.f32"), dtype="float32", mode="r",
        shape=(num, M, q, d),
    )

    sample_size = sample_size or cfg.data.scale_sample_size
    collected = []
    total = 0
    for i in range(num):
        mi = int(m_arr[i])
        if mi <= 0:
            continue
        collected.append(np.ascontiguousarray(target[i, :mi]).reshape(-1, d))
        total += mi * q
        if total >= sample_size:
            break
    latents = torch.from_numpy(np.concatenate(collected, axis=0)).float()
    whitening = compute_latent_whitening(latents, mode=mode)
    whitening.save(os.path.join(cache, WHITENING_FILENAME))
    return whitening


def load_whitening(cfg: Config, device="cpu") -> Whitening:
    path = os.path.join(cfg.data.latent_cache_dir, WHITENING_FILENAME)
    return Whitening.load(path).to(device)


# --------------------------------------------------------------------------- #
# Phase 2 — in-loop evaluation helpers (cache-only; no SONAR needed)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _val_flow_loss(predictor, count_head, val_loader, whitening, q, M, device,
                   max_batches: int = 20) -> tuple[float, float]:
    """Held-out flow/count loss on the validation cache (same masked MSE as train)."""
    predictor.eval()
    count_head.eval()
    tot_f = tot_c = 0.0
    nb = 0
    for batch in val_loader:
        if nb >= max_batches:
            break
        context = whitening.apply(batch["context"].to(device))
        target = whitening.apply(batch["target"].to(device))
        cmask = batch["context_mask"].to(device)
        tmask = batch["target_mask"].to(device)
        m = batch["m"].to(device)
        B, d = context.shape[0], context.shape[-1]
        C = context.reshape(B, -1, d)
        Z0 = target.reshape(B, -1, d)
        ctm = expand_chunk_mask(cmask, q)
        ttm = expand_chunk_mask(tmask, q)
        t = torch.rand(B, device=device)
        eps = torch.randn_like(Z0)
        z_t, v = flow_matching_target(Z0, eps, t)
        v_hat = predictor(z_t, t, C, ctm, ttm)
        w = ttm.float()[:, :, None]
        fl = ((v_hat - v) ** 2 * w).sum() / (w.sum().clamp(min=1.0) * d)
        cl = F.cross_entropy(count_head(C, ctm), m.clamp(min=0, max=M))
        tot_f += float(fl)
        tot_c += float(cl)
        nb += 1
    return tot_f / max(1, nb), tot_c / max(1, nb)


@torch.no_grad()
def _latent_metric_from_cache(predictor, count_head, whitening, train_ds, heldout_ds,
                              pcfg, n: int, guidance: float, steps: int, seed: int,
                              device) -> dict:
    """Decoder-free latent metric (the gate_latent computation) entirely from the
    cached latents — no SONAR. z* and identity come straight from the cache
    (target = encode(reference); context chunk-0 = encode(prompt lede)); zhat is
    the sampler's prediction un-whitened back to raw SONAR space.

    Caller is responsible for having EMA weights live in `predictor`."""
    q, M, d = pcfg.latents_per_chunk, pcfg.n_tgt_chunks, pcfg.latent_dim
    sampler = FlowSampler(predictor, count_head, pcfg)

    def gather(ds):
        return collate_pairs([ds[i] for i in range(min(n, len(ds)))])

    def cond(batch):
        ctx = batch["context"].to(device)            # [B,N,q,d] raw
        tgt = batch["target"].to(device)             # [B,M,q,d] raw
        cmask = batch["context_mask"].to(device)
        B, N = ctx.shape[0], ctx.shape[1]
        ctx_w = whitening.apply(ctx).reshape(B, N * q, d)
        ctm = expand_chunk_mask(cmask, q)
        gen = torch.Generator(device=device).manual_seed(seed)
        Zw, _m = sampler.sample(ctx_w, ctm, steps=steps, guidance_scale=guidance, generator=gen)
        Zraw = whitening.invert(Zw).reshape(B, M, q, d)   # raw SONAR space
        zhat0 = Zraw[:, 0].reshape(B, -1)
        zstar0 = tgt[:, 0].reshape(B, -1)
        ident0 = ctx[:, 0].reshape(B, -1)
        c = F.cosine_similarity(zhat0, zstar0, dim=1)
        return c, zstar0, ident0

    h_cos, h_zstar, h_ident = cond(gather(heldout_ds))
    t_cos, t_zstar, _ = cond(gather(train_ds))

    identity = float(F.cosine_similarity(h_ident, h_zstar, dim=1).mean())
    k = min(h_zstar.shape[0], t_zstar.shape[0])
    perm = torch.randperm(
        t_zstar.shape[0], generator=torch.Generator().manual_seed(seed)
    )[:k].to(t_zstar.device)
    marginal = float(F.cosine_similarity(h_zstar[:k], t_zstar[perm], dim=1).mean())
    return {
        "heldout_cos": float(h_cos.mean()),
        "train_cos": float(t_cos.mean()),
        "identity": identity,
        "marginal": marginal,
        "gap": float(t_cos.mean() - h_cos.mean()),
    }


# --------------------------------------------------------------------------- #
# Phase 2 — predictor
# --------------------------------------------------------------------------- #
def train_predictor(
    predictor: FlowMatchingPredictor,
    count_head: CountHead,
    loader: DataLoader,
    cfg: Config,
    *,
    heldout_ds=None,
    eval_n: int = 300,
    eval_guidance: float = 1.0,
    eval_steps: int = 50,
    eval_seed: int = 0,
    val_batches: int = 20,
    decode_readiness: float = 0.85,
    sample_eval: bool = False,
    sample_n: int = 5,
) -> None:
    """Flow-matching training on cached latent pairs (codec is frozen, not used
    here — latents are precomputed).

    Per step:
      batch -> {context[B,N,q,d], context_mask, target[B,M,q,d], target_mask, n, m}
      scale = cfg.latent_scale  (apply to context & target)
      flatten: C -> [B, N*q, d], Z0 -> [B, M*q, d]; expand masks to latent-tokens
      t   ~ U(0,1)                         [B]
      eps ~ N(0, I)                        [B, M*q, d]
      z_t, v = flow_matching_target(Z0, eps, t)
      # CFG dropout: with prob cfg.predictor.cfg_dropout, pass context=None
      v_hat = predictor(z_t, t, context_or_none, context_mask, target_mask)
      flow_loss = MSE(v_hat, v) averaged over VALID target latent-tokens only
      count_loss = CE(count_head(C, context_mask), m)
      loss = flow_loss + count_loss
      backward; clip; step; EMA.update(predictor)

    Validation gate (README §4/§8): decode + metric curves are done by the eval
    script (kept out of the train loop so SONAR need not be loaded during
    training); here we train, EMA, log losses, and checkpoint.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ocfg = cfg.train.phase2
    q = cfg.predictor.latents_per_chunk
    M = cfg.predictor.n_tgt_chunks
    pcfg = cfg.predictor

    predictor.to(device)
    count_head.to(device)
    whitening = load_whitening(cfg, device)
    train_ds = loader.dataset

    modules = nn.ModuleList([predictor, count_head])
    opt = make_optimizer(modules, ocfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, ocfg))
    ema = EmaModel(predictor, ocfg.ema_decay)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    amp_dtype = dtype_map[ocfg.amp_dtype]
    use_amp = device == "cuda" and amp_dtype != torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    cfg_drop = cfg.predictor.cfg_dropout
    out_dir = cfg.train.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # held-out evaluation plumbing (cache-only metric + val loss; SONAR-free)
    val_loader = None
    if heldout_ds is not None:
        val_loader = DataLoader(
            heldout_ds, batch_size=ocfg.batch_size, shuffle=False,
            collate_fn=collate_pairs, num_workers=0, drop_last=False,
        )
    sample_codec = None
    if sample_eval and heldout_ds is not None:
        from .codec import SonarCodecAdapter  # lazy: only when sampling is requested
        sample_codec = SonarCodecAdapter(cfg.codec, cfg.tokenizer, device=device)
    sample_sampler = FlowSampler(predictor, count_head, pcfg)
    best_heldout = -1.0

    def run_eval(step: int) -> None:
        nonlocal best_heldout
        # latent metric + sample dump use EMA weights; val loss uses live weights
        vf, vc = _val_flow_loss(predictor, count_head, val_loader, whitening, q, M, device, val_batches)
        backup = {k: v.detach().clone() for k, v in predictor.state_dict().items()}
        ema.copy_to(predictor)
        predictor.eval()
        met = _latent_metric_from_cache(
            predictor, count_head, whitening, train_ds, heldout_ds,
            pcfg, eval_n, eval_guidance, eval_steps, eval_seed, device,
        )
        hc = met["heldout_cos"]
        best_so_far = max(best_heldout, hc)
        print(
            f"[eval step {step}] val_flow {vf:.4f} | "
            f"heldout_cos {hc:.4f} / {decode_readiness:.2f} target "
            f"(best {best_so_far:.4f}) | "
            f"train_cos {met['train_cos']:.4f} | gap {met['gap']:+.4f} | "
            f"marginal {met['marginal']:.4f} | identity {met['identity']:.4f}"
        )
        if sample_codec is not None:
            items = collate_pairs([heldout_ds[i] for i in range(min(sample_n, len(heldout_ds)))])
            ctx = items["context"].to(device)
            cmask = items["context_mask"].to(device)
            Bs, N = ctx.shape[0], ctx.shape[1]
            ctx_w = whitening.apply(ctx).reshape(Bs, N * q, ctx.shape[-1])
            ctm = expand_chunk_mask(cmask, q)
            Zw, _m = sample_sampler.sample(ctx_w, ctm, steps=eval_steps, guidance_scale=eval_guidance)
            Zraw = whitening.invert(Zw).reshape(Bs, M, q, ctx.shape[-1])
            texts = sample_codec.decode_latents(Zraw[:, 0])
            print("  [lagging smell-check; sample quality trails latent cos, sharpens past ~0.85]")
            for tx in texts:
                print("   sample:", tx[:160])
        if hc > best_heldout:
            best_heldout = hc
            save_ckpt(
                os.path.join(out_dir, "predictor_best.pt"),
                predictor=predictor, count_head=count_head, ema=ema, step=step,
                heldout_cos=hc,
            )
            print(f"  new best heldout_cos {hc:.4f} -> predictor_best.pt")
        predictor.load_state_dict(backup)
        predictor.train()
        count_head.train()

    predictor.train()
    count_head.train()
    step = 0
    done = False
    while not done:
        for batch in loader:
            context = whitening.apply(batch["context"].to(device))   # [B,N,q,d]
            target = whitening.apply(batch["target"].to(device))     # [B,M,q,d]
            cmask = batch["context_mask"].to(device)                 # [B,N]
            tmask = batch["target_mask"].to(device)                  # [B,M]
            m = batch["m"].to(device)
            B = context.shape[0]
            d = context.shape[-1]

            C = context.reshape(B, -1, d)            # [B, N*q, d]
            Z0 = target.reshape(B, -1, d)            # [B, M*q, d]
            ctx_tok_mask = expand_chunk_mask(cmask, q)
            tgt_tok_mask = expand_chunk_mask(tmask, q)

            t = torch.rand(B, device=device)
            eps = torch.randn_like(Z0)
            z_t, v = flow_matching_target(Z0, eps, t)

            ctx_in, ctxmask_in = C, ctx_tok_mask
            if cfg_drop > 0 and torch.rand(()).item() < cfg_drop:
                ctx_in, ctxmask_in = None, None     # CFG unconditional branch

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                v_hat = predictor(z_t, t, ctx_in, ctxmask_in, tgt_tok_mask)
                w = tgt_tok_mask.float()[:, :, None]              # [B, M*q, 1]
                flow_loss = ((v_hat - v) ** 2 * w).sum() / (w.sum().clamp(min=1.0) * d)
                count_logits = count_head(C, ctx_tok_mask)
                count_loss = F.cross_entropy(count_logits, m.clamp(min=0, max=M))
                loss = flow_loss + count_loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modules.parameters(), ocfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(predictor)

            step += 1
            if step % 50 == 0:
                print(
                    f"step {step} | loss {loss.item():.4f} "
                    f"| flow {flow_loss.item():.4f} | count {count_loss.item():.4f} "
                    f"| lr {sched.get_last_lr()[0]:.2e}"
                )
            if val_loader is not None and step % ocfg.val_every == 0:
                run_eval(step)
            if step % ocfg.ckpt_every == 0:
                save_ckpt(
                    os.path.join(out_dir, f"predictor_{step}.pt"),
                    predictor=predictor, count_head=count_head, ema=ema, step=step,
                )
            if step >= ocfg.max_steps:
                done = True
                break

    if val_loader is not None:
        run_eval(step)
    save_ckpt(
        os.path.join(out_dir, "predictor_final.pt"),
        predictor=predictor, count_head=count_head, ema=ema, step=step,
    )


# --------------------------------------------------------------------------- #
# Phase 2 (hybrid) — mean + flow-residual: in-loop eval helpers (cache-only)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _hybrid_val_loss(hybrid, count_head, val_loader, whitening, q, M, device,
                     max_batches: int = 20) -> tuple[float, float, float]:
    """Held-out (mean MSE, residual-flow MSE, count CE) on the validation cache.
    Parallel to `_val_flow_loss`; residual target is z0 − μ.detach()."""
    hybrid.eval()
    count_head.eval()
    tot_m = tot_f = tot_c = 0.0
    nb = 0
    for batch in val_loader:
        if nb >= max_batches:
            break
        context = whitening.apply(batch["context"].to(device))
        target = whitening.apply(batch["target"].to(device))
        cmask = batch["context_mask"].to(device)
        tmask = batch["target_mask"].to(device)
        m = batch["m"].to(device)
        B, d = context.shape[0], context.shape[-1]
        C = context.reshape(B, -1, d)
        Z0 = target.reshape(B, -1, d)
        ctm = expand_chunk_mask(cmask, q)
        ttm = expand_chunk_mask(tmask, q)
        w = ttm.float()[:, :, None]
        denom = w.sum().clamp(min=1.0) * d

        mu = hybrid.mean(C, ctm, ttm)
        mean_loss = ((mu - Z0) ** 2 * w).sum() / denom
        r0 = Z0 - mu.detach()
        t = torch.rand(B, device=device)
        eps = torch.randn_like(r0)
        r_t, v = flow_matching_target(r0, eps, t)
        v_hat = hybrid.residual_velocity(r_t, t, C, ctm, ttm)
        flow_loss = ((v_hat - v) ** 2 * w).sum() / denom
        cl = F.cross_entropy(count_head(C, ctm), m.clamp(min=0, max=M))
        tot_m += float(mean_loss)
        tot_f += float(flow_loss)
        tot_c += float(cl)
        nb += 1
    n = max(1, nb)
    return tot_m / n, tot_f / n, tot_c / n


@torch.no_grad()
def _hybrid_latent_metric_from_cache(hybrid, count_head, whitening, train_ds, heldout_ds,
                                     pcfg, n: int, guidance: float, steps: int,
                                     temp: float, seed: int, device) -> dict:
    """Decoder-free hybrid metric from cached latents — no SONAR. Reports BOTH
    mean_cos (s=0, the accuracy read) and sample_cos (s=temp, the residual spread).
    identity/marginal/gap are computed on mean_cos, mirroring _latent_metric_from_cache.

    Caller is responsible for having EMA weights live in `hybrid`."""
    q, M, d = pcfg.latents_per_chunk, pcfg.n_tgt_chunks, pcfg.latent_dim
    sampler = HybridSampler(hybrid, count_head, pcfg)

    def gather(ds):
        return collate_pairs([ds[i] for i in range(min(n, len(ds)))])

    def cond(batch):
        ctx = batch["context"].to(device)            # [B,N,q,d] raw
        tgt = batch["target"].to(device)             # [B,M,q,d] raw
        cmask = batch["context_mask"].to(device)
        B, N = ctx.shape[0], ctx.shape[1]
        ctx_w = whitening.apply(ctx).reshape(B, N * q, d)
        ctm = expand_chunk_mask(cmask, q)
        gen = torch.Generator(device=device).manual_seed(seed)
        # s=0 -> deterministic mean (no flow); s=temp -> mean + residual sample
        mu_w, _m = sampler.sample(ctx_w, ctm, steps=steps, guidance_scale=guidance,
                                  temperature=0.0)
        smp_w, _m2 = sampler.sample(ctx_w, ctm, steps=steps, guidance_scale=guidance,
                                    temperature=temp, generator=gen)
        mu_raw = whitening.invert(mu_w).reshape(B, M, q, d)
        smp_raw = whitening.invert(smp_w).reshape(B, M, q, d)
        zstar0 = tgt[:, 0].reshape(B, -1)
        ident0 = ctx[:, 0].reshape(B, -1)
        mean_c = F.cosine_similarity(mu_raw[:, 0].reshape(B, -1), zstar0, dim=1)
        smp_c = F.cosine_similarity(smp_raw[:, 0].reshape(B, -1), zstar0, dim=1)
        return mean_c, smp_c, zstar0, ident0

    h_mean, h_smp, h_zstar, h_ident = cond(gather(heldout_ds))
    t_mean, _t_smp, t_zstar, _ = cond(gather(train_ds))

    identity = float(F.cosine_similarity(h_ident, h_zstar, dim=1).mean())
    k = min(h_zstar.shape[0], t_zstar.shape[0])
    perm = torch.randperm(
        t_zstar.shape[0], generator=torch.Generator().manual_seed(seed)
    )[:k].to(t_zstar.device)
    marginal = float(F.cosine_similarity(h_zstar[:k], t_zstar[perm], dim=1).mean())
    return {
        "mean_cos": float(h_mean.mean()),       # s=0 accuracy read (vs 0.62 oracle)
        "sample_cos": float(h_smp.mean()),      # s=temp; sits below mean_cos on 1-ref tasks
        "train_mean_cos": float(t_mean.mean()),
        "identity": identity,
        "marginal": marginal,
        "gap": float(t_mean.mean() - h_mean.mean()),
    }


# --------------------------------------------------------------------------- #
# Phase 2 (hybrid) — predictor: mean head + flow over the detached residual
# --------------------------------------------------------------------------- #
def train_hybrid(
    hybrid: HybridPredictor,
    count_head: CountHead,
    loader: DataLoader,
    cfg: Config,
    *,
    heldout_ds=None,
    eval_n: int = 300,
    eval_guidance: float = 1.0,
    eval_steps: int = 50,
    eval_temp: float = 1.0,
    eval_seed: int = 0,
    val_batches: int = 20,
    decode_readiness: float = 0.62,
    sample_eval: bool = False,
    sample_n: int = 5,
) -> None:
    """Additive hybrid training (README §8). Mirrors `train_predictor` (whitening,
    AMP, EMA, LR floor, checkpointing) with the combined objective:

        μ          = hybrid.mean(C, ctm, ttm)              # always conditional
        mean_loss  = masked_MSE(μ, Z0)
        r0         = Z0 − μ.detach()                       # residual around fixed mean
        r_t, v     = flow_matching_target(r0, eps, t)
        v_hat      = hybrid.residual_velocity(r_t, t, ctx_or_none, ...)  # cfg_dropout as usual
        flow_loss  = masked_MSE(v_hat, v)
        count_loss = CE(count_head(C, ctm), m)
        loss       = hybrid_mean_weight*mean_loss + flow_loss + count_loss

    EMA covers the whole hybrid; the count head stays separate and non-EMA'd.
    Best checkpoint (`hybrid_best.pt`) is saved on held-out mean_cos (the accuracy
    read against the 0.62 oracle — NOT 0.85, which is unreachable on 1-ref tasks)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ocfg = cfg.train.phase2
    q = cfg.predictor.latents_per_chunk
    M = cfg.predictor.n_tgt_chunks
    pcfg = cfg.predictor
    mean_weight = pcfg.hybrid_mean_weight

    hybrid.to(device)
    count_head.to(device)
    whitening = load_whitening(cfg, device)
    train_ds = loader.dataset

    modules = nn.ModuleList([hybrid, count_head])
    opt = make_optimizer(modules, ocfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, ocfg))
    ema = EmaModel(hybrid, ocfg.ema_decay)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    amp_dtype = dtype_map[ocfg.amp_dtype]
    use_amp = device == "cuda" and amp_dtype != torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    cfg_drop = cfg.predictor.cfg_dropout
    out_dir = cfg.train.out_dir
    os.makedirs(out_dir, exist_ok=True)

    val_loader = None
    if heldout_ds is not None:
        val_loader = DataLoader(
            heldout_ds, batch_size=ocfg.batch_size, shuffle=False,
            collate_fn=collate_pairs, num_workers=0, drop_last=False,
        )
    sample_codec = None
    if sample_eval and heldout_ds is not None:
        from .codec import SonarCodecAdapter  # lazy: only when sampling is requested
        sample_codec = SonarCodecAdapter(cfg.codec, cfg.tokenizer, device=device)
    sample_sampler = HybridSampler(hybrid, count_head, pcfg)
    best_mean = -1.0

    def run_eval(step: int) -> None:
        nonlocal best_mean
        vm, vf, vc = _hybrid_val_loss(hybrid, count_head, val_loader, whitening, q, M,
                                      device, val_batches)
        backup = {k: v.detach().clone() for k, v in hybrid.state_dict().items()}
        ema.copy_to(hybrid)
        hybrid.eval()
        met = _hybrid_latent_metric_from_cache(
            hybrid, count_head, whitening, train_ds, heldout_ds,
            pcfg, eval_n, eval_guidance, eval_steps, eval_temp, eval_seed, device,
        )
        mc = met["mean_cos"]
        best_so_far = max(best_mean, mc)
        print(
            f"[eval step {step}] val_mean {vm:.4f} val_flow {vf:.4f} | "
            f"mean_cos(s=0) {mc:.4f} / {decode_readiness:.2f} oracle "
            f"(best {best_so_far:.4f}) | "
            f"sample_cos(s={eval_temp:g}) {met['sample_cos']:.4f} (below mean_cos by "
            f"design on 1-ref) | train_mean_cos {met['train_mean_cos']:.4f} | "
            f"gap {met['gap']:+.4f} | marginal {met['marginal']:.4f} | "
            f"identity {met['identity']:.4f}"
        )
        if sample_codec is not None:
            items = collate_pairs([heldout_ds[i] for i in range(min(sample_n, len(heldout_ds)))])
            ctx = items["context"].to(device)
            cmask = items["context_mask"].to(device)
            Bs, N = ctx.shape[0], ctx.shape[1]
            ctx_w = whitening.apply(ctx).reshape(Bs, N * q, ctx.shape[-1])
            ctm = expand_chunk_mask(cmask, q)
            Zw, _m = sample_sampler.sample(ctx_w, ctm, steps=eval_steps,
                                           guidance_scale=eval_guidance, temperature=eval_temp)
            Zraw = whitening.invert(Zw).reshape(Bs, M, q, ctx.shape[-1])
            texts = sample_codec.decode_latents(Zraw[:, 0])
            print(f"  [lagging smell-check; s={eval_temp:g} samples]")
            for tx in texts:
                print("   sample:", tx[:160])
        if mc > best_mean:
            best_mean = mc
            save_ckpt(
                os.path.join(out_dir, "hybrid_best.pt"),
                hybrid=hybrid, count_head=count_head, ema=ema, step=step,
                mean_cos=mc,
            )
            print(f"  new best mean_cos {mc:.4f} -> hybrid_best.pt")
        hybrid.load_state_dict(backup)
        hybrid.train()
        count_head.train()

    hybrid.train()
    count_head.train()
    step = 0
    done = False
    while not done:
        for batch in loader:
            context = whitening.apply(batch["context"].to(device))   # [B,N,q,d]
            target = whitening.apply(batch["target"].to(device))     # [B,M,q,d]
            cmask = batch["context_mask"].to(device)                 # [B,N]
            tmask = batch["target_mask"].to(device)                  # [B,M]
            m = batch["m"].to(device)
            B = context.shape[0]
            d = context.shape[-1]

            C = context.reshape(B, -1, d)            # [B, N*q, d]
            Z0 = target.reshape(B, -1, d)            # [B, M*q, d]
            ctx_tok_mask = expand_chunk_mask(cmask, q)
            tgt_tok_mask = expand_chunk_mask(tmask, q)

            t = torch.rand(B, device=device)
            eps = torch.randn_like(Z0)

            # CFG dropout applies ONLY to the residual flow's context branch
            ctx_in, ctxmask_in = C, ctx_tok_mask
            if cfg_drop > 0 and torch.rand(()).item() < cfg_drop:
                ctx_in, ctxmask_in = None, None

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                w = tgt_tok_mask.float()[:, :, None]              # [B, M*q, 1]
                denom = w.sum().clamp(min=1.0) * d
                mu = hybrid.mean(C, ctx_tok_mask, tgt_tok_mask)  # always conditional
                mean_loss = ((mu - Z0) ** 2 * w).sum() / denom
                r0 = Z0 - mu.detach()                            # residual around fixed mean
                r_t, v = flow_matching_target(r0, eps, t)
                v_hat = hybrid.residual_velocity(r_t, t, ctx_in, ctxmask_in, tgt_tok_mask)
                flow_loss = ((v_hat - v) ** 2 * w).sum() / denom
                count_logits = count_head(C, ctx_tok_mask)
                count_loss = F.cross_entropy(count_logits, m.clamp(min=0, max=M))
                loss = mean_weight * mean_loss + flow_loss + count_loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modules.parameters(), ocfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(hybrid)

            step += 1
            if step % 50 == 0:
                print(
                    f"step {step} | loss {loss.item():.4f} "
                    f"| mean {mean_loss.item():.4f} | flow {flow_loss.item():.4f} "
                    f"| count {count_loss.item():.4f} | lr {sched.get_last_lr()[0]:.2e}"
                )
            if val_loader is not None and step % ocfg.val_every == 0:
                run_eval(step)
            if step % ocfg.ckpt_every == 0:
                save_ckpt(
                    os.path.join(out_dir, f"hybrid_{step}.pt"),
                    hybrid=hybrid, count_head=count_head, ema=ema, step=step,
                )
            if step >= ocfg.max_steps:
                done = True
                break

    if val_loader is not None:
        run_eval(step)
    save_ckpt(
        os.path.join(out_dir, "hybrid_final.pt"),
        hybrid=hybrid, count_head=count_head, ema=ema, step=step,
    )


# --------------------------------------------------------------------------- #
# Phase 3 — optional joint finetune
# --------------------------------------------------------------------------- #
def finetune_joint(
    codec: LatentCodec,
    predictor: FlowMatchingPredictor,
    count_head: CountHead,
    loader: DataLoader,
    cfg: Config,
) -> None:
    """Light, LAST-STEP coupling finetune. Unfreeze the DECODER only; backprop a
    token-level CE through sampled target latents so the decoder tolerates the
    predictor's slightly-off-manifold samples.

    Gradient path through sampling is non-trivial — use a straight-through /
    reparameterized single-step sample (or Gumbel on decoder logits). Keep LR
    tiny (cfg.train.phase3.lr) and steps few; this REINTRODUCES generative
    coupling and can destabilize, so monitor recon quality and stop early.

    AGENT TASK: implement only if Phase-2 eval shows off-manifold decode failures."""
    raise NotImplementedError("AGENT: finetune_joint")