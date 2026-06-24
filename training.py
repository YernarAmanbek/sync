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
from .data import Whitening, compute_latent_whitening
from .components import expand_chunk_mask
from .predictor import (
    CountHead,
    FlowMatchingPredictor,
    FlowSampler,
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
# Phase 2 — predictor
# --------------------------------------------------------------------------- #
def train_predictor(
    predictor: FlowMatchingPredictor,
    count_head: CountHead,
    loader: DataLoader,
    cfg: Config,
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

    predictor.to(device)
    count_head.to(device)
    whitening = load_whitening(cfg, device)

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
            if step % ocfg.ckpt_every == 0:
                save_ckpt(
                    os.path.join(out_dir, f"predictor_{step}.pt"),
                    predictor=predictor, count_head=count_head, ema=ema, step=step,
                )
            if step >= ocfg.max_steps:
                done = True
                break

    save_ckpt(
        os.path.join(out_dir, "predictor_final.pt"),
        predictor=predictor, count_head=count_head, ema=ema, step=step,
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