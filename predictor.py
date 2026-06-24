"""Stage B — the generative predictor. The only model trained on the task.

Maps prompt-latents (context `C`) to response-latents (target `Z`) by learning a
flow-matching velocity field, so it models the *distribution* of valid responses
rather than their mean (this is what fixes the JEPA mean-collapse failure).

All latents here are SCALED (README §7): the caller multiplies by `latent_scale`
before training/sampling and inverse-scales before decoding. The flow source is
unit Gaussian, which matches the ≈unit-variance scaled latents."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    ChunkAwarePositionalEmbedding,
    RMSNorm,
    TimestepEmbedding,
    TransformerStack,
    expand_chunk_mask,
)
from .config import PredictorConfig


# --------------------------------------------------------------------------- #
# Flow-matching target (concrete — this is the core math, keep it exact)
# --------------------------------------------------------------------------- #
def flow_matching_target(
    z0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectified-flow / linear-interpolant target.
      z0  : clean target latents      [B, M*q, d]
      eps : noise ~ N(0, I)           [B, M*q, d]
      t   : time in [0,1]             [B]
    Returns:
      z_t : interpolant (1-t)*eps + t*z0   [B, M*q, d]
      v   : target velocity  z0 - eps      [B, M*q, d]
    At inference we integrate dZ/dt = v_hat from t=0 (noise) to t=1 (data)."""
    tb = t[:, None, None]                      # [B,1,1]
    z_t = (1.0 - tb) * eps + tb * z0
    v = z0 - eps
    return z_t, v


# --------------------------------------------------------------------------- #
# Predictor backbone
# --------------------------------------------------------------------------- #
class FlowMatchingPredictor(nn.Module):
    """Transformer over the flattened TARGET latent-token sequence `[B, M*q, d]`
    that self-attends within the target and cross-attends to the flattened
    CONTEXT latent-tokens `[B, N_ctx*q, d]`. Time `t` is injected via
    TimestepEmbedding (added to every target token, FiLM, or a prepended token —
    agent's choice, but be consistent).

    Null-context handling: when `context` is None (CFG dropout / unconditional
    branch), use a learned null embedding in place of the cross stream.

    AGENT TASK:
      - in_proj: Linear(d, d_model) for target tokens; ctx_proj: Linear(d, d_model)
      - add ChunkAwarePositionalEmbedding to target (M canvas) and to context (N_ctx)
      - inject time embedding
      - TransformerStack(cross_attn=True): self over target (masked by target_mask),
        cross to context (masked by context_mask)
      - out_proj: Linear(d_model, d) -> predicted velocity"""

    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg
        self.q = cfg.latents_per_chunk
        self.in_proj = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.ctx_proj = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.tgt_pos = ChunkAwarePositionalEmbedding(cfg.n_tgt_chunks, cfg.latents_per_chunk, cfg.d_model)
        self.ctx_pos = ChunkAwarePositionalEmbedding(cfg.n_ctx_chunks, cfg.latents_per_chunk, cfg.d_model)
        self.time_embed = TimestepEmbedding(cfg.time_embed_dim, cfg.d_model)
        self.null_context = nn.Parameter(torch.zeros(1, 1, cfg.d_model))  # for CFG unconditional branch
        self.backbone = TransformerStack(
            cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.ffn_mult, cfg.dropout, cross_attn=True
        )
        self.out_proj = nn.Linear(cfg.d_model, cfg.latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,                       # [B, M*q, d] noised target
        t: torch.Tensor,                         # [B]
        context: Optional[torch.Tensor],         # [B, N_ctx*q, d] or None (unconditional)
        context_mask: Optional[torch.Tensor] = None,  # [B, N_ctx*q] bool
        target_mask: Optional[torch.Tensor] = None,   # [B, M*q] bool
    ) -> torch.Tensor:                           # v_hat [B, M*q, d]
        B = z_t.shape[0]

        # target stream: project, add chunk-aware positions, inject time
        h = self.in_proj(z_t)                    # [B, M*q, d_model]
        h = self.tgt_pos(h)
        h = h + self.time_embed(t)[:, None, :]   # broadcast time over tokens

        # context stream: either projected real context or the learned null token
        if context is None:
            ctx = self.null_context.expand(B, 1, -1)         # [B, 1, d_model]
            ctx_mask = None
        else:
            ctx = self.ctx_proj(context)                     # [B, N_ctx*q, d_model]
            ctx = self.ctx_pos(ctx)
            ctx_mask = context_mask

        h = self.backbone(h, self_mask=target_mask, context=ctx, context_mask=ctx_mask)
        return self.out_proj(h)                  # [B, M*q, d]


class CountHead(nn.Module):
    """Predicts the number of response chunks `m` from the context. At inference
    the sampler decodes only the first `m` of the M-slot canvas.

    AGENT TASK: pool context latent-tokens (mean over valid positions) -> MLP ->
    logits over 0..M (M+1 classes)."""

    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.n_tgt_chunks = cfg.n_tgt_chunks
        hidden = cfg.d_model
        self.mlp = nn.Sequential(
            nn.Linear(cfg.latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, cfg.n_tgt_chunks + 1),  # classes 0..M
        )

    def forward(
        self, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [B, N_ctx*q, d] -> [B, M+1]
        if context_mask is None:
            pooled = context.mean(dim=1)
        else:
            w = context_mask.float()[:, :, None]             # [B, T, 1]
            denom = w.sum(dim=1).clamp(min=1.0)
            pooled = (context * w).sum(dim=1) / denom         # [B, d]
        return self.mlp(pooled)


# --------------------------------------------------------------------------- #
# Sampling (inference) with classifier-free guidance
# --------------------------------------------------------------------------- #
class FlowSampler:
    """Integrates the learned velocity field from noise (t=0) to data (t=1).

    AGENT TASK: implement `sample`:
      1. m = argmax(count_head(context))                       # per item
      2. Z = randn([B, M*q, d])                                # t=0
      3. for k in range(steps): step the ODE solver, evaluating the velocity as
         v = v_uncond + guidance * (v_cond - v_uncond)         # CFG
         where v_cond = predictor(Z, t, context, ...) and
               v_uncond = predictor(Z, t, None, ...)
         (skip the uncond eval when guidance == 1.0)
      4. return Z (scaled target latents) and m
    Solvers: euler / midpoint / rk4 per cfg.ode_solver. dt = 1/steps."""

    def __init__(self, predictor: FlowMatchingPredictor, count_head: CountHead, cfg: PredictorConfig):
        self.predictor = predictor
        self.count_head = count_head
        self.cfg = cfg

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,                   # [B, N_ctx*q, d] (whitened)
        context_mask: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        m_override: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate dZ/dt = v_hat from noise (t=0) to data (t=1). CFG mixes the
        conditional and unconditional velocity fields. Returns
        (Z_hat [B, M*q, d] whitened, m [B] predicted chunk counts)."""
        cfg = self.cfg
        q = cfg.latents_per_chunk
        M = cfg.n_tgt_chunks
        d = cfg.latent_dim
        steps = steps or cfg.sample_steps
        guidance = cfg.guidance_scale if guidance_scale is None else guidance_scale
        B = context.shape[0]
        device = context.device

        # 1. chunk count -> per-target-token mask
        if m_override is not None:
            m = m_override.to(device).long()
        else:
            m = self.count_head(context, context_mask).argmax(dim=-1)  # [B]
        m = m.clamp(min=1, max=M)
        chunk_mask = torch.arange(M, device=device)[None, :] < m[:, None]  # [B, M]
        target_mask = expand_chunk_mask(chunk_mask, q)                     # [B, M*q]

        # 2. noise at t=0
        Z = torch.randn(B, M * q, d, device=device, generator=generator)

        def velocity(z: torch.Tensor, t_scalar: float) -> torch.Tensor:
            t = torch.full((B,), t_scalar, device=device)
            v_cond = self.predictor(z, t, context, context_mask, target_mask)
            if guidance == 1.0:
                return v_cond
            v_uncond = self.predictor(z, t, None, None, target_mask)
            return v_uncond + guidance * (v_cond - v_uncond)

        # 3. integrate
        dt = 1.0 / steps
        solver = cfg.ode_solver
        for k in range(steps):
            t0 = k * dt
            if solver == "euler":
                Z = Z + dt * velocity(Z, t0)
            elif solver == "midpoint":
                v1 = velocity(Z, t0)
                Z = Z + dt * velocity(Z + 0.5 * dt * v1, t0 + 0.5 * dt)
            elif solver == "rk4":
                k1 = velocity(Z, t0)
                k2 = velocity(Z + 0.5 * dt * k1, t0 + 0.5 * dt)
                k3 = velocity(Z + 0.5 * dt * k2, t0 + 0.5 * dt)
                k4 = velocity(Z + dt * k3, t0 + dt)
                Z = Z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"unknown ode_solver {solver!r}")

        return Z, m


# --------------------------------------------------------------------------- #
# Out-of-distribution gate (README §7 — there is no built-in abstention)
# --------------------------------------------------------------------------- #
def ood_score(
    context_latents: torch.Tensor,               # [B, N_ctx, q, d] UN-scaled encoder means
    context_mask: Optional[torch.Tensor] = None, # [B, N_ctx] bool
    train_stats: Optional[dict] = None,          # e.g. {"mean":[d], "cov_inv":[d,d]} or a kNN index
) -> torch.Tensor:                               # [B] higher = more OOD
    """Score how far a prompt's latents sit from the training distribution, so a
    bounded-input deployment can refuse/fallback. Two viable implementations:
      (a) negative log-likelihood under the VAE prior N(0,I) (cheap, weak), or
      (b) Mahalanobis distance to cached training-latent stats (better)."""
    # pool valid context latents -> [B, d]
    x = context_latents.mean(dim=2)              # average over q -> [B, N_ctx, d]
    if context_mask is None:
        pooled = x.mean(dim=1)
    else:
        w = context_mask.float()[:, :, None]
        pooled = (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)

    if train_stats is not None and "mean" in train_stats and "cov_inv" in train_stats:
        mean = torch.as_tensor(train_stats["mean"], device=pooled.device, dtype=pooled.dtype)
        cov_inv = torch.as_tensor(train_stats["cov_inv"], device=pooled.device, dtype=pooled.dtype)
        delta = pooled - mean                    # [B, d]
        # Mahalanobis distance: sqrt((x-mu)^T S^-1 (x-mu))
        maha = torch.einsum("bi,ij,bj->b", delta, cov_inv, delta).clamp(min=0.0)
        return torch.sqrt(maha)
    # fallback (a): NLL under N(0, I) up to a constant == 0.5 * ||x||^2
    return 0.5 * pooled.pow(2).sum(dim=-1)