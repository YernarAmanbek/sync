"""Shared nn building blocks. Codec and predictor both compose these so the two
backbones stay consistent. Signatures and shapes are fixed; agents fill bodies.

Convention: all blocks are batch-first, `[B, T, d_model]`. Attention masks are
boolean `[B, T]` where True = keep, False = pad (expanded internally to the
additive/key-padding form the attention impl needs)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm over the last dim. Body is a few lines; provided concretely so the
    backbones are unambiguous."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., dim] -> [..., dim]
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with self-attention, OPTIONAL cross-attention,
    and a gated MLP. When `cross_attn=True`, `forward` expects `context`.

    AGENT TASK: implement multi-head self-attn (+ optional cross-attn to
    `context`) and MLP. Respect `self_mask` (boolean key-padding for the self
    stream) and `context_mask` (for the cross stream)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        cross_attn: bool = False,
    ):
        super().__init__()
        self.cross_attn = cross_attn
        # AGENT: build norms, attention projections, (optional) cross-attn, MLP.
        raise NotImplementedError("AGENT: build TransformerBlock submodules")

    def forward(
        self,
        x: torch.Tensor,                         # [B, T, d_model]
        self_mask: Optional[torch.Tensor] = None,    # [B, T] bool, True=keep
        context: Optional[torch.Tensor] = None,      # [B, S, d_model] (if cross_attn)
        context_mask: Optional[torch.Tensor] = None, # [B, S] bool
    ) -> torch.Tensor:                           # [B, T, d_model]
        raise NotImplementedError("AGENT: forward of TransformerBlock")


class TransformerStack(nn.Module):
    """Stack of `n_layers` TransformerBlocks. Used as: codec encoder
    (self-only), codec decoder (self + cross to latents), predictor (self +
    cross to context latents)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        cross_attn: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, n_heads, ffn_mult, dropout, cross_attn)
            for _ in range(n_layers)
        )
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, self_mask=self_mask, context=context, context_mask=context_mask)
        return self.norm(x)


class PerceiverResampler(nn.Module):
    """Pools a variable-length sequence `[B, T, d_model]` into a FIXED set of `q`
    latent vectors `[B, q, d]` via `q` learned queries that cross-attend to the
    sequence. This is the codec's bottleneck (README §2: q latents per chunk).

    AGENT TASK: learned query bank `[q, d_model]`; one or more cross-attention
    blocks (queries attend to the masked input sequence); project to `d`."""

    def __init__(
        self,
        d_model: int,
        latent_dim: int,        # d
        num_latents: int,       # q
        n_heads: int,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_latents = num_latents
        # AGENT: nn.Parameter query bank, cross-attn blocks, output projection to d.
        raise NotImplementedError("AGENT: build PerceiverResampler")

    def forward(
        self,
        x: torch.Tensor,                     # [B, T, d_model]
        mask: Optional[torch.Tensor] = None, # [B, T] bool, True=keep
    ) -> torch.Tensor:                       # [B, q, d]
        raise NotImplementedError("AGENT: forward of PerceiverResampler")


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding of the flow time t∈[0,1] followed by an MLP. The
    sinusoidal part is provided concretely; only the MLP construction is left to
    keep the projection width explicit at the call site."""

    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # AGENT: small MLP embed_dim -> out_dim (e.g. Linear-SiLU-Linear).
        raise NotImplementedError("AGENT: build TimestepEmbedding MLP")

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:  # [B] -> [B, embed_dim]
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # [B] -> [B, out_dim]
        raise NotImplementedError("AGENT: run _sinusoidal(t) through the MLP")


class LearnedPositionalEmbedding(nn.Module):
    """Standard learned absolute positions for the codec (token positions in a
    chunk, length ≤ L). AGENT TASK: nn.Embedding(max_len, d_model), add to x."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, d_model] -> same
        raise NotImplementedError("AGENT: add positional embeddings to x")


class ChunkAwarePositionalEmbedding(nn.Module):
    """Positions for the PREDICTOR's flattened latent-token sequence `[B, M*q, d]`.
    Encodes BOTH the chunk index (0..M-1) and the within-chunk index (0..q-1),
    summed. This is what lets the predictor tell chunk boundaries apart after the
    `[B, M, q, d] -> [B, M*q, d]` flatten (README §2).

    AGENT TASK: two nn.Embeddings (chunk: M or N_ctx, within: q); given seq length
    M*q, build index tensors and add both embeddings."""

    def __init__(self, max_chunks: int, latents_per_chunk: int, d_model: int):
        super().__init__()
        self.max_chunks = max_chunks
        self.q = latents_per_chunk
        self.chunk_pos = nn.Embedding(max_chunks, d_model)
        self.within_pos = nn.Embedding(latents_per_chunk, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, n_chunks*q, d_model] -> same
        raise NotImplementedError("AGENT: add chunk + within-chunk positions")


# --------------------------------------------------------------------------- #
# Masking utilities
# --------------------------------------------------------------------------- #
def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """`lengths [B]` -> boolean `[B, max_len]`, True for valid positions.
    AGENT TASK: implement (arange < lengths[:, None])."""
    raise NotImplementedError("AGENT: lengths_to_mask")


def expand_chunk_mask(chunk_mask: torch.Tensor, q: int) -> torch.Tensor:
    """Expand a per-chunk mask `[B, n_chunks]` to per-latent-token
    `[B, n_chunks*q]` for the flattened predictor sequence.
    AGENT TASK: repeat_interleave along the chunk dim by q."""
    raise NotImplementedError("AGENT: expand_chunk_mask")