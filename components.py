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
import torch.nn.functional as F


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
        hidden = ffn_mult * d_model

        self.norm1 = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        if cross_attn:
            self.cross_norm = RMSNorm(d_model)
            self.cross_attn_mod = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
        self.norm2 = RMSNorm(d_model)
        # gated MLP (SwiGLU): project to 2*hidden, gate one half with SiLU
        self.w_in = nn.Linear(d_model, 2 * hidden)
        self.w_out = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _key_padding(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        # our masks are True=keep; nn.MultiheadAttention wants True=ignore
        return None if mask is None else ~mask

    def forward(
        self,
        x: torch.Tensor,                         # [B, T, d_model]
        self_mask: Optional[torch.Tensor] = None,    # [B, T] bool, True=keep
        context: Optional[torch.Tensor] = None,      # [B, S, d_model] (if cross_attn)
        context_mask: Optional[torch.Tensor] = None, # [B, S] bool
    ) -> torch.Tensor:                           # [B, T, d_model]
        h = self.norm1(x)
        a, _ = self.self_attn(
            h, h, h, key_padding_mask=self._key_padding(self_mask), need_weights=False
        )
        x = x + self.drop(a)

        if self.cross_attn and context is not None:
            h = self.cross_norm(x)
            a, _ = self.cross_attn_mod(
                h, context, context,
                key_padding_mask=self._key_padding(context_mask),
                need_weights=False,
            )
            x = x + self.drop(a)

        h = self.norm2(x)
        gate, up = self.w_in(h).chunk(2, dim=-1)
        x = x + self.drop(self.w_out(F.silu(gate) * up))
        return x


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
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:  # [B] -> [B, embed_dim]
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # [B] -> [B, out_dim]
        return self.mlp(self._sinusoidal(t))


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
        T = x.shape[1]
        assert T % self.q == 0, f"seq len {T} not a multiple of q={self.q}"
        idx = torch.arange(T, device=x.device)
        chunk_ids = idx // self.q          # [T] in 0..n_chunks-1
        within_ids = idx % self.q          # [T] in 0..q-1
        pos = self.chunk_pos(chunk_ids) + self.within_pos(within_ids)  # [T, d_model]
        return x + pos[None, :, :]


# --------------------------------------------------------------------------- #
# Masking utilities
# --------------------------------------------------------------------------- #
def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """`lengths [B]` -> boolean `[B, max_len]`, True for valid positions."""
    ar = torch.arange(max_len, device=lengths.device)
    return ar[None, :] < lengths[:, None]


def expand_chunk_mask(chunk_mask: torch.Tensor, q: int) -> torch.Tensor:
    """Expand a per-chunk mask `[B, n_chunks]` to per-latent-token
    `[B, n_chunks*q]` for the flattened predictor sequence."""
    return chunk_mask.repeat_interleave(q, dim=1)