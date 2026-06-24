"""Stage A — the frozen codec. Defines a smooth, decodable sentence-latent space.

`LatentCodec` (own VAE, README §6 default) and `SonarCodecAdapter` (frozen
pretrained) both satisfy `CodecInterface`. Everything downstream depends ONLY on
the interface, never on the concrete class.

Critical contract (README §7):
- `encode_chunk` returns the posterior MEAN (deterministic), used for conditioning
  and for caching Phase-2 latents.
- `decode_latent` renders a latent back to token ids (own VAE: parallel CMLM
  refinement; SONAR: autoregressive).
- Latents returned/consumed here are UN-scaled; the latent_scale (README §7) is
  applied by the caller (data/predictor/generate), not inside the codec."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    LearnedPositionalEmbedding,
    PerceiverResampler,
    RMSNorm,
    TransformerStack,
    lengths_to_mask,
)
from .config import CodecConfig, TokenizerConfig


# --------------------------------------------------------------------------- #
# Interface (the only thing downstream code imports)
# --------------------------------------------------------------------------- #
@runtime_checkable
class CodecInterface(Protocol):
    latent_dim: int          # d
    latents_per_chunk: int   # q

    def encode_chunk(
        self, tokens: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """`tokens [B, L]`, `pad_mask [B, L]` (True=keep) -> mean latent [B, q, d]."""
        ...

    def decode_latent(
        self, z: torch.Tensor, steps: Optional[int] = None
    ) -> torch.Tensor:
        """`z [B, q, d]` -> token ids `[B, L]` (padded). Parallel for own VAE,
        autoregressive for SONAR."""
        ...


# --------------------------------------------------------------------------- #
# Own VAE — encoder
# --------------------------------------------------------------------------- #
class VAEEncoder(nn.Module):
    """tokens -> transformer -> Perceiver pool to q vectors -> (mean, logvar).

    AGENT TASK: token embedding (V -> d_model), add positions, run TransformerStack
    (self-attn only), pool with PerceiverResampler to [B, q, d], then two linear
    heads producing mean and logvar, each [B, q, d]."""

    def __init__(self, cfg: CodecConfig, tok: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(tok.vocab_size, cfg.d_model, padding_idx=tok.pad_id)
        self.pos = LearnedPositionalEmbedding(cfg.max_tokens, cfg.d_model)
        self.backbone = TransformerStack(
            cfg.d_model, cfg.n_heads, cfg.enc_layers, cfg.ffn_mult, cfg.dropout, cross_attn=False
        )
        self.pool = PerceiverResampler(
            cfg.d_model, cfg.latent_dim, cfg.latents_per_chunk, cfg.n_heads
        )
        # AGENT: self.mean_head / self.logvar_head : Linear(d, d)
        raise NotImplementedError("AGENT: build mean/logvar heads")

    def forward(
        self, tokens: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> mean [B, q, d], logvar [B, q, d]."""
        raise NotImplementedError("AGENT: forward of VAEEncoder")


# --------------------------------------------------------------------------- #
# Own VAE — NAT (CMLM) decoder
# --------------------------------------------------------------------------- #
class NATDecoder(nn.Module):
    """Non-autoregressive decoder. Renders a full chunk in parallel, conditioned
    on the latent via cross-attention (token positions are queries; the q latent
    vectors are the cross-attention keys/values). Trained CMLM-style: a random
    subset of target tokens is masked and predicted from latent + unmasked tokens.

    AGENT TASK: token embedding shared/separate from encoder, add positions, run
    TransformerStack(cross_attn=True) with `context=z`, project to logits [B, L, V]."""

    def __init__(self, cfg: CodecConfig, tok: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(tok.vocab_size, cfg.d_model, padding_idx=tok.pad_id)
        self.pos = LearnedPositionalEmbedding(cfg.max_tokens, cfg.d_model)
        # latents may need projection d -> d_model before serving as cross KV
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.backbone = TransformerStack(
            cfg.d_model, cfg.n_heads, cfg.dec_layers, cfg.ffn_mult, cfg.dropout, cross_attn=True
        )
        # AGENT: self.out_head : Linear(d_model, V) (optionally tie with token_embed)
        raise NotImplementedError("AGENT: build output head")

    def forward(
        self,
        masked_tokens: torch.Tensor,   # [B, L] with mask_id at masked positions
        pad_mask: torch.Tensor,        # [B, L] bool, True=keep
        z: torch.Tensor,               # [B, q, d]
    ) -> torch.Tensor:                 # logits [B, L, V]
        raise NotImplementedError("AGENT: forward of NATDecoder")


class LengthHead(nn.Module):
    """Predicts the chunk's token count from the pooled latent (needed because a
    NAT decoder must know the output length up front).

    AGENT TASK: pool z [B, q, d] -> [B, d] (mean over q), MLP -> logits over
    0..L (length classes = L+1)."""

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.max_tokens = cfg.max_tokens
        # AGENT: small MLP latent_dim -> (L+1)
        raise NotImplementedError("AGENT: build LengthHead")

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # [B, q, d] -> [B, L+1]
        raise NotImplementedError("AGENT: forward of LengthHead")


# --------------------------------------------------------------------------- #
# Own VAE — composed codec with losses + iterative decode
# --------------------------------------------------------------------------- #
class LatentCodec(nn.Module):
    """Encoder + NAT decoder + length head, plus the Phase-1 training objective
    and the inference-time iterative decode. Satisfies CodecInterface."""

    def __init__(self, cfg: CodecConfig, tok: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = tok
        self.latent_dim = cfg.latent_dim
        self.latents_per_chunk = cfg.latents_per_chunk
        self.encoder = VAEEncoder(cfg, tok)
        self.decoder = NATDecoder(cfg, tok)
        self.length_head = LengthHead(cfg)

    # --- reparameterization (concrete) ---
    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    # --- CMLM masking (concrete spec; agent may optimize) ---
    def cmlm_mask(
        self, tokens: torch.Tensor, pad_mask: torch.Tensor, mask_ratio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace a `mask_ratio` fraction of non-pad tokens with mask_id.
        `mask_ratio` is `[B]` (sampled ~U(low, high] per example).
        Returns (masked_tokens [B, L], loss_mask [B, L] True where masked).
        AGENT TASK: implement the random selection respecting pad_mask."""
        raise NotImplementedError("AGENT: cmlm_mask")

    # --- losses ---
    def kl_loss(self, mean: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
        """Per-dim KL(q(z|x) || N(0,I)) with a free-bits floor, then summed.
        Provided concretely — this is the posterior-collapse-sensitive term."""
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())   # [B, q, d]
        kl = torch.clamp(kl, min=free_bits)                     # free bits per dim
        return kl.sum(dim=(1, 2)).mean()                        # scalar

    def forward(
        self,
        tokens: torch.Tensor,      # [B, L]
        pad_mask: torch.Tensor,    # [B, L] bool
        lengths: torch.Tensor,     # [B] true token counts (for length loss)
        beta: float,               # current KL weight (annealed by trainer)
        mask_ratio: torch.Tensor,  # [B]
    ) -> dict[str, torch.Tensor]:
        """Phase-1 step. Returns {"loss", "recon", "kl", "length"} (all scalars).

        AGENT TASK:
          1. mean, logvar = encoder(tokens, pad_mask)
          2. z = reparameterize(mean, logvar)
          3. masked, loss_mask = cmlm_mask(tokens, pad_mask, mask_ratio)
          4. logits = decoder(masked, pad_mask, z)
          5. recon = CE(logits, tokens) over loss_mask positions only
          6. kl = kl_loss(mean, logvar, free_bits)
          7. length = CE(length_head(z), lengths)
          8. loss = recon + beta * kl + length
        """
        raise NotImplementedError("AGENT: LatentCodec.forward (Phase-1 objective)")

    # --- CodecInterface ---
    @torch.no_grad()
    def encode_chunk(self, tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        mean, _ = self.encoder(tokens, pad_mask)   # MEAN, not a sample
        return mean                                # [B, q, d]

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """Mask-Predict inference (README §1, step 4):
          a. predict length per item -> build pad_mask
          b. start fully masked
          c. for `steps` iterations: predict all tokens, keep high-confidence,
             re-mask the lowest-confidence fraction, re-predict
        Returns token ids [B, L].
        AGENT TASK: implement the iterative refinement loop."""
        raise NotImplementedError("AGENT: LatentCodec.decode_latent (Mask-Predict)")


# --------------------------------------------------------------------------- #
# Buy — frozen SONAR adapter (same interface)
# --------------------------------------------------------------------------- #
class SonarCodecAdapter(nn.Module):
    """Wraps a frozen pretrained SONAR sentence encoder + decoder to expose the
    identical CodecInterface. Skips Phase 1 entirely (README §6). Decode is
    autoregressive (SONAR's decoder), so `steps` is ignored.

    AGENT TASK: load SONAR encoder/decoder via fairseq2/sonar, freeze params,
    map our (tokens, pad_mask) <-> SONAR's expected inputs, and `latent_dim`/`q`
    to SONAR's embedding size (q is typically 1)."""

    def __init__(self, cfg: CodecConfig, tok: TokenizerConfig):
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.latents_per_chunk = cfg.latents_per_chunk
        raise NotImplementedError("AGENT: load + freeze SONAR, wire I/O")

    @torch.no_grad()
    def encode_chunk(self, tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("AGENT: SONAR encode -> [B, q, d]")

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError("AGENT: SONAR autoregressive decode -> [B, L]")


def build_codec(cfg: CodecConfig, tok: TokenizerConfig) -> CodecInterface:
    """Factory honoring the build-vs-buy switch."""
    if cfg.use_pretrained_codec:
        return SonarCodecAdapter(cfg, tok)
    return LatentCodec(cfg, tok)