"""Inference — the end-to-end `str -> str` assembly (README §1, step list).

Holds the FROZEN codec, the EMA-weighted predictor + count head, the tokenizer,
the chunker, and the latent scale. Nothing here trains."""

from __future__ import annotations

import os
from typing import Optional

import torch

from .codec import CodecInterface
from .config import Config
from .data import Chunker, Tokenizer, Whitening
from .predictor import (
    CountHead,
    FlowMatchingPredictor,
    FlowSampler,
    HybridPredictor,
    HybridSampler,
    ood_score,
)


class TextGenerator:
    """Wire-up of the trained system for generation.

    Pure-flow path: pass `predictor` (FlowMatchingPredictor) -> FlowSampler.
    Hybrid path (additive): pass `hybrid` (HybridPredictor) -> HybridSampler, and
    `generate`/`sample_many` accept a `temperature` (s) kwarg (0=pure mean,
    1=full residual). Exactly one of `predictor` / `hybrid` must be given; when
    `hybrid` is None the flow behavior is unchanged."""

    def __init__(
        self,
        cfg: Config,
        codec,  # SonarCodecAdapter (text-native)
        predictor: Optional[FlowMatchingPredictor],
        count_head: CountHead,
        chunker: Chunker,
        tokenizer: Optional[Tokenizer] = None,
        device: str = "cuda",
        whitening: Optional[Whitening] = None,
        hybrid: Optional[HybridPredictor] = None,
    ):
        self.cfg = cfg
        self.codec = codec
        self.count_head = count_head.to(device).eval()
        self.chunker = chunker
        self.tok = tokenizer
        self.device = device
        if whitening is None:
            whitening = Whitening.load(
                os.path.join(cfg.data.latent_cache_dir, "whitening.npz")
            )
        self.whitening = whitening.to(device)
        self.is_hybrid = hybrid is not None
        if self.is_hybrid:
            self.hybrid = hybrid.to(device).eval()
            self.predictor = None
            self.sampler = HybridSampler(self.hybrid, count_head, cfg.predictor)
        else:
            assert predictor is not None, "pass either predictor or hybrid"
            self.predictor = predictor.to(device).eval()
            self.sampler = FlowSampler(predictor, count_head, cfg.predictor)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Full pipeline:
          1. chunks = chunker.chunk(prompt); tokens/pad_mask = tok.encode_batch(chunks)
          2. C = codec.encode_chunk(tokens, pad_mask)   # [n, q, d] means
             pad C to [1, N_ctx, q, d], build context_mask, apply latent_scale,
             flatten to [1, N_ctx*q, d]
          3. (optional) if cfg.infer.ood_gate: s = ood_score(C_unscaled, ...);
             if s > cfg.infer.ood_threshold: return a fallback / refusal string
          4. Z_scaled, m = sampler.sample(C, context_mask, steps, guidance_scale)
             unscale Z -> [M, q, d]; keep first m chunks
          5. for each of the first m latents: ids = codec.decode_latent(z_chunk)
             text_chunk = tok.decode(ids); join chunks (space/newline) -> response
        Returns the response string.

        Returns the response string."""
        pcfg = self.cfg.predictor
        N_ctx, q, d = pcfg.n_ctx_chunks, pcfg.latents_per_chunk, pcfg.latent_dim

        chunks = self.chunker.chunk(prompt)[:N_ctx]
        if not chunks:
            return ""

        # 1-2. encode + whiten + pad to the context canvas
        C_un = self.codec.encode_texts(chunks).to(self.device)   # [n, q, d]
        C_w = self.whitening.apply(C_un)
        n = C_w.shape[0]
        C = torch.zeros(1, N_ctx, q, d, device=self.device)
        C[0, :n] = C_w
        ctx_mask = torch.zeros(1, N_ctx, dtype=torch.bool, device=self.device)
        ctx_mask[0, :n] = True
        from .components import expand_chunk_mask

        ctx_tok_mask = expand_chunk_mask(ctx_mask, q)            # [1, N_ctx*q]
        C_flat = C.reshape(1, N_ctx * q, d)

        # 3. optional OOD gate (score the unpadded, un-whitened context latents)
        if self.cfg.infer.ood_gate:
            s = ood_score(C_un.unsqueeze(0))   # [1, n, q, d], all valid
            if float(s[0]) > self.cfg.infer.ood_threshold:
                return ""  # fallback / refusal

        # 4. sample target latents (whitened), then un-whiten
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)
        if self.is_hybrid:
            Z_w, m = self.sampler.sample(
                C_flat, ctx_tok_mask, steps=steps, guidance_scale=guidance_scale,
                generator=gen, temperature=temperature,
            )
        else:
            Z_w, m = self.sampler.sample(
                C_flat, ctx_tok_mask, steps=steps, guidance_scale=guidance_scale, generator=gen
            )
        Z = self.whitening.invert(Z_w).reshape(pcfg.n_tgt_chunks, q, d)
        m = int(m[0].clamp(max=self.cfg.infer.max_response_chunks))

        # 5. decode the first m chunks back to text
        texts = self.codec.decode_latents(Z[:m])
        return " ".join(t.strip() for t in texts if t.strip())

    @torch.no_grad()
    def sample_many(
        self,
        prompt: str,
        k: int,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> list[str]:
        """K independent samples for one prompt (for diversity/coverage eval).

        On the hybrid path at `temperature==0` the sampler is deterministic (pure
        mean), so all K samples are identical by construction — the s=0 coverage
        baseline (one effective output)."""
        outs = []
        for i in range(k):
            s = None if seed is None else seed + i
            outs.append(
                self.generate(
                    prompt, steps=steps, guidance_scale=guidance_scale, seed=s,
                    temperature=temperature,
                )
            )
        return outs

    @torch.no_grad()
    def generate_batch(self, prompts: list[str], **kw) -> list[str]:
        """Simple per-prompt loop (correctness over throughput for the MVP)."""
        return [self.generate(p, **kw) for p in prompts]