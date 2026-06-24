"""Inference — the end-to-end `str -> str` assembly (README §1, step list).

Holds the FROZEN codec, the EMA-weighted predictor + count head, the tokenizer,
the chunker, and the latent scale. Nothing here trains."""

from __future__ import annotations

from typing import Optional

import torch

from .codec import CodecInterface
from .config import Config
from .data import Chunker, Tokenizer
from .predictor import CountHead, FlowMatchingPredictor, FlowSampler, ood_score


class TextGenerator:
    """Wire-up of the trained system for generation.

    AGENT TASK (constructor): load frozen codec (build_codec + load_ckpt), load
    predictor + count_head with EMA weights, build Chunker/Tokenizer from cfg
    (SAME as training), load cfg.latent_scale, construct FlowSampler. Move to
    device, eval()."""

    def __init__(
        self,
        cfg: Config,
        codec: CodecInterface,
        predictor: FlowMatchingPredictor,
        count_head: CountHead,
        chunker: Chunker,
        tokenizer: Tokenizer,
        device: str = "cuda",
    ):
        self.cfg = cfg
        self.codec = codec
        self.predictor = predictor
        self.count_head = count_head
        self.chunker = chunker
        self.tok = tokenizer
        self.device = device
        self.scale = torch.as_tensor(cfg.latent_scale, device=device) if cfg.latent_scale else None
        self.sampler = FlowSampler(predictor, count_head, cfg.predictor)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
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

        AGENT TASK: implement steps 1-5; handle empty/over-long prompts; respect
        cfg.infer.max_response_chunks; seed the sampler's generator if seed given."""
        raise NotImplementedError("AGENT: TextGenerator.generate")

    @torch.no_grad()
    def generate_batch(self, prompts: list[str], **kw) -> list[str]:
        """Batched variant for throughput. AGENT TASK: pad across prompts in the
        chunk dimension, run the pipeline once, split results per prompt."""
        raise NotImplementedError("AGENT: TextGenerator.generate_batch")