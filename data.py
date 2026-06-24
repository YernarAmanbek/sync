"""Data pipeline for both phases.

Phase 1 consumes raw sentences (`CodecChunkDataset`). Phase 2 consumes
(prompt, response) pairs, but the predictor never sees text — it sees latents.
So `PairLatentDataset` is built in two stages: a one-time `precompute_and_cache`
pass that runs the FROZEN codec over every chunk and memmaps the latents, then
cheap random access at train time.

NON-NEGOTIABLE (README §7): the `Chunker` + `Tokenizer` instances here MUST be
identical to the ones used to train the codec. Construct them once from `Config`
and reuse."""

from __future__ import annotations

import os
from typing import Iterable, Iterator, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

from .codec import CodecInterface
from .config import ChunkConfig, DataConfig, PredictorConfig, TokenizerConfig


# --------------------------------------------------------------------------- #
# Chunking + tokenization
# --------------------------------------------------------------------------- #
class Chunker:
    """Sentence segmentation + length banding. Turns a document/string into a
    list of chunk strings each within [min_tokens, max_tokens] (approx; exact
    enforcement happens after tokenization).

    AGENT TASK: wrap the chosen segmenter (syntok/spacy/nltk); merge fragments
    below min_tokens with neighbors; split sentences over max_tokens at clause
    boundaries (fallback: hard token split)."""

    def __init__(self, cfg: ChunkConfig, tok: "Tokenizer"):
        self.cfg = cfg
        self.tok = tok

    def chunk(self, text: str) -> list[str]:
        raise NotImplementedError("AGENT: Chunker.chunk")


class Tokenizer:
    """Thin wrapper over a pretrained HF subword tokenizer. Fixes pad/bos/eos/mask
    ids into TokenizerConfig and enforces length L.

    AGENT TASK: load tokenizer by name; ensure a [MASK] token exists (add if not)
    and record mask_id; implement encode/decode with padding+truncation to L."""

    def __init__(self, cfg: TokenizerConfig, max_tokens: int):
        self.cfg = cfg
        self.max_tokens = max_tokens  # L
        raise NotImplementedError("AGENT: load HF tokenizer, set special-token ids on cfg")

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (ids [L], pad_mask [L] bool True=keep). Truncates/pads to L."""
        raise NotImplementedError("AGENT: Tokenizer.encode")

    def encode_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (ids [B, L], pad_mask [B, L])."""
        raise NotImplementedError("AGENT: Tokenizer.encode_batch")

    def decode(self, ids: torch.Tensor) -> str:
        raise NotImplementedError("AGENT: Tokenizer.decode")


# --------------------------------------------------------------------------- #
# Phase 1 dataset — raw chunks (self-supervised)
# --------------------------------------------------------------------------- #
class CodecChunkDataset(IterableDataset):
    """Streams the codec corpus, yields one chunk at a time. The chunk is both
    input and target.

    Yields dict: {"tokens": [L], "pad_mask": [L], "length": scalar}.

    AGENT TASK: stream shards from cfg.codec_corpus_paths (sharded by worker),
    Chunker.chunk each document, Tokenizer.encode each chunk, compute length."""

    def __init__(self, cfg: DataConfig, chunker: Chunker, tok: Tokenizer):
        super().__init__()
        self.cfg = cfg
        self.chunker = chunker
        self.tok = tok

    def __iter__(self) -> Iterator[dict]:
        raise NotImplementedError("AGENT: CodecChunkDataset.__iter__")


def collate_chunks(batch: list[dict]) -> dict:
    """Stack chunk dicts -> {"tokens":[B,L], "pad_mask":[B,L], "lengths":[B]}.
    AGENT TASK: torch.stack the fields."""
    raise NotImplementedError("AGENT: collate_chunks")


# --------------------------------------------------------------------------- #
# Latent scaling factor (README §7)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_latent_scale(
    codec: CodecInterface,
    loader: Iterable[dict],
    sample_size: int,
) -> torch.Tensor:
    """Encode ~sample_size chunks with the FROZEN codec, return per-dim 1/std as
    `[d]`. Stored into Config.latent_scale; applied to all latents before the
    predictor and inverted before decoding.

    AGENT TASK: accumulate encoder means over batches until sample_size reached,
    compute std over the (B*q) axis per latent dim, return reciprocal (eps-guarded)."""
    raise NotImplementedError("AGENT: compute_latent_scale")


# --------------------------------------------------------------------------- #
# Phase 2 dataset — precomputed latent pairs
# --------------------------------------------------------------------------- #
class PairLatentDataset(Dataset):
    """Random-access dataset over precomputed latent pairs. Build the cache ONCE
    with `precompute_and_cache`, then instantiate for training.

    __getitem__ returns dict:
      {"context": [N_ctx, q, d], "context_mask": [N_ctx] bool,
       "target":  [M, q, d],     "target_mask":  [M] bool,
       "n": scalar, "m": scalar}
    Latents in the cache are UN-scaled; the trainer applies Config.latent_scale.

    AGENT TASK: memmap reader keyed by example index; pad variable n/m to
    N_ctx/M; build masks."""

    def __init__(self, cfg: DataConfig, pcfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg
        self.pcfg = pcfg
        # AGENT: open memmaps in cfg.latent_cache_dir, read index/length table
        raise NotImplementedError("AGENT: open latent cache")

    def __len__(self) -> int:
        raise NotImplementedError("AGENT: PairLatentDataset.__len__")

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError("AGENT: PairLatentDataset.__getitem__")

    @staticmethod
    @torch.no_grad()
    def precompute_and_cache(
        cfg: DataConfig,
        pcfg: PredictorConfig,
        chunker: Chunker,
        tok: Tokenizer,
        codec: CodecInterface,
        device: str = "cuda",
    ) -> None:
        """One-time pass. For each (prompt, response) pair:
          1. chunker.chunk both sides (same chunker as Phase 1!)
          2. tok.encode each chunk -> tokens/pad_mask
          3. codec.encode_chunk -> latent means [n_i, q, d] and [m_i, q, d]
          4. append to memmap arrays in cfg.latent_cache_dir, record (n_i, m_i)
        Drop/clip pairs whose n > N_ctx or m > M.
        AGENT TASK: stream pairs, batch the encode for throughput, write memmaps."""
        raise NotImplementedError("AGENT: PairLatentDataset.precompute_and_cache")


def collate_pairs(batch: list[dict]) -> dict:
    """Stack pair dicts -> batched tensors:
      {"context":[B,N_ctx,q,d], "context_mask":[B,N_ctx],
       "target":[B,M,q,d], "target_mask":[B,M], "n":[B], "m":[B]}.
    AGENT TASK: torch.stack the fields."""
    raise NotImplementedError("AGENT: collate_pairs")