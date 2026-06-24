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

import json
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .codec import CodecInterface
from .config import ChunkConfig, DataConfig, PredictorConfig, TokenizerConfig


# --------------------------------------------------------------------------- #
# Whitening (README §8) — replaces naive per-dim scaling.
# --------------------------------------------------------------------------- #
@dataclass
class Whitening:
    """Full ZCA/PCA whitening of the SONAR latent space.

    SONAR is not a KL-regularized VAE latent: it is anisotropic and
    inter-correlated, so per-dim std scaling leaves the covariance intact and the
    flow still has to rotate isotropic noise onto a correlated shell. Whitening
    mean-centers and decorrelates so the flow's N(0, I) source matches the target
    by construction. Store `W` (apply) and `W_inv` (invert before decode).

    Shapes: mean [d], W [d, d], W_inv [d, d]. Operates on the last dim of any
    [..., d] tensor.
    """

    mean: torch.Tensor
    W: torch.Tensor
    W_inv: torch.Tensor

    def to(self, device, dtype=torch.float32) -> "Whitening":
        return Whitening(
            self.mean.to(device=device, dtype=dtype),
            self.W.to(device=device, dtype=dtype),
            self.W_inv.to(device=device, dtype=dtype),
        )

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """[..., d] un-whitened -> whitened (≈ zero-mean, identity-cov)."""
        return (x - self.mean) @ self.W.T

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """[..., d] whitened -> original SONAR space (do this before decode)."""
        return x @ self.W_inv.T + self.mean

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            mean=self.mean.cpu().numpy(),
            W=self.W.cpu().numpy(),
            W_inv=self.W_inv.cpu().numpy(),
        )

    @staticmethod
    def load(path: str) -> "Whitening":
        if not path.endswith(".npz"):
            path = path + ".npz"
        d = np.load(path)
        return Whitening(
            torch.from_numpy(d["mean"]).float(),
            torch.from_numpy(d["W"]).float(),
            torch.from_numpy(d["W_inv"]).float(),
        )


def compute_latent_whitening(
    latents: torch.Tensor, mode: str = "zca", eps: float = 1e-5
) -> Whitening:
    """Fit whitening on a sample of UN-scaled SONAR latents.

    `latents`: [N, d] (flatten the q dim before calling; q=1 here).
    `mode`: "zca" (symmetric, stays near the original basis) or "pca".
    Returns a `Whitening` with mean, W, W_inv.
    """
    x = latents.reshape(-1, latents.shape[-1]).to(torch.float64)
    mean = x.mean(dim=0)
    xc = x - mean
    n = xc.shape[0]
    cov = (xc.T @ xc) / max(1, n - 1)                  # [d, d]
    # symmetric eigendecomposition of the covariance
    evals, evecs = torch.linalg.eigh(cov)              # ascending
    inv_sqrt = torch.diag(1.0 / torch.sqrt(evals + eps))
    sqrt = torch.diag(torch.sqrt(evals + eps))
    if mode == "pca":
        W = inv_sqrt @ evecs.T
        W_inv = evecs @ sqrt
    elif mode == "zca":
        W = evecs @ inv_sqrt @ evecs.T
        W_inv = evecs @ sqrt @ evecs.T
    else:
        raise ValueError(f"unknown whitening mode {mode!r}")
    return Whitening(mean.float(), W.float(), W_inv.float())


# --------------------------------------------------------------------------- #
# Chunking + tokenization
# --------------------------------------------------------------------------- #
class Chunker:
    """Sentence segmentation + length banding. Turns a document/string into a
    list of chunk strings, each roughly within [min_tokens, max_tokens].

    On the SONAR path we do NOT need token-accurate banding (SONAR tokenizes
    internally), so length is approximated by whitespace word count unless a real
    `Tokenizer` is supplied. Sentences below `min_tokens` are merged with the
    next; sentences above `max_tokens` are split at word boundaries (or dropped
    if `band_overlong == "drop"`)."""

    def __init__(self, cfg: ChunkConfig, tok: Optional["Tokenizer"] = None):
        self.cfg = cfg
        self.tok = tok

    def _count(self, text: str) -> int:
        if self.tok is not None:
            return self.tok.length(text)
        return len(text.split())

    def _segment(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        try:
            from syntok import segmenter  # type: ignore

            sents: list[str] = []
            for paragraph in segmenter.process(text):
                for sentence in paragraph:
                    s = "".join(t.spacing + t.value for t in sentence).strip()
                    if s:
                        sents.append(s)
            if sents:
                return sents
        except Exception:
            pass
        # fallback: naive sentence split
        import re

        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def chunk(self, text: str) -> list[str]:
        sents = self._segment(text)
        lo, hi = self.cfg.min_tokens, self.cfg.max_tokens

        # merge runts forward
        merged: list[str] = []
        buf = ""
        for s in sents:
            buf = (buf + " " + s).strip() if buf else s
            if self._count(buf) >= lo:
                merged.append(buf)
                buf = ""
        if buf:
            if merged:
                merged[-1] = (merged[-1] + " " + buf).strip()
            else:
                merged.append(buf)

        # enforce upper bound
        out: list[str] = []
        for s in merged:
            if self._count(s) <= hi:
                out.append(s)
                continue
            if self.cfg.band_overlong == "drop":
                continue
            words = s.split()
            for i in range(0, len(words), hi):
                piece = " ".join(words[i : i + hi]).strip()
                if piece:
                    out.append(piece)
        return out


class Tokenizer:
    """Thin wrapper over a pretrained HF subword tokenizer. Only needed for the
    custom-codec path (and optional token-accurate banding); the SONAR path does
    not require it. Loads lazily so importing the package never pulls transformers.

    Adds a [MASK] token if absent and records special-token ids on `cfg`."""

    def __init__(self, cfg: TokenizerConfig, max_tokens: int):
        self.cfg = cfg
        self.max_tokens = max_tokens  # L
        from transformers import AutoTokenizer  # type: ignore

        self.hf = AutoTokenizer.from_pretrained(cfg.name)
        if self.hf.mask_token is None:
            self.hf.add_special_tokens({"mask_token": "[MASK]"})
        cfg.vocab_size = len(self.hf)
        cfg.pad_id = self.hf.pad_token_id if self.hf.pad_token_id is not None else 0
        cfg.bos_id = self.hf.bos_token_id if self.hf.bos_token_id is not None else -1
        cfg.eos_id = self.hf.eos_token_id if self.hf.eos_token_id is not None else -1
        cfg.mask_id = self.hf.mask_token_id

    def length(self, text: str) -> int:
        return len(self.hf.encode(text, add_special_tokens=False))

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (ids [L], pad_mask [L] bool True=keep). Truncates/pads to L."""
        ids, mask = self.encode_batch([text])
        return ids[0], mask[0]

    def encode_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (ids [B, L], pad_mask [B, L])."""
        enc = self.hf(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"].bool()

    def decode(self, ids: torch.Tensor) -> str:
        return self.hf.decode(ids.tolist(), skip_special_tokens=True)


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
# Task pair loaders (rung 0/1/2). Each yields (prompt, response, refs).
# `refs` is the full reference set for eval (multi-ref for MSCOCO); training
# uses (prompt, response).
# --------------------------------------------------------------------------- #
def _hf_load(path, split: str, name: Optional[str] = None):
    """Robust loader across `datasets` 3.x and 4.x.

    `path` may be a single repo id or a list of candidate ids tried in order
    (canonical owner-prefixed id first, legacy alias last). Owner-prefixed ids
    are required since HF moved legacy datasets under owner accounts (e.g.
    `gigaword` -> `Harvard/gigaword`); the bare aliases now 404 intermittently.

    `datasets>=4.0` removed script-based loading and `trust_remote_code`. For each
    candidate id we try, in order: (1) plain load (parquet-native repos and 3.x),
    (2) with trust_remote_code=True (3.x script datasets like classic gigaword),
    (3) the Hub's parquet export branch `refs/convert/parquet` (4.x, for datasets
    that only had a loader script on main but were auto-converted to parquet).

    NOTE: a script-only dataset with no parquet export (gigaword's case) cannot be
    loaded on `datasets>=4.0` by any path. Use `datasets>=3.5,<4` (see
    requirements.txt) for those tasks."""
    from datasets import load_dataset  # type: ignore

    paths = [path] if isinstance(path, str) else list(path)
    last = None
    for p in paths:
        base = dict(path=p, split=split)
        if name is not None:
            base["name"] = name
        for kw in (base, {**base, "trust_remote_code": True},
                   {**base, "revision": "refs/convert/parquet"}):
            try:
                return load_dataset(**kw)
            except Exception as e:  # TypeError (kwarg removed), DatasetNotFoundError, etc.
                last = e
    raise last


def load_task_pairs(
    task: str, split: str = "train", limit: Optional[int] = None
) -> Iterator[tuple[str, str, list[str]]]:
    """Stream (prompt, response, refs) for a task. Uses HF `datasets`; falls back
    cleanly across datasets 3.x/4.x (parquet export branch)."""

    def _take(it):
        for i, ex in enumerate(it):
            if limit is not None and i >= limit:
                break
            yield ex

    if task == "gigaword":
        ds = _hf_load(["Harvard/gigaword", "gigaword"], split)
        for ex in _take(ds):
            doc, summary = ex["document"], ex["summary"]
            if doc and summary:
                yield doc, summary, [summary]

    elif task == "xsum":
        ds = _hf_load(["EdinburghNLP/xsum", "xsum"], split)
        for ex in _take(ds):
            doc, summary = ex["document"], ex["summary"]
            if doc and summary:
                yield doc, summary, [summary]

    elif task == "cnn_dailymail":
        ds = _hf_load(["abisee/cnn_dailymail", "cnn_dailymail"], split, name="3.0.0")
        for ex in _take(ds):
            doc, summary = ex["article"], ex["highlights"]
            if doc and summary:
                yield doc, summary, [summary]

    elif task == "mscoco":
        # group the 5 captions per image; prompt = one caption, target = another,
        # refs = all captions (multi-reference coverage at eval).
        ds = _hf_load("yerevann/coco-karpathy", split)
        for ex in _take(ds):
            caps = ex.get("sentences") or ex.get("captions") or []
            caps = [c for c in caps if c]
            if len(caps) < 2:
                continue
            yield caps[0], caps[1], caps

    else:
        raise ValueError(f"unknown task {task!r}")


# --------------------------------------------------------------------------- #
# Phase 2 dataset — precomputed latent pairs (memmap, padded canvases)
# --------------------------------------------------------------------------- #
class PairLatentDataset(Dataset):
    """Random-access dataset over precomputed latent pairs. Build the cache ONCE
    with `precompute_and_cache`, then instantiate for training.

    __getitem__ returns dict:
      {"context": [N_ctx, q, d], "context_mask": [N_ctx] bool,
       "target":  [M, q, d],     "target_mask":  [M] bool,
       "n": scalar, "m": scalar}
    Latents in the cache are UN-whitened; the trainer applies the Whitening.
    """

    def __init__(self, cfg: DataConfig, pcfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg
        self.pcfg = pcfg
        meta_path = os.path.join(cfg.latent_cache_dir, "meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)
        d = self.meta["d"]
        q = self.meta["q"]
        self.N_ctx = self.meta["N_ctx"]
        self.M = self.meta["M"]
        self.num = self.meta["num"]
        cd = os.path.join(cfg.latent_cache_dir, "context.f32")
        td = os.path.join(cfg.latent_cache_dir, "target.f32")
        self.context = np.memmap(
            cd, dtype="float32", mode="r", shape=(self.num, self.N_ctx, q, d)
        )
        self.target = np.memmap(
            td, dtype="float32", mode="r", shape=(self.num, self.M, q, d)
        )
        self.n = np.load(os.path.join(cfg.latent_cache_dir, "n.npy"))
        self.m = np.load(os.path.join(cfg.latent_cache_dir, "m.npy"))

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, idx: int) -> dict:
        n = int(self.n[idx])
        m = int(self.m[idx])
        context = torch.from_numpy(np.ascontiguousarray(self.context[idx]))
        target = torch.from_numpy(np.ascontiguousarray(self.target[idx]))
        ctx_mask = torch.zeros(self.N_ctx, dtype=torch.bool)
        ctx_mask[:n] = True
        tgt_mask = torch.zeros(self.M, dtype=torch.bool)
        tgt_mask[:m] = True
        return {
            "context": context,
            "context_mask": ctx_mask,
            "target": target,
            "target_mask": tgt_mask,
            "n": n,
            "m": m,
        }

    @staticmethod
    @torch.no_grad()
    def precompute_and_cache(
        cfg: DataConfig,
        pcfg: PredictorConfig,
        chunker: Chunker,
        codec,  # SonarCodecAdapter (text-native)
        task: str,
        split: str = "train",
        limit: Optional[int] = None,
        encode_batch_size: int = 256,
    ) -> None:
        """One-time pass. For each (prompt, response) pair:
          1. chunker.chunk both sides (same chunker everywhere)
          2. codec.encode_texts -> latent means [n_i, q, d] and [m_i, q, d]
          3. write into padded memmaps; record (n_i, m_i); keep refs for eval
        Pairs with n > N_ctx or m > M are clipped to the canvas.
        Latents are stored UN-whitened. Encoding is batched across the chunks of
        many examples for throughput.
        """
        d = pcfg.latent_dim
        q = pcfg.latents_per_chunk
        N_ctx, M = pcfg.n_ctx_chunks, pcfg.n_tgt_chunks
        os.makedirs(cfg.latent_cache_dir, exist_ok=True)

        ctx_path = os.path.join(cfg.latent_cache_dir, "context.f32")
        tgt_path = os.path.join(cfg.latent_cache_dir, "target.f32")
        refs_path = os.path.join(cfg.latent_cache_dir, "refs.jsonl")

        # chunk all pairs first (cheap, CPU); then batch the GPU encode.
        rows: list[dict] = []
        for prompt, response, refs in load_task_pairs(task, split=split, limit=limit):
            ctx_chunks = chunker.chunk(prompt)[:N_ctx]
            tgt_chunks = chunker.chunk(response)[:M]
            if not ctx_chunks or not tgt_chunks:
                continue
            rows.append(
                {"ctx": ctx_chunks, "tgt": tgt_chunks, "refs": refs}
            )

        num = len(rows)
        if num == 0:
            raise RuntimeError(f"no usable pairs for task={task!r} split={split!r}")

        context = np.memmap(
            ctx_path, dtype="float32", mode="w+", shape=(num, N_ctx, q, d)
        )
        target = np.memmap(
            tgt_path, dtype="float32", mode="w+", shape=(num, M, q, d)
        )
        n_arr = np.zeros(num, dtype="int32")
        m_arr = np.zeros(num, dtype="int32")

        # encode in batches: flatten chunks across rows, remember offsets
        flat_texts: list[str] = []
        slots: list[tuple[int, str, int]] = []  # (row_idx, "ctx"/"tgt", within)
        for ri, row in enumerate(rows):
            for j, c in enumerate(row["ctx"]):
                flat_texts.append(c)
                slots.append((ri, "ctx", j))
            for j, c in enumerate(row["tgt"]):
                flat_texts.append(c)
                slots.append((ri, "tgt", j))
            n_arr[ri] = len(row["ctx"])
            m_arr[ri] = len(row["tgt"])

        for start in range(0, len(flat_texts), encode_batch_size):
            batch = flat_texts[start : start + encode_batch_size]
            emb = codec.encode_texts(batch).cpu().numpy()  # [b, q, d]
            for k, (ri, side, j) in enumerate(slots[start : start + encode_batch_size]):
                if side == "ctx":
                    context[ri, j] = emb[k]
                else:
                    target[ri, j] = emb[k]

        context.flush()
        target.flush()
        np.save(os.path.join(cfg.latent_cache_dir, "n.npy"), n_arr)
        np.save(os.path.join(cfg.latent_cache_dir, "m.npy"), m_arr)
        with open(refs_path, "w") as f:
            for row in rows:
                f.write(json.dumps({"refs": row["refs"], "tgt": row["tgt"]}) + "\n")
        with open(os.path.join(cfg.latent_cache_dir, "meta.json"), "w") as f:
            json.dump(
                {"num": num, "N_ctx": N_ctx, "M": M, "q": q, "d": d, "task": task,
                 "split": split},
                f,
            )


def collate_pairs(batch: list[dict]) -> dict:
    """Stack pair dicts -> batched tensors:
      {"context":[B,N_ctx,q,d], "context_mask":[B,N_ctx],
       "target":[B,M,q,d], "target_mask":[B,M], "n":[B], "m":[B]}.
    """
    return {
        "context": torch.stack([b["context"] for b in batch]),
        "context_mask": torch.stack([b["context_mask"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "target_mask": torch.stack([b["target_mask"] for b in batch]),
        "n": torch.tensor([b["n"] for b in batch], dtype=torch.long),
        "m": torch.tensor([b["m"] for b in batch], dtype=torch.long),
    }