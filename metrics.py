"""Evaluation harness (README §8). Deliberately NOT ROUGE/BLEU: those penalize
the lexical variation we are trying to produce on paraphrase/caption tasks.

- Quality = semantic similarity to reference(s) via an INDEPENDENT sentence
  embedder (sentence-transformers), never SONAR itself (scoring in SONAR space
  would be circular).
- Diversity = multi-reference coverage/recall (primary, non-gameable) +
  sample distinctness (secondary).
- Everything is reported as CURVES vs guidance_scale (CFG trades quality<->diversity).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


class SemanticScorer:
    """Independent sentence embedder for semantic-similarity scoring."""

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"
    ):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def embed(self, texts: list[str]) -> torch.Tensor:
        """-> L2-normalized embeddings [N, h]."""
        emb = self.model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
        )
        return emb

    def cos(self, a: list[str], b: list[str]) -> torch.Tensor:
        """Pairwise cosine sim between aligned lists a[i], b[i] -> [N]."""
        ea, eb = self.embed(a), self.embed(b)
        return (ea * eb).sum(dim=-1)


def quality_semantic_similarity(
    preds: list[str], refs: list[list[str]], scorer: SemanticScorer
) -> float:
    """Mean over examples of max cosine similarity between the prediction and any
    of its references. Multi-reference safe."""
    if not preds:
        return 0.0
    pred_emb = scorer.embed(preds)                       # [N, h]
    scores = []
    for i, ref_set in enumerate(refs):
        if not ref_set:
            continue
        r = scorer.embed(ref_set)                        # [R, h]
        sim = (pred_emb[i][None, :] * r).sum(dim=-1)     # [R]
        scores.append(float(sim.max()))
    return sum(scores) / max(1, len(scores))


def coverage(
    samples: list[list[str]], refs: list[list[str]], scorer: SemanticScorer
) -> float:
    """Multi-reference coverage/recall: for each human reference, the best
    similarity achieved by ANY of the K samples, averaged over refs and prompts.
    This is the non-gameable diversity signal — do K samples collectively recall
    the reference set?"""
    cov = []
    for samp, ref_set in zip(samples, refs):
        if not samp or not ref_set:
            continue
        s = scorer.embed(samp)                           # [K, h]
        r = scorer.embed(ref_set)                        # [R, h]
        sim = r @ s.T                                    # [R, K]
        best_per_ref = sim.max(dim=1).values             # [R]
        cov.append(float(best_per_ref.mean()))
    return sum(cov) / max(1, len(cov))


def self_distinctness(samples: list[list[str]], scorer: SemanticScorer) -> float:
    """Mean pairwise (1 - cosine) among the K samples for a prompt, averaged over
    prompts. Secondary diversity signal (gameable on its own — pair with coverage)."""
    dis = []
    for samp in samples:
        uniq = [s for s in samp if s]
        if len(uniq) < 2:
            continue
        e = scorer.embed(uniq)                           # [K, h]
        sim = e @ e.T                                    # [K, K]
        k = sim.shape[0]
        off = (sim.sum() - sim.diagonal().sum()) / (k * (k - 1))
        dis.append(float(1.0 - off))
    return sum(dis) / max(1, len(dis))


def validity(
    samples: list[list[str]], refs: list[list[str]], scorer: SemanticScorer
) -> float:
    """Per-sample faithfulness — the guard against fake diversity. For each of the
    K samples, its max semantic similarity to ANY reference; averaged over all
    samples and prompts. Coverage can rise with temperature simply by emitting
    junk that happens to spread; validity catches that — it must stay HIGH as `s`
    rises for the spread to count as *valid* diversity (README §8 hybrid)."""
    vals = []
    for samp, ref_set in zip(samples, refs):
        uniq = [s for s in samp if s]
        if not uniq or not ref_set:
            continue
        s = scorer.embed(uniq)                           # [K, h]
        r = scorer.embed(ref_set)                        # [R, h]
        sim = s @ r.T                                    # [K, R]
        best_per_sample = sim.max(dim=1).values          # [K]
        vals.append(float(best_per_sample.mean()))
    return sum(vals) / max(1, len(vals))


def distinct_n(samples: list[list[str]], n: int = 2) -> float:
    """Corpus distinct-n: unique n-grams / total n-grams across all samples."""
    total, uniq = 0, set()
    for samp in samples:
        for s in samp:
            toks = s.split()
            for i in range(len(toks) - n + 1):
                total += 1
                uniq.add(tuple(toks[i : i + n]))
    return len(uniq) / max(1, total)


def guidance_curve(
    prompts: list[str],
    refs: list[list[str]],
    sample_fn: Callable[[str, float], list[str]],
    guidance_values: list[float],
    scorer: SemanticScorer,
) -> dict:
    """Sweep guidance and report quality + diversity curves.

    `sample_fn(prompt, guidance) -> list[str]` returns K samples for one prompt.
    Returns {guidance: {quality, coverage, distinctness, distinct2}}.
    """
    curves = {}
    for g in guidance_values:
        samples = [sample_fn(p, g) for p in prompts]
        first = [s[0] if s else "" for s in samples]
        curves[g] = {
            "quality": quality_semantic_similarity(first, refs, scorer),
            "coverage": coverage(samples, refs, scorer),
            "distinctness": self_distinctness(samples, scorer),
            "distinct2": distinct_n(samples, n=2),
        }
    return curves


def temperature_curve(
    prompts: list[str],
    refs: list[list[str]],
    sample_fn: Callable[[str, float], list[str]],
    temps: list[float],
    scorer: SemanticScorer,
) -> dict:
    """Sweep the hybrid sampling temperature `s` and report the coverage/validity
    curves that decide whether the residual flow buys VALID diversity (README §8).

    `sample_fn(prompt, s) -> list[str]` returns K samples for one prompt at temp s.
    Returns {s: {coverage, validity, distinctness, distinct2, quality}}.

    Read (pre-committed decision rule):
      * coverage ↑ with s AND validity holds high -> residual buys valid diversity.
      * coverage ↑ but validity collapses        -> incoherent spread (fake).
      * coverage ~flat                            -> hybrid ≈ regressor + noise.
    coverage-against-references is the primary diversity signal; raw distinctness is
    a secondary descriptor only (it rewards incoherent variation on its own)."""
    curves = {}
    for s in temps:
        samples = [sample_fn(p, s) for p in prompts]
        first = [smp[0] if smp else "" for smp in samples]
        curves[s] = {
            "coverage": coverage(samples, refs, scorer),
            "validity": validity(samples, refs, scorer),
            "quality": quality_semantic_similarity(first, refs, scorer),
            "distinctness": self_distinctness(samples, scorer),
            "distinct2": distinct_n(samples, n=2),
        }
    return curves
