"""Gate 0 — reconstruction ceiling (README §8).

With SONAR q=1, a perfect predictor can do no better than SONAR's own
encode->decode round-trip on the TARGET sentences. Measure that ceiling with
semantic similarity (not ROUGE) so we know the headroom before training.

    python -m Sync.scripts.gate_ceiling --task gigaword --smoke
"""

from __future__ import annotations

import argparse

from ..codec import SonarCodecAdapter
from ..config import get_preset
from ..data import Chunker, load_task_pairs
from ..metrics import SemanticScorer, quality_semantic_similarity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gigaword")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke and args.limit is None:
        args.limit = 64

    cfg = get_preset(args.task)
    cfg.validate()
    M = cfg.predictor.n_tgt_chunks

    codec = SonarCodecAdapter(cfg.codec, cfg.tokenizer, device=args.device)
    chunker = Chunker(cfg.chunk)
    scorer = SemanticScorer(model_name=args.embedder)

    originals: list[str] = []
    prompts_first: list[str] = []
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for prompt, response, _refs in load_task_pairs(args.task, split=args.split, limit=args.limit):
        chunks = chunker.chunk(response)[:M]
        if not chunks:
            continue
        start = len(flat)
        flat.extend(chunks)
        spans.append((start, len(flat)))
        originals.append(" ".join(chunks))
        pc = chunker.chunk(prompt)
        prompts_first.append(pc[0] if pc else "")

    if not flat:
        raise RuntimeError("no targets collected")

    emb = codec.encode_texts(flat)
    decoded = codec.decode_latents(emb)
    recon = [" ".join(decoded[s:e]) for s, e in spans]

    ceiling = quality_semantic_similarity(recon, [[o] for o in originals], scorer)
    # floor reference: how similar is the raw input to the target (copy baseline)
    copy_floor = quality_semantic_similarity(prompts_first, [[o] for o in originals], scorer)

    print(f"task={args.task} split={args.split} n={len(originals)}")
    print(f"reconstruction CEILING (sem-sim, target->SONAR->target): {ceiling:.4f}")
    print(f"copy-input FLOOR       (sem-sim, prompt vs target)     : {copy_floor:.4f}")
    print("headroom = ceiling - floor =", round(ceiling - copy_floor, 4))
    for o, r in list(zip(originals, recon))[:5]:
        print("\n  orig :", o)
        print("  recon:", r)


if __name__ == "__main__":
    main()
