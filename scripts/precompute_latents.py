"""Build the Phase-2 latent cache and fit the whitening (README §8).

    python -m Sync.scripts.precompute_latents --task gigaword --smoke
"""

from __future__ import annotations

import argparse

from ..codec import SonarCodecAdapter
from ..config import get_preset
from ..data import Chunker, PairLatentDataset
from ..training import freeze_and_scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gigaword")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encode-batch-size", type=int, default=256)
    ap.add_argument("--whiten-mode", default="zca", choices=["zca", "pca"])
    ap.add_argument("--cache-dir", default=None,
                    help="override cache dir (use a separate dir for the held-out cache)")
    ap.add_argument("--no-whiten", action="store_true",
                    help="skip whitening fit (held-out cache must reuse the TRAIN whitening)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke and args.limit is None:
        args.limit = 256

    cfg = get_preset(args.task)
    if args.cache_dir is not None:
        cfg.data.latent_cache_dir = args.cache_dir
    cfg.validate()

    codec = SonarCodecAdapter(cfg.codec, cfg.tokenizer, device=args.device)
    chunker = Chunker(cfg.chunk)

    print(f"precomputing latents: task={args.task} split={args.split} limit={args.limit}")
    PairLatentDataset.precompute_and_cache(
        cfg.data, cfg.predictor, chunker, codec,
        task=args.task, split=args.split, limit=args.limit,
        encode_batch_size=args.encode_batch_size,
    )

    if args.no_whiten:
        print("skipping whitening fit (--no-whiten); held-out cache reuses TRAIN whitening")
    else:
        print("fitting whitening ...")
        w = freeze_and_scale(cfg, mode=args.whiten_mode)
        print(f"whitening fit (d={w.mean.shape[0]}); saved to {cfg.data.latent_cache_dir}")


if __name__ == "__main__":
    main()
