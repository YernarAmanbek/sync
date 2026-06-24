"""Phase-2 — train the flow-matching predictor on cached latents (README §8).

    python -m Sync.scripts.train_predictor --task gigaword --smoke
"""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from ..config import get_preset
from ..data import PairLatentDataset, collate_pairs
from ..predictor import CountHead, FlowMatchingPredictor
from ..training import train_predictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gigaword")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = get_preset(args.task)
    cfg.validate()
    if args.batch_size is not None:
        cfg.train.phase2.batch_size = args.batch_size
    if args.max_steps is not None:
        cfg.train.phase2.max_steps = args.max_steps
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    if args.smoke:
        cfg.train.phase2.max_steps = min(cfg.train.phase2.max_steps, 60)
        cfg.train.phase2.batch_size = min(cfg.train.phase2.batch_size, 16)
        cfg.train.phase2.warmup_steps = 10
        cfg.train.phase2.ckpt_every = 1000

    ds = PairLatentDataset(cfg.data, cfg.predictor)
    loader = DataLoader(
        ds,
        batch_size=cfg.train.phase2.batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )
    predictor = FlowMatchingPredictor(cfg.predictor)
    count_head = CountHead(cfg.predictor)
    print(f"training predictor: task={args.task} examples={len(ds)} "
          f"steps={cfg.train.phase2.max_steps} bs={cfg.train.phase2.batch_size}")
    train_predictor(predictor, count_head, loader, cfg)


if __name__ == "__main__":
    main()
