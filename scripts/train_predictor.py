"""Phase-2 — train the flow-matching predictor on cached latents (README §8).

    python -m Sync.scripts.train_predictor --task gigaword --smoke
"""

from __future__ import annotations

import argparse
from dataclasses import replace

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
    ap.add_argument("--lr", type=float, default=None, help="peak LR override")
    ap.add_argument("--min-lr-ratio", type=float, default=None,
                    help="cosine floor: LR decays to min_lr_ratio*lr instead of 0")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    # held-out evaluation (decoder-free latent metric + val loss, tracked in-loop)
    ap.add_argument("--heldout-cache-dir", default=None,
                    help="validation latent cache (disjoint from training); enables in-loop eval")
    ap.add_argument("--val-every", type=int, default=None, help="eval cadence in steps")
    ap.add_argument("--eval-n", type=int, default=300, help="prompts per split for the latent metric")
    ap.add_argument("--eval-guidance", type=float, default=1.0)
    ap.add_argument("--eval-steps", type=int, default=50, help="ODE steps for eval sampling")
    ap.add_argument("--decode-readiness", type=float, default=0.85,
                    help="held-out cosine target line printed each eval")
    ap.add_argument("--sample-eval", action="store_true",
                    help="enable the lagging SONAR sample dump (loads SONAR; OFF by default — "
                         "turn on for a spot-check once heldout_cos climbs past ~0.7)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = get_preset(args.task)
    cfg.validate()
    if args.batch_size is not None:
        cfg.train.phase2.batch_size = args.batch_size
    if args.max_steps is not None:
        cfg.train.phase2.max_steps = args.max_steps
    if args.lr is not None:
        cfg.train.phase2.lr = args.lr
    if args.min_lr_ratio is not None:
        cfg.train.phase2.min_lr_ratio = args.min_lr_ratio
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    if args.val_every is not None:
        cfg.train.phase2.val_every = args.val_every
    if args.smoke:
        cfg.train.phase2.max_steps = min(cfg.train.phase2.max_steps, 60)
        cfg.train.phase2.batch_size = min(cfg.train.phase2.batch_size, 16)
        cfg.train.phase2.warmup_steps = 10
        cfg.train.phase2.ckpt_every = 1000
        cfg.train.phase2.val_every = min(cfg.train.phase2.val_every, 30)

    ds = PairLatentDataset(cfg.data, cfg.predictor)
    loader = DataLoader(
        ds,
        batch_size=cfg.train.phase2.batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )

    heldout_ds = None
    if args.heldout_cache_dir is not None:
        heldout_cfg = replace(cfg.data, latent_cache_dir=args.heldout_cache_dir)
        heldout_ds = PairLatentDataset(heldout_cfg, cfg.predictor)

    predictor = FlowMatchingPredictor(cfg.predictor)
    count_head = CountHead(cfg.predictor)
    print(f"training predictor: task={args.task} examples={len(ds)} "
          f"steps={cfg.train.phase2.max_steps} bs={cfg.train.phase2.batch_size} "
          f"heldout={'none' if heldout_ds is None else len(heldout_ds)}")
    train_predictor(
        predictor, count_head, loader, cfg,
        heldout_ds=heldout_ds,
        eval_n=args.eval_n,
        eval_guidance=args.eval_guidance,
        eval_steps=args.eval_steps,
        decode_readiness=args.decode_readiness,
        sample_eval=args.sample_eval,
    )


if __name__ == "__main__":
    main()
