"""Scaled-run launcher: build the large train cache + a disjoint held-out cache,
then train the predictor with the in-loop decoder-free latent metric tracked
against the decode-readiness line.

Equivalent to running, in order:
  precompute_latents --split train      --limit <train-limit>
  precompute_latents --split validation --limit <heldout-limit> \
                     --cache-dir <heldout-cache-dir> --no-whiten
  train_predictor    --max-steps ... --lr ... --min-lr-ratio ... \
                     --heldout-cache-dir <heldout-cache-dir> --val-every ... ...

    python -m Sync.scripts.run_scaled --task gigaword            # all defaults below
    python -m Sync.scripts.run_scaled --task gigaword --train-limit 500000 --max-steps 40000
    python -m Sync.scripts.run_scaled --task gigaword --skip-precompute   # reuse caches

Run from the PARENT directory of this package (same rule as the other scripts).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

_ROOT_PKG = (__package__ or "").split(".")[0] or "Sync"


def _run(stage: str, stage_args: list[str]) -> float:
    cmd = [sys.executable, "-m", f"{_ROOT_PKG}.scripts.{stage}", *stage_args]
    print("\n" + "=" * 78)
    print(f"[run_scaled] STAGE: {stage}")
    print(f"[run_scaled] cmd: {' '.join(cmd)}")
    print("=" * 78, flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt = time.time() - t0
    print(f"\n[run_scaled] stage '{stage}' OK ({dt:.1f}s)", flush=True)
    return dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="gigaword")
    ap.add_argument("--device", default="cuda")

    # caches
    ap.add_argument("--train-limit", type=int, default=1_000_000)
    ap.add_argument("--heldout-limit", type=int, default=3000)
    ap.add_argument("--heldout-cache-dir", default="./cache/gigaword_val")
    ap.add_argument("--encode-batch-size", type=int, default=256)

    # training
    ap.add_argument("--max-steps", type=int, default=80_000)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--out-dir", default="./runs")

    # in-loop eval
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--eval-guidance", type=float, default=1.0)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--decode-readiness", type=float, default=0.85)
    ap.add_argument("--sample-eval", action="store_true",
                    help="enable the lagging SONAR sample dump (off by default)")

    # skips
    ap.add_argument("--skip-train-cache", action="store_true")
    ap.add_argument("--skip-heldout-cache", action="store_true")
    ap.add_argument("--skip-precompute", action="store_true", help="skip BOTH caches")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    stages: list[tuple[str, list[str]]] = []

    if not (args.skip_precompute or args.skip_train_cache):
        stages.append(("precompute_latents", [
            "--task", args.task, "--split", "train", "--device", args.device,
            "--limit", str(args.train_limit),
            "--encode-batch-size", str(args.encode_batch_size),
        ]))

    if not (args.skip_precompute or args.skip_heldout_cache):
        stages.append(("precompute_latents", [
            "--task", args.task, "--split", "validation", "--device", args.device,
            "--limit", str(args.heldout_limit),
            "--cache-dir", args.heldout_cache_dir, "--no-whiten",
            "--encode-batch-size", str(args.encode_batch_size),
        ]))

    if not args.skip_train:
        train_args = [
            "--task", args.task, "--out-dir", args.out_dir,
            "--max-steps", str(args.max_steps),
            "--lr", str(args.lr), "--min-lr-ratio", str(args.min_lr_ratio),
            "--heldout-cache-dir", args.heldout_cache_dir,
            "--val-every", str(args.val_every),
            "--eval-n", str(args.eval_n),
            "--eval-guidance", str(args.eval_guidance),
            "--eval-steps", str(args.eval_steps),
            "--decode-readiness", str(args.decode_readiness),
        ]
        if args.sample_eval:
            train_args.append("--sample-eval")
        stages.append(("train_predictor", train_args))

    if not stages:
        print("[run_scaled] nothing to do (all stages skipped)")
        return

    print(f"[run_scaled] task={args.task} stages={[s for s, _ in stages]}")
    timings: list[tuple[str, float]] = []
    t_start = time.time()
    for name, stage_args in stages:
        try:
            dt = _run(name, stage_args)
        except subprocess.CalledProcessError as e:
            print(f"\n[run_scaled] STAGE '{name}' FAILED (exit {e.returncode}). Stopping.", file=sys.stderr)
            print(f"[run_scaled] completed before failure: {[n for n, _ in timings]}", file=sys.stderr)
            sys.exit(e.returncode)
        timings.append((name, dt))

    total = time.time() - t_start
    print("\n" + "=" * 78)
    print("[run_scaled] ALL STAGES COMPLETE")
    for name, dt in timings:
        print(f"  {name:20s} {dt:8.1f}s")
    print(f"  {'TOTAL':20s} {total:8.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
