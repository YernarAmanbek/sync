"""Multi-reference coverage-validation launcher (the hybrid diversity thesis test).

Chains, in order:
  precompute_latents --task mscoco --split train      --limit <train-limit>
  precompute_latents --task mscoco --split validation --limit <heldout-limit> \
                     --cache-dir <heldout-cache-dir> --no-whiten
  train_hybrid       --task mscoco --out-dir <out-dir> --heldout-cache-dir ... \
                     --max-steps ... --lr ... --min-lr-ratio ... --eval-temp ...
  eval_curves        --task mscoco --ckpt <out-dir>/hybrid_best.pt \
                     --k ... --temps 0 0.5 1.0 1.5 --guidance ... --show ...

mscoco is the multi-reference task: load_task_pairs yields all captions as refs, so
coverage (the diversity signal) and validity (the fake-diversity guard) are scored
against the full reference set. The sweep over temperature s is the decisive read —
coverage ↑ with s while validity holds => the residual buys VALID diversity.

    python -m Sync.scripts.run_mscoco                       # all defaults below
    python -m Sync.scripts.run_mscoco --skip-precompute     # reuse caches
    python -m Sync.scripts.run_mscoco --skip-train          # just (re)run the sweep
    python -m Sync.scripts.run_mscoco --skip-precompute --skip-train   # sweep only

Default --out-dir is ./runs/mscoco so it does NOT clobber the gigaword hybrid_best.pt.
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
    print(f"[run_mscoco] STAGE: {stage}")
    print(f"[run_mscoco] cmd: {' '.join(cmd)}")
    print("=" * 78, flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt = time.time() - t0
    print(f"\n[run_mscoco] stage '{stage}' OK ({dt:.1f}s)", flush=True)
    return dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="mscoco")
    ap.add_argument("--device", default="cuda")

    # caches
    ap.add_argument("--train-limit", type=int, default=200_000)
    ap.add_argument("--heldout-limit", type=int, default=1000)
    ap.add_argument("--heldout-cache-dir", default="./cache/mscoco_val")
    ap.add_argument("--encode-batch-size", type=int, default=256)

    # training (hybrid)
    ap.add_argument("--max-steps", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--out-dir", default="./runs/mscoco")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--eval-guidance", type=float, default=1.0)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--eval-temp", type=float, default=1.0)
    ap.add_argument("--mean-weight", type=float, default=None)
    ap.add_argument("--sample-eval", action="store_true",
                    help="lagging SONAR sample dump during training (off by default)")

    # coverage sweep (eval_curves)
    ap.add_argument("--k", type=int, default=8, help="samples per prompt for coverage")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
    ap.add_argument("--eval-limit", type=int, default=200, help="held-out prompts to score")
    ap.add_argument("--show", type=int, default=10,
                    help="(input, refs, K-samples) examples to dump")
    ap.add_argument("--show-temp", type=float, default=1.0,
                    help="temperature s for the example dump (default 1.0)")

    # skips
    ap.add_argument("--skip-train-cache", action="store_true")
    ap.add_argument("--skip-heldout-cache", action="store_true")
    ap.add_argument("--skip-precompute", action="store_true", help="skip BOTH caches")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    ckpt = f"{args.out_dir.rstrip('/')}/hybrid_best.pt"
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
            "--eval-temp", str(args.eval_temp),
        ]
        if args.mean_weight is not None:
            train_args += ["--mean-weight", str(args.mean_weight)]
        if args.sample_eval:
            train_args.append("--sample-eval")
        stages.append(("train_hybrid", train_args))

    if not args.skip_sweep:
        sweep_args = [
            "--task", args.task, "--ckpt", ckpt,
            "--split", "validation", "--limit", str(args.eval_limit),
            "--k", str(args.k),
            "--temps", *[str(t) for t in args.temps],
            "--guidance", str(args.eval_guidance),
            "--show", str(args.show),
            "--show-temp", str(args.show_temp),
            "--device", args.device,
        ]
        stages.append(("eval_curves", sweep_args))

    if not stages:
        print("[run_mscoco] nothing to do (all stages skipped)")
        return

    print(f"[run_mscoco] task={args.task} out_dir={args.out_dir} ckpt={ckpt}")
    print(f"[run_mscoco] stages={[s for s, _ in stages]}")
    timings: list[tuple[str, float]] = []
    t_start = time.time()
    for name, stage_args in stages:
        try:
            dt = _run(name, stage_args)
        except subprocess.CalledProcessError as e:
            print(f"\n[run_mscoco] STAGE '{name}' FAILED (exit {e.returncode}). Stopping.",
                  file=sys.stderr)
            print(f"[run_mscoco] completed before failure: {[n for n, _ in timings]}",
                  file=sys.stderr)
            sys.exit(e.returncode)
        timings.append((name, dt))

    total = time.time() - t_start
    print("\n" + "=" * 78)
    print("[run_mscoco] ALL STAGES COMPLETE")
    for name, dt in timings:
        print(f"  {name:20s} {dt:8.1f}s")
    print(f"  {'TOTAL':20s} {total:8.1f}s")
    print("=" * 78)
    print("Read the sweep table: coverage ↑ with s AND validity holding => the residual "
          "buys VALID diversity (writeup-worthy); coverage flat => hybrid ≈ regressor.")


if __name__ == "__main__":
    main()
