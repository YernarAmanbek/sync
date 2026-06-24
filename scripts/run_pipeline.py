"""End-to-end pipeline runner (README §8): chains the whole SONAR validation
ladder in one command —

    gate_ceiling -> precompute_latents -> gate_uncond_flow -> train_predictor -> eval_curves

Each stage runs as an isolated subprocess (same as invoking the modules by hand),
so a failure in one stage is reported cleanly and does not corrupt the others.

    python -m Sync.scripts.run_pipeline --task gigaword --limit 100000 --max-steps 30000
    python -m Sync.scripts.run_pipeline --task gigaword --smoke          # tiny end-to-end
    python -m Sync.scripts.run_pipeline --task gigaword --skip-precompute # reuse a cache

Run from the PARENT directory of this package (same rule as the other scripts).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


# top-level package name ("Sync" or "sync"), resolved at runtime so the casing
# on disk does not matter.
_ROOT_PKG = (__package__ or "").split(".")[0] or "Sync"


def _module(stage: str) -> str:
    return f"{_ROOT_PKG}.scripts.{stage}"


def _run(stage: str, stage_args: list[str]) -> float:
    """Run one stage as `python -m <pkg>.scripts.<stage> <args>`; return elapsed
    seconds. Raises CalledProcessError (non-zero exit) to stop the pipeline."""
    cmd = [sys.executable, "-m", _module(stage), *stage_args]
    print("\n" + "=" * 78)
    print(f"[pipeline] STAGE: {stage}")
    print(f"[pipeline] cmd: {' '.join(cmd)}")
    print("=" * 78, flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt = time.time() - t0
    print(f"\n[pipeline] stage '{stage}' OK ({dt:.1f}s)", flush=True)
    return dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="gigaword")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run (passed to every stage)")

    # --- precompute_latents ---
    ap.add_argument("--limit", type=int, default=100_000, help="precompute: #pairs to cache")
    ap.add_argument("--precompute-split", default="train")
    ap.add_argument("--encode-batch-size", type=int, default=256)
    ap.add_argument("--whiten-mode", default="zca", choices=["zca", "pca"])

    # --- train_predictor ---
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--out-dir", default="./runs")

    # --- gate_ceiling ---
    ap.add_argument("--ceiling-limit", type=int, default=None)

    # --- eval_curves ---
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--eval-limit", type=int, default=200)
    ap.add_argument("--eval-k", type=int, default=5)
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--eval-steps", type=int, default=None, help="ODE steps override for eval")
    ap.add_argument("--eval-show", type=int, default=0,
                    help="print this many (prompt, reference, samples) examples per guidance")
    ap.add_argument("--ckpt", default=None, help="eval ckpt (default: <out-dir>/predictor_final.pt)")
    ap.add_argument("--no-ema", action="store_true")

    # --- stage skips ---
    ap.add_argument("--skip-ceiling", action="store_true")
    ap.add_argument("--skip-precompute", action="store_true")
    ap.add_argument("--skip-uncond", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    smoke = ["--smoke"] if args.smoke else []
    ckpt = args.ckpt or os.path.join(args.out_dir, "predictor_final.pt")

    # build the ordered stage list (stage_name, arg_list), honoring skips
    stages: list[tuple[str, list[str]]] = []

    if not args.skip_ceiling:
        a = ["--task", args.task, "--split", args.eval_split, "--device", args.device, *smoke]
        if args.ceiling_limit is not None:
            a += ["--limit", str(args.ceiling_limit)]
        stages.append(("gate_ceiling", a))

    if not args.skip_precompute:
        a = ["--task", args.task, "--split", args.precompute_split, "--device", args.device,
             "--encode-batch-size", str(args.encode_batch_size),
             "--whiten-mode", args.whiten_mode, *smoke]
        if args.limit is not None:
            a += ["--limit", str(args.limit)]
        stages.append(("precompute_latents", a))

    if not args.skip_uncond:
        stages.append(("gate_uncond_flow", ["--task", args.task, "--device", args.device, *smoke]))

    if not args.skip_train:
        a = ["--task", args.task, "--out-dir", args.out_dir, *smoke]
        if args.max_steps is not None:
            a += ["--max-steps", str(args.max_steps)]
        if args.batch_size is not None:
            a += ["--batch-size", str(args.batch_size)]
        if args.num_workers is not None:
            a += ["--num-workers", str(args.num_workers)]
        stages.append(("train_predictor", a))

    if not args.skip_eval:
        a = ["--task", args.task, "--ckpt", ckpt, "--split", args.eval_split,
             "--limit", str(args.eval_limit), "--k", str(args.eval_k),
             "--device", args.device, "--guidance", *[str(g) for g in args.guidance], *smoke]
        if args.eval_steps is not None:
            a += ["--steps", str(args.eval_steps)]
        if args.eval_show > 0:
            a += ["--show", str(args.eval_show)]
        if args.no_ema:
            a += ["--no-ema"]
        stages.append(("eval_curves", a))

    if not stages:
        print("[pipeline] nothing to do (all stages skipped)")
        return

    print(f"[pipeline] task={args.task} smoke={args.smoke} stages={[s for s, _ in stages]}")
    timings: list[tuple[str, float]] = []
    t_start = time.time()
    for name, stage_args in stages:
        try:
            dt = _run(name, stage_args)
        except subprocess.CalledProcessError as e:
            print(f"\n[pipeline] STAGE '{name}' FAILED (exit {e.returncode}). Stopping.", file=sys.stderr)
            print(f"[pipeline] completed before failure: {[n for n, _ in timings]}", file=sys.stderr)
            sys.exit(e.returncode)
        timings.append((name, dt))

    total = time.time() - t_start
    print("\n" + "=" * 78)
    print("[pipeline] ALL STAGES COMPLETE")
    for name, dt in timings:
        print(f"  {name:20s} {dt:8.1f}s")
    print(f"  {'TOTAL':20s} {total:8.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
