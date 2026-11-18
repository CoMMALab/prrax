#!/usr/bin/env python3
"""Batch solve: N MBM problems (spread over the robot's sets) in ONE prrtc_plan_batch call per
configuration: the loop checkers (binary, binary_coarse) against collision_checker="traced_edge"
(the whole waypoint check traced by mantis) at each mantis tier. Reports solved count, the batch's
kernel time (max over problems of the per-problem solve time: the launch's makespan), the
per-problem median, and the host wall time of the call (after a warm-up call that also builds).

    python benchmark/batch_edge_tiers.py --robot panda --problems 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_mbm_batched as M  # noqa: E402
from pyroffi.collision._obstacles import create_collision_environment  # noqa: E402

CONFIGS = {   # name: (collision_checker, warps_per_problem, edge_tier)
    "binary W=2": ("binary", 2, None),
    "binary_coarse W=2": ("binary_coarse", 2, None),
    "traced thread W=1": ("traced_edge", 1, ("thread", 1)),
    "traced thread W=2": ("traced_edge", 2, ("thread", 1)),
    "traced warp P=32 W=2": ("traced_edge", 2, ("warp", 32)),
    "traced warp P=16 W=2": ("traced_edge", 2, ("warp", 16)),
    "traced warp P=8 W=2": ("traced_edge", 2, ("warp", 8)),
    "traced warp P=4 W=2": ("traced_edge", 2, ("warp", 4)),
    "traced warp P=2 W=2": ("traced_edge", 2, ("warp", 2)),
    "traced warp P=1 W=2": ("traced_edge", 2, ("warp", 1)),
    "traced block k=2 P=32": ("traced_edge", 2, ("block", 32)),
    "traced block k=2 P=8": ("traced_edge", 2, ("block", 8)),
    "traced block k=4 P=32": ("traced_edge", 4, ("block", 32)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--robot", default="panda", choices=["panda", "fetch", "baxter"])
    ap.add_argument("--problems", type=int, default=100)
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated names of CONFIGS")
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--num-new-samples", type=int, default=64)
    ap.add_argument("--max-nodes", type=int, default=200_000)
    args = ap.parse_args()

    robot, coll, lo, hi = M.load_robot_models(args.robot)
    valid = [p for ps in M.load_robot_dataset(args.robot)["problems"].values() for p in ps if p.get("valid")]
    probs = [valid[i] for i in np.linspace(0, len(valid) - 1, args.problems).astype(int)]
    cache = {}
    ctxs = []
    for p in probs:
        sig = M._environment_signature(p)
        if sig not in cache:
            cache[sig] = M.PRRTC_UTILS.build_prrtc_collision_context(robot, coll, create_collision_environment(p))
        ctxs.append(cache[sig])
    kw = dict(start_configs=np.asarray([p["start"] for p in probs], np.float32),
              goal_configs=np.asarray([p["goals"] for p in probs], np.float32),
              collision_context=ctxs, max_iterations=args.max_iterations,
              step_size=M.STEP_SIZE_BY_ROBOT[args.robot], num_new_samples=args.num_new_samples,
              max_nodes=args.max_nodes, min_vals=lo, max_vals=hi)
    print(f"{args.robot}: {len(probs)} problems in one batch, {len(cache)} distinct worlds", flush=True)
    print(f"| config | solved | batch kernel (makespan) | per-problem median | call wall |\n|---|---|---|---|---|")
    for name in args.configs.split(","):
        checker, warps, tier = CONFIGS[name]
        extra = dict(collision_checker=checker, warps_per_problem=warps)
        if tier:
            extra["edge_tier"] = tier
        try:
            t0 = time.perf_counter()
            M.PRRTC.prrtc_plan_batch(**kw, **extra)                       # warm-up (and build)
            build = time.perf_counter() - t0
            t0 = time.perf_counter()
            rs = M.PRRTC.prrtc_plan_batch(**kw, **extra)
            wall = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001 - report and go on with the next tier
            print(f"| {name} | failed: {str(e).splitlines()[-1][:120]} | | | |", flush=True)
            continue
        k = np.array([r.kernel_time_ms for r in rs])
        print(f"| {name} | {sum(r.solved for r in rs)}/{len(rs)} | {k.max():.1f} ms | {np.median(k):.2f} ms | "
              f"{wall * 1e3:.0f} ms (first call {build:.0f} s) |", flush=True)


if __name__ == "__main__":
    main()
