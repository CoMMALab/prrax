#!/usr/bin/env python3
"""Benchmark CUDA pRRTC on MBM problems using batched solves per problem set."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyronot as pk
import yourdfpy

try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
except Exception:
    pass

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ModuleNotFoundError:
    HAS_TABULATE = False

    def _fmt_cell(v: Any) -> str:
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            fv = float(v)
            if abs(fv) >= 1000.0:
                return f"{fv:.1f}"
            return f"{fv:.6f}".rstrip("0").rstrip(".")
        return str(v)

    def _github_table_from_df(df: pd.DataFrame, headers: list[str]) -> str:
        cols = list(df.columns)
        display_headers = [""] + list(headers or cols)
        lines = ["| " + " | ".join(display_headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(display_headers)) + " |")

        for idx, row in df.iterrows():
            values = [f"{idx}"] + [_fmt_cell(row[c]) for c in cols]
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    def tabulate(data, headers=(), tablefmt=None):
        if isinstance(data, pd.DataFrame) and tablefmt == "github":
            return _github_table_from_df(data, list(headers))
        if isinstance(data, pd.DataFrame):
            return data.to_string()
        return str(data)

from pyronot.collision._obstacles import create_collision_environment
from pyronot.collision._robot_collision import RobotCollisionSpherized


ROOT = Path(__file__).resolve().parent
RESOURCES = ROOT / "pyronot" / "resources"
PRRTC_ROOT = ROOT / "cuda-rrtc" / "jax"
PRRTC_SCRIPTS = ROOT / "pRRTC" / "scripts"
SOLVE_BATCH_CHUNK_SIZE = 25


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRRTC = _load_module("cuda_rrtc_prrtc", PRRTC_ROOT / "prrtc.py")
PRRTC_UTILS = _load_module("cuda_rrtc_utils", PRRTC_ROOT / "utils.py")


def load_robot_dataset(robot: str) -> dict[str, Any]:
    """Load MBM dataset from pyronot/resources, with robust fallbacks for baxter."""
    robot_dir = RESOURCES / robot
    pkl_path = robot_dir / "problems.pkl"
    json_path = robot_dir / "problems.json"
    tar_path = robot_dir / "problems.tar.bz2"

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "problems" in data:
            return data

    if tar_path.exists():
        with tempfile.TemporaryDirectory(prefix=f"{robot}_mbm_") as tmpdir:
            with tarfile.open(tar_path, mode="r:bz2") as tf:
                tf.extractall(tmpdir)
            extracted = Path(tmpdir)
            pkl_candidates = list(extracted.rglob("problems.pkl"))
            json_candidates = list(extracted.rglob("problems.json"))

            if pkl_candidates:
                with open(pkl_candidates[0], "rb") as f:
                    return pickle.load(f)

            if json_candidates:
                with open(json_candidates[0], "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "problems" in data:
                    return data

    fallback = PRRTC_SCRIPTS / f"{robot}_problems.json"
    if fallback.exists():
        print(
            f"Warning: no parsed problems file found in {robot_dir}; "
            f"falling back to {fallback}"
        )
        with open(fallback, "r", encoding="utf-8") as f:
            return json.load(f)

    raise RuntimeError(
        f"Unable to load MBM dataset for robot '{robot}' from {robot_dir}"
    )


def load_robot_models(robot: str):
    urdf_path = RESOURCES / robot / f"{robot}_spherized.urdf"
    srdf_path = RESOURCES / robot / f"{robot}.srdf"

    if not urdf_path.exists():
        raise RuntimeError(f"URDF not found: {urdf_path}")
    if not srdf_path.exists():
        raise RuntimeError(f"SRDF not found: {srdf_path}")

    urdf = yourdfpy.URDF.load(str(urdf_path))
    robot_model = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=str(srdf_path))

    lo = jnp.asarray(robot_model.joints.lower_limits, dtype=jnp.float32)
    hi = jnp.asarray(robot_model.joints.upper_limits, dtype=jnp.float32)

    return robot_model, robot_coll, lo, hi


def _environment_signature(problem_data: dict[str, Any]) -> str:
    """Build a stable signature from obstacle geometry fields used by MBM scenes."""
    env_data = {
        "sphere": problem_data.get("sphere", []),
        "cylinder": problem_data.get("cylinder", []),
        "box": problem_data.get("box", []),
    }
    return json.dumps(env_data, sort_keys=True, separators=(",", ":"))


def _batched_solve_group(
    group_data: list[dict[str, Any]],
    lo,
    hi,
    collision_context,
    *,
    max_iterations: int,
    step_size: float,
    num_new_samples: int,
    granularity: int,
    max_nodes: int,
    balance_mode: int,
    tree_ratio: float,
    dynamic_domain: bool,
    dd_alpha: float,
    dd_radius: float,
    dd_min_radius: float,
    warmup: bool,
    timing_source: str,
    jit_trace: bool,
):
    starts = np.asarray([p["start"] for p in group_data], dtype=np.float32)
    goals = np.asarray([p["goals"] for p in group_data], dtype=np.float32)

    batch_kwargs = dict(
        start_configs=jnp.asarray(starts, dtype=jnp.float32),
        goal_configs=jnp.asarray(goals, dtype=jnp.float32),
        max_iterations=max_iterations,
        step_size=step_size,
        num_new_samples=num_new_samples,
        granularity=granularity,
        max_nodes=max_nodes,
        balance_mode=balance_mode,
        tree_ratio=tree_ratio,
        dynamic_domain=dynamic_domain,
        dd_alpha=dd_alpha,
        dd_radius=dd_radius,
        dd_min_radius=dd_min_radius,
        min_vals=lo,
        max_vals=hi,
        collision_context=collision_context,
        jit_trace=jit_trace,
    )

    if warmup:
        _ = PRRTC.prrtc_plan_batch(**batch_kwargs)

    t0 = time.perf_counter_ns()
    batch_results = PRRTC.prrtc_plan_batch(**batch_kwargs)
    PRRTC_UTILS.synchronize_prrtc_results(batch_results)
    elapsed_ns = time.perf_counter_ns() - t0

    # Host timing is one wall-time for the whole batch.
    # Kernel timing is provided per individual result.
    per_problem_ns = int(elapsed_ns / max(1, len(batch_results)))

    if timing_source == "kernel":
        per_problem_ns_list = []
        for result in batch_results:
            if getattr(result, "kernel_time_ms", None) is not None:
                per_problem_ns_list.append(int(float(result.kernel_time_ms) * 1e6))
            else:
                per_problem_ns_list.append(per_problem_ns)
    else:
        per_problem_ns_list = [per_problem_ns for _ in batch_results]

    return batch_results, per_problem_ns_list


def evaluate_robot(
    robot: str,
    *,
    max_iterations: int,
    step_size: float,
    num_new_samples: int,
    granularity: int,
    max_nodes: int,
    balance_mode: int,
    tree_ratio: float,
    dynamic_domain: bool,
    dd_alpha: float,
    dd_radius: float,
    dd_min_radius: float,
    warmup: bool,
    timing_source: str,
    jit_trace: bool,
    solve_batch_size: int,
    max_problems_per_set: int,
    print_failures: bool,
):
    data = load_robot_dataset(robot)
    robot_model, robot_coll, lo, hi = load_robot_models(robot)

    solve_batch_size = int(solve_batch_size)
    if solve_batch_size <= 0:
        raise ValueError(f"solve_batch_size must be > 0, got {solve_batch_size}")

    problems = data.get("problems", {})
    if not isinstance(problems, dict):
        raise RuntimeError(f"Unexpected problems structure for robot '{robot}'")

    total_problems = 0
    valid_problems = 0
    failed_problems = 0

    results: list[dict[str, Any]] = []

    for problem_name, pset in problems.items():
        if not isinstance(pset, list):
            continue

        failures: list[int] = []
        invalids: list[int] = []

        print(f"Evaluating {robot} on {problem_name}:")
        iterator = pset if max_problems_per_set <= 0 else pset[:max_problems_per_set]

        valid_data: list[dict[str, Any]] = []
        valid_global_indices: list[int] = []
        for i, problem_data in enumerate(iterator):
            total_problems += 1
            if not problem_data.get("valid", False):
                invalids.append(i)
                continue
            valid_problems += 1
            valid_data.append(problem_data)
            valid_global_indices.append(i)

        if not valid_data:
            if print_failures and invalids:
                print(f"  Invalid problems: {invalids}")
            continue

        # Build one collision context per problem, reusing identical scenes.
        context_cache: dict[str, Any] = {}
        per_problem_contexts: list[dict[str, Any]] = []
        for pdata in valid_data:
            sig = _environment_signature(pdata)
            if sig not in context_cache:
                obstacles = create_collision_environment(pdata)
                context_cache[sig] = PRRTC_UTILS.build_prrtc_collision_context(
                    robot_model,
                    robot_coll,
                    obstacles,
                )
            per_problem_contexts.append(context_cache[sig])

        if len(context_cache) > 1:
            print(
                f"  Note: detected {len(context_cache)} scene variants in {problem_name}; "
                "dispatching one batched solve with per-problem collision contexts."
            )

        if len(valid_data) > solve_batch_size:
            n_chunks = (len(valid_data) + solve_batch_size - 1) // solve_batch_size
            print(
                f"  Splitting {len(valid_data)} valid problems into {n_chunks} chunks "
                f"of up to {solve_batch_size}."
            )

        for chunk_start in range(0, len(valid_data), solve_batch_size):
            chunk_end = min(chunk_start + solve_batch_size, len(valid_data))
            chunk_valid_data = valid_data[chunk_start:chunk_end]
            chunk_contexts = per_problem_contexts[chunk_start:chunk_end]
            chunk_global_indices = valid_global_indices[chunk_start:chunk_end]

            batch_results, per_problem_ns_list = _batched_solve_group(
                chunk_valid_data,
                lo,
                hi,
                chunk_contexts,
                max_iterations=max_iterations,
                step_size=step_size,
                num_new_samples=num_new_samples,
                granularity=granularity,
                max_nodes=max_nodes,
                balance_mode=balance_mode,
                tree_ratio=tree_ratio,
                dynamic_domain=dynamic_domain,
                dd_alpha=dd_alpha,
                dd_radius=dd_radius,
                dd_min_radius=dd_min_radius,
                warmup=bool(warmup and chunk_start == 0),
                timing_source=timing_source,
                jit_trace=jit_trace,
            )

            for pos, result in enumerate(batch_results):
                global_idx = chunk_global_indices[pos]
                if not result.solved:
                    failures.append(global_idx)
                    continue

                planning_td = pd.Timedelta(nanoseconds=per_problem_ns_list[pos])
                simplification_td = pd.Timedelta(0)
                total_td = planning_td + simplification_td

                results.append(
                    {
                        "robot": robot,
                        "problem": problem_name,
                        "planning_time": planning_td,
                        "simplification_time": simplification_td,
                        "total_time": total_td,
                        "planning_iterations": int(result.iterations),
                        "initial_path_cost": float(result.cost),
                        "simplified_path_cost": float(result.cost),
                    }
                )

        failed_problems += len(failures)

        if print_failures:
            if invalids:
                print(f"  Invalid problems: {invalids}")
            if failures:
                print(f"  Failed on {sorted(failures)}")

    return results, total_problems, valid_problems, failed_problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pRRTC on MBM datasets (batched per problem)")
    parser.add_argument(
        "--robots",
        nargs="+",
        default=["panda", "fetch", "baxter"],
        choices=["panda", "fetch", "baxter"],
        help="Robots to evaluate",
    )
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--num-new-samples", type=int, default=64)
    parser.add_argument("--granularity", type=int, default=16)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    parser.add_argument("--balance-mode", type=int, default=2)
    parser.add_argument("--tree-ratio", type=float, default=1.0)
    parser.add_argument("--dynamic-domain", action="store_true", default=True)
    parser.add_argument("--no-dynamic-domain", action="store_false", dest="dynamic_domain")
    parser.add_argument("--dd-alpha", type=float, default=1e-4)
    parser.add_argument("--dd-radius", type=float, default=4.0)
    parser.add_argument("--dd-min-radius", type=float, default=1.0)
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", action="store_false", dest="warmup")
    parser.add_argument(
        "--solve-batch-size",
        type=int,
        default=SOLVE_BATCH_CHUNK_SIZE,
        help="Maximum number of problems per pRRTC batch dispatch.",
    )
    parser.add_argument(
        "--timing-source",
        choices=["host", "kernel"],
        default="host",
        help="Use host wall time (default) or kernel-reported time for planning_time.",
    )
    parser.add_argument(
        "--jit-trace",
        action="store_true",
        default=True,
        help="Use cached jax.jit tracing for batched pRRTC FFI dispatch.",
    )
    parser.add_argument(
        "--no-jit-trace",
        action="store_false",
        dest="jit_trace",
        help="Disable jax.jit tracing and call the FFI dispatch path directly.",
    )
    parser.add_argument("--max-problems-per-set", type=int, default=0)
    parser.add_argument("--print-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tick = time.perf_counter()

    all_results: list[dict[str, Any]] = []
    total = 0
    valid = 0
    failed = 0

    for robot in args.robots:
        (
            robot_results,
            robot_total,
            robot_valid,
            robot_failed,
        ) = evaluate_robot(
            robot,
            max_iterations=args.max_iterations,
            step_size=args.step_size,
            num_new_samples=args.num_new_samples,
            granularity=args.granularity,
            max_nodes=args.max_nodes,
            balance_mode=args.balance_mode,
            tree_ratio=args.tree_ratio,
            dynamic_domain=args.dynamic_domain,
            dd_alpha=args.dd_alpha,
            dd_radius=args.dd_radius,
            dd_min_radius=args.dd_min_radius,
            warmup=args.warmup,
            timing_source=args.timing_source,
            jit_trace=args.jit_trace,
            solve_batch_size=args.solve_batch_size,
            max_problems_per_set=args.max_problems_per_set,
            print_failures=args.print_failures,
        )
        all_results.extend(robot_results)
        total += robot_total
        valid += robot_valid
        failed += robot_failed

    if not all_results:
        raise RuntimeError("No solved plans were collected; cannot summarize results")

    df = pd.DataFrame.from_dict(all_results)

    df["planning_time"] = df["planning_time"].dt.microseconds
    df["simplification_time"] = df["simplification_time"].dt.microseconds
    df["avg_time_per_iteration"] = df["planning_iterations"] / df["planning_time"]
    df["total_time"] = df["total_time"].dt.microseconds

    time_stats = df[
        [
            "planning_time",
            "simplification_time",
            "total_time",
            "planning_iterations",
            "avg_time_per_iteration",
        ]
    ].describe(percentiles=[0.25, 0.5, 0.75, 0.95])
    time_stats.drop(index=["count"], inplace=True)

    cost_stats = df[["initial_path_cost", "simplified_path_cost"]].describe(
        percentiles=[0.25, 0.5, 0.75, 0.95]
    )
    cost_stats.drop(index=["count"], inplace=True)

    print()
    print(
        tabulate(
            time_stats,
            headers=[
                "Planning Time (us)",
                "Simplification Time (us)",
                "Total Time (us)",
                "Planning Iters.",
                "Time per Iter. (us)",
            ],
            tablefmt="github",
            **({"floatfmt": ".6f"} if HAS_TABULATE else {}),
        )
    )

    print(
        tabulate(
            cost_stats,
            headers=[
                " Initial Cost (L2)",
                "    Simplified Cost (L2)",
            ],
            tablefmt="github",
            **({"floatfmt": ".6f"} if HAS_TABULATE else {}),
        )
    )

    tock = time.perf_counter()

    print(f"Timing source for planning_time: {args.timing_source}")
    print(f"JIT tracing for batched pRRTC dispatch: {args.jit_trace}")
    print(f"Configured solve batch size: {args.solve_batch_size}")
    print(f"Solved / Valid / Total # Problems: {valid - failed} / {valid} / {total}")
    print(f"Completed all problems in {df['total_time'].sum() / 1000:.3f} milliseconds")
    print(f"Total time including Python overhead: {(tock - tick) * 1000:.3f} milliseconds")


if __name__ == "__main__":
    main()