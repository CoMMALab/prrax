#!/usr/bin/env python3
"""Benchmark CUDA pRRTC on MBM problems from pyroffi resources."""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyroffi as pk
import yourdfpy
from tqdm import tqdm

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

from pyroffi.collision._obstacles import create_collision_environment
from pyroffi.collision._robot_collision import RobotCollisionSpherized

# Maximum joints the CUDA FK kernel supports (PRRTC_MAX_JOINTS in prrtc_planner.cu).
_PRRTC_MAX_JOINTS = 64


def verify_robot_collision_context(
    robot_name: str,
    robot_model,
    robot_coll,
    lo: "np.ndarray",
    hi: "np.ndarray",
    collision_context: dict,
    *,
    start: "np.ndarray | None" = None,
    goals: "np.ndarray | None" = None,
) -> None:
    """Hard assertions verifying pyroffi↔cuda-rrtc collision-context consistency.

    Catches URDF/SRDF parsing mismatches that silently break multi-EEF robots
    by making every configuration appear to be in collision.

    Checks
    ------
    1.  Joint-limit ordering sanity (lo < hi for all actuated joints).
    2.  Config-dimension agreement: start/goal dim == n_act == lo/hi dim.
    3.  fk_act_idx range: every value is -1 (fixed) or in [0, n_act).
    4.  All actuated joint indices 0..n_act-1 appear in fk_act_idx (no orphaned DOFs).
    5.  fk_mimic_act_idx range: every value is -1 or in [0, n_act).
    6.  sphere_link_idx range: every value is in [0, n_joints) -- no -1 base-link refs
        that would corrupt FK lookups.
    7.  n_joints <= PRRTC_MAX_JOINTS (CUDA stack array limit).
    8.  No always-colliding self-collision pairs at the zero configuration -- a
        structural SRDF gap that permanently blocks all configurations.
    9.  If start/goals provided: every start and every goal is collision-free
        according to pyroffi (world + self).  Catches joint-ordering mismatches
        between the dataset and the robot model.
    """
    fk_act_idx = np.asarray(collision_context["fk_act_idx"])
    fk_mimic_act_idx = np.asarray(collision_context["fk_mimic_act_idx"])
    sphere_link_idx = np.asarray(collision_context["sphere_link_idx"])

    n_act = int(robot_model.joints.num_actuated_joints)
    n_joints = int(robot_model.joints.num_joints)
    lo_np = np.asarray(lo)
    hi_np = np.asarray(hi)

    # 1. Joint-limit ordering
    assert np.all(lo_np < hi_np), (
        f"[{robot_name}] Joint limits violated: lo >= hi for joints "
        f"{np.where(lo_np >= hi_np)[0].tolist()}"
    )

    # 2. Config-dimension agreement
    assert lo_np.shape[0] == n_act, (
        f"[{robot_name}] lo.shape[0]={lo_np.shape[0]} != n_act={n_act}"
    )
    assert hi_np.shape[0] == n_act, (
        f"[{robot_name}] hi.shape[0]={hi_np.shape[0]} != n_act={n_act}"
    )
    if start is not None:
        start_np = np.asarray(start)
        assert start_np.shape[-1] == n_act, (
            f"[{robot_name}] start dim {start_np.shape[-1]} != n_act {n_act}. "
            "Dataset joint ordering may not match pyroffi's actuated_joints ordering."
        )
    if goals is not None:
        goals_np = np.asarray(goals)
        assert goals_np.shape[-1] == n_act, (
            f"[{robot_name}] goals dim {goals_np.shape[-1]} != n_act {n_act}. "
            "Dataset joint ordering may not match pyroffi's actuated_joints ordering."
        )

    # 3. fk_act_idx range
    assert np.all((fk_act_idx == -1) | ((fk_act_idx >= 0) & (fk_act_idx < n_act))), (
        f"[{robot_name}] fk_act_idx contains values outside [-1, n_act={n_act}): "
        f"{fk_act_idx[(fk_act_idx != -1) & ((fk_act_idx < 0) | (fk_act_idx >= n_act))].tolist()}"
    )

    # 4. All actuated joint indices covered (no orphaned DOFs in multi-EEF robots)
    covered = set(fk_act_idx[fk_act_idx >= 0].tolist())
    missing = set(range(n_act)) - covered
    assert not missing, (
        f"[{robot_name}] Actuated joint indices {sorted(missing)} do not appear in "
        "fk_act_idx -- those DOFs are invisible to the CUDA FK. "
        "Check actuated_joints ordering in the spherized URDF."
    )

    # 5. fk_mimic_act_idx range
    assert np.all(
        (fk_mimic_act_idx == -1) | ((fk_mimic_act_idx >= 0) & (fk_mimic_act_idx < n_act))
    ), (
        f"[{robot_name}] fk_mimic_act_idx contains out-of-range values."
    )

    # 6. sphere_link_idx range: -1 is allowed (base-link spheres; the CUDA kernel
    #    skips them via `if (link_idx < 0 || link_idx >= n_joints) continue`).
    #    Any value < -1 or >= n_joints is genuinely out of range and will corrupt
    #    T_world array accesses.
    if sphere_link_idx.size > 0:
        assert np.all((sphere_link_idx >= -1) & (sphere_link_idx < n_joints)), (
            f"[{robot_name}] sphere_link_idx contains values outside [-1, n_joints={n_joints}): "
            f"min={sphere_link_idx.min()}, max={sphere_link_idx.max()}. "
            "Values < -1 or >= n_joints will produce out-of-bounds T_world reads in the CUDA FK."
        )

    # 7. CUDA FK stack limit
    assert n_joints <= _PRRTC_MAX_JOINTS, (
        f"[{robot_name}] n_joints={n_joints} exceeds PRRTC_MAX_JOINTS={_PRRTC_MAX_JOINTS}. "
        "The CUDA FK allocates a fixed stack of that size."
    )

    # 8. No always-colliding self-collision pairs at zero configuration.
    #    A constant negative distance means the SRDF is missing an adjacent-link
    #    disable entry -- the CUDA planner will reject every configuration.
    zero_cfg = jnp.zeros(n_act, dtype=jnp.float32)
    self_dists_zero = np.asarray(
        robot_coll.compute_self_collision_distance(robot_model, zero_cfg)
    )
    always_coll_mask = self_dists_zero < 0
    if np.any(always_coll_mask):
        active_i = np.asarray(robot_coll.active_idx_i)
        active_j = np.asarray(robot_coll.active_idx_j)
        bad_pairs = [
            (robot_coll.link_names[active_i[k]], robot_coll.link_names[active_j[k]],
             float(self_dists_zero[k]))
            for k in np.where(always_coll_mask)[0]
        ]
        raise AssertionError(
            f"[{robot_name}] {len(bad_pairs)} self-collision pair(s) are always in collision "
            f"at the zero configuration. These missing SRDF disable_collisions entries will "
            f"cause the CUDA planner to reject EVERY configuration:\n"
            + "\n".join(
                f"  {ln1} <-> {ln2}  (dist={d:.6f})" for ln1, ln2, d in bad_pairs
            )
        )

    # 9. Start/goal collision-free check (self-collision only; world obstacles are
    #    problem-specific so we test structural validity against an empty environment).
    def _self_collision_report(cfg):
        min_self = float(np.min(np.asarray(
            robot_coll.compute_self_collision_distance(robot_model, jnp.asarray(cfg, dtype=jnp.float32))
        )))
        return min_self

    if start is not None:
        min_self = _self_collision_report(start)
        assert min_self > -1e-3, (
            f"[{robot_name}] Start configuration is in self-collision "
            f"(min_self={min_self:.6f}). "
            "The dataset may use a different joint ordering than pyroffi's actuated_joints."
        )
    if goals is not None:
        goals_arr = np.asarray(goals).reshape(-1, n_act)
        for gi, g in enumerate(goals_arr):
            min_self = _self_collision_report(g)
            assert min_self > -1e-3, (
                f"[{robot_name}] Goal[{gi}] is in self-collision "
                f"(min_self={min_self:.6f}). "
                "The dataset may use a different joint ordering than pyroffi's actuated_joints."
            )


ROOT = Path(__file__).resolve().parent
RESOURCES = ROOT / "pyroffi" / "resources"
PRRTC_ROOT = ROOT / "cuda-rrtc" / "jax"
TQDM_DISABLE = not sys.stdout.isatty()
STEP_SIZE_BY_ROBOT = {
    "panda": 0.5,
    "fetch": 0.5,
    "baxter": 0.5,
}


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
    """Load MBM dataset from pyroffi/resources."""
    robot_dir = RESOURCES / robot
    pkl_path = robot_dir / "problems.pkl"

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    raise RuntimeError(
        f"Unable to load MBM dataset for robot '{robot}' from {robot_dir}; "
        "expected problems.pkl"
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
    max_problems_per_set: int,
    print_failures: bool,
):
    data = load_robot_dataset(robot)
    robot_model, robot_coll, lo, hi = load_robot_models(robot)

    problems = data.get("problems", {})
    if not isinstance(problems, dict):
        raise RuntimeError(f"Unexpected problems structure for robot '{robot}'")

    total_problems = 0
    valid_problems = 0
    failed_problems = 0
    did_warmup = not warmup

    results: list[dict[str, Any]] = []

    # Run structural verification once per robot before any planning.
    # Build a dummy context (no world obstacles) to check FK array invariants.
    _dummy_ctx = PRRTC_UTILS.build_prrtc_collision_context(robot_model, robot_coll, [])
    verify_robot_collision_context(
        robot, robot_model, robot_coll, np.asarray(lo), np.asarray(hi), _dummy_ctx
    )

    _first_valid_verified = False

    for problem_name, pset in problems.items():
        if not isinstance(pset, list):
            continue

        failures: list[int] = []
        invalids: list[int] = []

        print(f"Evaluating {robot} on {problem_name}:")
        iterator = pset if max_problems_per_set <= 0 else pset[:max_problems_per_set]

        for i, problem_data in tqdm(list(enumerate(iterator)), disable=TQDM_DISABLE):
            total_problems += 1

            if not problem_data.get("valid", False):
                invalids.append(i)
                continue

            valid_problems += 1

            start = jnp.asarray(problem_data["start"], dtype=jnp.float32)
            goals = jnp.asarray(problem_data["goals"], dtype=jnp.float32)
            obstacles = create_collision_environment(problem_data)
            collision_context = PRRTC_UTILS.build_prrtc_collision_context(
                robot_model,
                robot_coll,
                obstacles,
            )

            # Verify start/goal collision-freedom once per robot (first valid problem).
            if not _first_valid_verified:
                verify_robot_collision_context(
                    robot,
                    robot_model,
                    robot_coll,
                    np.asarray(lo),
                    np.asarray(hi),
                    collision_context,
                    start=np.asarray(start),
                    goals=np.asarray(goals),
                )
                _first_valid_verified = True

            plan_kwargs = dict(
                start_config=start,
                goal_configs=goals,
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
                collision_checker="binary_coarse_coop",
                jit_trace=jit_trace,
            )

            if not did_warmup:
                _ = PRRTC.prrtc_plan(**plan_kwargs)
                did_warmup = True

            t0 = time.perf_counter_ns()
            result = PRRTC.prrtc_plan(**plan_kwargs)
            PRRTC_UTILS.synchronize_prrtc_result(result)
            host_planning_ns = time.perf_counter_ns() - t0
            kernel_planning_ns = None
            if getattr(result, "kernel_time_ms", None) is not None:
                kernel_planning_ns = int(float(result.kernel_time_ms) * 1e6)

            if timing_source == "kernel" and kernel_planning_ns is not None:
                planning_ns = kernel_planning_ns
            else:
                planning_ns = host_planning_ns

            if not result.solved:
                failures.append(i)
                continue

            planning_td = pd.Timedelta(nanoseconds=int(planning_ns))
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
                print(f"  Failed on {failures}")

    return results, total_problems, valid_problems, failed_problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pRRTC on MBM datasets")
    parser.add_argument(
        "--robots",
        nargs="+",
        default=["panda", "fetch", "baxter"],
        choices=["panda", "fetch", "baxter"],
        help="Robots to evaluate",
    )
    parser.add_argument("--max-iterations", type=int, default=5000)
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
        "--timing-source",
        choices=["host", "kernel"],
        default="host",
        help="Use host wall time (default) or kernel-reported time for planning_time.",
    )
    parser.add_argument(
        "--jit-trace",
        action="store_true",
        default=True,
        help="Use cached jax.jit tracing for pRRTC FFI dispatch.",
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
    per_robot_counts: dict[str, tuple[int, int, int]] = {}

    for robot in args.robots:
        robot_step_size = STEP_SIZE_BY_ROBOT[robot]
        (
            robot_results,
            robot_total,
            robot_valid,
            robot_failed,
        ) = evaluate_robot(
            robot,
            max_iterations=args.max_iterations,
            step_size=robot_step_size,
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
            max_problems_per_set=args.max_problems_per_set,
            print_failures=args.print_failures,
        )
        all_results.extend(robot_results)
        total += robot_total
        valid += robot_valid
        failed += robot_failed
        per_robot_counts[robot] = (robot_valid - robot_failed, robot_valid, robot_total)

    if not all_results:
        raise RuntimeError("No solved plans were collected; cannot summarize results")

    df = pd.DataFrame.from_dict(all_results)

    # Match vamp_evaluate_mbm.py output math and units.
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

    for robot_name, robot_df in df.groupby("robot", sort=True):
        robot_time_stats = robot_df[
            [
                "planning_time",
                "simplification_time",
                "total_time",
                "planning_iterations",
                "avg_time_per_iteration",
            ]
        ].describe(percentiles=[0.25, 0.5, 0.75, 0.95])
        robot_time_stats.drop(index=["count"], inplace=True)

        robot_cost_stats = robot_df[["initial_path_cost", "simplified_path_cost"]].describe(
            percentiles=[0.25, 0.5, 0.75, 0.95]
        )
        robot_cost_stats.drop(index=["count"], inplace=True)

        print()
        print(f"Per-robot breakdown: {robot_name}")
        print(
            tabulate(
                robot_time_stats,
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
                robot_cost_stats,
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
    print(f"JIT tracing for pRRTC dispatch: {args.jit_trace}")
    print(f"Solved / Valid / Total # Problems: {valid - failed} / {valid} / {total}")
    for robot in args.robots:
        solved_robot, valid_robot, total_robot = per_robot_counts[robot]
        print(f"  {robot}: {solved_robot} / {valid_robot} / {total_robot}")
    print(f"Completed all problems in {df['total_time'].sum() / 1000:.3f} milliseconds")
    print(f"Total time including Python overhead: {(tock - tick) * 1000:.3f} milliseconds")


if __name__ == "__main__":
    main()
