#!/usr/bin/env python3
"""Plan a batch of MBM paths with CUDA pRRTC and bundle them for pyroffi.

pyroffi's TOPP-RA example needs a realistic batch of geometric paths — the kind
a sampling-based planner actually produces: variable length, unevenly spaced,
and jagged. Rather than make pyroffi depend on prrax (and on a CUDA planner
build) at example time, this script runs the planner once here and writes a
single self-contained ``.npz`` into pyroffi's resources.

The padding is done at bundle time, because that is where the variable lengths
are: paths are right-padded to the longest in the batch by repeating the final
configuration, and the true length of each is stored alongside as ``n_valid``.
That is exactly the layout ``pyroffi.topp.topp_ra_batched`` consumes.

Usage::

    python generate_topp_path_bundle.py --robot panda --max-problems 64
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pyroffi as pk
import yourdfpy

try:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
except Exception:
    pass

from pyroffi.collision._obstacles import create_collision_environment
from pyroffi.collision._robot_collision import RobotCollisionSpherized

ROOT = Path(__file__).resolve().parent
RESOURCES = ROOT / "pyroffi" / "resources"
PRRTC_ROOT = ROOT / "cuda-rrtc" / "jax"

STEP_SIZE_BY_ROBOT = {"panda": 0.5, "fetch": 0.6, "baxter": 0.25}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PRRTC = _load_module("prrtc_mod", PRRTC_ROOT / "prrtc.py")
PRRTC_UTILS = _load_module("prrtc_utils_mod", PRRTC_ROOT / "utils.py")


def load_dataset(robot: str) -> dict[str, Any]:
    with open(RESOURCES / robot / "problems.pkl", "rb") as f:
        return pickle.load(f)


def load_models(robot: str):
    urdf = yourdfpy.URDF.load(str(RESOURCES / robot / f"{robot}_spherized.urdf"))
    robot_model = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollisionSpherized.from_urdf(
        urdf, srdf_path=str(RESOURCES / robot / f"{robot}.srdf")
    )
    lo = jnp.asarray(robot_model.joints.lower_limits, dtype=jnp.float32)
    hi = jnp.asarray(robot_model.joints.upper_limits, dtype=jnp.float32)
    return robot_model, robot_coll, lo, hi


def collect_paths(robot: str, max_problems: int, problem_sets: list[str] | None):
    data = load_dataset(robot)
    robot_model, robot_coll, lo, hi = load_models(robot)

    paths: list[np.ndarray] = []
    labels: list[str] = []

    candidates = [
        (name, [p for p in pset if p.get("valid", False)])
        for name, pset in data.get("problems", {}).items()
        if isinstance(pset, list)
        and (not problem_sets or name in problem_sets)
    ]
    candidates = [(n, v) for n, v in candidates if v]
    if not candidates:
        return [], []

    # Spread the quota across problem sets rather than draining the first one:
    # a batch drawn from a single scene has near-identical path lengths and
    # geometry, which is exactly the variation the padded solve exists to handle.
    per_set = max(1, -(-max_problems // len(candidates)))

    for problem_name, valid in candidates:
        remaining = max_problems - len(paths)
        if remaining <= 0:
            break
        valid = valid[: min(per_set, remaining)]

        contexts = [
            PRRTC_UTILS.build_prrtc_collision_context(
                robot_model, robot_coll, create_collision_environment(p)
            )
            for p in valid
        ]

        results = PRRTC.prrtc_plan_batch(
            start_configs=jnp.asarray(
                np.asarray([p["start"] for p in valid], dtype=np.float32)
            ),
            goal_configs=jnp.asarray(
                np.asarray([p["goals"] for p in valid], dtype=np.float32)
            ),
            max_iterations=500_000,
            step_size=STEP_SIZE_BY_ROBOT.get(robot, 0.5),
            min_vals=lo,
            max_vals=hi,
            collision_context=contexts,
        )

        solved = 0
        for r in results:
            if r.solved and r.path is not None and np.asarray(r.path).shape[0] >= 2:
                paths.append(np.asarray(r.path, dtype=np.float32))
                labels.append(problem_name)
                solved += 1
        print(f"  {problem_name}: {solved}/{len(valid)} solved")

    return paths, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", default="panda")
    ap.add_argument("--max-problems", type=int, default=64)
    ap.add_argument("--problem-sets", nargs="*", default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .npz; defaults to pyroffi/resources/<robot>/topp_paths.npz",
    )
    args = ap.parse_args()

    print(f"Planning {args.robot} MBM problems with CUDA pRRTC...")
    paths, labels = collect_paths(args.robot, args.max_problems, args.problem_sets)
    if not paths:
        raise SystemExit("no paths were solved; nothing to bundle")

    lengths = np.asarray([p.shape[0] for p in paths], dtype=np.int32)
    n_wp_max = int(lengths.max())
    dof = paths[0].shape[1]

    padded = np.empty((len(paths), n_wp_max, dof), dtype=np.float32)
    for b, p in enumerate(paths):
        padded[b, : p.shape[0]] = p
        # Repeat the final configuration so the padding is a stationary
        # continuation rather than a jump, for any consumer that ignores
        # n_valid. See pyroffi.topp.pad_paths, which uses the same convention.
        padded[b, p.shape[0] :] = p[-1]

    out = args.out or (RESOURCES / args.robot / "topp_paths.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        waypoints=padded,
        n_valid=lengths,
        problem=np.asarray(labels),
        robot=np.asarray(args.robot),
    )

    print(
        f"\nWrote {out}\n"
        f"  {len(paths)} paths, {dof} DOF, padded to {n_wp_max} waypoints\n"
        f"  waypoint counts: min {lengths.min()}, median "
        f"{int(np.median(lengths))}, max {lengths.max()}\n"
        f"  padding overhead: {100 * (1 - lengths.sum() / lengths.size / n_wp_max):.1f}%"
    )


if __name__ == "__main__":
    main()
