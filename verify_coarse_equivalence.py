#!/usr/bin/env python3
"""Check the coarse→fine binary checker is *sound* vs the flat one.

Exact iteration/cost equality is NOT a valid test here: the planner is a concurrent
race across 64 blocks, so a checker with different per-config timing makes a different
block win first → different tree, even when the collision *decision* function is
identical (the same effect that made binary vs differentiable differ earlier).

The meaningful correctness property is soundness: the coarse check must never report
"free" when the fine model says "collision". So every configuration the coarse planner
*accepts* (its tree nodes — each added only after passing the coarse edge check) must be
collision-free under the fine pyroffi ground-truth model. We verify that for a large
sample of accepted nodes; binary is checked too as a baseline.
"""

import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import pyroffi as pk
import yourdfpy
from pyroffi.collision._obstacles import create_collision_environment
from pyroffi.collision._robot_collision import RobotCollisionSpherized

_here = Path(__file__).resolve().parent


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, _here / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prrtc = _load("cuda_rrtc_prrtc", "cuda-rrtc/jax/prrtc.py")
utils = _load("cuda_rrtc_utils", "cuda-rrtc/jax/utils.py")

RESOURCE_ROOT = _here / "pyroffi" / "resources"
PANDA_URDF = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"


def min_margin_over_nodes(robot, robot_coll, nodes, obstacles, sample=400):
    """Smallest fine-model collision margin over a sample of configs (nodes: [N, dim])."""
    nodes = np.asarray(nodes, dtype=np.float32)
    if nodes.shape[0] == 0:
        return np.inf
    if nodes.shape[0] > sample:
        sel = np.linspace(0, nodes.shape[0] - 1, sample).astype(int)
        nodes = nodes[sel]
    worst = np.inf
    for q in nodes:
        rep = utils.config_collision_report(robot, robot_coll, jnp.asarray(q), obstacles)
        worst = min(worst, rep["min_margin"])
    return worst


def main():
    urdf = yourdfpy.URDF.load(str(PANDA_URDF))
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollisionSpherized.from_urdf(
        urdf, srdf_path=str(RESOURCE_ROOT / "panda" / "panda.srdf")
    )
    lo = np.array(robot.joints.lower_limits)
    hi = np.array(robot.joints.upper_limits)

    problem = sys.argv[1] if len(sys.argv) > 1 else "bookshelf_tall"
    indices = [int(x) for x in sys.argv[2:]] or [1, 2, 3, 4]

    # Soundness tolerance: the CUDA planner and pyroffi share the same sphere model, so an
    # accepted (collision-free) config should have fine margin >= 0. Allow a tiny negative
    # band for float drift / edge-midpoint vs node sampling.
    TOL = -2e-3

    n_fail = 0
    n_checked = 0
    for idx in indices:
        vp = utils.load_vamp_problem(RESOURCE_ROOT, problem=problem, index=idx)
        if vp is None or "start" not in vp or "goals" not in vp:
            print(f"  [{problem}:{idx}] skipped (not found)")
            continue
        obstacles = create_collision_environment(vp)
        cc = utils.build_prrtc_collision_context(robot, robot_coll, obstacles)
        start = jnp.array(vp["start"], dtype=jnp.float32)
        goal = jnp.array(vp["goals"][0], dtype=jnp.float32)

        kw = dict(
            start_config=start,
            goal_configs=goal.reshape(1, -1),
            max_iterations=5000,
            step_size=0.5,
            num_new_samples=64,
            dynamic_domain=False,
            min_vals=jnp.array(lo, dtype=jnp.float32),
            max_vals=jnp.array(hi, dtype=jnp.float32),
            collision_context=cc,
            jit_trace=False,
        )
        rb = prrtc.prrtc_plan(**kw, collision_checker="binary")
        nodes_b = np.concatenate(
            [np.asarray(rb.tree_a_configs).T, np.asarray(rb.tree_b_configs).T], axis=0
        )
        worst_b = min_margin_over_nodes(robot, robot_coll, nodes_b, obstacles)

        for checker in ("binary_coarse", "binary_coarse_coop"):
            rc = prrtc.prrtc_plan(**kw, collision_checker=checker)
            nodes_c = np.concatenate(
                [np.asarray(rc.tree_a_configs).T, np.asarray(rc.tree_b_configs).T], axis=0
            )
            worst_c = min_margin_over_nodes(robot, robot_coll, nodes_c, obstacles)
            n_checked += 1
            sound = worst_c >= TOL
            if not sound:
                n_fail += 1
            print(
                f"  [{problem}:{idx}] {checker:18s} {'SOUND' if sound else 'UNSOUND'}  "
                f"solved b/c={rb.solved}/{rc.solved}  it b/c={rb.iterations}/{rc.iterations}  "
                f"worst fine margin={worst_c:+.4f} (baseline binary={worst_b:+.4f})"
            )

    n_coarse = int(np.asarray(cc["coarse_sphere_link_idx"]).shape[0])
    n_fine = int(np.asarray(cc["sphere_radius"]).shape[0])
    n_self = int(np.asarray(cc["self_pairs"]).shape[0])
    n_coarse_self = int(np.asarray(cc["coarse_self_pairs"]).shape[0])
    print(
        f"\nReduction factors: world {n_fine}→{n_coarse} spheres "
        f"({n_fine / max(n_coarse,1):.1f}x), self {n_self}→{n_coarse_self} pairs "
        f"({n_self / max(n_coarse_self,1):.1f}x)"
    )
    if n_fail:
        print(f"\nFAIL: coarse under-reported collisions on {n_fail}/{n_checked} problems.")
        sys.exit(1)
    print(f"\nPASS: coarse-accepted nodes are collision-free on all {n_checked} problems.")


if __name__ == "__main__":
    main()
