#!/usr/bin/env python3
"""Constrained pRRTC on McVAMP's Panda benchmark problem sets.

The problem sets and settings are McVAMP's (mcvamp/experiments/assets/problems/benchmark/,
cpp/benchmark_line_plane.cc, cpp/benchmark_maze.cc):

    plane    1000 problems, end effector free in a plane, its own cuboids each
    line      766 problems, end effector free along a line
    maze      100 problems, 9-sphere pen on the hand, hand at a fixed height and orientation

Line/plane open the orientation bounds to +-1000 unless --keep-orientation. Records match
McVAMP's --results files (success, solve_time_ns, iterations, num_cuboid_obstacles, path_cost,
on_manifold), so compare_mcvamp.py can join them; solve_time_ns is host wall time of the
solve (kernel time is recorded too). path_cost is unsimplified (McVAMP reports simplified).

    python benchmark_constrained.py --problems plane [--keep-orientation] [--mode mcvamp|cprrtc]
                                    [--batch] [--limit N] [--results FILE]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "mcvamp" / "experiments" / "assets"
PANDA = ROOT / "pyroffi" / "resources" / "panda"
EE_LINK = "panda_grasptarget"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRRTC = _load("cuda_rrtc_prrtc", ROOT / "cuda-rrtc" / "jax" / "prrtc.py")
UTILS = _load("cuda_rrtc_utils", ROOT / "cuda-rrtc" / "jax" / "utils.py")


def load_panda():
    import pyroffi as pk
    import yourdfpy
    from pyroffi.collision._robot_collision import RobotCollisionSpherized

    urdf = yourdfpy.URDF.load(str(PANDA / "panda_spherized.urdf"))
    robot = pk.Robot.from_urdf(urdf)
    coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=str(PANDA / "panda.srdf"))
    return robot, coll


def _boxes(centers, half_extents) -> np.ndarray:
    """Axis-aligned cuboids as the planner's [N, 15] box rows."""
    c = np.asarray(centers, np.float32).reshape(-1, 3)
    h = np.asarray(half_extents, np.float32).reshape(-1, 3)
    eye = np.broadcast_to(np.eye(3, dtype=np.float32).reshape(1, 9), (len(c), 9))
    return np.concatenate((c, eye, h), axis=1)


def load_problems(name: str, keep_orientation: bool = False) -> list[dict]:
    """McVAMP's problems: start, goal, boxes [N, 15], TSR (offset, reference, lower, upper)."""
    if name == "maze":
        cub = json.loads((ASSETS / "environments" / "maze" / "cuboids.json").read_text())
        boxes = _boxes([[o["x"], o["y"], o["z"]] for o in cub],
                       [[o["dx"] / 2, o["dy"] / 2, o["dz"] / 2] for o in cub])
        tsr = ([1, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0.0, 0.29276255, -0.55347496, 0.20607783],
               [-10.01, -10.01, -0.01, -0.01, -0.01, -10.01], [10.01, 10.01, 0.01, 0.01, 0.01, 10.01])
        items = json.loads((ASSETS / "problems" / "benchmark" / "maze_pen_problems.json").read_text())
        return [dict(start=p["problem_start"], goal=p["problem_end"], boxes=boxes, tsr=tsr,
                     n_obstacles=len(cub)) for p in items]

    items = json.loads((ASSETS / "problems" / "benchmark" / f"{name}_problems.json").read_text())
    out = []
    for p in items:
        cub = np.asarray(p["cuboid_obstacles"], np.float32).reshape(-1, 6)
        lo, hi = list(p["tsr_lower_bound"]), list(p["tsr_upper_bound"])
        if not keep_orientation:
            lo[3:], hi[3:] = [-1000.0] * 3, [1000.0] * 3
        # McVAMP's constructor order: (rTe, wTr) = (ref_frame_w_world, eef_transforms).
        tsr = (p["eef_transforms_ref_frame_w_world"], p["eef_transforms"], lo, hi)
        out.append(dict(start=p["problem_start"], goal=p["problem_end"],
                        boxes=_boxes(cub[:, :3], cub[:, 3:]), tsr=tsr, n_obstacles=len(cub)))
    return out


# McVAMP's pen: nine 1 cm spheres, 2 cm apart along the end-effector z axis.
PEN = np.array([[0.0, 0.0, 0.02 * i, 0.01] for i in range(9)], np.float32)


def make_context(robot, coll, maze: bool):
    ee_frame = int(robot.links.parent_joint_indices[list(robot.links.names).index(EE_LINK)])
    return UTILS.build_prrtc_collision_context(robot, coll, [],
                                               attached_spheres=(ee_frame, PEN) if maze else None)


def tsr_constraint(robot, tsr):
    offset, reference, lo, hi = tsr
    return PRRTC.make_tsr_constraint(robot, EE_LINK, reference, lo, hi, offset=offset)


def reference_error(robot, con, q) -> np.ndarray:
    """|d| of a TSR constraint at configurations q (B, dim), in JAX (pyroffi FK + jaxlie)."""
    link = list(robot.links.names).index(EE_LINK)
    T_ee = jaxlie.SE3(robot.forward_kinematics(jnp.asarray(q, jnp.float32))[:, link])
    T = (jaxlie.SE3(jnp.asarray(con.reference)).inverse() @ T_ee
         @ jaxlie.SE3(jnp.asarray(con.offset)).inverse())
    x = jnp.concatenate((T.translation(), T.rotation().log()), axis=-1)
    d = x - jnp.clip(x, con.lower, con.upper)
    return np.asarray(jnp.linalg.norm(d, axis=-1))


def box_clearance(points: np.ndarray, radii: np.ndarray, boxes: np.ndarray) -> float:
    """Min signed distance of spheres (points [P, 3], radii [P]) to axis-aligned boxes."""
    if len(boxes) == 0:
        return np.inf
    q = np.abs(points[:, None, :] - boxes[None, :, :3]) - boxes[None, :, 12:15]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return float((outside + inside - radii[:, None]).min())


def validate_path(robot, coll, con, path, boxes, tol, pen: bool) -> dict:
    """Independent check of a solved path: constraint error (JAX), robot collision (pyroffi),
    pen collision (numpy), and the largest gap between consecutive waypoints."""
    from pyroffi.collision._geometry import Box

    path = np.asarray(path, np.float32)
    err = reference_error(robot, con, path)
    world = Box.from_center_and_half_lengths(boxes[:, :3], boxes[:, 12:15]) if len(boxes) else None
    self_d = np.asarray(jax.vmap(lambda q: coll.compute_self_collision_distance(robot, q))(jnp.asarray(path)))
    min_d = float(self_d.min()) if self_d.size else np.inf
    if world is not None:
        world_d = np.asarray(jax.vmap(lambda q: coll.compute_world_collision_distance(robot, q, world))(jnp.asarray(path)))
        min_d = min(min_d, float(world_d.min()))
    if pen:
        link = list(robot.links.names).index(EE_LINK)
        T = jaxlie.SE3(robot.forward_kinematics(jnp.asarray(path))[:, link])
        pts = np.asarray(jax.vmap(lambda t: jax.vmap(t.apply)(jnp.asarray(PEN[:, :3])))(T))
        min_d = min(min_d, box_clearance(pts.reshape(-1, 3), np.tile(PEN[:, 3], len(path)), boxes))
    gaps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return dict(max_err=float(err.max()), on_manifold=bool(err.max() <= tol),
                min_clearance=min_d, collision_free=bool(min_d >= -1e-4),
                max_gap=float(gaps.max()) if gaps.size else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", choices=["plane", "line", "maze"], default="plane")
    ap.add_argument("--keep-orientation", action="store_true")
    ap.add_argument("--mode", choices=["mcvamp", "cprrtc"], default="mcvamp")
    ap.add_argument("--batch", action="store_true", help="solve the whole set in one batched call")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-iterations", type=int, default=2000, help="per block")
    ap.add_argument("--num-new-samples", type=int, default=0,
                    help="blocks per problem (default: 256 single, 64 batched; both measured best)")
    ap.add_argument("--granularity", type=int, default=16)
    ap.add_argument("--max-nodes", type=int, default=200_000)
    ap.add_argument("--collision-checker", default="binary_coarse")
    ap.add_argument("--validate", action="store_true", help="independently check every path")
    ap.add_argument("--results", default="")
    args = ap.parse_args()

    maze = args.problems == "maze"
    robot, coll = load_panda()
    base_ctx = make_context(robot, coll, maze)
    problems = load_problems(args.problems, args.keep_orientation)
    if args.limit:
        problems = problems[: args.limit]
    lo = jnp.asarray(robot.joints.lower_limits, jnp.float32)
    hi = jnp.asarray(robot.joints.upper_limits, jnp.float32)
    # McVAMP: range 1 and 15 projection iterations (line/plane), range 0.4 and 20 (maze).
    step = 0.4 if maze else 1.0
    proj = PRRTC.ProjectionSettings(mode=args.mode, iters=20 if maze else 15)
    blocks = args.num_new_samples or (64 if args.batch else 256)
    kw = dict(max_iterations=args.max_iterations, step_size=step, num_new_samples=blocks,
              granularity=args.granularity, max_nodes=args.max_nodes, dynamic_domain=False,
              min_vals=lo, max_vals=hi, collision_checker=args.collision_checker,
              projection=proj, jit_trace=True)

    ctxs = [dict(base_ctx, world_boxes=p["boxes"]) for p in problems]
    cons = [[tsr_constraint(robot, p["tsr"])] for p in problems]
    starts = np.asarray([p["start"] for p in problems], np.float32)
    goals = np.asarray([p["goal"] for p in problems], np.float32)

    # Warm-up: per-robot build + JIT, not timed.
    PRRTC.prrtc_plan(starts[0], goals[:1], collision_context=ctxs[0], constraints=cons[0], **kw)

    records, results = [], []
    if args.batch:
        t0 = time.perf_counter_ns()
        results = PRRTC.prrtc_plan_batch(starts, goals[:, None], collision_context=ctxs, constraints=cons, **kw)
        batch_ns = time.perf_counter_ns() - t0
        times = [batch_ns] * len(problems)
    else:
        times = []
        for i in range(len(problems)):
            t0 = time.perf_counter_ns()
            try:
                results.append(PRRTC.prrtc_plan(starts[i], goals[i:i + 1], collision_context=ctxs[i],
                                                constraints=cons[i], **kw))
            except ValueError as e:   # seed not projectable: not planned (as McVAMP counts it)
                print(f"problem {i}: {e}")
                results.append(PRRTC.PRRTCResult(False, None, 0, 0, 0, float("inf"), 0.0))
            times.append(time.perf_counter_ns() - t0)

    n_bad = 0
    for i, (p, r, t) in enumerate(zip(problems, results, times)):
        rec = dict(problem_number=i, success=bool(r.solved), solve_time_ns=int(t) if r.solved else 0,
                   kernel_time_ns=int(r.kernel_time_ms * 1e6) if r.solved else 0,
                   iterations=r.iterations, num_cuboid_obstacles=p["n_obstacles"],
                   path_cost=r.cost if r.solved else 0.0, on_manifold=True)
        if r.solved and args.validate:
            v = validate_path(robot, coll, cons[i][0], r.path, p["boxes"], proj.tol * 1.01, maze)
            rec.update(v)
            n_bad += (not v["on_manifold"]) or (not v["collision_free"])
        records.append(rec)

    solved = [r for r in records if r["success"]]
    ms = np.asarray([r["solve_time_ns"] for r in solved]) / 1e6
    kms = np.asarray([r["kernel_time_ns"] for r in solved]) / 1e6
    label = f"{args.problems}{'_ori' if args.keep_orientation else ''} {args.mode}{' batch' if args.batch else ''}"
    print(f"{label}: solved {len(solved)}/{len(records)}", end="")
    if args.batch:
        print(f", batch wall {times[0] / 1e6:.1f} ms ({times[0] / 1e6 / len(records):.3f} ms/problem)", end="")
    elif len(ms):
        print(f", median {np.median(ms):.2f} ms (kernel {np.median(kms):.2f}), p95 {np.percentile(ms, 95):.2f} ms", end="")
    if args.validate:
        print(f", invalid paths {n_bad}", end="")
    print()
    if args.results:
        Path(args.results).write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
