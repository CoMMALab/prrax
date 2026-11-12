#!/usr/bin/env python3
"""cpRRTC on McVAMP's line and plane problems, run like McVAMP's
experiments/python/benchmark_line_plane.py and shown like the McVAMP project page's recordings
(commalab.org/papers/mcvamp): the Panda's end effector stays on a plane or a line through 100
cuboids.

Scenes: by default the page's own. Plane: plane_problems.json problem 902 (the page's
plane_no_ori/plane_ori recordings). Line: the page's 100-cuboid line scene
(line_no_ori/recording_8), which is not in line_problems.json; it is rebuilt from the recording in
demos/assets/mcvamp_page_line_scene.json. --first I [--limit N] runs benchmark problems instead.

As in McVAMP, the orientation bounds are opened up (+-1000) unless --keep-orientation, start and
goal are projected onto the manifold first (200 iterations) and must be collision-free, and each
path is simplified. --visualize shows the first solved problem the way the page does: cuboids
in the page's palette, the constraint as a green slab, the end-effector path as copper spheres
(red start, magenta goal), a label and the page's camera; the path is on a waypoint slider.

If McVAMP's Python module is installed (pip install . in mcvamp/vamp), every problem is also solved
by McVAMP with McVAMP's own settings (one CPU core) and the times are printed side by side;
--no-mcvamp skips it.

    python demos/demo_line_plane.py [--problems plane|line] [--keep-orientation] [--visualize]
                                    [--first I] [--limit N] [--range 1.0] [--shortcut-rounds 1]
                                    [--no-mcvamp]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "mcvamp" / "experiments" / "python"))
import benchmark_constrained as B  # noqa: E402
import mcvamp_side  # noqa: E402

P = B.PRRTC
VAMP_PANDA = ROOT / "mcvamp" / "vamp" / "resources" / "panda"
PAGE_LINE = Path(__file__).resolve().parent / "assets" / "mcvamp_page_line_scene.json"
PAGE_PLANE_INDEX = 902
# The page's constraint slabs (BoxMessage "constraint" in its recordings), in the TSR frame.
SLAB = {"plane": (2.0, 0.0002, 4.0), "line": (0.01, 0.02, 40.0)}


def page_line_case(keep_orientation: bool) -> dict:
    s = json.loads(PAGE_LINE.read_text())
    lo, hi = list(s["tsr_lower_bound"]), list(s["tsr_upper_bound"])
    if not keep_orientation:
        lo[3:], hi[3:] = [-1000.0] * 3, [1000.0] * 3
    cub = np.asarray(s["cuboids"], np.float32)
    return dict(start=s["start"], goal=s["goal"], boxes=B._boxes(cub[:, :3], cub[:, 3:]),
                tsr=(s["eef_transforms_ref_frame_w_world"], s["eef_transforms"], lo, hi),
                n_obstacles=len(cub))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", choices=["plane", "line"], default="plane")
    ap.add_argument("--keep-orientation", action="store_true")
    ap.add_argument("--first", type=int, default=None, help="benchmark problem to start at (default: the page's scene)")
    ap.add_argument("--limit", type=int, default=1, help="problems to run (0 = all)")
    ap.add_argument("--range", type=float, default=1.0, help="extension step (McVAMP's range_)")
    ap.add_argument("--shortcut-rounds", type=int, default=1, help="simplification (McVAMP simplifies)")
    ap.add_argument("--visualize", action="store_true", help="show the first solved problem in viser")
    ap.add_argument("--no-mcvamp", action="store_true", help="do not also run McVAMP (when installed)")
    args = ap.parse_args()
    mcv = None
    if mcvamp_side.AVAILABLE and not args.no_mcvamp:
        # benchmark_line_plane.py's knobs (benchmark_common.Knobs)
        mcv = mcvamp_side.McVamp(args.range, max_iterations=200_000, seed_iterations=200, knobs=dict(
            method="OuterLM", descend_rate=1.0, max_iterations=15, emit_all_waypoints=True,
            perturbation_scale=0.1))
    elif not args.no_mcvamp:
        print("McVAMP not installed (pip install . in mcvamp/vamp): prrax only", flush=True)

    robot, coll = B.load_panda()
    base_ctx = B.make_context(robot, coll, maze=False)
    if args.problems == "line" and args.first is None:
        cases, names, source = [page_line_case(args.keep_orientation)], ["page"], PAGE_LINE.name
    else:
        cases = B.load_problems(args.problems, keep_orientation=args.keep_orientation)
        first = PAGE_PLANE_INDEX if args.first is None else args.first
        end = len(cases) if args.limit == 0 else min(len(cases), first + args.limit)
        cases, names = cases[first:end], list(range(first, end))
        source = f"{args.problems}_problems.json"
    lo = jnp.asarray(robot.joints.lower_limits, jnp.float32)
    hi = jnp.asarray(robot.joints.upper_limits, jnp.float32)
    # McVAMP's knobs: 15 projection iterations, no dynamic domain.
    proj = P.ProjectionSettings(mode="cprrtc", iters=15)
    kw = dict(max_iterations=2000, step_size=args.range, num_new_samples=256, granularity=16,
              max_nodes=200_000, dynamic_domain=False, min_vals=lo, max_vals=hi,
              collision_checker="binary_coarse", projection=proj, shortcut_rounds=args.shortcut_rounds,
              jit_trace=True)

    print(f"line/plane problems ({source}): cpRRTC, problems {names}", flush=True)
    times_ms, iterations, costs = [], [], []
    mc_ms, mc_costs, mc_attempted = [], [], 0
    attempted = off_manifold = 0
    shown = None
    warm = False

    for name, p in zip(names, cases):
        ctx = dict(base_ctx, world_boxes=p["boxes"])
        con = B.tsr_constraint(robot, p["tsr"])
        attempted += 1
        n_obs = p["n_obstacles"]

        _, ends, ok = P.constraint_error(ctx, [con], np.asarray([p["start"], p["goal"]], np.float32),
                                         proj._replace(iters=200), lo, hi)
        if not ok.all():
            print(f"problem {name} ({n_obs} obstacles): not planned (seed projection failed)", flush=True)
            continue
        if P.config_in_collision(ctx, ends, [con], "binary_coarse").any():
            print(f"problem {name} ({n_obs} obstacles): not planned (seed in collision)", flush=True)
            continue

        if not warm:   # the first call builds the planner for this robot: not timed
            P.prrtc_plan(ends[0], ends[1:], collision_context=ctx, constraints=[con], **kw)
            warm = True
        t0 = time.perf_counter()
        r = P.prrtc_plan(ends[0], ends[1:], collision_context=ctx, constraints=[con], **kw)
        ms = (time.perf_counter() - t0) * 1e3
        if not r.solved:
            print(f"problem {name} ({n_obs} obstacles): no path after {r.iterations} iterations", flush=True)
            continue

        path = np.asarray(r.path, np.float32)
        err, _, _ = P.constraint_error(ctx, [con], path, proj, lo, hi)
        on_manifold = bool(err.max() <= proj.tol)
        cost = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        times_ms.append(ms)
        iterations.append(int(r.iterations))
        costs.append(cost)
        off_manifold += 0 if on_manifold else 1
        print(f"problem {name} ({n_obs} obstacles): solved in {ms:.3f} ms, {r.iterations} iterations, "
              f"simplified cost {cost:.4f}", flush=True)
        if mcv is not None:
            mc_attempted += 1
            m = mcv.solve(p["start"], p["goal"], [(b[:3], b[12:15]) for b in p["boxes"]], p["tsr"])
            if m["status"] != "solved":
                print(f"    McVAMP: {m['status']}", flush=True)
            else:
                tot = m["plan_ms"] + m["simplify_ms"]
                mc_ms.append(tot)
                mc_costs.append(m["cost"])
                print(f"    McVAMP: solved in {tot:.3f} ms ({m['plan_ms']:.3f} plan + {m['simplify_ms']:.3f} "
                      f"simplify), {m['iterations']} iterations, simplified cost {m['cost']:.4f}; "
                      + (f"prrax {tot / ms:.1f}x faster" if tot > ms else f"McVAMP {ms / tot:.1f}x faster"),
                      flush=True)
        if shown is None:
            shown = (name, p, path)

    solved = len(times_ms)
    print("------ Summary ------")
    print(f"Total problems: {attempted}")
    print(f"Successful problems: {solved}")
    print(f"Success rate: {100.0 * solved / attempted if attempted else 0.0:g}%")
    if solved:
        print(f"Average time (ms) for successful problems: {statistics.mean(times_ms):.4f}")
        print(f"Average iterations for successful problems: {int(statistics.mean(iterations))}")
        print(f"Median time (ms) for successful problems: {statistics.median(times_ms):.4f}")
        print(f"Median iterations for successful problems: {int(statistics.median(iterations))}")
        print(f"Mean simplified path cost: {statistics.mean(costs):.4f}")
        print(f"Paths with a waypoint off the manifold: {off_manifold}", flush=True)
    if mcv is not None:
        print(f"McVAMP (1 CPU core, plan + simplify): {len(mc_ms)}/{mc_attempted} solved", end="")
        print(f", median {statistics.median(mc_ms):.4f} ms, mean cost {statistics.mean(mc_costs):.4f}"
              if mc_ms else "", flush=True)

    if not args.visualize:
        return
    if shown is None:
        print("nothing to visualize: no problem was solved")
        return
    show_problem(robot, args.problems, args.keep_orientation, *shown)


def copper(x: np.ndarray) -> np.ndarray:
    """matplotlib's 'copper' colormap (the page's path colours), as uint8 RGB."""
    x = np.asarray(x, np.float64)[:, None]
    return (255 * np.clip(x * np.array([1.25, 0.7812, 0.4975]), 0, 1)).astype(np.uint8)


def show_problem(robot, kind: str, keep_orientation: bool, name, p: dict, path: np.ndarray) -> None:
    """The page's look: palette cuboids, green constraint slab, copper path spheres, label, camera."""
    from viser_utils import setup_viser_with_robot

    print(f"visualizing problem {name} (the first one solved; pick another with --first and --limit)")
    server, urdf_vis = setup_viser_with_robot(VAMP_PANDA, "panda_spherized.urdf")
    server.scene.world_axes.visible = True
    server.scene.add_grid("/floor_grid", width=10.0, height=10.0, plane="xy")
    orient = "with" if keep_orientation else "without"
    server.scene.add_label("/handle", f"{kind.capitalize()} Position {orient} Orientation Constraint",
                           position=(0.0, -0.25, 0.75))

    palette = json.loads(PAGE_LINE.read_text())["cuboid_palette"]
    for i, b in enumerate(p["boxes"]):
        color = palette[int(i * len(palette) / len(p["boxes"]))]
        server.scene.add_box(f"/cuboid_{i}", color=tuple(color), dimensions=2 * b[12:15], position=b[:3])

    _, ref, _, _ = p["tsr"]
    ref = np.asarray(ref, np.float32)
    server.scene.add_box("/constraint", color=(0, 255, 0), dimensions=SLAB[kind], opacity=0.5,
                         wxyz=ref[:4], position=ref[4:])

    ee = list(robot.links.names).index(B.EE_LINK)
    pts = np.asarray(jax.jit(lambda q: robot.forward_kinematics(q)[:, ee, 4:7])(jnp.asarray(path)))
    for i, (pt, c) in enumerate(zip(pts, copper(np.linspace(0.0, 1.0, len(pts))))):
        server.scene.add_icosphere(f"/trajectory_waypoint_{i}", radius=0.01, color=tuple(int(v) for v in c),
                                   position=tuple(pt))
    server.scene.add_icosphere("/trajectory_waypoint_first", radius=0.05, color=(255, 0, 0), position=tuple(pts[0]))
    server.scene.add_icosphere("/trajectory_waypoint_last", radius=0.05, color=(255, 0, 255), position=tuple(pts[-1]))

    @server.on_client_connect
    def _(client) -> None:   # the page's camera
        client.camera.position = (1.7664506, -0.01426106, 0.94084048)
        client.camera.look_at = (0.0, 0.0, 0.0)
        client.camera.fov = 1.3089

    urdf_vis.update_cfg(path[0])
    slider = server.gui.add_slider("Current Waypoint", min=0, max=len(path) - 1, step=1, initial_value=0)

    @slider.on_update
    def _(_event) -> None:
        urdf_vis.update_cfg(path[slider.value])

    print(f"visualization at http://localhost:{server.get_port()}; ctrl-c to exit", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
