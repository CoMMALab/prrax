#!/usr/bin/env python3
"""cpRRTC replica of the McVAMP project page's dynamic maze demo (commalab.org/papers/mcvamp;
mcvamp's viser_dynamic_obstacle_avoid.py + viser_ros_listener.py, in one process, no ROS).

A Panda holding a marker (crispee's marker panda and VAMP's marker sphere model, in
demos/assets/panda_marker) shuttles between two goals through the maze with the marker tip on
the table plane (z = 0.04, tilt +-0.01 rad, yaw free: crispee's/VAMP's marker maze TSR). Drag the red sphere obstacle: every planner tick (10 Hz)
the rest of the current path is re-checked against it and, if blocked, cpRRTC replans from
where the arm is. Playback is geometric, a fixed time per waypoint (no dynamics), at 60 fps.

If McVAMP's Python module is installed (pip install . in mcvamp/vamp), each replan is also handed to
McVAMP (one CPU core, its own thread, so it never delays prrax) on McVAMP's own maze model: the
plain Panda with its 10-sphere pen and its hand TSR (pen tip on the table), settings of
viser_dynamic_obstacle_avoid.py. Its time is shown next to prrax's; --no-mcvamp skips it.

    python demos/demo_maze.py [--hard] [--waypoint-dt 0.01] [--planner-hz 10] [--no-mcvamp]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
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
from viser_utils import setup_viser_with_robot  # noqa: E402

P, U = B.PRRTC, B.UTILS
MARKER_PANDA = Path(__file__).resolve().parent / "assets" / "panda_marker"

# crispee benchmark_maze/vamp_tsr.py (VAMP's benchmark_maze_marker_vamp.cc), on the marker tip;
# z at the C++ original's +-5 mm. Reference frame: ref y -> world z.
TIP_LINK = "panda_tip"
TIP_Z = 0.04
TSR = ([0.0, 0.0, 0.7071068, 0.7071068, 0.6, 0.3, TIP_Z],
       [-10.01, -0.005, -10.01, -0.01, -10.01, -0.01], [10.01, 0.005, 10.01, 0.01, 10.01, 0.01])
# Start/end of crispee's tsr_panda_marker_maze_problems.json problem 0, the one McVAMP's marker
# maze results are on (39 ms there; prrax ~8 ms). The curated sets pick hard problems (~2 s).
GOALS = np.asarray([[0.740415, 0.892249, -1.1401451, -1.8482553, 0.9408859, 2.0794082, 2.9670999],
                    [0.3952203, 0.7529666, 0.2910529, -1.4438856, -0.2262138, 2.1491058, 0.7215207]], np.float32)
# --hard: crispee's tsr_panda_marker_maze_problems_curated_100.json problem 0, across the whole
# maze (tip y 0.35 -> -0.46).
HARD_GOALS = np.asarray([[2.5562892, -1.5311456, -1.5348423, -2.021972, -1.519433, 1.555756, -1.1147763],
                         [2.0414813, -1.0720757, -2.359721, -1.1092368, -0.7198169, 1.9229773, 0.6187202]], np.float32)
OBSTACLE_RADIUS = 0.05
# McVAMP's maze model (viser_dynamic_obstacle_avoid.py): TSR on the hand, 10-sphere pen.
MCVAMP_TSR = ([1, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0.0, 0.29276255, -0.55347496, 0.20607783],
              [-10.01, -10.01, -0.01, -0.01, -0.01, -10.01], [10.01, 10.01, 0.01, 0.01, 0.01, 10.01])
MCVAMP_PEN = [((0.0, 0.0, z), 0.01) for z in np.linspace(0.0, 0.16, 10)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waypoint-dt", type=float, default=0.01, help="seconds per path waypoint")
    ap.add_argument("--planner-hz", type=float, default=10.0)
    ap.add_argument("--hard", action="store_true", help="crispee's curated problem across the whole maze")
    ap.add_argument("--no-mcvamp", action="store_true", help="do not also run McVAMP (when installed)")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--simplify-rounds", type=int, default=1,
                    help="simplification rounds per plan (B-spline smoothing + shortcutting), prrax and McVAMP alike "
                         "(~10 ms each in the maze, 0 = off for both)")
    args = ap.parse_args()

    import pyroffi as pk
    import yourdfpy
    from pyroffi.collision._robot_collision import RobotCollisionSpherized

    urdf = yourdfpy.URDF.load(str(MARKER_PANDA / "panda_expo_spherized_vamp.urdf"))
    robot = pk.Robot.from_urdf(urdf)
    coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=str(MARKER_PANDA / "panda_marker_vamp.srdf"))
    tip = list(robot.links.names).index(TIP_LINK)
    base_ctx = U.build_prrtc_collision_context(robot, coll, [])
    cub = json.loads((B.ASSETS / "environments" / "maze" / "cuboids.json").read_text())
    base_ctx["world_boxes"] = B._boxes([[o["x"], o["y"], o["z"]] for o in cub],
                                       [[o["dx"] / 2, o["dy"] / 2, o["dz"] / 2] for o in cub])
    con = P.make_tsr_constraint(robot, TIP_LINK, TSR[0], TSR[1], TSR[2])
    lo = jnp.asarray(robot.joints.lower_limits, jnp.float32)
    hi = jnp.asarray(robot.joints.upper_limits, jnp.float32)
    proj = P.ProjectionSettings(mode="cprrtc", iters=20)
    plan_kw = dict(max_iterations=2000, step_size=0.2, num_new_samples=256, granularity=16,
                   max_nodes=200_000, dynamic_domain=False, min_vals=lo, max_vals=hi,
                   collision_checker="binary_coarse", constraints=[con], projection=proj, jit_trace=True)

    def project(q):
        err, qp, ok = P.constraint_error(base_ctx, [con], q, proj._replace(iters=50), lo, hi)
        return np.where((err <= proj.tol)[:, None], np.atleast_2d(q), qp), ok | (err <= proj.tol)

    goals, ok = project(HARD_GOALS if args.hard else GOALS)
    assert ok.all(), "goal projection failed"
    tip_pos = jax.jit(lambda q: robot.forward_kinematics(q)[:, tip, 4:7])
    import trimesh
    dot = trimesh.creation.icosphere(subdivisions=1, radius=0.0075)
    dot.visual.vertex_colors = (0, 255, 0, 255)

    print("Warming up (first call builds the planner for this robot)...")
    P.prrtc_plan(goals[0], goals[1:], collision_context=dict(base_ctx, world_spheres=np.zeros((0, 4), np.float32)),
                 **plan_kw)

    server, urdf_vis = setup_viser_with_robot(MARKER_PANDA, "panda_expo.urdf")
    # mcvamp's green constraint slab, its top face on the tip plane.
    server.scene.add_box("my_cuboid", dimensions=(10.02, 10.5, 0.06), position=(0.6, 0.3, TIP_Z - 0.03),
                         color=(0, 255, 0), opacity=0.5)
    for i, o in enumerate(cub):
        server.scene.add_box(f"/{o.get('name', f'cuboid_{i}')}", dimensions=(o["dx"], o["dy"], o["dz"]),
                             color=(12, 89, 178), position=(o["x"], o["y"], o["z"]))
    # Clear of the shuttle path (mcvamp's (0.4, 0.3, 0.4) hits this robot's start configuration).
    obstacle = server.scene.add_transform_controls("/obstacle", scale=0.2, position=(0.7, 0.0, 0.5))
    server.scene.add_icosphere("/obstacle/mesh", radius=OBSTACLE_RADIUS, color=(255, 0, 0))
    elapsed = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # McVAMP: the latest replan request, solved on a worker thread (stale requests are dropped).
    mc_req, mc_cond = [None], threading.Condition()
    if mcvamp_side.AVAILABLE and not args.no_mcvamp:
        mcv = mcvamp_side.McVamp(0.2, max_iterations=100_000, seed_iterations=50,
                                 simplify_rounds=args.simplify_rounds)
        mc_cuboids = [((o["x"], o["y"], o["z"]), (o["dx"] / 2, o["dy"] / 2, o["dz"] / 2)) for o in cub]
        mc_elapsed = server.gui.add_number("McVAMP (ms)", 0.001, disabled=True)

        def mcvamp_loop() -> None:
            while True:
                with mc_cond:
                    while mc_req[0] is None:
                        mc_cond.wait()
                    (start, goal, obs, prrax_ms), mc_req[0] = mc_req[0], None
                m = mcv.solve(start, goal, mc_cuboids, MCVAMP_TSR, spheres=[(obs, OBSTACLE_RADIUS)],
                              attachment=MCVAMP_PEN, simplify_min=3)
                if m["status"] != "solved":
                    print(f"    McVAMP: {m['status']}")
                    continue
                tot = m["plan_ms"] + m["simplify_ms"]
                mc_elapsed.value = tot
                print(f"    McVAMP: planned in {tot:.2f} ms ({m['plan_ms']:.2f} plan + {m['simplify_ms']:.2f} "
                      f"simplify) vs prrax {prrax_ms:.2f} ms")

        threading.Thread(target=mcvamp_loop, daemon=True).start()
    elif not args.no_mcvamp:
        print("McVAMP not installed (pip install . in mcvamp/vamp): prrax only")

    lock = threading.Lock()
    st = dict(traj=goals[:1], t=0.0, goal=1, q=goals[0])

    def planner_tick() -> None:
        t1 = time.perf_counter()
        with lock:
            traj, t, goal_i, q_now = st["traj"], st["t"], st["goal"], st["q"].copy()
        goal = goals[goal_i]
        ctx = dict(base_ctx, world_spheres=np.asarray([[*obstacle.position, OBSTACLE_RADIUS]], np.float32))
        # Keep the current path if it still ends at the goal and its remainder is collision-free.
        i = int(t / args.waypoint_dt)
        if np.linalg.norm(traj[-1] - goal) < 1e-3 and (i >= len(traj) - 1 or not P.config_in_collision(
                ctx, np.concatenate([q_now[None], traj[i + 1:]]), [con], "binary_coarse").any()):
            print(f"No change, computed in {(time.perf_counter() - t1) * 1e3:.2f} ms")
            return
        start, _ = project(q_now[None].astype(np.float32))
        # Like VAMP, reject a colliding start or goal at once: prrax would search its whole budget.
        hit = P.config_in_collision(ctx, np.concatenate([start, goal[None]]), [con], "binary_coarse")
        if hit.any():
            with lock:
                st["traj"], st["t"] = start, 0.0
            elapsed.value = (time.perf_counter() - t1) * 1e3
            print(f"Not planned: the obstacle hits the {' and '.join(n for n, h in zip(('start', 'goal'), hit) if h)}")
            return
        r = P.prrtc_plan(start[0], goal[None], collision_context=ctx, shortcut_rounds=args.simplify_rounds, **plan_kw)
        new = np.asarray(r.path, np.float32) if r.solved else start
        with lock:
            st["traj"], st["t"] = new, 0.0
        elapsed.value = (time.perf_counter() - t1) * 1e3
        if not r.solved:
            print(f"Failed to find a solution! start_config={start[0]}, goal_config={goal}, "
                  f"cur_obstacle={np.asarray(obstacle.position)}")
            return
        print(f"Planned trajectory with {len(new)} waypoints in {elapsed.value:.2f} mseconds")
        with mc_cond:   # the same start and goal for McVAMP
            mc_req[0] = (start[0], goal, np.asarray(obstacle.position, np.float32), elapsed.value)
            mc_cond.notify()
        # The listener's waypoint spheres, at the marker tip: one instanced mesh, replaced whole.
        pts = np.asarray(tip_pos(jnp.asarray(new)), np.float32)
        server.scene.add_batched_meshes_trimesh("/trajectory_waypoints", dot, np.tile([1.0, 0, 0, 0], (len(pts), 1)),
                                                pts, cast_shadow=False, receive_shadow=False)

    def planner_loop() -> None:
        while True:
            t0 = time.perf_counter()
            planner_tick()
            time.sleep(max(0.0, 1.0 / args.planner_hz - (time.perf_counter() - t0)))

    threading.Thread(target=planner_loop, daemon=True).start()
    print(f"visualization at http://localhost:{server.get_port()}; drag the red sphere; ctrl-c to exit")

    dt = 1.0 / args.fps
    last_q = None
    try:
        while True:
            t0 = time.perf_counter()
            with lock:
                traj = st["traj"]
                x = min(st["t"] / args.waypoint_dt, len(traj) - 1)
                k = min(int(x), len(traj) - 2) if len(traj) > 1 else 0
                q = traj[k] + (x - k) * (traj[k + 1] - traj[k]) if len(traj) > 1 else traj[0]
                st["q"] = q
                st["t"] += dt
                if np.linalg.norm(q - goals[st["goal"]]) < 1e-3:   # reached: head for the other goal
                    st["goal"] = 1 - st["goal"]
            if last_q is None or not np.array_equal(q, last_q):   # parked: nothing to send
                with server.atomic():   # every link transform in one websocket batch
                    urdf_vis.update_cfg(q)
                last_q = q
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
