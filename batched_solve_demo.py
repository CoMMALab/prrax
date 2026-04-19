#!/usr/bin/env python3
"""
Batched pRRTC demo: run N parallel Panda planning solves on a single VAMP
problem and visualize every bidirectional tree with a distinct per-batch color.
"""

import argparse
import colorsys
import importlib.util
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

try:
    import pyroffi as pk
    import yourdfpy
    from pyroffi.collision._obstacles import create_collision_environment
    from pyroffi.collision._robot_collision import RobotCollisionSpherized
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install pyroffi and yourdfpy")
    sys.exit(1)

try:
    prrtc_impl = Path(__file__).parent / "cuda-rrtc" / "jax" / "prrtc.py"
    spec = importlib.util.spec_from_file_location("cuda_rrtc_prrtc", prrtc_impl)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {prrtc_impl}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prrtc_plan_batch = mod.prrtc_plan_batch

    utils_impl = Path(__file__).parent / "cuda-rrtc" / "jax" / "utils.py"
    utils_spec = importlib.util.spec_from_file_location("cuda_rrtc_utils", utils_impl)
    if utils_spec is None or utils_spec.loader is None:
        raise ImportError(f"Failed to load module spec from {utils_impl}")
    utils_mod = importlib.util.module_from_spec(utils_spec)
    utils_spec.loader.exec_module(utils_mod)
    load_vamp_problem = utils_mod.load_vamp_problem
    build_prrtc_collision_context = utils_mod.build_prrtc_collision_context
    config_collision_report = utils_mod.config_collision_report
except Exception as e:
    print(f"cuda-rrtc import error: {e}")
    print("Make sure the library is compiled: cd cuda-rrtc && bash build.sh")
    sys.exit(1)


RESOURCE_ROOT = Path("/home/scoumar/Work/rrax/pyroffi/resources")
PANDA_URDF = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"


def distinct_colors(n: int) -> np.ndarray:
    """Generate n visually distinct RGB colors as uint8 in [0, 255]."""
    colors = np.empty((n, 3), dtype=np.uint8)
    for i in range(n):
        h = (i / max(n, 1)) % 1.0
        # Mild offset in s/v to avoid pastels and fully-saturated clashes
        s = 0.75
        v = 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def dir_to_wxyz(direction) -> tuple:
    """Quaternion (w, x, y, z) rotating -z (viser's default light forward) to ``direction``."""
    d = np.asarray(direction, dtype=np.float64)
    d /= np.linalg.norm(d) + 1e-12
    forward = np.array([0.0, 0.0, -1.0])
    axis = np.cross(forward, d)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return (1.0, 0.0, 0.0, 0.0) if np.dot(forward, d) > 0 else (0.0, 1.0, 0.0, 0.0)
    axis /= axis_norm
    angle = np.arccos(np.clip(np.dot(forward, d), -1.0, 1.0))
    s = np.sin(angle / 2.0)
    return (float(np.cos(angle / 2.0)), float(axis[0] * s), float(axis[1] * s), float(axis[2] * s))


def select_diverse_trees(results, k: int = 20):
    """Pick up to ``k`` trees that maximize pairwise diversity in joint-space
    statistics via farthest-point sampling. Seeded with a solved result when
    available so the slider's solution path is non-empty."""
    valid = [(i, r) for i, r in enumerate(results)
             if r.tree_a_configs is not None and r.tree_b_configs is not None]
    if len(valid) <= k:
        return [r for _, r in valid]

    feats = []
    for _, r in valid:
        a = np.asarray(r.tree_a_configs).T
        b = np.asarray(r.tree_b_configs).T
        combined = np.concatenate([a, b], axis=0)
        feats.append(np.concatenate([combined.mean(axis=0), combined.std(axis=0)]))
    feats = np.stack(feats).astype(np.float64)

    seed = next((idx for idx, (_, r) in enumerate(valid) if r.solved), 0)
    selected = [seed]
    dists = np.linalg.norm(feats - feats[seed], axis=1)
    dists[seed] = -np.inf
    for _ in range(k - 1):
        idx = int(np.argmax(dists))
        selected.append(idx)
        dists = np.minimum(dists, np.linalg.norm(feats - feats[idx], axis=1))
        dists[idx] = -np.inf

    return [valid[i][1] for i in selected]


def _build_edge_segments(pts: np.ndarray, parents: np.ndarray):
    non_root = np.where(np.arange(len(parents)) != parents)[0]
    if len(non_root) == 0:
        return None
    starts = pts[non_root]
    ends = pts[parents[non_root]]
    return np.stack([starts, ends], axis=1)


def visualize_batched_trees(robot, urdf, results, obstacles=None, hz: float = 10.0):
    """Render every batch element's bidirectional tree in task space with a unique color."""
    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as e:
        print(f"  Visualization unavailable (missing dependency): {e}")
        return

    server = viser.ViserServer(host="0.0.0.0", port=8080)
    server.scene.set_up_direction("+z")

    obs_centroids = []
    for i, obs in enumerate(obstacles or []):
        if hasattr(obs, "to_trimesh"):
            tm = obs.to_trimesh()
            tm.visual.face_colors = np.array([60, 60, 60, 255], dtype=np.uint8)
            obs_centroids.append(np.asarray(tm.centroid, dtype=np.float32))
            server.scene.add_mesh_trimesh(f"/world/obstacles/obj_{i}", mesh=tm)

    server.scene.add_transform_controls("/ctrl/key",  position=(1.5,  -1.5, 2.5), scale=0.4)
    server.scene.add_label("/ctrl/key/label",  "Key light")
    server.scene.add_transform_controls("/ctrl/fill", position=(-1.5,  0.5, 1.5), scale=0.4)
    server.scene.add_label("/ctrl/fill/label", "Fill light")

    server.scene.add_light_directional(
        "/ctrl/key/light",
        color=(255, 228, 185),
        intensity=3.5,
        cast_shadow=True,
        wxyz=dir_to_wxyz([0.55, -0.4, -1.0]),
    )
    server.scene.add_light_directional(
        "/ctrl/fill/light",
        color=(160, 200, 255),
        intensity=1.4,
        cast_shadow=False,
        wxyz=dir_to_wxyz([-1.0, 0.3, -0.5]),
    )
    server.scene.add_light_directional(
        "/lights/rim",
        color=(255, 255, 255),
        intensity=1.8,
        cast_shadow=False,
        wxyz=dir_to_wxyz([0.05, 1.0, 0.5]),
    )

    chain_mid = (
        np.mean(np.stack(obs_centroids), axis=0)
        if obs_centroids
        else np.array([0.3, 0.0, 0.5], dtype=np.float32)
    )
    server.scene.add_light_point(
        "/lights/accent",
        color=(200, 220, 255),
        intensity=8.0,
        distance=1.2,
        decay=2.0,
        cast_shadow=True,
        position=tuple(chain_mid + np.array([0.0, 0.0, 0.5])),
    )

    try:
        ee_idx = robot.links.names.index("panda_hand")
    except (ValueError, AttributeError):
        ee_idx = -1

    palette = distinct_colors(len(results))

    first_solution_path = None
    for i, result in enumerate(results):
        if result.tree_a_configs is None or result.tree_b_configs is None:
            continue
        color = tuple(int(c) for c in palette[i])

        configs_a = np.array(result.tree_a_configs).T  # (size_a, dim)
        configs_b = np.array(result.tree_b_configs).T
        parents_a = np.array(result.tree_a_parents)
        parents_b = np.array(result.tree_b_parents)

        fk_a = np.array(robot.forward_kinematics(jnp.array(configs_a, dtype=jnp.float32)))
        fk_b = np.array(robot.forward_kinematics(jnp.array(configs_b, dtype=jnp.float32)))
        pts_a = fk_a[:, ee_idx, 4:7].astype(np.float32)
        pts_b = fk_b[:, ee_idx, 4:7].astype(np.float32)

        colors_a = np.tile(np.array([color], dtype=np.uint8), (len(pts_a), 1))
        colors_b = np.tile(np.array([color], dtype=np.uint8), (len(pts_b), 1))

        server.scene.add_point_cloud(
            f"/batch_{i:03d}/tree_a/nodes", points=pts_a, colors=colors_a, point_size=0.006
        )
        server.scene.add_point_cloud(
            f"/batch_{i:03d}/tree_b/nodes", points=pts_b, colors=colors_b, point_size=0.006
        )

        segs_a = _build_edge_segments(pts_a, parents_a)
        segs_b = _build_edge_segments(pts_b, parents_b)
        if segs_a is not None:
            server.scene.add_line_segments(
                f"/batch_{i:03d}/tree_a/edges", points=segs_a, colors=color, line_width=1.0
            )
        if segs_b is not None:
            server.scene.add_line_segments(
                f"/batch_{i:03d}/tree_b/edges", points=segs_b, colors=color, line_width=1.0
            )

        if first_solution_path is None and result.solved and result.path is not None:
            first_solution_path = np.array(result.path, dtype=np.float32)

    if first_solution_path is not None:
        fk_path = np.array(robot.forward_kinematics(jnp.array(first_solution_path)))
        ee_path = fk_path[:, ee_idx, 4:7]
        server.scene.add_spline_catmull_rom(
            "/solution_path", positions=ee_path, color=(0, 220, 60), line_width=4.0
        )

    # URDF slider — cycle through the first solved path (or a default config otherwise).
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    slider_configs = (
        first_solution_path
        if first_solution_path is not None
        else np.array(results[0].tree_a_configs).T
    )
    slider = server.gui.add_slider(
        "Node", min=0, max=len(slider_configs) - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Playing", initial_value=True)

    solved_count = sum(1 for r in results if r.solved)
    print(
        f"\nBatched tree visualization: {solved_count}/{len(results)} solved "
        f"(colored per batch)."
    )
    print("Viewer running at http://localhost:8080  |  Press Ctrl+C to exit.")
    urdf_vis.update_cfg(slider_configs[0])
    try:
        while True:
            if playing.value:
                slider.value = (slider.value + 1) % len(slider_configs)
            urdf_vis.update_cfg(slider_configs[slider.value])
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        print("\nStopping visualization.")


def main():
    parser = argparse.ArgumentParser(
        description="Batched pRRTC demo: render N parallel tree solves on one VAMP problem"
    )
    parser.add_argument("--vamp-problem", default="bookshelf_tall", help="VAMP problem name")
    parser.add_argument("--vamp-index", type=int, default=1, help="VAMP problem index")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of parallel solves")
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--num-new-samples", type=int, default=64)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument(
        "--jit-trace",
        action="store_true",
        default=True,
        help="Use cached jax.jit tracing for pRRTC FFI dispatch.",
    )
    parser.add_argument("--no-jit-trace", action="store_false", dest="jit_trace")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Batched pRRTC demo — {args.batch_size} parallel solves")
    print("=" * 70)

    if not PANDA_URDF.exists():
        print(f"ERROR: URDF not found at {PANDA_URDF}")
        sys.exit(1)

    print(f"\nLoading robot from {PANDA_URDF}")
    urdf = yourdfpy.URDF.load(str(PANDA_URDF))
    robot = pk.Robot.from_urdf(urdf)
    srdf_path = str(RESOURCE_ROOT / "panda" / "panda.srdf")
    robot_coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=srdf_path)
    n_act = robot.joints.num_actuated_joints
    print(f"  {n_act} actuated joints")

    vamp_problem = load_vamp_problem(
        RESOURCE_ROOT, problem=args.vamp_problem, index=args.vamp_index
    )
    if vamp_problem is None:
        print(f"ERROR: VAMP problem {args.vamp_problem}[{args.vamp_index}] not found")
        sys.exit(1)
    obstacles = create_collision_environment(vamp_problem)
    collision_context = build_prrtc_collision_context(robot, robot_coll, obstacles)
    print(
        f"  Loaded VAMP problem {args.vamp_problem}[{args.vamp_index}] "
        f"with {len(obstacles)} obstacles"
    )

    lo = np.array(robot.joints.lower_limits)
    hi = np.array(robot.joints.upper_limits)

    if "start" not in vamp_problem or "goals" not in vamp_problem:
        print("ERROR: VAMP problem missing start/goals")
        sys.exit(1)
    start_config = jnp.array(vamp_problem["start"], dtype=jnp.float32)
    goal_config = jnp.array(vamp_problem["goals"][0], dtype=jnp.float32)

    start_report = config_collision_report(robot, robot_coll, start_config, obstacles)
    goal_report = config_collision_report(robot, robot_coll, goal_config, obstacles)
    print(
        f"  Start margin={start_report['min_margin']:.5f}, "
        f"goal margin={goal_report['min_margin']:.5f}"
    )

    # Replicate the same start/goal across the batch — each solve runs on its own
    # CUDA stream with independent planner state, producing a different tree.
    starts = jnp.broadcast_to(start_config, (args.batch_size, n_act))
    goals = jnp.broadcast_to(goal_config, (args.batch_size, 1, n_act))

    plan_kwargs = dict(
        start_configs=starts,
        goal_configs=goals,
        max_iterations=args.max_iterations,
        step_size=args.step_size,
        num_new_samples=args.num_new_samples,
        dynamic_domain=False,
        min_vals=jnp.array(lo, dtype=jnp.float32),
        max_vals=jnp.array(hi, dtype=jnp.float32),
        collision_context=collision_context,
        jit_trace=args.jit_trace,
    )

    print(f"\nWarmup solve (batch={args.batch_size}) — pays JIT compile + FFI registration...")
    t_warm = time.perf_counter()
    _ = prrtc_plan_batch(**plan_kwargs)
    warm_ms = (time.perf_counter() - t_warm) * 1000.0
    print(f"  Warmup wall: {warm_ms:.2f} ms")

    print("Timed JIT-compiled solve (using cached kernel)...")
    t0 = time.perf_counter()
    results = prrtc_plan_batch(**plan_kwargs)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    solved = [r for r in results if r.solved]
    kernel_times = [r.kernel_time_ms for r in results if r.kernel_time_ms is not None]
    print(f"  Solved: {len(solved)}/{len(results)}  (wall={wall_ms:.2f} ms)")
    if kernel_times:
        arr = np.asarray(kernel_times, dtype=np.float32)
        print(
            f"  Kernel time: mean={arr.mean():.3f} ms, "
            f"min={arr.min():.3f} ms, max={arr.max():.3f} ms"
        )
    tree_sizes = [r.tree_a_size + r.tree_b_size for r in results]
    print(
        f"  Tree sizes (A+B): mean={np.mean(tree_sizes):.1f}, "
        f"min={np.min(tree_sizes)}, max={np.max(tree_sizes)}"
    )

    if not args.no_viz:
        diverse_results = select_diverse_trees(results, k=20)
        print(f"  Visualizing {len(diverse_results)} most diverse trees (of {len(results)}).")
        visualize_batched_trees(
            robot=robot, urdf=urdf, results=diverse_results, obstacles=obstacles
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
