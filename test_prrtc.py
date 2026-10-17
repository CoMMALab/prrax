#!/usr/bin/env python3
"""
Test script for pRRTC with a VAMP problem from pyroffi.
"""

import sys
import importlib.util
import argparse
import time
from pathlib import Path
import jax.numpy as jnp
import numpy as np

# Import pyroffi
try:
    import pyroffi as pk
    import yourdfpy
    from pyroffi.collision._obstacles import create_collision_environment
    from pyroffi.collision._robot_collision import RobotCollisionSpherized
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install pyroffi and yourdfpy")
    sys.exit(1)

# Import cuda-rrtc
try:
    prrtc_impl = Path(__file__).parent / "cuda-rrtc" / "jax" / "prrtc.py"
    spec = importlib.util.spec_from_file_location("cuda_rrtc_prrtc", prrtc_impl)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {prrtc_impl}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prrtc_plan = mod.prrtc_plan
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
    print("The planner is built per robot on first use (needs nvcc, pyroffi-grim with GRiM and cricket2).")
    sys.exit(1)


RESOURCE_ROOT = Path(__file__).resolve().parent / "pyroffi" / "resources"
PANDA_URDF = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"


def _slice_collision_context(base_ctx: dict, *, world_spheres_keep: int) -> dict:
    """Clone collision context while trimming its world spheres (the robot model is baked)."""
    out = dict(base_ctx)
    ws = np.asarray(base_ctx.get("world_spheres", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
    out["world_spheres"] = ws[:max(0, min(int(world_spheres_keep), int(ws.shape[0])))]
    return out


def validate_path_collision(robot, robot_coll, path, world_obstacles):
    """Validate solved path against full pyroffi collision model."""
    path_np = np.asarray(path, dtype=np.float32)
    min_self = np.inf
    min_world = np.inf
    for q in path_np:
        q_jax = jnp.asarray(q)
        self_d = robot_coll.compute_self_collision_distance(robot, q_jax)
        min_self = min(min_self, float(np.min(np.asarray(self_d))))
        if world_obstacles:
            for obs in world_obstacles:
                world_d = robot_coll.compute_world_collision_distance(robot, q_jax, obs)
                min_world = min(min_world, float(np.min(np.asarray(world_d))))
    if not world_obstacles:
        min_world = np.inf
    min_margin = min(min_self, min_world)
    return {
        "min_self": min_self,
        "min_world": min_world,
        "min_margin": min_margin,
        "collision_free": bool(min_margin > 0.0),
    }


def visualize_tree_with_viser(robot, urdf, result, obstacles=None, hz: float = 10.0):
    """Visualize RRT trees as pointcloud + edges in task space, with optional solved path."""
    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as e:
        print(f"  Visualization unavailable (missing dependency): {e}")
        return

    if result.tree_a_configs is None or result.tree_b_configs is None:
        print("  Visualization skipped: no tree data in result")
        return

    server = viser.ViserServer(host="0.0.0.0", port=8080)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    server.scene.add_frame("/world", show_axes=True)

    for i, obs in enumerate(obstacles or []):
        if hasattr(obs, "to_trimesh"):
            server.scene.add_mesh_trimesh(f"/world/obstacles/obj_{i}", mesh=obs.to_trimesh())

    # Determine EE link index for FK projection to task space
    try:
        ee_idx = robot.links.names.index("panda_hand")
    except (ValueError, AttributeError):
        ee_idx = -1

    # Configs: (dim, size) -> (size, dim)
    configs_a = np.array(result.tree_a_configs).T
    configs_b = np.array(result.tree_b_configs).T
    parents_a = np.array(result.tree_a_parents)
    parents_b = np.array(result.tree_b_parents)

    # FK to get EE positions for all nodes
    fk_a = np.array(robot.forward_kinematics(jnp.array(configs_a, dtype=jnp.float32)))
    fk_b = np.array(robot.forward_kinematics(jnp.array(configs_b, dtype=jnp.float32)))
    pts_a = fk_a[:, ee_idx, 4:7].astype(np.float32)  # (size_a, 3)
    pts_b = fk_b[:, ee_idx, 4:7].astype(np.float32)  # (size_b, 3)

    # Pointclouds
    colors_a = np.tile(np.array([[30, 100, 255]], dtype=np.uint8), (len(pts_a), 1))
    colors_b = np.tile(np.array([[255, 80, 30]], dtype=np.uint8), (len(pts_b), 1))
    server.scene.add_point_cloud("/tree_a/nodes", points=pts_a, colors=colors_a, point_size=0.008)
    server.scene.add_point_cloud("/tree_b/nodes", points=pts_b, colors=colors_b, point_size=0.008)

    # Edges: skip root (self-parent) nodes
    def _build_edge_segments(pts, parents):
        non_root = np.where(np.arange(len(parents)) != parents)[0]
        if len(non_root) == 0:
            return None
        starts = pts[non_root]
        ends = pts[parents[non_root]]
        return np.stack([starts, ends], axis=1)  # (N, 2, 3)

    segs_a = _build_edge_segments(pts_a, parents_a)
    segs_b = _build_edge_segments(pts_b, parents_b)
    if segs_a is not None:
        server.scene.add_line_segments("/tree_a/edges", points=segs_a, colors=(80, 140, 255), line_width=1.0)
    if segs_b is not None:
        server.scene.add_line_segments("/tree_b/edges", points=segs_b, colors=(255, 140, 80), line_width=1.0)

    # Solved path overlay
    if result.solved and result.path is not None:
        path_np = np.array(result.path, dtype=np.float32)
        fk_path = np.array(robot.forward_kinematics(jnp.array(path_np)))
        ee_path = fk_path[:, ee_idx, 4:7]
        server.scene.add_spline_catmull_rom("/solution_path", positions=ee_path, color=(0, 220, 60), line_width=3.0)

    # URDF slider over all nodes (path first if solved, then tree nodes)
    all_configs = configs_a if not (result.solved and result.path is not None) else np.concatenate([np.array(result.path, dtype=np.float32), configs_a], axis=0)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    slider = server.gui.add_slider("Node", min=0, max=len(all_configs) - 1, step=1, initial_value=0)

    status = "SOLVED" if result.solved else "FAILED"
    print(f"\nTree visualization [{status}] — tree_a={result.tree_a_size} nodes, tree_b={result.tree_b_size} nodes")
    print("Viewer running at http://localhost:8080  |  Press Ctrl+C to exit.")
    urdf_vis.update_cfg(all_configs[0])
    try:
        while True:
            urdf_vis.update_cfg(all_configs[slider.value])
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        print("\nStopping visualization.")


def visualize_path_with_viser(robot, urdf, path, obstacles=None, hz: float = 10.0):
    """Visualize planned joint path with Viser and optional VAMP obstacles."""
    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as e:
        print(f"  Visualization unavailable (missing dependency): {e}")
        return

    traj = np.asarray(path, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] == 0:
        print("  Visualization skipped: empty or invalid path array")
        return

    server = viser.ViserServer(host="0.0.0.0", port=8080)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    server.scene.add_frame("/world", show_axes=True)

    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    for i, obs in enumerate(obstacles or []):
        if hasattr(obs, "to_trimesh"):
            server.scene.add_mesh_trimesh(
                name=f"/world/obstacles/obj_{i}",
                mesh=obs.to_trimesh(),
                visible=True,
            )

    try:
        ee_link_name = "panda_hand"
        ee_link_index = robot.links.names.index(ee_link_name)
        fk_all = robot.forward_kinematics(jnp.array(traj))
        ee_positions = np.array(fk_all[:, ee_link_index, 4:7])
        server.scene.add_spline_catmull_rom(
            "/trajectory_path",
            positions=ee_positions,
            color=(0, 120, 255),
            line_width=3.0,
        )
    except Exception as e:
        print(f"  Could not draw end-effector spline path: {e}")

    slider = server.gui.add_slider(
        "Timestep", min=0, max=traj.shape[0] - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Playing", initial_value=True)

    print("\nViewer running at http://localhost:8080  |  Press Ctrl+C to exit.")
    urdf_vis.update_cfg(traj[0])
    try:
        while True:
            if playing.value:
                slider.value = (slider.value + 1) % traj.shape[0]
            urdf_vis.update_cfg(traj[slider.value])
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        print("\nStopping visualization.")


def main():
    parser = argparse.ArgumentParser(description="Test pRRTC and visualize one planned path")
    parser.add_argument("--no-viz", action="store_true", help="Disable Viser visualization")
    parser.add_argument("--vamp-problem", default="bookshelf_tall", help="VAMP problem name")
    parser.add_argument("--vamp-index", type=int, default=1, help="VAMP problem index")
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
    args = parser.parse_args()

    print("=" * 70)
    print("Testing pRRTC with PyRoFFI VAMP problem")
    print("=" * 70)
    print(f"JIT tracing for pRRTC dispatch: {args.jit_trace}")

    if not PANDA_URDF.exists():
        print(f"ERROR: URDF not found at {PANDA_URDF}")
        print("Please run pyroffi resource generation first")
        sys.exit(1)

    print(f"\nLoading robot from {PANDA_URDF}")
    urdf = yourdfpy.URDF.load(str(PANDA_URDF))
    robot = pk.Robot.from_urdf(urdf)
    srdf_path = str(RESOURCE_ROOT / "panda" / "panda.srdf")
    robot_coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=srdf_path)
    n_act = robot.joints.num_actuated_joints
    n_joints_total = int(robot.joints.twists.shape[0])
    n_links_total = int(robot.links.num_links)
    print(f"  {n_act} actuated joints, {n_joints_total} total joints, {n_links_total} links")

    vamp_problem = load_vamp_problem(
        RESOURCE_ROOT,
        problem=args.vamp_problem,
        index=args.vamp_index,
    )
    obstacles = create_collision_environment(vamp_problem) if vamp_problem is not None else []
    collision_context = build_prrtc_collision_context(robot, robot_coll, obstacles)
    ws = collision_context['world_spheres'].shape[0]
    wc = collision_context.get('world_capsules', np.empty((0,))).shape[0]
    wb = collision_context.get('world_boxes', np.empty((0,))).shape[0]
    wh = collision_context.get('world_halfspaces', np.empty((0,))).shape[0]
    print(
        "  Collision context: "
        f"robot_spheres={collision_context['sphere_radius'].shape[0]}, "
        f"world=[spheres={ws}, capsules={wc}, boxes={wb}, halfspaces={wh}], "
        f"self_pairs={collision_context['self_pairs'].shape[0]}"
    )

    # Define start and goal configurations
    print("\nDefining planning problem...")
    rng = np.random.default_rng(42)
    lo = np.array(robot.joints.lower_limits)
    hi = np.array(robot.joints.upper_limits)

    # Prefer VAMP-provided start/goal when available (typically collision-valid seeds).
    if vamp_problem is not None and "start" in vamp_problem and "goals" in vamp_problem:
        start_config = jnp.array(vamp_problem["start"], dtype=jnp.float32)
        goal_config = jnp.array(vamp_problem["goals"][0], dtype=jnp.float32)
        print("  Using VAMP start/goal configuration")
    else:
        # Fallback: simple interpolation across joint range.
        start_config = jnp.array(lo + 0.1 * (hi - lo), dtype=jnp.float32)
        goal_config = jnp.array(hi - 0.1 * (hi - lo), dtype=jnp.float32)
        print("  Using fallback synthetic start/goal configuration")

    print(f"  Start config: {np.array(start_config)}")
    print(f"  Goal config: {np.array(goal_config)}")

    start_report = config_collision_report(robot, robot_coll, start_config, obstacles)
    goal_report = config_collision_report(robot, robot_coll, goal_config, obstacles)
    print(
        "  Start collision margin: "
        f"{start_report['min_margin']:.5f} "
        f"(self={start_report['min_self']:.5f}, world={start_report['min_world']:.5f})"
    )
    print(
        "  Goal collision margin: "
        f"{goal_report['min_margin']:.5f} "
        f"(self={goal_report['min_self']:.5f}, world={goal_report['min_world']:.5f})"
    )

    # Trust the VAMP-provided `valid` flag when available: it was computed with the
    # reference collision checker.  The spherized model can report small negative
    # margins due to sphere-approximation artifacts even for genuinely free configs.
    vamp_valid = vamp_problem.get("valid", False) if vamp_problem is not None else False
    roots_valid = start_report["collision_free"] and goal_report["collision_free"]
    if not roots_valid and vamp_valid:
        print(
            "  Sphere model reports marginal overlap "
            f"(start={start_report['min_margin']:.5f}, goal={goal_report['min_margin']:.5f}) "
            "but VAMP problem is marked valid=True — trusting VAMP validity."
        )
        roots_valid = True
    elif not roots_valid:
        print(
            "  Root config collision detected. Source pRRTC behavior expects caller-side "
            "prevalidated start/goal and does not auto-repair roots."
        )
        print("  Skipping planning for this problem to mirror source benchmark behavior.")

    # Test single planning
    print("\nTesting single planning...")
    single_result = None
    if roots_valid:
        try:
            single_plan_kwargs = dict(return_trees=True,
                start_config=start_config,
                goal_configs=goal_config.reshape(1, -1),
                max_iterations=5000,
                step_size=0.5,
                num_new_samples=64,
                dynamic_domain=False,
                min_vals=jnp.array(lo, dtype=jnp.float32),
                max_vals=jnp.array(hi, dtype=jnp.float32),
                collision_context=collision_context,
                jit_trace=args.jit_trace,
            )
            # First call pays setup overhead (e.g., FFI target registration/JIT plumbing).
            _ = prrtc_plan(**single_plan_kwargs)

            result = prrtc_plan(**single_plan_kwargs)
            single_result = result
            print(f"  Result: solved={result.solved}")
            print(f"  Tree A size: {result.tree_a_size}")
            print(f"  Tree B size: {result.tree_b_size}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Cost: {result.cost:.4f}")
            if result.kernel_time_ms is not None:
                print(
                    "  GPU planner-kernel time (no host dispatch): "
                    f"{result.kernel_time_ms:.3f} ms"
                )
                if result.iterations > 0:
                    us_per_iter = (result.kernel_time_ms * 1000.0) / float(result.iterations)
                    print(f"  Timing detail: {us_per_iter:.3f} us/iteration")
            if not result.solved:
                print(
                    "  Planner failed: "
                    f"tree_a_size={result.tree_a_size}, "
                    f"tree_b_size={result.tree_b_size}"
                )
            if result.solved and result.path is not None:
                print(f"  Path length: {int(result.path.shape[0])}")
                report = validate_path_collision(robot, robot_coll, result.path, obstacles)
                print(
                    "  Collision check: "
                    f"free={report['collision_free']} "
                    f"min_margin={report['min_margin']:.5f} "
                    f"(self={report['min_self']:.5f}, world={report['min_world']:.5f})"
                )
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  Skipped: invalid start/goal roots for source-like evaluation")

    # Test multiple planning problems
    print("\nTesting multiple planning problems...")
    try:
        # Generate batch of problems
        batch_size = 4
        starts_np = rng.uniform(lo, hi, size=(batch_size, n_act)).astype(np.float32)
        goals_np = rng.uniform(lo, hi, size=(batch_size, n_act)).astype(np.float32)

        # Source-like caller behavior: ensure roots are valid before calling planner.
        for i in range(batch_size):
            srep = config_collision_report(robot, robot_coll, starts_np[i], obstacles)
            retries = 0
            while not srep["collision_free"] and retries < 2000:
                starts_np[i] = rng.uniform(lo, hi, size=(n_act,)).astype(np.float32)
                srep = config_collision_report(robot, robot_coll, starts_np[i], obstacles)
                retries += 1
            grep = config_collision_report(robot, robot_coll, goals_np[i], obstacles)
            retries = 0
            while not grep["collision_free"] and retries < 2000:
                goals_np[i] = rng.uniform(lo, hi, size=(n_act,)).astype(np.float32)
                grep = config_collision_report(robot, robot_coll, goals_np[i], obstacles)
                retries += 1

        max_ws = int(collision_context.get("world_spheres", np.empty((0, 4), dtype=np.float32)).shape[0])
        per_problem_contexts = []
        for i in range(batch_size):
            if i % 2 == 0:
                per_problem_contexts.append(collision_context)
            else:
                ws_keep = max(0, max_ws - 1)
                per_problem_contexts.append(
                    _slice_collision_context(
                        collision_context,
                        world_spheres_keep=ws_keep,
                    )
                )

        batch_plan_kwargs = dict(
            start_configs=jnp.array(starts_np),
            goal_configs=jnp.array(goals_np[:, None, :]),  # [batch, 1, dim] — one goal per problem
            max_iterations=1000,
            step_size=0.6,
            num_new_samples=16,
            dynamic_domain=False,
            min_vals=jnp.array(lo, dtype=jnp.float32),
            max_vals=jnp.array(hi, dtype=jnp.float32),
            collision_context=per_problem_contexts,
            jit_trace=args.jit_trace,
        )
        _ = prrtc_plan_batch(**batch_plan_kwargs)
        results = prrtc_plan_batch(**batch_plan_kwargs)
        print(
            "  Batch collision contexts: mixed per-problem "
            f"(unique_world_spheres={[c.get('world_spheres', np.empty((0,4))).shape[0] for c in per_problem_contexts]})"
        )
        kernel_times = [r.kernel_time_ms for r in results if r.kernel_time_ms is not None]
        if kernel_times:
            total_ms = float(np.sum(np.asarray(kernel_times, dtype=np.float32)))
            mean_ms = float(np.mean(np.asarray(kernel_times, dtype=np.float32)))
            max_ms = float(np.max(np.asarray(kernel_times, dtype=np.float32)))
            print(
                "  Batch GPU planner-kernel times: "
                f"mean={mean_ms:.3f} ms, max={max_ms:.3f} ms, sum={total_ms:.3f} ms"
            )
        for i, result in enumerate(results):
            status = "✓" if result.solved else "✗"
            if result.solved:
                print(
                    f"  Problem {i}: {status} "
                    f"(tree_a_size={result.tree_a_size}, tree_b_size={result.tree_b_size}, "
                    f"iters={result.iterations}, kernel_ms={result.kernel_time_ms:.3f})"
                )
            else:
                print(
                    f"  Problem {i}: {status} "
                    f"(planner failed, tree_a_size={result.tree_a_size}, tree_b_size={result.tree_b_size}, "
                    f"iters={result.iterations}, kernel_ms={result.kernel_time_ms:.3f})"
                )

        success_rate = sum(1 for r in results if r.solved) / batch_size
        print(f"\n  Success rate: {success_rate*100:.1f}%")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    if not args.no_viz:
        print("\nPreparing visualization for single planning result...")
        if single_result is None:
            print("  Skipping visualization: planner was not run")
        else:
            print(f"  Loaded {len(obstacles)} VAMP obstacles for visualization")
            visualize_tree_with_viser(
                robot=robot,
                urdf=urdf,
                result=single_result,
                obstacles=obstacles,
            )

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
