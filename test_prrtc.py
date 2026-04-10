#!/usr/bin/env python3
"""
Test script for pRRTC with a VAMP problem from pyronot.
"""

import sys
import importlib.util
import argparse
import pickle
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

# Import pyronot
try:
    import pyronot as pk
    import yourdfpy
    from pyronot.collision._geometry import Box, Capsule, HalfSpace, Sphere
    from pyronot.collision._obstacles import create_collision_environment
    from pyronot.collision._robot_collision import RobotCollisionSpherized
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install pyronot and yourdfpy")
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
except Exception as e:
    print(f"cuda-rrtc import error: {e}")
    print("Make sure the library is compiled: cd cuda-rrtc && bash build.sh")
    sys.exit(1)


RESOURCE_ROOT = Path("/home/scoumar/Work/rrax/pyronot/resources")
PANDA_URDF = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"


def load_vamp_problem(problem: str = "bookshelf_tall", index: int = 1):
    """Load a single VAMP problem dict for obstacle visualization."""
    problems_path = RESOURCE_ROOT / "panda" / "problems.pkl"
    if not problems_path.exists():
        print(f"  VAMP problems file not found: {problems_path}")
        return None

    try:
        with open(problems_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  Failed to load VAMP problems: {e}")
        return None

    problem_map = data.get("problems", {})
    if problem not in problem_map:
        print(f"  VAMP problem '{problem}' not found. Available: {list(problem_map.keys())[:5]}")
        return None

    try:
        return next(p for p in problem_map[problem] if p.get("index") == index)
    except StopIteration:
        print(f"  VAMP problem '{problem}' has no index {index}")
        return None


def _sphere_geom_to_array(sphere_geom) -> np.ndarray:
    """Convert a (possibly batched) Sphere geom to [N,4] float32 array."""
    axes = sphere_geom.get_batch_axes()
    if len(axes) == 0:
        sphere_geom = sphere_geom.broadcast_to((1,))
    elif len(axes) > 1:
        sphere_geom = sphere_geom.reshape((-1,))
    centers = np.asarray(sphere_geom.pose.translation(), dtype=np.float32)
    radii = np.asarray(sphere_geom.radius, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([centers, radii], axis=-1)


def world_obstacles_to_exact(obstacles) -> dict[str, np.ndarray]:
    """Extract world obstacles in their native geometry types for CUDA pRRTC.

    The CUDA kernel supports sphere, capsule, box, and halfspace primitives
    natively.  Converting everything to spheres (the old approach) inflates
    obstacle volumes and makes collision checking overly conservative in
    narrow passages.
    """
    spheres: list[np.ndarray] = []
    capsules: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    halfspaces: list[np.ndarray] = []

    for obs in obstacles or []:
        if isinstance(obs, Sphere):
            spheres.append(_sphere_geom_to_array(obs))
        elif isinstance(obs, Capsule):
            # Capsule: [x1, y1, z1, x2, y2, z2, radius]
            axes = obs.get_batch_axes()
            if len(axes) == 0:
                obs = obs.broadcast_to((1,))
            elif len(axes) > 1:
                obs = obs.reshape((-1,))
            centers = np.asarray(obs.pose.translation(), dtype=np.float32)
            axis_dir = np.asarray(obs.axis, dtype=np.float32)
            half_h = np.asarray(obs.height, dtype=np.float32).reshape(-1, 1) / 2.0
            radii = np.asarray(obs.radius, dtype=np.float32).reshape(-1, 1)
            p1 = centers - axis_dir * half_h
            p2 = centers + axis_dir * half_h
            capsules.append(
                np.concatenate([p1, p2, radii], axis=-1).astype(np.float32)
            )
        elif isinstance(obs, Box):
            # Box: [cx, cy, cz, a1x..a1z, a2x..a2z, a3x..a3z, hl1, hl2, hl3]
            axes = obs.get_batch_axes()
            if len(axes) == 0:
                obs = obs.broadcast_to((1,))
            elif len(axes) > 1:
                obs = obs.reshape((-1,))
            centers = np.asarray(obs.pose.translation(), dtype=np.float32)
            rot = np.asarray(obs.pose.rotation().as_matrix(), dtype=np.float32)
            hl = np.asarray(obs.half_lengths, dtype=np.float32)
            a1 = rot[..., :, 0]
            a2 = rot[..., :, 1]
            a3 = rot[..., :, 2]
            boxes.append(
                np.concatenate([centers, a1, a2, a3, hl], axis=-1).astype(np.float32)
            )
        elif isinstance(obs, HalfSpace):
            # HalfSpace: [nx, ny, nz, px, py, pz]
            axes = obs.get_batch_axes()
            if len(axes) == 0:
                obs = obs.broadcast_to((1,))
            elif len(axes) > 1:
                obs = obs.reshape((-1,))
            normals = np.asarray(obs.normal, dtype=np.float32)
            points = np.asarray(obs.pose.translation(), dtype=np.float32)
            halfspaces.append(
                np.concatenate([normals, points], axis=-1).astype(np.float32)
            )
        elif hasattr(obs, "to_trimesh"):
            # Fallback: approximate unknown geometry as spheres
            spheres.append(_sphere_geom_to_array(Sphere.from_trimesh(obs.to_trimesh())))

    def _concat_or_empty(arrs, cols):
        if arrs:
            return np.concatenate(arrs, axis=0).astype(np.float32)
        return np.zeros((0, cols), dtype=np.float32)

    return {
        "world_spheres": _concat_or_empty(spheres, 4),
        "world_capsules": _concat_or_empty(capsules, 7),
        "world_boxes": _concat_or_empty(boxes, 15),
        "world_halfspaces": _concat_or_empty(halfspaces, 6),
    }


def build_prrtc_collision_context(robot, robot_coll, world_obstacles):
    """Build collision_context dict expected by cuda-rrtc jax/prrtc.py."""
    fk_twists = np.asarray(robot.joints.twists, dtype=np.float32)
    fk_parent_tf = np.asarray(robot.joints.parent_transforms, dtype=np.float32)
    fk_parent_idx = np.asarray(robot.joints.parent_indices, dtype=np.int32)
    fk_act_idx = np.asarray(robot.joints.actuated_indices, dtype=np.int32)
    fk_mimic_mul = np.asarray(robot.joints.mimic_multiplier, dtype=np.float32)
    fk_mimic_off = np.asarray(robot.joints.mimic_offset, dtype=np.float32)
    fk_mimic_act_idx = np.asarray(robot.joints.mimic_act_indices, dtype=np.int32)
    fk_topo_inv = np.asarray(robot.joints._topo_sort_inv, dtype=np.int32)

    sphere_local_full = np.asarray(robot_coll.coll.pose.translation(), dtype=np.float32)
    sphere_radius_full = np.asarray(robot_coll.coll.radius, dtype=np.float32)
    n_links, n_spheres_per_link = sphere_radius_full.shape
    valid = sphere_radius_full > 0.0

    link_ids = np.broadcast_to(np.arange(n_links, dtype=np.int32)[:, None], (n_links, n_spheres_per_link))
    sphere_link_idx_raw = link_ids[valid]
    sphere_local = sphere_local_full[valid]
    sphere_radius = sphere_radius_full[valid]

    # The CUDA FK produces T_world indexed by JOINT (n_joints entries), but
    # the collision model is indexed by LINK (n_links entries).  For Panda,
    # n_joints=12 but n_links=13, so link index 12 would read uninitialised
    # memory.  Remap each sphere's link index to its parent joint index so
    # the CUDA kernel indexes into T_world correctly.
    parent_joint_indices = np.asarray(robot.links.parent_joint_indices, dtype=np.int32)
    sphere_link_idx = parent_joint_indices[sphere_link_idx_raw]
    # Base-link spheres get parent_joint=-1; the CUDA kernel already skips
    # negative indices (guard: if link_idx < 0 ... continue).

    flat_index = np.full((n_links, n_spheres_per_link), -1, dtype=np.int32)
    flat_index[valid] = np.arange(sphere_radius.shape[0], dtype=np.int32)

    pairs = []
    active_i = np.asarray(robot_coll.active_idx_i, dtype=np.int32)
    active_j = np.asarray(robot_coll.active_idx_j, dtype=np.int32)
    for li, lj in zip(active_i, active_j):
        a = flat_index[li]
        b = flat_index[lj]
        a = a[a >= 0]
        b = b[b >= 0]
        if a.size == 0 or b.size == 0:
            continue
        pairs.append(np.stack(np.meshgrid(a, b, indexing="ij"), axis=-1).reshape(-1, 2))
    self_pairs = np.concatenate(pairs, axis=0).astype(np.int32) if pairs else np.zeros((0, 2), dtype=np.int32)

    world_geom = world_obstacles_to_exact(world_obstacles)

    return {
        "fk_twists": fk_twists,
        "fk_parent_tf": fk_parent_tf,
        "fk_parent_idx": fk_parent_idx,
        "fk_act_idx": fk_act_idx,
        "fk_mimic_mul": fk_mimic_mul,
        "fk_mimic_off": fk_mimic_off,
        "fk_mimic_act_idx": fk_mimic_act_idx,
        "fk_topo_inv": fk_topo_inv,
        "sphere_link_idx": sphere_link_idx,
        "sphere_local": sphere_local,
        "sphere_radius": sphere_radius,
        "self_pairs": self_pairs,
        **world_geom,
    }


def validate_path_collision(robot, robot_coll, path, world_obstacles):
    """Validate solved path against full pyronot collision model."""
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


def config_collision_report(robot, robot_coll, cfg, world_obstacles):
    """Collision report for a single configuration."""
    q_jax = jnp.asarray(cfg, dtype=jnp.float32)
    min_self = float(np.min(np.asarray(robot_coll.compute_self_collision_distance(robot, q_jax))))
    min_world = np.inf
    for obs in world_obstacles or []:
        world_d = robot_coll.compute_world_collision_distance(robot, q_jax, obs)
        min_world = min(min_world, float(np.min(np.asarray(world_d))))
    min_margin = min(min_self, min_world)
    return {
        "min_self": min_self,
        "min_world": min_world,
        "min_margin": min_margin,
        "collision_free": bool(min_margin > 0.0),
    }


def repair_collision_config(robot, robot_coll, cfg, lo, hi, world_obstacles, rng):
    """Try to find a nearby collision-free config; fallback to global random samples."""
    base = np.asarray(cfg, dtype=np.float32)
    span = (hi - lo).astype(np.float32)

    # First search locally around the original configuration.
    for scale in (0.01, 0.03, 0.08):
        for _ in range(400):
            candidate = np.clip(base + rng.normal(0.0, scale * span), lo, hi)
            report = config_collision_report(robot, robot_coll, candidate, world_obstacles)
            if report["collision_free"]:
                return jnp.asarray(candidate, dtype=jnp.float32), report

    # Then fallback to global samples inside joint limits.
    for _ in range(1500):
        candidate = rng.uniform(lo, hi).astype(np.float32)
        report = config_collision_report(robot, robot_coll, candidate, world_obstacles)
        if report["collision_free"]:
            return jnp.asarray(candidate, dtype=jnp.float32), report

    return None, None


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
    args = parser.parse_args()

    print("=" * 70)
    print("Testing pRRTC with PyRoNot VAMP problem")
    print("=" * 70)

    if not PANDA_URDF.exists():
        print(f"ERROR: URDF not found at {PANDA_URDF}")
        print("Please run pyronot resource generation first")
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

    vamp_problem = load_vamp_problem(problem=args.vamp_problem, index=args.vamp_index)
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
            result = prrtc_plan(
                start_config=start_config,
                goal_configs=goal_config.reshape(1, -1),
                max_iterations=5000,
                step_size=1,
                num_new_samples=64,
                dynamic_domain=False,
                min_vals=jnp.array(lo, dtype=jnp.float32),
                max_vals=jnp.array(hi, dtype=jnp.float32),
                collision_context=collision_context,
            )
            single_result = result
            print(f"  Result: solved={result.solved}")
            print(f"  Tree A size: {result.tree_a_size}")
            print(f"  Tree B size: {result.tree_b_size}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Cost: {result.cost:.4f}")
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

        results = []
        for i in range(batch_size):
            result = prrtc_plan(
                start_config=jnp.array(starts_np[i]),
                goal_configs=jnp.array(goals_np[i]).reshape(1, -1),
                max_iterations=1000,
                step_size=0.6,
                num_new_samples=16,
                dynamic_domain=False,
                min_vals=jnp.array(lo, dtype=jnp.float32),
                max_vals=jnp.array(hi, dtype=jnp.float32),
                collision_context=collision_context,
            )
            results.append(result)
            status = "✓" if result.solved else "✗"
            if result.solved:
                print(
                    f"  Problem {i}: {status} "
                    f"(tree_a_size={result.tree_a_size}, tree_b_size={result.tree_b_size}, iters={result.iterations})"
                )
            else:
                print(
                    f"  Problem {i}: {status} "
                    f"(planner failed, tree_a_size={result.tree_a_size}, tree_b_size={result.tree_b_size}, iters={result.iterations})"
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
