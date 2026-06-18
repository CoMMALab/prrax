"""Utilities for pRRTC problem setup and timing."""

from __future__ import annotations

import pickle
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import jax.numpy as jnp
import numpy as np

from pyroffi.collision._geometry import Box, Capsule, HalfSpace, Sphere


def load_vamp_problem(
    resource_root: Path,
    problem: str = "bookshelf_tall",
    index: int = 1,
):
    """Load a single VAMP problem dict for obstacle visualization."""
    problems_path = resource_root / "panda" / "problems.pkl"
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
    """Extract world obstacles in native geometry types for CUDA pRRTC."""
    spheres: list[np.ndarray] = []
    capsules: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    halfspaces: list[np.ndarray] = []

    for obs in obstacles or []:
        if isinstance(obs, Sphere):
            spheres.append(_sphere_geom_to_array(obs))
        elif isinstance(obs, Capsule):
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
            capsules.append(np.concatenate([p1, p2, radii], axis=-1).astype(np.float32))
        elif isinstance(obs, Box):
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
            boxes.append(np.concatenate([centers, a1, a2, a3, hl], axis=-1).astype(np.float32))
        elif isinstance(obs, HalfSpace):
            axes = obs.get_batch_axes()
            if len(axes) == 0:
                obs = obs.broadcast_to((1,))
            elif len(axes) > 1:
                obs = obs.reshape((-1,))
            normals = np.asarray(obs.normal, dtype=np.float32)
            points = np.asarray(obs.pose.translation(), dtype=np.float32)
            halfspaces.append(np.concatenate([normals, points], axis=-1).astype(np.float32))
        elif hasattr(obs, "to_trimesh"):
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
    """Build collision_context dict expected by cuda-rrtc/jax/prrtc.py."""
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

    link_ids = np.broadcast_to(
        np.arange(n_links, dtype=np.int32)[:, None],
        (n_links, n_spheres_per_link),
    )
    raw_link_per_sphere = link_ids[valid]              # raw link id per valid fine sphere
    local_per_sphere = sphere_local_full[valid]        # [n_valid, 3]
    radius_per_sphere = sphere_radius_full[valid]      # [n_valid]

    parent_joint_indices = np.asarray(robot.links.parent_joint_indices, dtype=np.int32)
    joint_per_sphere = parent_joint_indices[raw_link_per_sphere]  # FK frame per fine sphere

    # --- Group fine spheres by FK frame and sort so each group is contiguous. ---
    # The CUDA planner can then gate the fine check on a per-group "dirty" flag set by a
    # coarse bounding-sphere pre-pass (one coarse sphere per group). Sorting by joint frame
    # keeps each group's fine spheres in a single [offset[k], offset[k+1]) range.
    order = np.argsort(joint_per_sphere, kind="stable")
    sphere_link_idx = joint_per_sphere[order].astype(np.int32)
    sphere_local = local_per_sphere[order].astype(np.float32)
    sphere_radius = radius_per_sphere[order].astype(np.float32)

    # New (sorted) fine index for each old valid-sphere position, for remapping self-pairs.
    new_index_of_old = np.empty(order.shape[0], dtype=np.int32)
    new_index_of_old[order] = np.arange(order.shape[0], dtype=np.int32)

    # Contiguous group boundaries → one coarse bounding sphere per group.
    if sphere_link_idx.shape[0] > 0:
        boundaries = np.flatnonzero(np.diff(sphere_link_idx)) + 1
        group_starts = np.concatenate(([0], boundaries))
        group_ends = np.concatenate((boundaries, [sphere_link_idx.shape[0]]))
    else:
        group_starts = np.zeros((0,), dtype=np.int64)
        group_ends = np.zeros((0,), dtype=np.int64)

    coarse_sphere_link_idx = sphere_link_idx[group_starts].astype(np.int32)
    fine_link_offsets = np.concatenate(
        (group_starts, [sphere_link_idx.shape[0]])
    ).astype(np.int32)

    coarse_sphere_local = np.zeros((group_starts.shape[0], 3), dtype=np.float32)
    coarse_sphere_radius = np.zeros((group_starts.shape[0],), dtype=np.float32)
    for k, (s, e) in enumerate(zip(group_starts, group_ends)):
        locs = sphere_local[s:e]
        rads = sphere_radius[s:e]
        # AABB-center bounding sphere that *encloses* every fine sphere in the group, so a
        # coarse "clear" provably implies a fine "clear" (sound culling).
        c = 0.5 * (locs.min(axis=0) + locs.max(axis=0))
        coarse_sphere_local[k] = c
        coarse_sphere_radius[k] = float(np.max(np.linalg.norm(locs - c, axis=1) + rads))

    group_of_joint = {int(j): k for k, j in enumerate(coarse_sphere_link_idx)}

    # --- Self-collision pairs (fine), grouped by coarse link-pair with [start,end) ranges. ---
    flat_index = np.full((n_links, n_spheres_per_link), -1, dtype=np.int32)
    flat_index[valid] = new_index_of_old  # raw (link, slot) -> NEW sorted fine index

    active_i = np.asarray(robot_coll.active_idx_i, dtype=np.int32)
    active_j = np.asarray(robot_coll.active_idx_j, dtype=np.int32)
    self_pair_blocks = []
    coarse_self_pairs = []
    self_pair_ranges = []
    cursor = 0
    for li, lj in zip(active_i, active_j):
        a = flat_index[li]
        b = flat_index[lj]
        a = a[a >= 0]
        b = b[b >= 0]
        if a.size == 0 or b.size == 0:
            continue
        gi = group_of_joint.get(int(parent_joint_indices[li]), -1)
        gj = group_of_joint.get(int(parent_joint_indices[lj]), -1)
        if gi < 0 or gj < 0:
            continue
        block = np.stack(np.meshgrid(a, b, indexing="ij"), axis=-1).reshape(-1, 2)
        self_pair_blocks.append(block)
        coarse_self_pairs.append((gi, gj))
        self_pair_ranges.append((cursor, cursor + block.shape[0]))
        cursor += block.shape[0]

    self_pairs = (
        np.concatenate(self_pair_blocks, axis=0).astype(np.int32)
        if self_pair_blocks
        else np.zeros((0, 2), dtype=np.int32)
    )
    coarse_self_pairs = (
        np.asarray(coarse_self_pairs, dtype=np.int32)
        if coarse_self_pairs
        else np.zeros((0, 2), dtype=np.int32)
    )
    self_pair_ranges = (
        np.asarray(self_pair_ranges, dtype=np.int32)
        if self_pair_ranges
        else np.zeros((0, 2), dtype=np.int32)
    )

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
        # Coarse→fine culling tensors (optional; planner falls back to a flat sweep if absent)
        "coarse_sphere_link_idx": coarse_sphere_link_idx,
        "coarse_sphere_local": coarse_sphere_local,
        "coarse_sphere_radius": coarse_sphere_radius,
        "fine_link_offsets": fine_link_offsets,
        "coarse_self_pairs": coarse_self_pairs,
        "self_pair_ranges": self_pair_ranges,
        **world_geom,
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
    """Try local then global random repair for a collision-free configuration."""
    base = np.asarray(cfg, dtype=np.float32)
    span = (hi - lo).astype(np.float32)

    for scale in (0.01, 0.03, 0.08):
        for _ in range(400):
            candidate = np.clip(base + rng.normal(0.0, scale * span), lo, hi)
            report = config_collision_report(robot, robot_coll, candidate, world_obstacles)
            if report["collision_free"]:
                return jnp.asarray(candidate, dtype=jnp.float32), report

    for _ in range(1500):
        candidate = rng.uniform(lo, hi).astype(np.float32)
        report = config_collision_report(robot, robot_coll, candidate, world_obstacles)
        if report["collision_free"]:
            return jnp.asarray(candidate, dtype=jnp.float32), report

    return None, None


def synchronize_prrtc_result(result: Any) -> None:
    """Block until all JAX arrays inside a PRRTCResult are ready."""
    for name in (
        "path",
        "tree_a_configs",
        "tree_b_configs",
        "tree_a_parents",
        "tree_b_parents",
    ):
        value = getattr(result, name, None)
        if value is not None and hasattr(value, "block_until_ready"):
            value.block_until_ready()


def synchronize_prrtc_results(results: list[Any]) -> None:
    """Block until all JAX arrays across a list of PRRTCResult are ready."""
    for result in results:
        synchronize_prrtc_result(result)


def timed_prrtc_solve(callable_fn, *args, **kwargs):
    """Measure solve wall time including GPU kernel completion."""
    t0_ns = perf_counter_ns()
    result = callable_fn(*args, **kwargs)
    synchronize_prrtc_result(result)
    elapsed_ms = (perf_counter_ns() - t0_ns) / 1e6
    return result, elapsed_ms


def timed_prrtc_solve_batch(callable_fn, *args, **kwargs):
    """Measure batched solve wall time including GPU kernel completion."""
    t0_ns = perf_counter_ns()
    results = callable_fn(*args, **kwargs)
    synchronize_prrtc_results(results)
    elapsed_ms = (perf_counter_ns() - t0_ns) / 1e6
    return results, elapsed_ms
