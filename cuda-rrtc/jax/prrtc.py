"""
pRRTC - Parallel Rapidly-exploring Random Tree JAX Interface.

This module provides a JAX FFI wrapper for the pRRTC motion planner, which finds
collision-free paths in configuration space using a parallel CUDA implementation.

The planner is compiled per robot through GRiM (see ``_grim_build``): the robot's kinematics
(cricket2's straight-line trace) and sphere model are baked into the build, so a call passes
only starts, goals, joint limits and world obstacles.

Key features:
  - Two-tree bidirectional planning (start and goal trees)
  - Parallel tree expansion using CUDA, one block per sample
  - Halton sequence sampling for better configuration space coverage
  - Dynamic tree balancing for efficient search
  - Batched planning: every problem has its own world and runs on its own stream
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

try:
    from . import _grim_build
except ImportError:   # loaded by file path (importlib spec), not as a package
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import _grim_build


class PRRTCResult(NamedTuple):
    """Result of pRRTC planning (host numpy arrays)."""

    solved: bool
    path: Optional[np.ndarray]
    tree_a_size: int
    tree_b_size: int
    iterations: int
    cost: float
    kernel_time_ms: Optional[float] = None
    # Raw tree data for visualization (shape: (dim, max_nodes), sliced to tree_*_size)
    tree_a_configs: Optional[Array] = None
    tree_b_configs: Optional[Array] = None
    tree_a_parents: Optional[Array] = None
    tree_b_parents: Optional[Array] = None


# Collision-checker selection passed to the CUDA kernel as ``collision_mode``.
#   "binary"             — pRRTC-style early-exit boolean check (fewest ops/config).
#   "differentiable"     — full smooth signed-distance-field sweep (no early exit).
#   "binary_coarse"      — coarse bounding-sphere pre-pass gating the fine check.
#   "binary_coarse_coop" — the coarse check split over lane-groups of the block.
_COLLISION_MODES = {"binary": 0, "differentiable": 1, "binary_coarse": 2, "binary_coarse_coop": 3}
_WORLD = (("world_spheres", 4), ("world_capsules", 7), ("world_boxes", 15), ("world_halfspaces", 6))


def _resolve_collision_mode(collision_checker: str) -> int:
    try:
        return _COLLISION_MODES[collision_checker]
    except KeyError:
        if collision_checker == "robogpu":
            raise ValueError(
                "collision_checker='robogpu' is not available in the monolithic "
                "in-kernel planner: OptiX ray-tracing cannot be invoked from "
                "inside a CUDA kernel. Use the host-driven planner instead:\n"
                "    from cuda_rrtc.jax import prrtc_plan_robogpu\n"
                "which validates edges with pyroffi's RoboGPUCollisionChecker "
                "(OptiX point-cloud BVH)."
            ) from None
        raise ValueError(
            f"collision_checker must be one of {sorted(_COLLISION_MODES)}, "
            f"got {collision_checker!r}"
        ) from None


_targets: dict[str, tuple[str, ...]] = {}


def _planner_targets(collision_context, ee_joints: tuple[int, ...]) -> tuple[str, ...]:
    """The build ``(planner, fk, tsr)`` for a context's robot and sphere model with constraint
    frames ``ee_joints`` (built on first use)."""
    h = hashlib.sha1(f"{id(collision_context['robot'])}{ee_joints}".encode())
    for k in _grim_build.ROBOT_KEYS:
        h.update(np.ascontiguousarray(collision_context[k]).tobytes())
    key = h.hexdigest()
    if key not in _targets:
        _targets[key] = _grim_build.planner_targets(collision_context, ee_joints)
    return _targets[key]


# --- Manifold constraints (CUDA/prrtc_constraint.cuh) ---------------------------------------
_CONS_MAX, _CONS_HDR, _CONS_STRIDE = 4, 12, 28
_CONS_SIZE = _CONS_HDR + _CONS_MAX * _CONS_STRIDE
_PROJ_MODES = {"mcvamp": 0, "cprrtc": 1}


class TSRConstraint(NamedTuple):
    """Task Space Region: x = [t; log R] of ``reference^-1 * T_ee * offset^-1`` must lie in
    [lower, upper] (McVAMP's TaskSpaceConstraint). Poses are [qw, qx, qy, qz, x, y, z]."""

    ee_joint: int             # joint frame posing the end-effector link
    offset: np.ndarray        # rTe, the end-effector offset frame (7,)
    reference: np.ndarray     # wTr, the TSR reference frame in the world (7,)
    lower: np.ndarray         # (6,)
    upper: np.ndarray         # (6,)


def make_tsr_constraint(robot, ee_link: str, reference, lower, upper,
                        offset=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) -> TSRConstraint:
    """A TSR constraint on link ``ee_link`` of a pyroffi ``robot``."""
    link = list(robot.links.names).index(ee_link)
    return TSRConstraint(int(robot.links.parent_joint_indices[link]),
                         np.asarray(offset, np.float32), np.asarray(reference, np.float32),
                         np.asarray(lower, np.float32), np.asarray(upper, np.float32))


class ProjectionSettings(NamedTuple):
    """Projection onto the constraint manifold (defaults: McVAMP's line/plane benchmark)."""

    mode: str = "mcvamp"      # "mcvamp" two-stage particles, or "cprrtc" smoothed joint
    tol: float = 1e-3         # converged when |d| < tol for every constraint
    iters: int = 15
    damping: float = 1e-6     # LM lambda
    rate: float = 1.0         # descent rate
    max_step: float = 2.0     # a longer projection step means divergence
    perturbation: float = 0.1 # McVAMP steer-particle scale
    gap: float = 2.0          # max waypoint gap = gap * edge length / granularity
    connect_eps: float = 1e-3 # connect reached the other tree within this distance


def _se3_inv(T) -> np.ndarray:
    """Inverse of a [qw, qx, qy, qz, x, y, z] pose."""
    T = np.asarray(T, np.float64)
    w, v, t = T[0], -T[1:4], T[4:]
    # R^T t = rotation of t by the conjugate quaternion (w, v).
    t_inv = -(t + 2.0 * np.cross(v, np.cross(v, t) + w * t))
    return np.concatenate(([w], v, t_inv)).astype(np.float32)


def _pack_constraints(cons: list[TSRConstraint], ee_joints: tuple[int, ...],
                      proj: ProjectionSettings) -> np.ndarray:
    if len(cons) > _CONS_MAX:
        raise ValueError(f"at most {_CONS_MAX} constraints per problem, got {len(cons)}")
    out = np.zeros(_CONS_SIZE, np.float32)
    out[:10] = [len(cons), _PROJ_MODES[proj.mode], proj.tol, proj.iters, proj.damping,
                proj.rate, proj.max_step, proj.perturbation, proj.gap, proj.connect_eps]
    for c, con in enumerate(cons):
        o = _CONS_HDR + c * _CONS_STRIDE
        out[o] = ee_joints.index(con.ee_joint)
        out[o + 1:o + 8] = _se3_inv(con.offset)
        out[o + 8:o + 15] = _se3_inv(con.reference)
        out[o + 15:o + 21] = con.lower
        out[o + 21:o + 27] = con.upper
    return out


def constraint_error(collision_context, constraints: list[TSRConstraint], q,
                     projection: ProjectionSettings = ProjectionSettings(),
                     min_vals=None, max_vals=None):
    """``(err, q_proj, ok)`` at configurations ``q`` (B, dim): |d| over the constraints, and
    the planner's projection of q onto their manifold."""
    ee = tuple(sorted({c.ee_joint for c in constraints}))
    tsr = _planner_targets(collision_context, ee)[2]
    q = np.atleast_2d(np.asarray(q, np.float32))
    dim = q.shape[1]
    lo = np.asarray(-np.pi * np.ones(dim) if min_vals is None else min_vals, np.float32)
    hi = np.asarray(np.pi * np.ones(dim) if max_vals is None else max_vals, np.float32)
    err2, q_proj, ok = jax.device_get(_tsr_call(tsr, q.shape)(
        q, _pack_constraints(list(constraints), ee, projection), lo, hi))
    return np.sqrt(err2), q_proj, ok.astype(bool)


def config_in_collision(collision_context, q, constraints: list[TSRConstraint] = (),
                        collision_checker: str = "binary"):
    """``(B,)`` bool: the planner build's own collision verdict at configurations ``q`` (B, dim)
    -- the exact model the planner checks edges with. ``constraints`` only selects the build
    (pass the planner's, so no second build compiles)."""
    ee = tuple(sorted({c.ee_joint for c in constraints}))
    name = _planner_targets(collision_context, ee)[3]
    q = np.atleast_2d(np.asarray(q, np.float32))
    worlds = _pack_worlds([collision_context])
    hit = jax.ffi.ffi_call(name, jax.ShapeDtypeStruct((q.shape[0],), jnp.int32))(
        q, *worlds[:4], worlds[4].astype(np.float32), mode=np.int32(_resolve_collision_mode(collision_checker)))
    return np.asarray(hit).astype(bool)


@lru_cache(maxsize=64)
def _tsr_call(name: str, shape: tuple[int, int]):
    return jax.jit(jax.ffi.ffi_call(name, (
        jax.ShapeDtypeStruct((shape[0],), jnp.float32),
        jax.ShapeDtypeStruct(shape, jnp.float32),
        jax.ShapeDtypeStruct((shape[0],), jnp.int32),
    )))


@lru_cache(maxsize=256)
def _planner_call(name: str, batch: int, dim: int, max_nodes: int, jit: bool, wps: int,
                  max_path: int, **attrs):
    shapes = (
        jax.ShapeDtypeStruct((batch, dim, max_nodes), jnp.float32),  # tree_a_configs
        jax.ShapeDtypeStruct((batch, dim, max_nodes), jnp.float32),  # tree_b_configs
        jax.ShapeDtypeStruct((batch, max_nodes), jnp.int32),         # tree_a_parents
        jax.ShapeDtypeStruct((batch, max_nodes), jnp.int32),         # tree_b_parents
        jax.ShapeDtypeStruct((batch, 2), jnp.int32),                 # tree_sizes
        jax.ShapeDtypeStruct((batch, 2), jnp.int32),                 # completed
        jax.ShapeDtypeStruct((batch, 1), jnp.int32),                 # iter_count
        jax.ShapeDtypeStruct((batch, 3), jnp.int32),                 # connection_info
        jax.ShapeDtypeStruct((batch, 1), jnp.int32),                 # solved_flag
        jax.ShapeDtypeStruct((batch,), jnp.float32),                 # kernel_time_ms
        jax.ShapeDtypeStruct((batch, wps), jnp.float32),             # edge_wps_a
        jax.ShapeDtypeStruct((batch, wps), jnp.float32),             # edge_wps_b
        jax.ShapeDtypeStruct((batch, max_path, dim), jnp.float32),   # path
        jax.ShapeDtypeStruct((batch,), jnp.int32),                   # path_len
    )
    attrs = {k: (np.float32(v) if isinstance(v, float) else np.int32(v)) for k, v in attrs.items()}

    def call(*args):
        return jax.ffi.ffi_call(name, shapes)(*args, max_nodes=np.int32(max_nodes),
                                              batch=np.int32(batch), dim=np.int32(dim), **attrs)

    return jax.jit(call) if jit else call


@lru_cache(maxsize=None)
def _take_path(n: int):
    """Problem i's first n path points, as one compiled gather."""
    return jax.jit(lambda path, i: jax.lax.dynamic_index_in_dim(path, i, keepdims=False)[:n])


@lru_cache(maxsize=None)
def _take_trees(na: int, nb: int):
    """Problem i's first na / nb nodes of both trees, as one compiled gather."""
    def take(ta, tb, pa, pb, i):
        return (jax.lax.dynamic_index_in_dim(ta, i, keepdims=False)[:, :na],
                jax.lax.dynamic_index_in_dim(tb, i, keepdims=False)[:, :nb],
                jax.lax.dynamic_index_in_dim(pa, i, keepdims=False)[:na],
                jax.lax.dynamic_index_in_dim(pb, i, keepdims=False)[:nb])
    return jax.jit(take)


def _pack_worlds(contexts) -> list[np.ndarray]:
    """Per-problem world obstacles as padded ``[batch, max_n, width]`` tensors + counts."""
    out, counts = [], np.zeros((len(contexts), 4), np.int32)
    for w, (key, width) in enumerate(_WORLD):
        arrays = [np.asarray(c.get(key, np.zeros((0, width))), np.float32).reshape(-1, width)
                  for c in contexts]
        counts[:, w] = [a.shape[0] for a in arrays]
        packed = np.zeros((len(contexts), max(counts[:, w].max(), 1), width), np.float32)
        for i, a in enumerate(arrays):
            packed[i, : a.shape[0]] = a
        out.append(packed)
    return out + [counts]


def _solve(
    start_configs, goal_configs, contexts, *, max_iterations, step_size, num_new_samples,
    granularity, max_nodes, balance_mode, tree_ratio, dynamic_domain, dd_alpha, dd_radius,
    dd_min_radius, min_vals, max_vals, jit_trace, collision_checker, collision_margin,
    constraints, projection, return_trees=False, max_path_points=8192, ao_passes=0,
    cost_bound=None, shortcut_rounds=0,
) -> list[PRRTCResult]:
    collision_mode = _resolve_collision_mode(collision_checker)
    if any(c is None for c in contexts):
        raise ValueError("collision_context is required: build it with "
                         "utils.build_prrtc_collision_context(robot, robot_coll, obstacles)")
    if step_size <= 0.0:
        raise ValueError("step_size must be > 0")
    # 4 threads per waypoint (PRRAX_WP_LANES in CUDA/prrtc_constraint.cuh), at most 128 per block.
    if granularity < 1 or (granularity & (granularity - 1)) != 0 or granularity > 32:
        raise ValueError(f"granularity must be a power of 2 <= 32, got {granularity}")
    if projection.mode not in _PROJ_MODES:
        raise ValueError(f"projection.mode must be one of {sorted(_PROJ_MODES)}")
    ee = tuple(sorted({c.ee_joint for cons in constraints for c in cons}))
    names = {_planner_targets(c, ee) for c in contexts}
    if len(names) != 1:
        raise ValueError("every problem in a batch must share one robot and sphere model")
    names = names.pop()

    start_configs = np.array(start_configs, np.float32)   # copies: roots may be projected
    goal_configs = np.array(goal_configs, np.float32)
    batch, dim = start_configs.shape
    n_act = int(contexts[0]["robot"].joints.num_actuated_joints)
    if dim != n_act or goal_configs.shape[-1] != n_act:
        raise ValueError(f"configurations must have the robot's {n_act} actuated joints, got {dim}")
    min_vals = np.asarray(-np.pi * np.ones(dim) if min_vals is None else min_vals, np.float32)
    max_vals = np.asarray(np.pi * np.ones(dim) if max_vals is None else max_vals, np.float32)
    packed = np.stack([_pack_constraints(list(c), ee, projection) for c in constraints])
    wps = int(max_nodes) * (int(granularity) - 1) * dim if any(constraints) else 1
    worlds = _pack_worlds(contexts)
    call = _planner_call(
        names[0], batch, dim, int(max_nodes), bool(jit_trace), wps, int(max_path_points),
        max_iterations=int(max_iterations), step_size=float(step_size),
        num_new_samples=int(num_new_samples), balance_mode=int(balance_mode),
        tree_ratio=float(tree_ratio), dynamic_domain=int(bool(dynamic_domain)),
        dd_alpha=float(dd_alpha), dd_radius=float(dd_radius), dd_min_radius=float(dd_min_radius),
        granularity=int(granularity), collision_mode=collision_mode,
        collision_margin=float(collision_margin), shortcut_rounds=int(shortcut_rounds),
        num_goals=int(goal_configs.shape[1]), mw0=worlds[0].shape[1], mw1=worlds[1].shape[1],
        mw2=worlds[2].shape[1], mw3=worlds[3].shape[1],
    )
    # AO-x (MR. POP / pAORRTC): pass 0 finds any path; each further pass replans from scratch
    # for a path cheaper than the best so far (informed sampling, cost-bounded trees). A
    # problem that is unsolved, or not improved in a pass, keeps its best result.
    def run(best):
        # One host->device copy: every input in one float32 buffer (layout: PrrtcPlannerImpl).
        raw = call(np.concatenate([a.astype(np.float32, copy=False).ravel() for a in (
            start_configs, goal_configs, min_vals, max_vals, *worlds, packed, best)]))
        # Only the small outputs and the traced path leave the device (trees on request), in one
        # transfer for small batches (whole path buffers), per problem for large ones.
        small = (raw[4], raw[6], raw[7], raw[8], raw[9], raw[13])
        if batch <= 16:
            sizes, iters, conn, solved_flags, ktimes, plen, paths = jax.device_get(small + (raw[12],))
        else:
            (sizes, iters, conn, solved_flags, ktimes, plen), paths = jax.device_get(small), None

        results = []
        for i in range(batch):
            size_a, size_b = (int(x) for x in sizes[i])
            solved = bool(solved_flags[i][0] == 1)
            trees = (None,) * 4
            if return_trees or (solved and plen[i] < 0):
                # Slices of power-of-two length: a few compiled shapes, not one per tree size.
                na, nb = (min(1 << max(n - 1, 0).bit_length(), max_nodes) for n in (size_a, size_b))
                ta_cfg, tb_cfg, ta_par, tb_par = jax.device_get(_take_trees(na, nb)(*raw[:4], i))
                trees = (ta_cfg[:, :size_a], tb_cfg[:, :size_b], ta_par[:size_a], tb_par[:size_b])
            path = None
            if solved and plen[i] > 0:
                n = int(plen[i])
                path = (paths[i, :n] if paths is not None
                        else np.asarray(_take_path(1 << (n - 1).bit_length())(raw[12], i))[:n])
            elif solved:   # longer than the path buffer: trace it on the host
                ta_cfg, tb_cfg, ta_par, tb_par = trees
                nodes_a, nodes_b = _path_nodes(ta_par, tb_par, conn[i])
                path_a, path_b = ta_cfg[:, nodes_a].T, tb_cfg[:, nodes_b].T
                if constraints[i]:   # expand each edge's projected interior waypoints
                    W1 = int(granularity) - 1
                    wa, wb = (np.asarray(_take_wps(len(n), W1, dim)(raw[k], i, np.asarray(n, np.int32)))
                              for k, n in ((10, nodes_a), (11, nodes_b)))
                    path_a = np.concatenate([path_a[:1]] + [np.concatenate((wa[j], path_a[j:j + 1]))
                                                            for j in range(1, len(nodes_a))])
                    path_b = np.concatenate([np.concatenate((path_b[j:j + 1], wb[j][::-1]))
                                             for j in range(len(nodes_b) - 1)] + [path_b[-1:]])
                path = np.concatenate((path_a, path_b))
            results.append(PRRTCResult(
                solved=solved,
                path=path,
                tree_a_size=size_a,
                tree_b_size=size_b,
                iterations=int(iters[i][0]),
                cost=float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if solved else float("inf"),
                kernel_time_ms=float(ktimes[i]),   # GPU time of setup + root projection + planning
                tree_a_configs=trees[0] if return_trees else None,
                tree_b_configs=trees[1] if return_trees else None,
                tree_a_parents=trees[2] if return_trees else None,
                tree_b_parents=trees[3] if return_trees else None,
            ))
        return results

    # cost_bound: the first run is already cost-bounded (e.g. a bound shared from an outer
    # planner's incumbent); a path is only returned if it is cheaper.
    results = run(np.full(batch, np.finfo(np.float32).max if cost_bound is None else cost_bound,
                          np.float32))
    for _ in range(int(ao_passes)):
        best = np.array([r.cost if r.solved else -1.0 for r in results], np.float32)
        if (best < 0).all():
            break
        for i, r in enumerate(run(best)):
            old = results[i]
            t = (old.kernel_time_ms or 0.0) + (r.kernel_time_ms or 0.0)
            results[i] = (r if r.solved and r.cost < old.cost else old)._replace(
                kernel_time_ms=t, iterations=old.iterations + r.iterations)
    return results


def prrtc_plan(
    start_config: Float[Array, "*batch dim"],
    goal_configs: Float[Array, "num_goals dim"],
    max_iterations: int = 1_000_000,
    step_size: float = 0.5,
    num_new_samples: int = 128,
    granularity: int = 16,
    max_nodes: int = 1_000_000,
    balance_mode: int = 1,
    tree_ratio: float = 0.5,
    dynamic_domain: bool = True,
    dd_alpha: float = 1e-4,
    dd_radius: float = 4.0,
    dd_min_radius: float = 1.0,
    min_vals: Optional[Float[Array, "dim"]] = None,
    max_vals: Optional[Float[Array, "dim"]] = None,
    collision_context: Optional[dict] = None,
    jit_trace: bool = False,
    collision_checker: str = "binary",
    collision_margin: float = 0.02,
    constraints: Sequence[TSRConstraint] = (),
    projection: ProjectionSettings = ProjectionSettings(),
    return_trees: bool = False,
    ao_passes: int = 0,
    shortcut_rounds: int = 0,
) -> PRRTCResult:
    """
    Plan a path using the parallel RRTC (pRRTC) algorithm.

    Args:
        start_config: Start configuration, shape ``(dim,)`` or ``(1, dim)``.
        goal_configs: Goal configurations, shape ``(num_goals, dim)``.
        max_iterations: Maximum number of planning iterations (per block).
        step_size: Maximum extension step size.
        num_new_samples: Number of blocks, each extending toward one sample per iteration.
        granularity: Waypoints checked per edge (4 threads each). Power of 2, max 32.
        max_nodes: Maximum nodes per tree.
        balance_mode: Tree balancing mode (0/1/2) matching pRRTC settings.
        tree_ratio: Balance threshold ratio used by mode 1/2.
        dynamic_domain: Enable dynamic-domain radius adaptation.
        dd_alpha, dd_radius, dd_min_radius: Dynamic-domain parameters.
        min_vals, max_vals: Sampling bounds (default ``[-pi, pi]``).
        collision_context: From ``utils.build_prrtc_collision_context(robot, robot_coll,
            obstacles)``. Its robot and sphere model select (and on first use build) the
            planner; its world obstacles are inputs of this call.
        jit_trace: Dispatch through a cached ``jax.jit`` wrapper.
        collision_checker: ``"binary"``, ``"differentiable"``, ``"binary_coarse"`` or
            ``"binary_coarse_coop"`` (see ``_COLLISION_MODES``).
        collision_margin: Safety margin (metres) for the ``"differentiable"`` checker.
        constraints: Manifold constraints (``make_tsr_constraint``, at most 4). Every edge is
            then projected onto their intersection and every projected waypoint is a tree node,
            so the path lies on the manifold at waypoint resolution. A start or goal off the
            manifold is projected onto it first (the path then starts/ends there); one that
            does not project leaves the problem unsolved.
        projection: Projection settings (``ProjectionSettings``).
        return_trees: Also return both trees (``tree_*_configs`` / ``tree_*_parents``), e.g. to
            visualize them; off by default (the path is traced on the GPU).
        shortcut_rounds: MR. POP path-simplification rounds on the GPU before the path is
            returned. Each round runs 3 sweeps of B-spline smoothing, then takes the shortest
            path through the shortcut graph between up to 64 anchor waypoints. Every change is
            validated by the planner's own connect (projected steps plus edge collision
            checks). With ``ao_passes``, each pass simplifies its path, so the next pass is
            bounded by the simplified cost.
        ao_passes: AO-x optimizing passes after the first solution (MR. POP / pAORRTC): each
            replans for a path cheaper than the best so far. ``kernel_time_ms`` and
            ``iterations`` then sum over the passes.

    Returns:
        PRRTCResult (path from start root to a goal root when solved).
    """
    start = np.atleast_2d(np.asarray(start_config, np.float32))
    if start.shape[0] != 1:
        raise ValueError("prrtc_plan takes one start; use prrtc_plan_batch for several")
    goals = np.asarray(goal_configs, np.float32).reshape(1, -1, start.shape[1])
    return _solve(
        start, goals, [collision_context], max_iterations=max_iterations, step_size=step_size,
        num_new_samples=num_new_samples, granularity=granularity, max_nodes=max_nodes,
        balance_mode=balance_mode, tree_ratio=tree_ratio, dynamic_domain=dynamic_domain,
        dd_alpha=dd_alpha, dd_radius=dd_radius, dd_min_radius=dd_min_radius,
        min_vals=min_vals, max_vals=max_vals, jit_trace=jit_trace,
        collision_checker=collision_checker, collision_margin=collision_margin,
        constraints=[list(constraints)], projection=projection, return_trees=return_trees,
        ao_passes=ao_passes,
        shortcut_rounds=shortcut_rounds,
    )[0]


def prrtc_plan_batch(
    start_configs: Float[Array, "batch dim"],
    goal_configs: Float[Array, "batch num_goals dim"],
    max_iterations: int = 1_000_000,
    step_size: float = 0.5,
    num_new_samples: int = 128,
    granularity: int = 16,
    max_nodes: int = 1_000_000,
    balance_mode: int = 1,
    tree_ratio: float = 0.5,
    dynamic_domain: bool = True,
    dd_alpha: float = 1e-4,
    dd_radius: float = 4.0,
    dd_min_radius: float = 1.0,
    min_vals: Optional[Float[Array, "dim"]] = None,
    max_vals: Optional[Float[Array, "dim"]] = None,
    collision_context: Optional[dict | list | tuple] = None,
    jit_trace: bool = False,
    collision_checker: str = "binary",
    collision_margin: float = 0.02,
    constraints: Sequence = (),
    projection: ProjectionSettings = ProjectionSettings(),
    return_trees: bool = False,
    ao_passes: int = 0,
    cost_bound: Optional[float] = None,
    shortcut_rounds: int = 0,
) -> list[PRRTCResult]:
    """
    Plan paths for a batch of independent start/goal pairs in parallel on the GPU.

    Each problem runs on its own CUDA stream. ``goal_configs`` is ``(batch, num_goals, dim)``
    (or ``(batch, dim)`` for one goal each). ``collision_context`` is one context shared by
    every problem, or a list of ``batch`` contexts with their own world obstacles (they must
    share the robot and sphere model). ``constraints`` is one list of constraints shared by
    every problem, or a list of ``batch`` lists. All other arguments are as in ``prrtc_plan``.

    Returns:
        One ``PRRTCResult`` per problem, in order.
    """
    starts = np.atleast_2d(np.asarray(start_configs, np.float32))
    goals = np.asarray(goal_configs, np.float32)
    if goals.ndim == 2:
        goals = goals[:, None, :]
    if goals.ndim != 3 or goals.shape[0] != starts.shape[0]:
        raise ValueError(f"goal_configs must be (batch, num_goals, dim), got {goals.shape}")
    contexts = (list(collision_context) if isinstance(collision_context, (list, tuple))
                else [collision_context] * starts.shape[0])
    if len(contexts) != starts.shape[0]:
        raise ValueError(f"need {starts.shape[0]} collision contexts, got {len(contexts)}")
    constraints = list(constraints)
    if not (constraints and isinstance(constraints[0], (list, tuple))
            and not isinstance(constraints[0], TSRConstraint)):
        constraints = [constraints] * starts.shape[0]
    if len(constraints) != starts.shape[0]:
        raise ValueError(f"need {starts.shape[0]} constraint lists, got {len(constraints)}")
    return _solve(
        starts, goals, contexts, max_iterations=max_iterations, step_size=step_size,
        num_new_samples=num_new_samples, granularity=granularity, max_nodes=max_nodes,
        balance_mode=balance_mode, tree_ratio=tree_ratio, dynamic_domain=dynamic_domain,
        dd_alpha=dd_alpha, dd_radius=dd_radius, dd_min_radius=dd_min_radius,
        min_vals=min_vals, max_vals=max_vals, jit_trace=jit_trace,
        collision_checker=collision_checker, collision_margin=collision_margin,
        constraints=constraints, projection=projection, return_trees=return_trees,
        ao_passes=ao_passes,
        cost_bound=cost_bound,
        shortcut_rounds=shortcut_rounds,
    )


def _path_nodes(tree_a_parents, tree_b_parents, connection_info):
    """Node indices from the start root (tree A, index 0) to tree A's connection node, and from
    tree B's connection node to its goal root. connection_info: [node in A, node in B, tree
    expanded during connect]."""
    def walk(parents, node):
        nodes = []
        for _ in range(parents.shape[0]):
            nodes.append(node)
            parent = int(parents[node])
            if parent == node or parent < 0:
                break
            node = parent
        return nodes

    a, b = int(connection_info[0]), int(connection_info[1])
    if a < 0 or b < 0:
        return [0], []
    return walk(np.asarray(tree_a_parents), a)[::-1], walk(np.asarray(tree_b_parents), b)


@lru_cache(maxsize=None)
def _take_wps_jit(n: int, w1: int, dim: int):
    def take(wps, i, nodes):
        row = jax.lax.dynamic_index_in_dim(wps, i, keepdims=False)
        return row.reshape(-1, w1, dim)[nodes]
    return jax.jit(take)


def _take_wps(n: int, w1: int, dim: int):
    """Interior waypoints [n, w1, dim] of the edges into `nodes` (padded to a power of two)."""
    m = 1 << max(n - 1, 0).bit_length()
    f = _take_wps_jit(m, w1, dim)
    return lambda wps, i, nodes: f(wps, i, np.pad(nodes, (0, m - n)))[:n]



def simplify_paths(paths, collision_context, constraints: Sequence = (), *, rounds: int = 1,
                   projection: ProjectionSettings = ProjectionSettings(), min_vals=None, max_vals=None,
                   step_size: float = 0.5, granularity: int = 16, collision_checker: str = "binary",
                   collision_margin: float = 0.02, max_path_points: Optional[int] = None):
    """MR. POP simplification (the ``shortcut_rounds`` pass) of caller-supplied paths.

    ``paths``: list of ``(n_i, dim)`` arrays (None passed through), all in one world.
    ``constraints``: one constraint list shared by every path. Each change is validated by the
    planner's own connect, exactly as on a planner path. Returns the simplified list.
    """
    live = [i for i, p in enumerate(paths) if p is not None and len(p) > 2]
    out = [None if p is None else np.asarray(p, np.float32) for p in paths]
    if not live or rounds <= 0:
        return out
    cons = list(constraints)
    ee = tuple(sorted({c.ee_joint for c in cons}))
    name = _planner_targets(collision_context, ee)[4]
    batch, dim = len(live), out[live[0]].shape[1]
    n_max = max(len(out[i]) for i in live)
    max_path = int(max_path_points or max(1024, 1 << (2 * n_max - 1).bit_length()))
    P = np.zeros((batch, max_path, dim), np.float32)
    for b, i in enumerate(live):
        P[b, :len(out[i])] = out[i]
    lens = np.array([len(out[i]) for i in live], np.int32)
    lo = np.asarray(-np.pi * np.ones(dim) if min_vals is None else min_vals, np.float32)
    hi = np.asarray(np.pi * np.ones(dim) if max_vals is None else max_vals, np.float32)
    worlds = _pack_worlds([collision_context] * batch)
    packed = np.stack([_pack_constraints(cons, ee, projection)] * batch)
    inputs = np.concatenate([a.astype(np.float32, copy=False).ravel()
                             for a in (lo, hi, *worlds, packed)])
    f = jax.ffi.ffi_call(name, (jax.ShapeDtypeStruct((batch, max_path, dim), jnp.float32),
                                jax.ShapeDtypeStruct((batch,), jnp.int32)))
    Pn, ln = jax.device_get(f(inputs, P, lens, rounds=np.int32(rounds), step_size=np.float32(step_size),
                              granularity=np.int32(granularity),
                              collision_mode=np.int32(_resolve_collision_mode(collision_checker)),
                              collision_margin=np.float32(collision_margin), batch=np.int32(batch),
                              dim=np.int32(dim), mw0=np.int32(worlds[0].shape[1]),
                              mw1=np.int32(worlds[1].shape[1]), mw2=np.int32(worlds[2].shape[1]),
                              mw3=np.int32(worlds[3].shape[1])))
    for b, i in enumerate(live):
        out[i] = np.asarray(Pn[b, :int(ln[b])])
    return out
