"""RoboGPU-validated pRRTC planner (host-driven, OptiX point-cloud collision).

The monolithic CUDA :func:`prrtc_plan` kernel checks collisions inline using
analytic robot/world geometry only — it has no point-cloud support, and OptiX
ray-tracing (``optixTrace``) cannot be invoked from inside an ordinary CUDA
kernel.  To use pyroffi's RT-core point-cloud checker
(:class:`pyroffi.collision.RoboGPUCollisionChecker`) we therefore drive the RRT
loop from the host and validate batches of candidate edges with the robogpu
OptiX kernel each round.

This trades the single-kernel-launch speed of :func:`prrtc_plan` for the
ability to collision-check against a dense environment point cloud on the GPU's
ray-tracing cores.  Every collision query (the loop's bottleneck) still runs on
the GPU and is batched across all candidate edges in an iteration, so the host
only orchestrates tree bookkeeping.

Algorithm: batched RRT-Connect.  Each iteration samples ``num_new_samples``
random configurations, extends the currently smaller tree toward them
(step-size limited), validates all extension edges in a single robogpu call,
inserts the collision-free ones, then runs a batched greedy "connect" of the
other tree toward each freshly added node.  A connection that reaches a node
within ``step_size`` (with a collision-free edge) solves the problem.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import jax.numpy as jnp
from jax import Array

from .prrtc import PRRTCResult


def _steer(q_from: np.ndarray, q_to: np.ndarray, step_size: float) -> np.ndarray:
    """Step from ``q_from`` toward ``q_to``, clamped to ``step_size`` (vectorised).

    ``q_from``/``q_to`` are ``[S, dim]``.  Returns ``[S, dim]`` new configs.
    """
    delta = q_to - q_from
    dist = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, step_size / np.maximum(dist, 1e-8))
    return q_from + delta * scale


def _edge_waypoints(q_a: np.ndarray, q_b: np.ndarray, granularity: int) -> np.ndarray:
    """Discretise edges ``q_a→q_b`` into ``granularity`` waypoints.

    Inputs ``[S, dim]``; output ``[S, G, dim]`` (endpoints included).
    """
    ts = np.linspace(0.0, 1.0, granularity, dtype=np.float32)  # [G]
    return (
        q_a[:, None, :] * (1.0 - ts)[None, :, None]
        + q_b[:, None, :] * ts[None, :, None]
    ).astype(np.float32)


class _Tree:
    """Minimal preallocated tree (Structure-of-Arrays, host-side)."""

    def __init__(self, root: np.ndarray, max_nodes: int, dim: int):
        self.configs = np.zeros((max_nodes, dim), dtype=np.float32)
        self.parents = np.full((max_nodes,), -1, dtype=np.int32)
        self.size = 1
        self.configs[0] = root
        self.parents[0] = 0  # root is its own parent (matches _trace_path convention)

    def add(self, config: np.ndarray, parent: int) -> int:
        idx = self.size
        if idx >= self.configs.shape[0]:
            return -1
        self.configs[idx] = config
        self.parents[idx] = parent
        self.size += 1
        return idx

    @property
    def view(self) -> np.ndarray:
        return self.configs[: self.size]

    def nearest(self, queries: np.ndarray) -> np.ndarray:
        """Index of the nearest tree node for each query ``[S, dim] → [S]``."""
        # [size, S] squared distances; argmin over the tree axis.
        diff = self.view[:, None, :] - queries[None, :, :]
        d2 = np.einsum("ksd,ksd->ks", diff, diff)
        return np.argmin(d2, axis=0)

    def path_to(self, node: int) -> list[np.ndarray]:
        """Walk parents from ``node`` back to the root; returns root→node order."""
        chain: list[np.ndarray] = []
        curr = int(node)
        for _ in range(self.size):
            chain.append(self.configs[curr])
            parent = int(self.parents[curr])
            if parent == curr:
                break
            curr = parent
        chain.reverse()
        return chain


def prrtc_plan_robogpu(
    start_config: Array,
    goal_configs: Array,
    robot,
    robogpu_checker,
    world_geom=None,
    point_cloud: Optional[Array] = None,
    r_env: float = 0.02,
    max_iterations: int = 5_000,
    step_size: float = 0.5,
    num_new_samples: int = 128,
    granularity: int = 16,
    max_nodes: int = 1_000_000,
    connect_max_steps: int = 32,
    min_vals: Optional[Array] = None,
    max_vals: Optional[Array] = None,
    seed: int = 0,
) -> PRRTCResult:
    """Plan a path using RRT-Connect with robogpu (OptiX) point-cloud collision.

    Unlike :func:`prrtc_plan`, this host-driven planner validates edges against
    an environment point cloud on the GPU's ray-tracing cores via
    :class:`pyroffi.collision.RoboGPUCollisionChecker`.

    Args:
        start_config: Start configuration, shape ``(dim,)``.
        goal_configs: Goal configurations, shape ``(num_goals, dim)`` or ``(dim,)``.
            The first goal seeds the goal tree's root.
        robot: pyroffi robot model (provides FK joint arrays to the checker).
        robogpu_checker: A :class:`RoboGPUCollisionChecker` built from the
            robot's spherized collision model.
        world_geom: Optional analytic world geometry (spheres/capsules/boxes/
            halfspaces) checked in robogpu's CUDA stage 1.
        point_cloud: ``[Mp, 3]`` float32 environment point cloud checked on the
            OptiX BVH (stage 2).  ``None`` runs analytic-only validation.
        r_env: Radius of each environment point sphere.
        max_iterations: Maximum outer RRT-Connect iterations.
        step_size: Maximum extension step in configuration space.
        num_new_samples: Random samples drawn (and edges validated) per iteration.
        granularity: Waypoints per edge passed to the checker.
        max_nodes: Per-tree capacity.
        connect_max_steps: Greedy-connect step cap per newly added node.
        min_vals / max_vals: Sampling bounds; default to the robot joint limits,
            falling back to ``[-pi, pi]``.
        seed: RNG seed for sampling.

    Returns:
        :class:`PRRTCResult`.  ``tree_a`` is the start tree, ``tree_b`` the goal
        tree (configs returned as ``(dim, size)`` to match :func:`prrtc_plan`).
    """
    # Allocate the point cloud once: the checker caches its JIT + OptiX BVH on
    # the array's identity/device-pointer, so reusing this exact object keeps the
    # BVH warm across every per-iteration check (a fresh array would rebuild it).
    pc_arr = (
        jnp.asarray(point_cloud, dtype=jnp.float32).reshape(-1, 3)
        if point_cloud is not None
        else None
    )

    start = np.asarray(start_config, dtype=np.float32).reshape(-1)
    goals = np.asarray(goal_configs, dtype=np.float32).reshape(-1, start.shape[0])
    goal = goals[0]
    dim = start.shape[0]

    if min_vals is None:
        lo = getattr(getattr(robot, "joints", None), "lower_limits", None)
        min_vals = np.asarray(lo, dtype=np.float32) if lo is not None else np.full(dim, -np.pi, np.float32)
    else:
        min_vals = np.asarray(min_vals, dtype=np.float32)
    if max_vals is None:
        hi = getattr(getattr(robot, "joints", None), "upper_limits", None)
        max_vals = np.asarray(hi, dtype=np.float32) if hi is not None else np.full(dim, np.pi, np.float32)
    else:
        max_vals = np.asarray(max_vals, dtype=np.float32)

    rng = np.random.default_rng(seed)

    tree_a = _Tree(start, max_nodes, dim)  # start tree
    tree_b = _Tree(goal, max_nodes, dim)   # goal tree

    def edges_free(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
        """Batched collision check: ``[S] bool`` — True if the edge is free."""
        if q_a.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        edges = jnp.asarray(_edge_waypoints(q_a, q_b, granularity))
        free = robogpu_checker.check_edges_collision_free(
            robot, edges, world_geom=world_geom, point_cloud=pc_arr, r_env=r_env
        )
        return np.asarray(free).reshape(-1) == 1

    solved = False
    connection = None  # (a_node_idx, b_node_idx)
    iterations = 0

    # `active`/`other` swap each iteration so both trees grow (RRT-Connect).
    active, other = tree_a, tree_b

    for iterations in range(1, max_iterations + 1):
        # ── 1. Sample and extend the active tree toward each sample ──────────
        samples = rng.uniform(min_vals, max_vals, size=(num_new_samples, dim)).astype(np.float32)
        near_idx = active.nearest(samples)
        q_near = active.view[near_idx]
        q_new = _steer(q_near, samples, step_size)

        free = edges_free(q_near, q_new)
        new_nodes: list[int] = []
        new_configs: list[np.ndarray] = []
        for s in np.nonzero(free)[0]:
            idx = active.add(q_new[s], int(near_idx[s]))
            if idx < 0:
                break  # tree full
            new_nodes.append(idx)
            new_configs.append(q_new[s])

        if not new_nodes:
            active, other = other, active
            continue

        # ── 2. Greedy-connect the other tree toward each new active node ─────
        targets = np.stack(new_configs, axis=0)           # [N, dim]
        active_node_ids = np.asarray(new_nodes, dtype=np.int32)
        # Per target, the current frontier node in the *other* tree.
        other_idx = other.nearest(targets)                # [N]
        live = np.ones(targets.shape[0], dtype=bool)

        for _ in range(connect_max_steps):
            if not live.any():
                break
            li = np.nonzero(live)[0]
            q_from = other.view[other_idx[li]]
            q_tgt = targets[li]
            q_step = _steer(q_from, q_tgt, step_size)
            free_c = edges_free(q_from, q_step)

            for k, s in enumerate(li):
                if not free_c[k]:
                    live[s] = False
                    continue
                idx = other.add(q_step[k], int(other_idx[s]))
                if idx < 0:
                    live[s] = False
                    continue
                other_idx[s] = idx
                # Reached the target node ⇒ trees connected.
                if np.linalg.norm(q_step[k] - targets[s]) <= step_size * 1e-3 + 1e-6:
                    live[s] = False
                    a_node = int(active_node_ids[s])
                    b_node = idx
                    # Map back to (start_tree_node, goal_tree_node).
                    if active is tree_a:
                        connection = (a_node, b_node)
                    else:
                        connection = (b_node, a_node)
                    solved = True
                    break
            if solved:
                break

        if solved:
            break

        active, other = other, active

    # ── Build result ─────────────────────────────────────────────────────────
    size_a = tree_a.size
    size_b = tree_b.size
    ta_cfg = jnp.asarray(tree_a.configs[:size_a].T)  # (dim, size_a)
    tb_cfg = jnp.asarray(tree_b.configs[:size_b].T)
    ta_par = jnp.asarray(tree_a.parents[:size_a])
    tb_par = jnp.asarray(tree_b.parents[:size_b])

    if solved and connection is not None:
        a_node, b_node = connection
        path_a = tree_a.path_to(a_node)                 # start → connection
        path_b = tree_b.path_to(b_node)                 # goal  → connection
        path_b.reverse()                                # connection → goal
        full = np.stack(path_a + path_b, axis=0)
        path = jnp.asarray(full)
        cost = float(np.sum(np.linalg.norm(np.diff(full, axis=0), axis=1)))
    else:
        path = None
        cost = float("inf")

    return PRRTCResult(
        solved=solved,
        path=path,
        tree_a_size=size_a,
        tree_b_size=size_b,
        iterations=iterations,
        cost=cost,
        kernel_time_ms=None,
        tree_a_configs=ta_cfg,
        tree_b_configs=tb_cfg,
        tree_a_parents=ta_par,
        tree_b_parents=tb_par,
    )
