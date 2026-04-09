"""
pRRTC - Parallel Rapidly-exploring Random Tree JAX Interface.

This module provides a JAX FFI wrapper for the pRRTC motion planner, which finds
collision-free paths in configuration space using a parallel CUDA implementation.

Key features:
  - Two-tree bidirectional planning (start and goal trees)
  - Parallel tree expansion using CUDA
  - Halton sequence sampling for better configuration space coverage
  - Dynamic tree balancing for efficient search
  - CUDA Graph support for minimal kernel launch overhead
  - Batched planning via JAX vmap
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, Int


class PRRTCResult(NamedTuple):
    """Result of pRRTC planning."""

    solved: bool
    path: Optional[Array]
    tree_a_size: int
    tree_b_size: int
    iterations: int
    cost: float
    # Raw tree data for visualization (shape: (dim, max_nodes), sliced to tree_*_size)
    tree_a_configs: Optional[Array] = None
    tree_b_configs: Optional[Array] = None
    tree_a_parents: Optional[Array] = None
    tree_b_parents: Optional[Array] = None


_LIB_NAME = "_prrtc_planner_lib.so"


@lru_cache(maxsize=1)
def _load_and_register() -> None:
    """Load the pRRTC shared library and register FFI targets (runs once)."""
    lib_path = Path(__file__).parent.parent / _LIB_NAME
    
    if not lib_path.exists():
        raise RuntimeError(
            f"pRRTC library not found at {lib_path}.\n"
            f"Files in directory: {list(lib_path.parent.iterdir())}\n"
            "Compile it first with:\n"
            "  cd cuda-rrtc && bash build.sh\n"
            "(This produces _prrtc_planner_lib.so in the cuda-rrtc directory.)"
        )
    
    lib = ctypes.CDLL(str(lib_path))

    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    def _register(ffi_name: str, symbol_name: str) -> None:
        capsule = _PyCapsule_New(
            ctypes.cast(getattr(lib, symbol_name), ctypes.c_void_p),
            b"xla._CUSTOM_CALL_TARGET",
            None,
        )
        jax.ffi.register_ffi_target(ffi_name, capsule, platform="CUDA")

    _register("prrtc_planner", "PrrtcPlannerFfi")
    _register("prrtc_nearest_neighbor", "PrrtcNearestNeighborFfi")
    _register("prrtc_extend", "PrrtcExtendFfi")
    _register("prrtc_iteration", "PrrtcIterationFfi")


def prrtc_plan(
    start_config: Float[Array, "*batch dim"],
    goal_configs: Float[Array, "num_goals dim"],
    max_iterations: int = 10000,
    step_size: float = 0.5,
    num_new_samples: int = 64,
    balance_mode: int = 1,
    tree_ratio: float = 0.5,
    dynamic_domain: bool = True,
    dd_alpha: float = 1e-4,
    dd_radius: float = 4.0,
    dd_min_radius: float = 1.0,
    min_vals: Optional[Float[Array, "dim"]] = None,
    max_vals: Optional[Float[Array, "dim"]] = None,
    collision_context: Optional[dict[str, Array]] = None,
    allow_unsafe_no_collision: bool = False,
) -> PRRTCResult:
    """
    Plan a path using the parallel RRTC (pRRTC) algorithm.

    Args:
        start_config: Start configuration, shape ``(*batch, dim)``.
        goal_configs: Goal configurations, shape ``(num_goals, dim)``.
        max_iterations: Maximum number of planning iterations.
        step_size: Maximum extension step size.
        num_new_samples: Number of new samples to generate per iteration.
        balance_mode: Tree balancing mode (0/1/2) matching pRRTC settings.
        tree_ratio: Balance threshold ratio used by mode 1/2.
        dynamic_domain: Enable dynamic-domain radius adaptation.
        dd_alpha: Dynamic-domain radius adaptation rate.
        dd_radius: Initial radius used after first rejection.
        dd_min_radius: Minimum allowed dynamic-domain radius.
        min_vals: Minimum configuration values (optional, defaults to -pi).
        max_vals: Maximum configuration values (optional, defaults to pi).
        collision_context: Pyronot collision tensors. Collision-aware planning
            is enabled by default and requires this context.
        allow_unsafe_no_collision: If True, disables default collision-context
            requirement and runs geometric-only planning.

    Returns:
        PRRTCResult containing:
            - solved: Whether a path was found
            - path: Sequence of configurations if solved, else None
            - tree_a_size: Number of nodes in start tree
            - tree_b_size: Number of nodes in goal tree
            - iterations: Number of iterations performed
            - cost: Total path cost (sum of segment lengths)

    Notes:
        - Uses two-tree bidirectional planning: start tree and goal tree
        - Samples configuration space using Halton low-discrepancy sequence
        - Extends tree toward samples with step-size limiting
                - Root validity is caller-managed (source pRRTC style): start/goal
                    roots are inserted directly and collision checks apply during
                    expansion/connect edge validation.
        - Supports batched planning via JAX vmap
    """
    _load_and_register()

    # Extract robot properties
    start_config = jnp.atleast_1d(start_config)
    if start_config.ndim == 1:
        start_config = start_config.reshape(1, -1)

    batch_size_in = start_config.shape[0]
    dim = start_config.shape[-1]

    if batch_size_in != 1:
        raise ValueError(
            "prrtc_plan currently accepts a single start configuration per call. "
            "For batches, call this function in a Python loop or via jax.vmap "
            "with one start per mapped call."
        )

    start_vec = start_config.reshape(dim)
    goal_flat = goal_configs.reshape(-1, dim)
    num_goals = goal_flat.shape[0]

    # Get joint limits from robot or use defaults
    if min_vals is None:
        min_vals = jnp.ones(dim) * -jnp.pi
    if max_vals is None:
        max_vals = jnp.ones(dim) * jnp.pi

    # Ensure proper shapes
    min_vals = jnp.asarray(min_vals, dtype=jnp.float32)
    max_vals = jnp.asarray(max_vals, dtype=jnp.float32)

    # Compute max nodes based on iterations and batch size
    max_nodes = max_iterations + num_goals + 1000

    # Determine output shapes for FFI call
    result_shapes = (
        jax.ShapeDtypeStruct((dim, max_nodes), jnp.float32),  # tree_a_configs
        jax.ShapeDtypeStruct((dim, max_nodes), jnp.float32),  # tree_b_configs
        jax.ShapeDtypeStruct((max_nodes,), jnp.int32),        # tree_a_parents
        jax.ShapeDtypeStruct((max_nodes,), jnp.int32),        # tree_b_parents
        jax.ShapeDtypeStruct((2,), jnp.int32),                # tree_sizes
        jax.ShapeDtypeStruct((2,), jnp.int32),                # completed
        jax.ShapeDtypeStruct((1,), jnp.int32),                # iter_count
        jax.ShapeDtypeStruct((3,), jnp.int32),                # connection_info
        jax.ShapeDtypeStruct((1,), jnp.int32),                # solved_flag
    )

    # Optional collision/FK context; zero-sized buffers disable collision path.
    fk_twists = jnp.zeros((0, 6), dtype=jnp.float32)
    fk_parent_tf = jnp.zeros((0, 7), dtype=jnp.float32)
    fk_parent_idx = jnp.zeros((0,), dtype=jnp.int32)
    fk_act_idx = jnp.zeros((0,), dtype=jnp.int32)
    fk_mimic_mul = jnp.zeros((0,), dtype=jnp.float32)
    fk_mimic_off = jnp.zeros((0,), dtype=jnp.float32)
    fk_mimic_act_idx = jnp.zeros((0,), dtype=jnp.int32)
    fk_topo_inv = jnp.zeros((0,), dtype=jnp.int32)
    sphere_link_idx = jnp.zeros((0,), dtype=jnp.int32)
    sphere_local = jnp.zeros((0, 3), dtype=jnp.float32)
    sphere_radius = jnp.zeros((0,), dtype=jnp.float32)
    world_spheres = jnp.zeros((0, 4), dtype=jnp.float32)
    self_pairs = jnp.zeros((0, 2), dtype=jnp.int32)

    required_keys = (
        "fk_twists",
        "fk_parent_tf",
        "fk_parent_idx",
        "fk_act_idx",
        "fk_mimic_mul",
        "fk_mimic_off",
        "fk_mimic_act_idx",
        "fk_topo_inv",
        "sphere_link_idx",
        "sphere_local",
        "sphere_radius",
        "world_spheres",
    )

    if collision_context is None and not allow_unsafe_no_collision:
        raise ValueError(
            "Collision-aware planning is the default. Provide collision_context "
            "with required keys or set allow_unsafe_no_collision=True to opt out. "
            f"Required keys: {required_keys}"
        )

    if collision_context is not None:
        missing = [k for k in required_keys if k not in collision_context]
        if missing:
            raise ValueError(f"collision_context missing required keys: {missing}")
        fk_twists = jnp.asarray(collision_context["fk_twists"], dtype=jnp.float32)
        fk_parent_tf = jnp.asarray(collision_context["fk_parent_tf"], dtype=jnp.float32)
        fk_parent_idx = jnp.asarray(collision_context["fk_parent_idx"], dtype=jnp.int32)
        fk_act_idx = jnp.asarray(collision_context["fk_act_idx"], dtype=jnp.int32)
        fk_mimic_mul = jnp.asarray(collision_context["fk_mimic_mul"], dtype=jnp.float32)
        fk_mimic_off = jnp.asarray(collision_context["fk_mimic_off"], dtype=jnp.float32)
        fk_mimic_act_idx = jnp.asarray(collision_context["fk_mimic_act_idx"], dtype=jnp.int32)
        fk_topo_inv = jnp.asarray(collision_context["fk_topo_inv"], dtype=jnp.int32)
        sphere_link_idx = jnp.asarray(collision_context["sphere_link_idx"], dtype=jnp.int32)
        sphere_local = jnp.asarray(collision_context["sphere_local"], dtype=jnp.float32)
        sphere_radius = jnp.asarray(collision_context["sphere_radius"], dtype=jnp.float32)
        world_spheres = jnp.asarray(collision_context["world_spheres"], dtype=jnp.float32)
        if "self_pairs" in collision_context:
            self_pairs = jnp.asarray(collision_context["self_pairs"], dtype=jnp.int32)

    # Call FFI kernel via JAX FFI.
    result = jax.ffi.ffi_call(
        "prrtc_planner",
        result_shapes,
    )(
        start_vec,
        goal_flat,
        min_vals,
        max_vals,
        fk_twists,
        fk_parent_tf,
        fk_parent_idx,
        fk_act_idx,
        fk_mimic_mul,
        fk_mimic_off,
        fk_mimic_act_idx,
        fk_topo_inv,
        sphere_link_idx,
        sphere_local,
        sphere_radius,
        world_spheres,
        self_pairs,
        max_iterations=np.int32(max_iterations),
        step_size=np.float32(step_size),
        num_new_samples=np.int32(num_new_samples),
        balance_mode=np.int32(balance_mode),
        tree_ratio=np.float32(tree_ratio),
        dynamic_domain=np.int32(1 if dynamic_domain else 0),
        dd_alpha=np.float32(dd_alpha),
        dd_radius=np.float32(dd_radius),
        dd_min_radius=np.float32(dd_min_radius),
        dim=np.int32(dim),
        max_nodes=np.int32(max_nodes),
    )

    # Extract result
    tree_a_final = result[0]
    tree_b_final = result[1]
    solved = result[8][0] == 1

    if solved:
        path = _trace_path(
            tree_a_final,
            tree_b_final,
            result[2],
            result[3],
            result[7],
        )
        cost = jnp.sum(jnp.linalg.norm(jnp.diff(path, axis=0), axis=1))
    else:
        path = None
        cost = jnp.inf

    size_a = int(result[4][0])
    size_b = int(result[4][1])
    return PRRTCResult(
        solved=solved,
        path=path,
        tree_a_size=size_a,
        tree_b_size=size_b,
        iterations=int(result[6][0]),
        cost=float(cost),
        tree_a_configs=tree_a_final[:, :size_a],
        tree_b_configs=tree_b_final[:, :size_b],
        tree_a_parents=result[2][:size_a],
        tree_b_parents=result[3][:size_b],
    )


def _trace_path(
    tree_a_configs: Float[Array, "dim max_nodes"],
    tree_b_configs: Float[Array, "dim max_nodes"],
    tree_a_parents: Int[Array, "max_nodes"],
    tree_b_parents: Int[Array, "max_nodes"],
    connection_info: Int[Array, "3"],
) -> Float[Array, "path_len dim"]:
    """
    Trace path from start root (tree A root index 0) to a goal root in tree B.

    connection_info stores:
      - index 0: connection node in tree A
      - index 1: connection node in tree B
      - index 2: id of tree expanded during connect (diagnostic)
    """
    a_connect = int(connection_info[0])
    b_connect = int(connection_info[1])

    if a_connect < 0 or b_connect < 0:
        # Should not happen when solved, but return start node as safe fallback.
        return tree_a_configs[:, :1].T

    max_nodes = int(tree_a_configs.shape[1])

    # Walk from connection in tree A back to start root (index 0 expected).
    a_nodes = []
    curr = a_connect
    for _ in range(max_nodes):
        a_nodes.append(tree_a_configs[:, curr])
        parent = int(tree_a_parents[curr])
        if parent == curr:
            break
        curr = parent
    a_nodes.reverse()  # start -> ... -> a_connect

    # Walk from connection in tree B toward its root goal node.
    b_nodes = []
    curr = b_connect
    for _ in range(max_nodes):
        b_nodes.append(tree_b_configs[:, curr])
        parent = int(tree_b_parents[curr])
        if parent == curr:
            break
        curr = parent

    if not a_nodes:
        return jnp.stack(b_nodes, axis=0)
    if not b_nodes:
        return jnp.stack(a_nodes, axis=0)

    path_a = jnp.stack(a_nodes, axis=0)
    path_b = jnp.stack(b_nodes, axis=0)
    return jnp.concatenate((path_a, path_b), axis=0)


def prrtc_nearest_neighbor(
    tree_configs: Float[Array, "dim max_nodes"],
    query_configs: Float[Array, "batch dim"],
    tree_size: int,
) -> tuple[Float[Array, "batch"], Int[Array, "batch"]]:
    """
    Find nearest node in tree for each query configuration.

    Args:
        tree_configs: Tree node configurations, shape ``(dim, max_nodes)`` (SoA layout).
        query_configs: Query configurations, shape ``(batch, dim)``.
        tree_size: Current number of nodes in tree.

    Returns:
        Tuple of (distances, indices) with shape ``(batch,)`` each.
    """
    _load_and_register()

    tree_size_arr = jnp.array([tree_size], dtype=jnp.int32)

    indices, dists = jax.ffi.ffi_call(
        "prrtc_nearest_neighbor",
        (
            jax.ShapeDtypeStruct((query_configs.shape[0],), jnp.int32),
            jax.ShapeDtypeStruct((query_configs.shape[0],), jnp.float32),
        ),
    )(
        tree_configs,
        query_configs,
        tree_size_arr,
    )

    return dists, indices


def prrtc_extend(
    tree_configs: Float[Array, "dim max_nodes"],
    nearest_indices: Int[Array, "batch"],
    samples: Float[Array, "batch dim"],
    step_size: float,
) -> tuple[Float[Array, "batch dim"], Int[Array, "batch"], Int[Array, "batch"]]:
    """
    Extend tree from nearest nodes toward samples.

    Args:
        tree_configs: Tree configurations, shape ``(dim, max_nodes)``.
        nearest_indices: Indices of nearest nodes, shape ``(batch,)``.
        samples: Target sample configurations, shape ``(batch, dim)``.
        step_size: Maximum extension step.

    Returns:
        Tuple of (new_configs, parent_indices, valid_flags).
    """
    _load_and_register()

    step_size_arr = jnp.array([step_size], dtype=jnp.float32)

    new_configs, parent_indices, valid_flags = jax.ffi.ffi_call(
        "prrtc_extend",
        (
            jax.ShapeDtypeStruct(samples.shape, jnp.float32),
            jax.ShapeDtypeStruct(samples.shape[:1], jnp.int32),
            jax.ShapeDtypeStruct(samples.shape[:1], jnp.int32),
        ),
    )(
        tree_configs,
        nearest_indices,
        samples,
        step_size_arr,
    )

    return new_configs, parent_indices, valid_flags
