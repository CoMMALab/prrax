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
import hashlib
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
    kernel_time_ms: Optional[float] = None
    # Raw tree data for visualization (shape: (dim, max_nodes), sliced to tree_*_size)
    tree_a_configs: Optional[Array] = None
    tree_b_configs: Optional[Array] = None
    tree_a_parents: Optional[Array] = None
    tree_b_parents: Optional[Array] = None


_LIB_NAME = "_prrtc_planner_lib.so"

# Collision-checker selection passed to the CUDA kernel as ``collision_mode``.
#   "binary"        — pRRTC-style early-exit boolean check (fewest ops/config).
#   "differentiable"— full smooth signed-distance-field sweep (no early exit),
#                     mirroring pyroffi's differentiable collision kernel.
_COLLISION_MODES = {"binary": 0, "differentiable": 1, "binary_coarse": 2, "binary_coarse_coop": 3}


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


def _coarse_buffers(collision_context):
    """Extract the optional coarse→fine culling tensors (zero-sized if absent).

    Returns ``(coarse_sphere_link_idx, coarse_sphere_local, coarse_sphere_radius,
    fine_link_offsets, coarse_self_pairs, self_pair_ranges)``. These power the
    ``"binary_coarse"`` checker (collision_mode 2); when absent the CUDA planner
    falls back to the flat binary sweep.
    """
    coarse_sphere_link_idx = jnp.zeros((0,), dtype=jnp.int32)
    coarse_sphere_local = jnp.zeros((0, 3), dtype=jnp.float32)
    coarse_sphere_radius = jnp.zeros((0,), dtype=jnp.float32)
    fine_link_offsets = jnp.zeros((0,), dtype=jnp.int32)
    coarse_self_pairs = jnp.zeros((0, 2), dtype=jnp.int32)
    self_pair_ranges = jnp.zeros((0, 2), dtype=jnp.int32)
    if collision_context is not None:
        if "coarse_sphere_link_idx" in collision_context:
            coarse_sphere_link_idx = jnp.asarray(collision_context["coarse_sphere_link_idx"], dtype=jnp.int32)
        if "coarse_sphere_local" in collision_context:
            coarse_sphere_local = jnp.asarray(collision_context["coarse_sphere_local"], dtype=jnp.float32)
        if "coarse_sphere_radius" in collision_context:
            coarse_sphere_radius = jnp.asarray(collision_context["coarse_sphere_radius"], dtype=jnp.float32)
        if "fine_link_offsets" in collision_context:
            fine_link_offsets = jnp.asarray(collision_context["fine_link_offsets"], dtype=jnp.int32)
        if "coarse_self_pairs" in collision_context:
            coarse_self_pairs = jnp.asarray(collision_context["coarse_self_pairs"], dtype=jnp.int32)
        if "self_pair_ranges" in collision_context:
            self_pair_ranges = jnp.asarray(collision_context["self_pair_ranges"], dtype=jnp.int32)
    return (
        coarse_sphere_link_idx,
        coarse_sphere_local,
        coarse_sphere_radius,
        fine_link_offsets,
        coarse_self_pairs,
        self_pair_ranges,
    )


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
    _register("prrtc_planner_batch", "PrrtcPlannerBatchFfi")
    _register("prrtc_planner_batch_ctx", "PrrtcPlannerBatchCtxFfi")
    _register("prrtc_nearest_neighbor", "PrrtcNearestNeighborFfi")
    _register("prrtc_extend", "PrrtcExtendFfi")
    _register("prrtc_iteration", "PrrtcIterationFfi")


@lru_cache(maxsize=256)
def _get_prrtc_single_jit_kernel(
    *,
    max_iterations: int,
    step_size: float,
    num_new_samples: int,
    granularity: int,
    max_nodes: int,
    balance_mode: int,
    tree_ratio: float,
    dynamic_domain: bool,
    dd_alpha: float,
    dd_radius: float,
    dd_min_radius: float,
    collision_mode: int,
    collision_margin: float,
):
    """Build and cache a JIT-traced single-problem FFI kernel."""

    def _call(
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
        world_capsules,
        world_boxes,
        world_halfspaces,
        self_pairs,
        coarse_sphere_link_idx,
        coarse_sphere_local,
        coarse_sphere_radius,
        fine_link_offsets,
        coarse_self_pairs,
        self_pair_ranges,
    ):
        dim = int(start_vec.shape[0])
        result_shapes = (
            jax.ShapeDtypeStruct((dim, max_nodes), jnp.float32),
            jax.ShapeDtypeStruct((dim, max_nodes), jnp.float32),
            jax.ShapeDtypeStruct((max_nodes,), jnp.int32),
            jax.ShapeDtypeStruct((max_nodes,), jnp.int32),
            jax.ShapeDtypeStruct((2,), jnp.int32),
            jax.ShapeDtypeStruct((2,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.float32),
        )
        return jax.ffi.ffi_call("prrtc_planner", result_shapes)(
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
            world_capsules,
            world_boxes,
            world_halfspaces,
            self_pairs,
            coarse_sphere_link_idx,
            coarse_sphere_local,
            coarse_sphere_radius,
            fine_link_offsets,
            coarse_self_pairs,
            self_pair_ranges,
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
            granularity=np.int32(granularity),
            collision_mode=np.int32(collision_mode),
            collision_margin=np.float32(collision_margin),
        )

    return jax.jit(_call)


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
    collision_context: Optional[dict[str, Array]] = None,
    allow_unsafe_no_collision: bool = False,
    jit_trace: bool = False,
    collision_checker: str = "binary",
    collision_margin: float = 0.02,
) -> PRRTCResult:
    """
    Plan a path using the parallel RRTC (pRRTC) algorithm.

    Args:
        start_config: Start configuration, shape ``(*batch, dim)``.
        goal_configs: Goal configurations, shape ``(num_goals, dim)``.
        max_iterations: Maximum number of planning iterations.
        step_size: Maximum extension step size.
        num_new_samples: Number of new samples to generate per iteration.
        granularity: Threads per block (= waypoints checked per edge). Must be a
            power of 2, max 64. Matches pRRTC's ``granularity`` parameter.
        max_nodes: Maximum nodes per tree. Defaults to 1,000,000 matching pRRTC.
        balance_mode: Tree balancing mode (0/1/2) matching pRRTC settings.
        tree_ratio: Balance threshold ratio used by mode 1/2.
        dynamic_domain: Enable dynamic-domain radius adaptation.
        dd_alpha: Dynamic-domain radius adaptation rate.
        dd_radius: Initial radius used after first rejection.
        dd_min_radius: Minimum allowed dynamic-domain radius.
        min_vals: Minimum configuration values (optional, defaults to -pi).
        max_vals: Maximum configuration values (optional, defaults to pi).
        collision_context: PyRoFFI collision tensors. Collision-aware planning
            is enabled by default and requires this context. World geometry keys
            are all optional (default empty): ``world_spheres`` [Ms,4],
            ``world_capsules`` [Mc,7], ``world_boxes`` [Mb,15],
            ``world_halfspaces`` [Mh,6]. At least one must be non-empty for
            collision checking to be active.
        allow_unsafe_no_collision: If True, disables default collision-context
            requirement and runs geometric-only planning.
        collision_checker: Which CUDA collision checker to use during edge
            validation. ``"binary"`` (default) runs the pRRTC-style early-exit
            boolean check; ``"differentiable"`` runs the full smooth
            signed-distance-field sweep (no early exit) used by pyroffi's
            differentiable collision kernel.
        collision_margin: Safety margin (metres) for the ``"differentiable"``
            checker. A configuration counts as in-collision when any robot
            geometry lies within this margin of an obstacle. Ignored by the
            ``"binary"`` checker.
        jit_trace: If True, run the planner FFI dispatch through a cached
            ``jax.jit`` wrapper keyed by planner hyper-parameters and input
            shape. This mostly reduces Python dispatch overhead for repeated
            calls.

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
    # Validate the checker name first so an unsupported choice (e.g. "robogpu")
    # reports a clear error before we attempt to dlopen the planner library.
    collision_mode = _resolve_collision_mode(collision_checker)

    _load_and_register()

    collision_margin = float(collision_margin)

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

    # Get joint limits from robot or use defaults
    if min_vals is None:
        min_vals = jnp.ones(dim) * -jnp.pi
    if max_vals is None:
        max_vals = jnp.ones(dim) * jnp.pi

    # Ensure proper shapes
    min_vals = jnp.asarray(min_vals, dtype=jnp.float32)
    max_vals = jnp.asarray(max_vals, dtype=jnp.float32)

    step_size = float(step_size)
    if step_size <= 0.0:
        raise ValueError("step_size must be > 0")

    granularity = int(granularity)
    if granularity < 1 or (granularity & (granularity - 1)) != 0:
        raise ValueError(f"granularity must be a power of 2, got {granularity}")
    if granularity > 64:
        raise ValueError(f"granularity must be <= 64 (PRRTC_BLOCK_THREADS_MAX), got {granularity}")

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
        jax.ShapeDtypeStruct((1,), jnp.float32),              # kernel_time_ms
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
    world_capsules = jnp.zeros((0, 7), dtype=jnp.float32)
    world_boxes = jnp.zeros((0, 15), dtype=jnp.float32)
    world_halfspaces = jnp.zeros((0, 6), dtype=jnp.float32)
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
        world_spheres = jnp.asarray(collision_context["world_spheres"], dtype=jnp.float32) if "world_spheres" in collision_context else world_spheres
        world_capsules = jnp.asarray(collision_context["world_capsules"], dtype=jnp.float32) if "world_capsules" in collision_context else world_capsules
        world_boxes = jnp.asarray(collision_context["world_boxes"], dtype=jnp.float32) if "world_boxes" in collision_context else world_boxes
        world_halfspaces = jnp.asarray(collision_context["world_halfspaces"], dtype=jnp.float32) if "world_halfspaces" in collision_context else world_halfspaces
        if "self_pairs" in collision_context:
            self_pairs = jnp.asarray(collision_context["self_pairs"], dtype=jnp.int32)

    (coarse_sphere_link_idx, coarse_sphere_local, coarse_sphere_radius,
     fine_link_offsets, coarse_self_pairs, self_pair_ranges) = _coarse_buffers(collision_context)

    # Call FFI kernel via JAX FFI.
    if jit_trace:
        traced_call = _get_prrtc_single_jit_kernel(
            max_iterations=int(max_iterations),
            step_size=float(step_size),
            num_new_samples=int(num_new_samples),
            granularity=int(granularity),
            max_nodes=int(max_nodes),
            balance_mode=int(balance_mode),
            tree_ratio=float(tree_ratio),
            dynamic_domain=bool(dynamic_domain),
            dd_alpha=float(dd_alpha),
            dd_radius=float(dd_radius),
            dd_min_radius=float(dd_min_radius),
            collision_mode=int(collision_mode),
            collision_margin=float(collision_margin),
        )
        result = traced_call(
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
            world_capsules,
            world_boxes,
            world_halfspaces,
            self_pairs,
            coarse_sphere_link_idx,
            coarse_sphere_local,
            coarse_sphere_radius,
            fine_link_offsets,
            coarse_self_pairs,
            self_pair_ranges,
        )
    else:
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
            world_capsules,
            world_boxes,
            world_halfspaces,
            self_pairs,
            coarse_sphere_link_idx,
            coarse_sphere_local,
            coarse_sphere_radius,
            fine_link_offsets,
            coarse_self_pairs,
            self_pair_ranges,
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
            granularity=np.int32(granularity),
            collision_mode=np.int32(collision_mode),
            collision_margin=np.float32(collision_margin),
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
        kernel_time_ms=float(result[9][0]),
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


def _collision_context_signature(collision_context: Optional[dict[str, Array]]) -> str:
    """Create a stable signature for potential context-cache keys."""
    if collision_context is None:
        return "__none__"

    hasher = hashlib.sha1()
    for key in sorted(collision_context.keys()):
        arr = np.asarray(collision_context[key])
        hasher.update(key.encode("utf-8"))
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(str(arr.dtype).encode("utf-8"))
        hasher.update(np.ascontiguousarray(arr).view(np.uint8).tobytes())
    return hasher.hexdigest()


@lru_cache(maxsize=256)
def _get_prrtc_batch_jit_kernel(
    *,
    max_iterations: int,
    step_size: float,
    num_new_samples: int,
    granularity: int,
    max_nodes: int,
    balance_mode: int,
    tree_ratio: float,
    dynamic_domain: bool,
    dd_alpha: float,
    dd_radius: float,
    dd_min_radius: float,
    collision_mode: int,
    collision_margin: float,
):
    """Build and cache a JIT-traced shared-context batch kernel."""

    def _call(
        start_configs,
        goal_batched,
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
        world_capsules,
        world_boxes,
        world_halfspaces,
        self_pairs,
        coarse_sphere_link_idx,
        coarse_sphere_local,
        coarse_sphere_radius,
        fine_link_offsets,
        coarse_self_pairs,
        self_pair_ranges,
    ):
        batch_size = int(start_configs.shape[0])
        dim = int(start_configs.shape[1])
        result_shapes = (
            jax.ShapeDtypeStruct((batch_size, dim, max_nodes), jnp.float32),
            jax.ShapeDtypeStruct((batch_size, dim, max_nodes), jnp.float32),
            jax.ShapeDtypeStruct((batch_size, max_nodes), jnp.int32),
            jax.ShapeDtypeStruct((batch_size, max_nodes), jnp.int32),
            jax.ShapeDtypeStruct((batch_size, 2), jnp.int32),
            jax.ShapeDtypeStruct((batch_size, 2), jnp.int32),
            jax.ShapeDtypeStruct((batch_size, 1), jnp.int32),
            jax.ShapeDtypeStruct((batch_size, 3), jnp.int32),
            jax.ShapeDtypeStruct((batch_size, 1), jnp.int32),
            jax.ShapeDtypeStruct((batch_size,), jnp.float32),
        )
        return jax.ffi.ffi_call("prrtc_planner_batch", result_shapes)(
            start_configs,
            goal_batched,
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
            world_capsules,
            world_boxes,
            world_halfspaces,
            self_pairs,
            coarse_sphere_link_idx,
            coarse_sphere_local,
            coarse_sphere_radius,
            fine_link_offsets,
            coarse_self_pairs,
            self_pair_ranges,
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
            granularity=np.int32(granularity),
            collision_mode=np.int32(collision_mode),
            collision_margin=np.float32(collision_margin),
        )

    return jax.jit(_call)


@lru_cache(maxsize=256)
def _get_prrtc_batch_ctx_jit_kernel(
    *,
    max_iterations: int,
    step_size: float,
    num_new_samples: int,
    granularity: int,
    max_nodes: int,
    balance_mode: int,
    tree_ratio: float,
    dynamic_domain: bool,
    dd_alpha: float,
    dd_radius: float,
    dd_min_radius: float,
    collision_mode: int,
    collision_margin: float,
):
    """Build and cache a JIT-traced per-problem-context batch kernel."""

    def _call(
        start_configs,
        goal_batched,
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
        world_capsules,
        world_boxes,
        world_halfspaces,
        self_pairs,
        world_spheres_count,
        world_capsules_count,
        world_boxes_count,
        world_halfspaces_count,
        self_pairs_count,
        coarse_sphere_link_idx,
        coarse_sphere_local,
        coarse_sphere_radius,
        fine_link_offsets,
        coarse_self_pairs,
        self_pair_ranges,
    ):
        batch_size = int(start_configs.shape[0])
        dim = int(start_configs.shape[1])
        result_shapes = (
            jax.ShapeDtypeStruct((batch_size, dim, max_nodes), jnp.float32),
            jax.ShapeDtypeStruct((batch_size, dim, max_nodes), jnp.float32),
            jax.ShapeDtypeStruct((batch_size, max_nodes), jnp.int32),
            jax.ShapeDtypeStruct((batch_size, max_nodes), jnp.int32),
            jax.ShapeDtypeStruct((batch_size, 2), jnp.int32),
            jax.ShapeDtypeStruct((batch_size, 2), jnp.int32),
            jax.ShapeDtypeStruct((batch_size, 1), jnp.int32),
            jax.ShapeDtypeStruct((batch_size, 3), jnp.int32),
            jax.ShapeDtypeStruct((batch_size, 1), jnp.int32),
            jax.ShapeDtypeStruct((batch_size,), jnp.float32),
        )
        return jax.ffi.ffi_call("prrtc_planner_batch_ctx", result_shapes)(
            start_configs,
            goal_batched,
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
            world_capsules,
            world_boxes,
            world_halfspaces,
            self_pairs,
            world_spheres_count,
            world_capsules_count,
            world_boxes_count,
            world_halfspaces_count,
            self_pairs_count,
            coarse_sphere_link_idx,
            coarse_sphere_local,
            coarse_sphere_radius,
            fine_link_offsets,
            coarse_self_pairs,
            self_pair_ranges,
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
            granularity=np.int32(granularity),
            collision_mode=np.int32(collision_mode),
            collision_margin=np.float32(collision_margin),
        )

    return jax.jit(_call)


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
    collision_context: Optional[dict[str, Array] | list[Optional[dict[str, Array]]] | tuple[Optional[dict[str, Array]], ...]] = None,
    allow_unsafe_no_collision: bool = False,
    jit_trace: bool = False,
    collision_checker: str = "binary",
    collision_margin: float = 0.02,
) -> list[PRRTCResult]:
    """
    Plan paths for a batch of independent start/goal pairs in parallel on the GPU.

    Each planning problem is launched on its own CUDA stream so all batch
    elements run concurrently.  Every problem has its own start and its own
    set of goals. Collision context can be shared across the whole batch, or
    provided per-problem as a list/tuple of length ``batch``.

    Args:
        start_configs: Batch of start configurations, shape ``(batch, dim)``.
        goal_configs: Per-problem goal configurations, shape
            ``(batch, num_goals, dim)``.  Pass shape ``(batch, 1, dim)`` for
            one goal per problem, or ``(batch, G, dim)`` to give each problem
            G candidate goals.
        collision_context: Either a single context dict shared by all problems,
            or a list/tuple of per-problem contexts with length ``batch``.
            Per-problem contexts are packed into padded batched world/self
            tensors and dispatched in one JAX FFI batch call.
        jit_trace: If True, run the FFI dispatch through a cached ``jax.jit``
            wrapper keyed by planner hyper-parameters and input shapes. This
            mostly reduces Python dispatch overhead for repeated calls.
        ...: All other arguments are identical to ``prrtc_plan``.

    Returns:
        List of ``PRRTCResult`` objects, one per start configuration, in the
        same order as ``start_configs``.

    Example::

        results = prrtc_plan_batch(
            start_configs,                        # [B, dim]
            goal_configs[:, None, :],             # [B, 1, dim]  — one goal each
            max_iterations=500_000,
            collision_context=cc,
        )
        solved_mask = [r.solved for r in results]
    """
    _load_and_register()

    collision_mode = _resolve_collision_mode(collision_checker)
    collision_margin = float(collision_margin)

    start_configs = jnp.atleast_2d(jnp.asarray(start_configs, dtype=jnp.float32))
    batch_size = int(start_configs.shape[0])
    dim = int(start_configs.shape[1])

    # goal_configs must be [batch, num_goals, dim]
    goal_configs_arr = jnp.asarray(goal_configs, dtype=jnp.float32)
    if goal_configs_arr.ndim == 2:
        # [batch, dim] → [batch, 1, dim]
        goal_configs_arr = goal_configs_arr[:, None, :]
    if goal_configs_arr.ndim != 3 or goal_configs_arr.shape[0] != batch_size:
        raise ValueError(
            f"goal_configs must have shape (batch, num_goals, dim), "
            f"got {goal_configs_arr.shape} with batch_size={batch_size}"
        )
    num_goals = int(goal_configs_arr.shape[1])
    # Contiguous [batch, num_goals, dim] tensor — the CUDA handler slices per problem
    goal_batched = goal_configs_arr.reshape(batch_size, num_goals, dim)

    # Optional per-problem collision contexts: pack into one batched-context FFI call.
    if isinstance(collision_context, (list, tuple)):
        if len(collision_context) != batch_size:
            raise ValueError(
                "When collision_context is a list/tuple, it must have length "
                f"equal to batch_size ({batch_size}), got {len(collision_context)}"
            )
        contexts = list(collision_context)
        first_ctx = next((ctx for ctx in contexts if ctx is not None), None)
        if first_ctx is None:
            if not allow_unsafe_no_collision:
                raise ValueError(
                    "Collision-aware planning is the default. Provide collision_context "
                    "with required keys or set allow_unsafe_no_collision=True to opt out."
                )
            collision_context = None
        else:
            required_keys = (
                "fk_twists", "fk_parent_tf", "fk_parent_idx", "fk_act_idx",
                "fk_mimic_mul", "fk_mimic_off", "fk_mimic_act_idx", "fk_topo_inv",
                "sphere_link_idx", "sphere_local", "sphere_radius",
            )
            for i, ctx in enumerate(contexts):
                if ctx is None:
                    continue
                missing = [k for k in required_keys if k not in ctx]
                if missing:
                    raise ValueError(f"collision_context[{i}] missing required keys: {missing}")

            fk_twists = jnp.asarray(first_ctx["fk_twists"], dtype=jnp.float32)
            fk_parent_tf = jnp.asarray(first_ctx["fk_parent_tf"], dtype=jnp.float32)
            fk_parent_idx = jnp.asarray(first_ctx["fk_parent_idx"], dtype=jnp.int32)
            fk_act_idx = jnp.asarray(first_ctx["fk_act_idx"], dtype=jnp.int32)
            fk_mimic_mul = jnp.asarray(first_ctx["fk_mimic_mul"], dtype=jnp.float32)
            fk_mimic_off = jnp.asarray(first_ctx["fk_mimic_off"], dtype=jnp.float32)
            fk_mimic_act_idx = jnp.asarray(first_ctx["fk_mimic_act_idx"], dtype=jnp.int32)
            fk_topo_inv = jnp.asarray(first_ctx["fk_topo_inv"], dtype=jnp.int32)
            sphere_link_idx = jnp.asarray(first_ctx["sphere_link_idx"], dtype=jnp.int32)
            sphere_local = jnp.asarray(first_ctx["sphere_local"], dtype=jnp.float32)
            sphere_radius = jnp.asarray(first_ctx["sphere_radius"], dtype=jnp.float32)

            # Coarse model is robot-level (shared across the batch), like the fk_/sphere_ arrays.
            (coarse_sphere_link_idx, coarse_sphere_local, coarse_sphere_radius,
             fine_link_offsets, coarse_self_pairs, self_pair_ranges) = _coarse_buffers(first_ctx)

            world_specs: list[tuple[str, int]] = [
                ("world_spheres", 4),
                ("world_capsules", 7),
                ("world_boxes", 15),
                ("world_halfspaces", 6),
            ]
            packed_world: dict[str, jnp.ndarray] = {}
            packed_counts: dict[str, jnp.ndarray] = {}

            for key, feat in world_specs:
                counts = np.zeros((batch_size,), dtype=np.int32)
                arrays: list[np.ndarray] = []
                max_count = 0
                for i, ctx in enumerate(contexts):
                    if ctx is None or key not in ctx:
                        arr = np.zeros((0, feat), dtype=np.float32)
                    else:
                        arr = np.asarray(ctx[key], dtype=np.float32)
                        if arr.ndim != 2 or arr.shape[1] != feat:
                            raise ValueError(
                                f"collision_context[{i}]['{key}'] must have shape (N, {feat}), got {arr.shape}"
                            )
                    arrays.append(arr)
                    counts[i] = int(arr.shape[0])
                    max_count = max(max_count, int(arr.shape[0]))

                packed = np.zeros((batch_size, max_count, feat), dtype=np.float32)
                for i, arr in enumerate(arrays):
                    n = arr.shape[0]
                    if n > 0:
                        packed[i, :n, :] = arr

                packed_world[key] = jnp.asarray(packed, dtype=jnp.float32)
                packed_counts[key] = jnp.asarray(counts, dtype=jnp.int32)

            self_counts = np.zeros((batch_size,), dtype=np.int32)
            self_arrays: list[np.ndarray] = []
            max_self_pairs = 0
            for i, ctx in enumerate(contexts):
                if ctx is None or "self_pairs" not in ctx:
                    arr = np.zeros((0, 2), dtype=np.int32)
                else:
                    arr = np.asarray(ctx["self_pairs"], dtype=np.int32)
                    if arr.ndim != 2 or arr.shape[1] != 2:
                        raise ValueError(
                            f"collision_context[{i}]['self_pairs'] must have shape (N, 2), got {arr.shape}"
                        )
                self_arrays.append(arr)
                self_counts[i] = int(arr.shape[0])
                max_self_pairs = max(max_self_pairs, int(arr.shape[0]))

            packed_self = np.zeros((batch_size, max_self_pairs, 2), dtype=np.int32)
            for i, arr in enumerate(self_arrays):
                n = arr.shape[0]
                if n > 0:
                    packed_self[i, :n, :] = arr

            self_pairs_batched = jnp.asarray(packed_self, dtype=jnp.int32)
            self_pairs_count = jnp.asarray(self_counts, dtype=jnp.int32)

            if min_vals is None:
                min_vals = jnp.ones(dim, dtype=jnp.float32) * -jnp.pi
            if max_vals is None:
                max_vals = jnp.ones(dim, dtype=jnp.float32) * jnp.pi
            min_vals = jnp.asarray(min_vals, dtype=jnp.float32)
            max_vals = jnp.asarray(max_vals, dtype=jnp.float32)

            step_size = float(step_size)
            if step_size <= 0.0:
                raise ValueError("step_size must be > 0")

            granularity = int(granularity)
            if granularity < 1 or (granularity & (granularity - 1)) != 0:
                raise ValueError(f"granularity must be a power of 2, got {granularity}")
            if granularity > 64:
                raise ValueError(
                    f"granularity must be <= 64 (PRRTC_BLOCK_THREADS_MAX), got {granularity}"
                )

            result_shapes = (
                jax.ShapeDtypeStruct((batch_size, dim, max_nodes), jnp.float32),
                jax.ShapeDtypeStruct((batch_size, dim, max_nodes), jnp.float32),
                jax.ShapeDtypeStruct((batch_size, max_nodes), jnp.int32),
                jax.ShapeDtypeStruct((batch_size, max_nodes), jnp.int32),
                jax.ShapeDtypeStruct((batch_size, 2), jnp.int32),
                jax.ShapeDtypeStruct((batch_size, 2), jnp.int32),
                jax.ShapeDtypeStruct((batch_size, 1), jnp.int32),
                jax.ShapeDtypeStruct((batch_size, 3), jnp.int32),
                jax.ShapeDtypeStruct((batch_size, 1), jnp.int32),
                jax.ShapeDtypeStruct((batch_size,), jnp.float32),
            )

            if jit_trace:
                traced_call = _get_prrtc_batch_ctx_jit_kernel(
                    max_iterations=int(max_iterations),
                    step_size=float(step_size),
                    num_new_samples=int(num_new_samples),
                    granularity=int(granularity),
                    max_nodes=int(max_nodes),
                    balance_mode=int(balance_mode),
                    tree_ratio=float(tree_ratio),
                    dynamic_domain=bool(dynamic_domain),
                    dd_alpha=float(dd_alpha),
                    dd_radius=float(dd_radius),
                    dd_min_radius=float(dd_min_radius),
                    collision_mode=int(collision_mode),
                    collision_margin=float(collision_margin),
                )
                raw = traced_call(
                    start_configs,
                    goal_batched,
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
                    packed_world["world_spheres"],
                    packed_world["world_capsules"],
                    packed_world["world_boxes"],
                    packed_world["world_halfspaces"],
                    self_pairs_batched,
                    packed_counts["world_spheres"],
                    packed_counts["world_capsules"],
                    packed_counts["world_boxes"],
                    packed_counts["world_halfspaces"],
                    self_pairs_count,
                    coarse_sphere_link_idx,
                    coarse_sphere_local,
                    coarse_sphere_radius,
                    fine_link_offsets,
                    coarse_self_pairs,
                    self_pair_ranges,
                )
            else:
                raw = jax.ffi.ffi_call("prrtc_planner_batch_ctx", result_shapes)(
                    start_configs,
                    goal_batched,
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
                    packed_world["world_spheres"],
                    packed_world["world_capsules"],
                    packed_world["world_boxes"],
                    packed_world["world_halfspaces"],
                    self_pairs_batched,
                    packed_counts["world_spheres"],
                    packed_counts["world_capsules"],
                    packed_counts["world_boxes"],
                    packed_counts["world_halfspaces"],
                    self_pairs_count,
                    coarse_sphere_link_idx,
                    coarse_sphere_local,
                    coarse_sphere_radius,
                    fine_link_offsets,
                    coarse_self_pairs,
                    self_pair_ranges,
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
                    granularity=np.int32(granularity),
                    collision_mode=np.int32(collision_mode),
                    collision_margin=np.float32(collision_margin),
                )

            raw = jax.device_get(raw)

            results: list[PRRTCResult] = []
            for i in range(batch_size):
                ta_cfg = raw[0][i]
                tb_cfg = raw[1][i]
                ta_par = raw[2][i]
                tb_par = raw[3][i]
                tsizes = raw[4][i]
                conn = raw[7][i]
                solved = bool(raw[8][i][0] == 1)

                if solved:
                    path = _trace_path(ta_cfg, tb_cfg, ta_par, tb_par, conn)
                    cost = float(jnp.sum(jnp.linalg.norm(jnp.diff(path, axis=0), axis=1)))
                else:
                    path = None
                    cost = float("inf")

                size_a = int(tsizes[0])
                size_b = int(tsizes[1])
                results.append(PRRTCResult(
                    solved=solved,
                    path=path,
                    tree_a_size=size_a,
                    tree_b_size=size_b,
                    iterations=int(raw[6][i][0]),
                    cost=cost,
                    kernel_time_ms=float(raw[9][i]),
                    tree_a_configs=ta_cfg[:, :size_a],
                    tree_b_configs=tb_cfg[:, :size_b],
                    tree_a_parents=ta_par[:size_a],
                    tree_b_parents=tb_par[:size_b],
                ))
            return results

    if min_vals is None:
        min_vals = jnp.ones(dim, dtype=jnp.float32) * -jnp.pi
    if max_vals is None:
        max_vals = jnp.ones(dim, dtype=jnp.float32) * jnp.pi
    min_vals = jnp.asarray(min_vals, dtype=jnp.float32)
    max_vals = jnp.asarray(max_vals, dtype=jnp.float32)

    step_size = float(step_size)
    if step_size <= 0.0:
        raise ValueError("step_size must be > 0")

    granularity = int(granularity)
    if granularity < 1 or (granularity & (granularity - 1)) != 0:
        raise ValueError(f"granularity must be a power of 2, got {granularity}")
    if granularity > 64:
        raise ValueError(f"granularity must be <= 64 (PRRTC_BLOCK_THREADS_MAX), got {granularity}")

    # --- Build collision buffers (identical logic to prrtc_plan) ---
    fk_twists        = jnp.zeros((0, 6),  dtype=jnp.float32)
    fk_parent_tf     = jnp.zeros((0, 7),  dtype=jnp.float32)
    fk_parent_idx    = jnp.zeros((0,),    dtype=jnp.int32)
    fk_act_idx       = jnp.zeros((0,),    dtype=jnp.int32)
    fk_mimic_mul     = jnp.zeros((0,),    dtype=jnp.float32)
    fk_mimic_off     = jnp.zeros((0,),    dtype=jnp.float32)
    fk_mimic_act_idx = jnp.zeros((0,),    dtype=jnp.int32)
    fk_topo_inv      = jnp.zeros((0,),    dtype=jnp.int32)
    sphere_link_idx  = jnp.zeros((0,),    dtype=jnp.int32)
    sphere_local     = jnp.zeros((0, 3),  dtype=jnp.float32)
    sphere_radius    = jnp.zeros((0,),    dtype=jnp.float32)
    world_spheres    = jnp.zeros((0, 4),  dtype=jnp.float32)
    world_capsules   = jnp.zeros((0, 7),  dtype=jnp.float32)
    world_boxes      = jnp.zeros((0, 15), dtype=jnp.float32)
    world_halfspaces = jnp.zeros((0, 6),  dtype=jnp.float32)
    self_pairs       = jnp.zeros((0, 2),  dtype=jnp.int32)

    required_keys = (
        "fk_twists", "fk_parent_tf", "fk_parent_idx", "fk_act_idx",
        "fk_mimic_mul", "fk_mimic_off", "fk_mimic_act_idx", "fk_topo_inv",
        "sphere_link_idx", "sphere_local", "sphere_radius",
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
        fk_twists        = jnp.asarray(collision_context["fk_twists"],        dtype=jnp.float32)
        fk_parent_tf     = jnp.asarray(collision_context["fk_parent_tf"],     dtype=jnp.float32)
        fk_parent_idx    = jnp.asarray(collision_context["fk_parent_idx"],    dtype=jnp.int32)
        fk_act_idx       = jnp.asarray(collision_context["fk_act_idx"],       dtype=jnp.int32)
        fk_mimic_mul     = jnp.asarray(collision_context["fk_mimic_mul"],     dtype=jnp.float32)
        fk_mimic_off     = jnp.asarray(collision_context["fk_mimic_off"],     dtype=jnp.float32)
        fk_mimic_act_idx = jnp.asarray(collision_context["fk_mimic_act_idx"], dtype=jnp.int32)
        fk_topo_inv      = jnp.asarray(collision_context["fk_topo_inv"],      dtype=jnp.int32)
        sphere_link_idx  = jnp.asarray(collision_context["sphere_link_idx"],  dtype=jnp.int32)
        sphere_local     = jnp.asarray(collision_context["sphere_local"],     dtype=jnp.float32)
        sphere_radius    = jnp.asarray(collision_context["sphere_radius"],    dtype=jnp.float32)
        if "world_spheres"    in collision_context:
            world_spheres    = jnp.asarray(collision_context["world_spheres"],    dtype=jnp.float32)
        if "world_capsules"   in collision_context:
            world_capsules   = jnp.asarray(collision_context["world_capsules"],   dtype=jnp.float32)
        if "world_boxes"      in collision_context:
            world_boxes      = jnp.asarray(collision_context["world_boxes"],      dtype=jnp.float32)
        if "world_halfspaces" in collision_context:
            world_halfspaces = jnp.asarray(collision_context["world_halfspaces"], dtype=jnp.float32)
        if "self_pairs"       in collision_context:
            self_pairs       = jnp.asarray(collision_context["self_pairs"],       dtype=jnp.int32)

    (coarse_sphere_link_idx, coarse_sphere_local, coarse_sphere_radius,
     fine_link_offsets, coarse_self_pairs, self_pair_ranges) = _coarse_buffers(collision_context)

    # Output shapes — leading batch dimension on every buffer
    result_shapes = (
        jax.ShapeDtypeStruct((batch_size, dim, max_nodes), jnp.float32),  # tree_a_configs
        jax.ShapeDtypeStruct((batch_size, dim, max_nodes), jnp.float32),  # tree_b_configs
        jax.ShapeDtypeStruct((batch_size, max_nodes),      jnp.int32),    # tree_a_parents
        jax.ShapeDtypeStruct((batch_size, max_nodes),      jnp.int32),    # tree_b_parents
        jax.ShapeDtypeStruct((batch_size, 2),              jnp.int32),    # tree_sizes
        jax.ShapeDtypeStruct((batch_size, 2),              jnp.int32),    # completed
        jax.ShapeDtypeStruct((batch_size, 1),              jnp.int32),    # iter_count
        jax.ShapeDtypeStruct((batch_size, 3),              jnp.int32),    # connection_info
        jax.ShapeDtypeStruct((batch_size, 1),              jnp.int32),    # solved_flag
        jax.ShapeDtypeStruct((batch_size,),                jnp.float32),  # kernel_time_ms
    )

    if jit_trace:
        traced_call = _get_prrtc_batch_jit_kernel(
            max_iterations=int(max_iterations),
            step_size=float(step_size),
            num_new_samples=int(num_new_samples),
            granularity=int(granularity),
            max_nodes=int(max_nodes),
            balance_mode=int(balance_mode),
            tree_ratio=float(tree_ratio),
            dynamic_domain=bool(dynamic_domain),
            dd_alpha=float(dd_alpha),
            dd_radius=float(dd_radius),
            dd_min_radius=float(dd_min_radius),
            collision_mode=int(collision_mode),
            collision_margin=float(collision_margin),
        )
        raw = traced_call(
            start_configs,
            goal_batched,
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
            world_capsules,
            world_boxes,
            world_halfspaces,
            self_pairs,
            coarse_sphere_link_idx,
            coarse_sphere_local,
            coarse_sphere_radius,
            fine_link_offsets,
            coarse_self_pairs,
            self_pair_ranges,
        )
    else:
        raw = jax.ffi.ffi_call("prrtc_planner_batch", result_shapes)(
            start_configs,
            goal_batched,
            min_vals,
            max_vals,
            fk_twists, fk_parent_tf, fk_parent_idx, fk_act_idx,
            fk_mimic_mul, fk_mimic_off, fk_mimic_act_idx, fk_topo_inv,
            sphere_link_idx, sphere_local, sphere_radius,
            world_spheres, world_capsules, world_boxes, world_halfspaces, self_pairs,
            coarse_sphere_link_idx, coarse_sphere_local, coarse_sphere_radius,
            fine_link_offsets, coarse_self_pairs, self_pair_ranges,
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
            granularity=np.int32(granularity),
            collision_mode=np.int32(collision_mode),
            collision_margin=np.float32(collision_margin),
        )

    # Materialize all outputs from device before Python post-processing
    raw = jax.device_get(raw)

    results = []
    for i in range(batch_size):
        ta_cfg  = raw[0][i]   # [dim, max_nodes]
        tb_cfg  = raw[1][i]   # [dim, max_nodes]
        ta_par  = raw[2][i]   # [max_nodes]
        tb_par  = raw[3][i]   # [max_nodes]
        tsizes  = raw[4][i]   # [2]
        conn    = raw[7][i]   # [3]
        solved  = bool(raw[8][i][0] == 1)

        if solved:
            path = _trace_path(ta_cfg, tb_cfg, ta_par, tb_par, conn)
            cost = float(jnp.sum(jnp.linalg.norm(jnp.diff(path, axis=0), axis=1)))
        else:
            path = None
            cost = float("inf")

        size_a = int(tsizes[0])
        size_b = int(tsizes[1])
        results.append(PRRTCResult(
            solved=solved,
            path=path,
            tree_a_size=size_a,
            tree_b_size=size_b,
            iterations=int(raw[6][i][0]),
            cost=cost,
            kernel_time_ms=float(raw[9][i]),
            tree_a_configs=ta_cfg[:, :size_a],
            tree_b_configs=tb_cfg[:, :size_b],
            tree_a_parents=ta_par[:size_a],
            tree_b_parents=tb_par[:size_b],
        ))

    return results


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
