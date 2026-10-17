"""Per-robot builds of the pRRTC planner through GRiM.

The planner (``CUDA/prrtc_planner.cu``) is compiled once per robot collision model. GRiM's
``robot_header`` supplies the kinematics (cricket2's straight-line FK when ``traced``, its baked
``__constant__`` tables otherwise); ``prrax_gen.cuh`` bakes prrax's sphere model (fine spheres
grouped by FK frame, coarse bounding spheres, self pairs) with compile-time counts. World
obstacles stay per-call inputs.

GRiM's ``compile_kernel`` caches on its own kernel sources only, so the prrax sources go into
the build flags (``-DPRRAX_SRC=<sha1>``): editing them forces a rebuild.

ponytail: relies on GRiM's private ``grim.motion._build`` (compile_kernel, register,
robot_header); pin the GRiM commit (see README) and revisit if GRiM grows a public build API.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

CUDA_DIR = Path(__file__).resolve().parent.parent / "CUDA"
_SOURCE = CUDA_DIR / "prrtc_planner"
SYMBOLS = ("PrrtcPlannerFfi", "PrraxFkFfi", "PrraxTsrFfi", "PrraxCcFfi", "PrraxSimplifyFfi")
# Baked tables above this go to global memory: __constant__ is 64 KB per module.
_CONSTANT_BYTES = 48 * 1024

# Robot-side keys of a collision context (see utils.build_prrtc_collision_context).
ROBOT_KEYS = ("sphere_link_idx", "sphere_local", "sphere_radius", "self_pairs",
              "coarse_sphere_link_idx", "coarse_sphere_local", "coarse_sphere_radius",
              "fine_link_offsets", "coarse_self_pairs", "self_pair_ranges")

_cache: dict[str, tuple[str, ...]] = {}


def _table(space: str, ctype: str, name: str, values) -> str:
    values = np.asarray(values).reshape(-1)
    fmt = (lambda v: f"{float(v):.9e}f") if ctype == "float" else (lambda v: str(int(v)))
    return (f"{space} {ctype} {name}[{max(values.size, 1)}] = "
            f"{{{', '.join(fmt(v) for v in values) or '0'}}};\n")


def prrax_header(ctx) -> str:
    """``prrax_gen.cuh``: the sphere model of a collision context as compile-time tables."""
    t = {k: np.asarray(ctx[k]) for k in ROBOT_KEYS}
    n_coarse = len(t["coarse_sphere_link_idx"])
    nbytes = sum(a.size * 4 for a in t.values())
    space = "__device__ __constant__" if nbytes <= _CONSTANT_BYTES else "__device__ const"
    return ("#pragma once\nnamespace prrax {\n"
            f"constexpr int n_spheres = {len(t['sphere_link_idx'])};\n"
            f"constexpr int n_self_pairs = {len(t['self_pairs'])};\n"
            f"constexpr int n_coarse = {n_coarse};\n"
            f"constexpr int n_coarse_self_pairs = {len(t['coarse_self_pairs'])};\n"
            f"constexpr bool coarse_enabled = {'true' if 0 < n_coarse <= 64 else 'false'};\n"
            + _table(space, "int", "kSphereJoint", t["sphere_link_idx"])
            + _table(space, "float", "kSphereLocal", t["sphere_local"])
            + _table(space, "float", "kSphereRadius", t["sphere_radius"])
            + _table(space, "int", "kSelfPairs", t["self_pairs"])
            + _table(space, "int", "kCoarseJoint", t["coarse_sphere_link_idx"])
            + _table(space, "float", "kCoarseLocal", t["coarse_sphere_local"])
            + _table(space, "float", "kCoarseRadius", t["coarse_sphere_radius"])
            + _table(space, "int", "kFineOffsets", t["fine_link_offsets"])
            + _table(space, "int", "kCoarseSelfPairs", t["coarse_self_pairs"])
            + _table(space, "int", "kSelfPairRanges", t["self_pair_ranges"])
            + "}  // namespace prrax\n")


def _sources_hash() -> str:
    h = hashlib.sha1()
    for p in sorted(CUDA_DIR.glob("*.cu*")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def planner_targets(ctx, ee_joints: tuple[int, ...] = (), traced: bool = True) -> tuple[str, ...]:
    """FFI target names ``(planner, fk, tsr)`` of the build for ``ctx``'s robot and sphere
    model, with the constraint Jacobians of the joint frames ``ee_joints`` (cricket2's traced
    end-effector Jacobian for one frame, the geometric Jacobian of traced frame poses for
    several)."""
    from pyroffi import _grim
    _grim.ensure_importable()
    from grim.motion import _build

    robot = ctx["robot"]
    mr = _grim.motion_robot(robot)
    header = _build.robot_header(mr, _build.Problem(ee_joints=tuple(ee_joints)), traced)
    gen = prrax_header(ctx)
    # MAX_ACT >= 6: GRiM's chol_solve scratch also holds the 6x6 LM system of a TSR step.
    flags = (f"-DMAX_JOINTS={mr.n_joints}", f"-DMAX_ACT={max(mr.n_act, 6)}",
             f"-DMAX_EE={max(len(ee_joints), 1)}",
             "-DGRIM_SOLVE_N_BUCKETS(X)=X(1)", f"-DPRRAX_SRC={_sources_hash()}")
    key = hashlib.sha1("\x00".join((header, gen, *flags)).encode()).hexdigest()
    if key not in _cache:
        so = _build.compile_kernel(str(_SOURCE), {"grim_robot_gen.cuh": header,
                                                  "prrax_gen.cuh": gen}, flags)
        names = _build.register(so, SYMBOLS)
        if traced:
            _validate_fk(robot, mr.n_joints, names[1])
        _cache[key] = names
    return _cache[key]


def fk(name: str, n_frames: int, q) -> jax.Array:
    """Frame poses ``(B, n_frames, 7)`` of a planner build at ``q`` ``(B, dim)``."""
    q = jnp.asarray(q, jnp.float32)
    return jax.ffi.ffi_call(name, jax.ShapeDtypeStruct((q.shape[0], n_frames, 7), jnp.float32))(q)


def _validate_fk(robot, n_frames: int, name: str, n: int = 256, tol: float = 1e-4) -> None:
    """The build's (traced) FK against pyroffi's at random configurations."""
    rng = np.random.default_rng(0)
    lo = np.asarray(robot.joints.lower_limits, np.float32)
    hi = np.asarray(robot.joints.upper_limits, np.float32)
    q = rng.uniform(lo, hi, size=(n, lo.shape[0])).astype(np.float32)
    T = np.asarray(fk(name, n_frames, q))
    ref = np.asarray(jax.vmap(robot.forward_kinematics)(jnp.asarray(q)))   # (n, n_links, 7)
    pj = np.asarray(robot.links.parent_joint_indices)
    links = np.flatnonzero(pj >= 0)
    got = T[:, pj[links]]
    want = ref[:, links]
    pos_err = np.abs(got[..., 4:] - want[..., 4:]).max()
    rot_err = (1.0 - np.abs(np.sum(got[..., :4] * want[..., :4], axis=-1))).max()
    if pos_err > tol or rot_err > tol:
        raise RuntimeError(f"traced planner FK disagrees with pyroffi's (position {pos_err:.2e}, "
                           f"rotation {rot_err:.2e}): rebuild without tracing or report the "
                           "miscompile")
