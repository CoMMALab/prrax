"""Per-robot builds of the pRRTC planner through GRiM.

The planner (``CUDA/prrtc_planner.cu``) is compiled once per robot collision model. GRiM's
``robot_header`` supplies the kinematics (cricket2's straight-line FK when ``traced``, its baked
``__constant__`` tables otherwise); ``prrax_gen.cuh`` bakes prrax's sphere model (fine spheres
grouped by FK frame, coarse bounding spheres, self pairs) with compile-time counts, plus
mantis's traced FK of the sphere frames for the binary check. World obstacles stay per-call
inputs: any world runs without a rebuild.

GRiM's ``compile_kernel`` caches on its own kernel sources only, so the prrax sources go into
the build flags (``-DPRRAX_SRC=<sha1>``): editing them forces a rebuild.

ponytail: relies on GRiM's private ``grim.motion._build`` (compile_kernel, register,
robot_header); pin the GRiM commit (see README) and revisit if GRiM grows a public build API.
"""

from __future__ import annotations

import hashlib
import tempfile
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


def prrax_header(ctx, ee_joints: tuple[int, ...] = (), edge=None) -> str:
    """``prrax_gen.cuh``: the sphere model of a collision context as compile-time tables, with
    prrax's mantis-traced kernels (sphere-frame FK; a TSR step per frame of ``ee_joints``)."""
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
            + "}  // namespace prrax\n" + _traced_spheres(ctx, t, space, tuple(ee_joints), edge))


def _traced_spheres(ctx, t, space: str, ee_joints: tuple[int, ...] = (), edge=None) -> str:
    """The fused FK of the collision check (``prrax::fused_frames`` / ``fused_centre`` /
    ``fused_coarse_centre``): the traced pose of every frame carrying a fine or coarse sphere,
    12 floats ([t, R column-major]), plus an identity slot for base spheres (link -1); a
    sphere's centre is its frame slot's ``t + R c`` from a table. World obstacles stay per-call.

    The sphere / pair walks stay loops over tables: fully unrolled, the per-check instruction
    footprint thrashed the planner kernel's i-cache (68k -> 110k SASS, 2.5x slower), although a
    standalone check got faster.

    With ``ee_joints``, also ``prrax::tsr_step_traced``: the fused TSR LM step of
    ``_mantis_algorithms.tsr_step`` for constraint end-effector ``e`` (CON_EE).

    With ``edge = (tier, problems, warps, (n_spheres, n_capsules, n_boxes, n_halfspaces))``, also
    ``prrax::edge_margin``: the whole waypoint check of ``_mantis_algorithms.edge_margin``
    (collision mode 4, "traced_edge") at that mantis tier, for worlds padded to those counts."""
    from pyroffi import _grim
    _grim.ensure_importable()
    import mantis
    from grim.motion import _build

    from . import _mantis_algorithms as algs
    mr = _grim.motion_robot(ctx["robot"])
    link, local = t["sphere_link_idx"].reshape(-1).astype(int), t["sphere_local"].reshape(-1, 3)
    clink = t["coarse_sphere_link_idx"].reshape(-1).astype(int)
    clocal = t["coarse_sphere_local"].reshape(-1, 3)
    bad = sorted({int(j) for j in np.concatenate([link, clink]) if j >= mr.n_joints})
    if bad:
        raise ValueError(f"sphere frames {bad} are not joints of the robot ({mr.n_joints} joints)")
    frames = sorted({int(j) for j in np.concatenate([link, clink]) if j >= 0})
    nf = len(frames)
    src, gather, tsr = "", "", ""
    if frames or ee_joints or edge is not None:
        with tempfile.TemporaryDirectory() as tmp:
            urdf = Path(tmp) / "robot.urdf"
            urdf.write_text(_build.kinematic_urdf(mr.urdf_xml))
            model = mantis.Robot(urdf, name="prrax_spheres")
        todo = {algs.tsr_step(i): {"frame": mr.joint_names[j]} for i, j in enumerate(ee_joints)}
        if frames:
            todo[algs.sphere_frames()] = {"frames": tuple(mr.joint_names[j] for j in frames)}
        variants = {mantis.resolve(a).name: [mantis.Variant("thread")] for a in todo}
        if edge is not None:
            tier, P, warps, mw = edge
            rows = tuple((mr.joint_names[j] if j >= 0 else None, tuple(map(float, c)), float(r))
                         for j, c, r in zip(link, local, np.asarray(t["sphere_radius"]).reshape(-1)))
            pairs = tuple((int(i), int(j)) for i, j in np.asarray(t["self_pairs"]).reshape(-1, 2)
                          if 0 <= i < len(rows) and 0 <= j < len(rows))
            ea = algs.edge_margin()
            todo[ea] = dict(rows=rows, pairs=pairs, n_spheres=mw[0], n_capsules=mw[1], n_boxes=mw[2],
                            n_halfspaces=mw[3])
            variants[mantis.resolve(ea).name] = [mantis.Variant(tier, 32 * warps if tier == "block" else 1,
                                                                problems=P)]
        src = mantis.generate(model, todo, variants=variants, name="prrax_spheres") + "\n"
        # mantis's joint order <-> the planner's configuration order (the robot's actuated joints)
        load_q = ("    float q[mantis::robots::prrax_spheres::n_q];\n"
                  + "".join(f"    q[{i}] = cfg[{mr.actuated_names.index(n)}];\n"
                            for i, n in enumerate(model.joint_names)))
    if ee_joints:
        tsr += ("#define PRRAX_TRACED_TSR 1\n"
               "// e2 = |d|^2 of constraint end-effector e at cfg; g (configuration order) the LM step.\n"
               "__device__ __forceinline__ float tsr_step_traced(int e, const float* cfg, const float* con,\n"
               "                                                 float lam, float* g_cfg) {\n"
               + load_q +
               "    float e2 = 0.f, g[mantis::robots::prrax_spheres::n_q];\n"
               "    switch (e) {\n"
               + "".join(f"    case {i}: mantis::robots::prrax_spheres::prrax_tsr_step_{i}_thread<float>(q, con, &lam, &e2, g); break;\n"
                         for i in range(len(ee_joints)))
               + "    }\n"
               + "".join(f"    g_cfg[{mr.actuated_names.index(n)}] = g[{i}];\n" for i, n in enumerate(model.joint_names))
               + "    return e2;\n}\n")
    if frames:
        gather = load_q + "    mantis::robots::prrax_spheres::prrax_sphere_frames_thread<float>(q, X);\n"
    if edge is not None:
        tier, P, warps, mw = edge
        fn = f"mantis::robots::prrax_spheres::prrax_edge_margin_{tier}"
        coop = tier != "thread"
        tsr += (f"#define PRRAX_EDGE_TRACED 1\n#define PRRAX_EDGE_TIER {('thread', 'warp', 'block').index(tier)}\n"
                f"#define PRRAX_EDGE_P {P if coop else 1}\n#define PRRAX_EDGE_K {warps}\n"
                f"constexpr int edge_tables = {f'{fn}_tables<float>()' if coop else 0};\n"
                f"constexpr int edge_smem = {f'{fn}_smem' if coop else 0};\n"
                "__device__ __forceinline__ void edge_load_tables(float* tb, int rank, int size) {"
                + (f" {fn}_load_tables<float>(tb, rank, size);" if coop else " (void)tb; (void)rank; (void)size;") + " }\n"
                "// margin <= 0: waypoint base + t * delta (configuration order) collides with the padded world.\n"
                "__device__ __forceinline__ void edge_margin(const float* base, const float* delta, float t,\n"
                "    const float* ws, const float* wc, const float* wb, const float* wh, float* m, float* s, const float* tables) {\n"
                "    float qm[mantis::robots::prrax_spheres::n_q], dm[mantis::robots::prrax_spheres::n_q];\n"
                + "".join(f"    qm[{i}] = base[{mr.actuated_names.index(n)}]; dm[{i}] = delta[{mr.actuated_names.index(n)}];\n"
                          for i, n in enumerate(model.joint_names))
                + (f"    {fn}<float>(qm, dm, &t, ws, wc, wb, wh, m, s, tables);\n" if coop else
                   f"    (void)s; (void)tables;\n    {fn}<float>(qm, dm, &t, ws, wc, wb, wh, m);\n")
                + "}\n")
    slot = {j: s for s, j in enumerate(frames)}

    def table(lk, lc, r):   # rows: frame slot, centre (3), radius
        return np.concatenate([np.array([slot.get(int(j), nf) for j in lk], np.float32).reshape(-1, 1),
                               lc.astype(np.float32), np.asarray(r, np.float32).reshape(-1, 1)], 1)
    return (src + "namespace prrax {\n"
            f"constexpr int n_fused_frames = {nf};\n"
            + _table(space, "float", "kFusedSphere", table(link, local, t["sphere_radius"]))
            + _table(space, "float", "kFusedCoarse", table(clink, clocal, t["coarse_sphere_radius"])) +
            "__device__ __forceinline__ void fused_frames(const float* cfg, float* X) {\n"
            + gather +
            "    for (int i = 0; i < 12; ++i) X[12 * n_fused_frames + i] = (i == 3 || i == 7 || i == 11) ? 1.f : 0.f;\n"
            "}\n"
            "__device__ __forceinline__ void fused_point(const float* X, const float* e, float* p) {\n"
            "    const float* T = X + 12 * (int)e[0];\n"
            "    for (int i = 0; i < 3; ++i) p[i] = T[i] + T[3 + i] * e[1] + T[6 + i] * e[2] + T[9 + i] * e[3];\n"
            "}\n"
            "__device__ __forceinline__ void fused_centre(const float* X, int s, float* p) { fused_point(X, &kFusedSphere[5 * s], p); }\n"
            "__device__ __forceinline__ void fused_coarse_centre(const float* X, int k, float* p) { fused_point(X, &kFusedCoarse[5 * k], p); }\n"
            + tsr + "}  // namespace prrax\n")


def _sources_hash() -> str:
    h = hashlib.sha1()
    for p in sorted(CUDA_DIR.glob("*.cu*")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def planner_targets(ctx, ee_joints: tuple[int, ...] = (), traced: bool = True, edge=None) -> tuple[str, ...]:
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
    gen = prrax_header(ctx, tuple(ee_joints), edge)
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
