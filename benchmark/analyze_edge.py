#!/usr/bin/env python3
"""Grama parallel analysis (mantis.parallel_analysis.dump) of prrax's fully traced edge validator
(_mantis_algorithms.edge_margin: interpolation + FK + every sphere/obstacle and self distance),
traced for a robot with worlds padded to the largest of N MBM problems. No kernels are built:
tables of T_p, speedup, efficiency, p T_p cost and overheads per emittable layout, the batch
model, and the layout to pick per batch size.

    python benchmark/analyze_edge.py --robot panda --problems 100 [--exact-smem]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_mbm_batched as M  # noqa: E402
from prrax.jax import _grim_build, _mantis_algorithms as algs  # noqa: E402
from prrax.jax.prrtc import _pack_worlds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--robot", default="panda", choices=["panda", "fetch", "baxter"])
    ap.add_argument("--problems", type=int, default=100)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 32, 256, 1024, 4096, 16384, 65536, 262144])
    ap.add_argument("--objective", choices=["throughput", "latency", "both"], default="both")
    ap.add_argument("--exact-smem", action="store_true")
    ap.add_argument("--program", choices=["edge", "fk"], default="edge",
                    help="edge: the whole traced waypoint check; fk: only the traced FK of the sphere "
                         "frames (prrax::fused_frames, what binary / binary_coarse trace today)")
    args = ap.parse_args()

    from pyroffi import _grim
    _grim.ensure_importable()
    import mantis
    from mantis import parallel_analysis as pa
    from grim.motion import _build
    from pyroffi.collision._obstacles import create_collision_environment

    robot, coll, _, _ = M.load_robot_models(args.robot)
    valid = [p for ps in M.load_robot_dataset(args.robot)["problems"].values() for p in ps if p.get("valid")]
    probs = [valid[i] for i in np.linspace(0, len(valid) - 1, args.problems).astype(int)]
    cache = {}
    for p in probs:
        cache.setdefault(M._environment_signature(p), p)
    ctxs = [M.PRRTC_UTILS.build_prrtc_collision_context(robot, coll, create_collision_environment(p))
            for p in cache.values()]
    mw = [w.shape[1] for w in _pack_worlds(ctxs)[:4]]           # the planner's padded counts
    ctx = ctxs[0]
    mr = _grim.motion_robot(robot)
    t = {k: np.asarray(ctx[k]) for k in _grim_build.ROBOT_KEYS}
    link = t["sphere_link_idx"].reshape(-1).astype(int)
    rows = tuple((mr.joint_names[j] if j >= 0 else None, tuple(map(float, c)), float(r))
                 for j, c, r in zip(link, t["sphere_local"].reshape(-1, 3), t["sphere_radius"].reshape(-1)))
    pairs = tuple((int(i), int(j)) for i, j in t["self_pairs"].reshape(-1, 2) if 0 <= i < len(rows) and 0 <= j < len(rows))
    with tempfile.TemporaryDirectory() as tmp:
        urdf = Path(tmp) / "robot.urdf"
        urdf.write_text(_build.kinematic_urdf(mr.urdf_xml))
        model = mantis.Robot(urdf, name=args.robot)
    if args.program == "fk":
        clink = t["coarse_sphere_link_idx"].reshape(-1).astype(int)
        frames = tuple(mr.joint_names[j] for j in sorted({int(j) for j in np.concatenate([link, clink]) if j >= 0}))
        prog = mantis.trace(model, algs.sphere_frames(), frames=frames)
    else:
        prog = mantis.trace(model, algs.edge_margin(), rows=rows, pairs=pairs, n_spheres=mw[0], n_capsules=mw[1],
                            n_boxes=mw[2], n_halfspaces=mw[3])
    print(f"{args.robot}: {len(rows)} spheres, {len(pairs)} self pairs, world padded to "
          f"spheres/capsules/boxes/halfspaces {mw} ({len(cache)} distinct worlds in {len(probs)} problems)")
    pa.dump(prog, args.batch, args.objective, exact_smem=args.exact_smem)


if __name__ == "__main__":
    main()
