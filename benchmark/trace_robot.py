"""Trace and build prrax's planner for a robot, then validate and time its binary collision check.

    python benchmark/trace_robot.py robot.urdf [--srdf robot.srdf] [--boxes 8] [--batch 262144]

The planner traces the robot through crocket and builds on first use anyway (cached per robot
and sphere model); this prebuilds it and checks it. Random configurations are checked against
random boxes by the binary check (mode 0: crocket's traced FK) and by the differentiable check
at margin 0 (mode 1: GRiM's FK and the sphere tables), an independent path in the same build.
The boxes are per-call inputs, never compiled in: any world runs without a rebuild. Exits
non-zero if the two disagree.
"""

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pyroffi as pk
import yourdfpy
from pyroffi.collision._robot_collision import RobotCollisionSpherized

from prrax.jax import _grim_build as gb
from prrax.jax import utils as U


def random_boxes(n, rng, lo=(0.3, -0.8, 0.0), hi=(1.0, 0.8, 1.2)):
    """n boxes [centre, unit axes (3x3, row k = axis k), half-extents] (prrax's 15-float row)."""
    out = np.zeros((max(n, 1), 15), np.float32)
    for i in range(n):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        out[i] = np.concatenate([rng.uniform(lo, hi), q.T.reshape(-1), rng.uniform(0.02, 0.08, 3)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urdf")
    ap.add_argument("--srdf", help="self-collision exclusions (default: none)")
    ap.add_argument("--boxes", type=int, default=8)
    ap.add_argument("--batch", type=int, default=262144)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    urdf = yourdfpy.URDF.load(a.urdf)
    robot = pk.Robot.from_urdf(urdf)
    coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=a.srdf)
    ctx = U.build_prrtc_collision_context(robot, coll, [])
    print(f"{a.urdf}: {len(ctx['sphere_link_idx'])} spheres, {len(ctx['self_pairs'])} self pairs")

    t = time.perf_counter()
    name = gb.planner_targets(ctx, ())[3]   # PrraxCcFfi of this robot's build
    print(f"build {time.perf_counter() - t:.1f} s")

    rng = np.random.default_rng(a.seed)
    lo = np.asarray(robot.joints.lower_limits, np.float32)
    hi = np.asarray(robot.joints.upper_limits, np.float32)
    lo, hi = np.where(np.isfinite(lo), lo, -np.pi), np.where(np.isfinite(hi), hi, np.pi)
    q = jnp.asarray(rng.uniform(lo, hi, (a.batch, len(lo))).astype(np.float32))
    world = [jnp.zeros(4), jnp.zeros(7), jnp.asarray(random_boxes(a.boxes, rng).reshape(-1)), jnp.zeros(6)]
    counts = jnp.asarray([0, 0, a.boxes, 0], jnp.float32)

    hits = {}
    for mode, label in ((0, "binary (traced)"), (1, "differentiable")):
        f = jax.jit(lambda q, mode=mode: jax.ffi.ffi_call(name, jax.ShapeDtypeStruct((a.batch,), jnp.int32))(
            q, *world, counts, mode=np.int32(mode)))
        hits[mode] = np.asarray(jax.block_until_ready(f(q)))
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            jax.block_until_ready(f(q))
            ts.append(time.perf_counter() - t)
        print(f"{label:16s}: {np.median(ts) * 1e3:7.3f} ms at B={a.batch}, {hits[mode].mean():.3f} in collision")

    bad = int((hits[0] != hits[1]).sum())
    print(f"verdict mismatches: {bad} / {a.batch}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
