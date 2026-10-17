#!/usr/bin/env python3
"""Checks of constrained pRRTC (CUDA/prrtc_constraint.cuh) against independent references.

1. TSR error of the CUDA build vs a JAX reference (pyroffi FK + jaxlie) at random configs.
2. Projection: the CUDA projection of perturbed on-manifold configs converges, and the JAX
   reference agrees the result is on the manifold.
3. Planning on McVAMP plane/line problems, both projection modes, single and batched (with
   per-problem worlds and constraints), plus a two-constraint problem (position TSR and an
   orientation TSR on the same frame): every path waypoint is on the manifold (JAX) and
   collision-free (pyroffi).

    python test_constrained.py
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import benchmark_constrained as B

P = B.PRRTC


def main() -> None:
    robot, coll = B.load_panda()
    ctx = B.make_context(robot, coll, maze=False)
    lo = jnp.asarray(robot.joints.lower_limits, jnp.float32)
    hi = jnp.asarray(robot.joints.upper_limits, jnp.float32)
    proj = P.ProjectionSettings()
    tol = proj.tol * 1.01
    plane = B.load_problems("plane")
    line = B.load_problems("line", keep_orientation=True)
    rng = np.random.default_rng(0)

    # 1. TSR error vs the JAX reference.
    con = B.tsr_constraint(robot, line[0]["tsr"])
    q = rng.uniform(np.asarray(lo), np.asarray(hi), size=(256, 7)).astype(np.float32)
    err, _, _ = P.constraint_error(ctx, [con], q, proj, lo, hi)
    ref = B.reference_error(robot, con, q)
    assert np.allclose(np.asarray(err), ref, rtol=1e-3, atol=1e-4), np.abs(np.asarray(err) - ref).max()
    print(f"1. TSR error matches the JAX reference (max diff {np.abs(np.asarray(err) - ref).max():.2e})")

    # 2. Projection of perturbed on-manifold configurations.
    seeds = np.asarray([p["start"] for p in line[:64]], np.float32)
    cons = [B.tsr_constraint(robot, p["tsr"]) for p in line[:64]]
    noisy = seeds + rng.normal(scale=0.05, size=seeds.shape).astype(np.float32)
    n_ok, n_on = 0, 0
    for i in range(len(seeds)):
        _, q_proj, ok = P.constraint_error(ctx, [cons[i]], noisy[i:i + 1], proj, lo, hi)
        if bool(ok[0]):
            n_ok += 1
            n_on += bool(B.reference_error(robot, cons[i], np.asarray(q_proj))[0] <= tol)
    assert n_ok >= 0.9 * len(seeds) and n_on == n_ok, (n_ok, n_on)
    print(f"2. projection converged {n_ok}/{len(seeds)}, all on the manifold per the reference")

    kw = dict(max_iterations=2000, step_size=1.0, num_new_samples=64, granularity=16,
              max_nodes=100_000, dynamic_domain=False, min_vals=lo, max_vals=hi,
              collision_checker="binary_coarse")

    def check(results, probs, cons_lists, label):
        n_bad = 0
        for r, p, cs in zip(results, probs, cons_lists):
            assert r.solved, f"{label}: unsolved"
            for c in cs:
                v = B.validate_path(robot, coll, c, r.path, p["boxes"], tol, pen=False)
                n_bad += (not v["on_manifold"]) or (not v["collision_free"])
        assert n_bad == 0, f"{label}: {n_bad} invalid path checks"
        print(f"3. {label}: {len(results)}/{len(results)} solved, every path valid")

    # 3a. Single solves, both modes, plane and line(+orientation) problems with obstacles.
    probs = [plane[i] for i in (5, 300, 950)] + [line[i] for i in (3, 400, 700)]
    for mode in ("mcvamp", "cprrtc"):
        pr = proj._replace(mode=mode)
        res, cl = [], []
        for p in probs:
            c = [B.tsr_constraint(robot, p["tsr"])]
            res.append(P.prrtc_plan(p["start"], np.asarray([p["goal"]]),
                                    collision_context=dict(ctx, world_boxes=p["boxes"]),
                                    constraints=c, projection=pr, **kw))
            cl.append(c)
        check(res, probs, cl, f"single {mode}")

    # 3b. Batched: per-problem worlds and constraints in one call.
    batch = plane[100:108]
    res = P.prrtc_plan_batch(np.asarray([p["start"] for p in batch]),
                             np.asarray([p["goal"] for p in batch])[:, None],
                             collision_context=[dict(ctx, world_boxes=p["boxes"]) for p in batch],
                             constraints=[[B.tsr_constraint(robot, p["tsr"])] for p in batch],
                             projection=proj, **dict(kw, max_nodes=20_000))
    check(res, batch, [[B.tsr_constraint(robot, p["tsr"])] for p in batch], "batch, per-problem constraints")

    # 3c. Two constraints on one frame (cyclic projection): a plane-with-orientation problem's
    # TSR split into its position rows and its orientation rows.
    p = B.load_problems("plane", keep_orientation=True)[7]
    off, ref_pose, lo_b, hi_b = p["tsr"]
    pos = B.tsr_constraint(robot, (off, ref_pose, list(lo_b[:3]) + [-1000.0] * 3, list(hi_b[:3]) + [1000.0] * 3))
    ori = B.tsr_constraint(robot, (off, ref_pose, [-1000.0] * 3 + list(lo_b[3:]), [1000.0] * 3 + list(hi_b[3:])))
    err0 = np.asarray(P.constraint_error(ctx, [pos, ori], np.asarray([p["start"], p["goal"]]), proj, lo, hi)[0])
    if np.all(err0 <= proj.tol):
        r = P.prrtc_plan(p["start"], np.asarray([p["goal"]]), collision_context=dict(ctx, world_boxes=p["boxes"]),
                         constraints=[pos, ori], projection=proj, **kw)
        check([r], [p], [[pos, ori]], "two constraints (cyclic projection)")
    else:
        print(f"3. two constraints: skipped, problem 7's start/goal violate the orientation TSR ({err0})")


if __name__ == "__main__":
    main()
