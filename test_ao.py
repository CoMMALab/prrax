#!/usr/bin/env python3
"""Check of the AO-x passes (``ao_passes``, MR. POP / pAORRTC) on McVAMP's Panda problems.

Unconstrained and constrained (cpRRTC projection), single and batched: the kept path stays
valid (collision-free along densified edges; on the manifold when constrained) and is cheaper
than a plain solve's on most problems (solves are not deterministic across blocks). The same
for MR. POP's path simplification (``shortcut_rounds``), alone and between AO-x passes.

    python test_ao.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import benchmark_constrained as B

P = B.PRRTC


def main() -> None:
    from pyroffi.collision._geometry import Box

    robot, coll = B.load_panda()
    ctx = B.make_context(robot, coll, maze=False)
    lo = jnp.asarray(robot.joints.lower_limits, jnp.float32)
    hi = jnp.asarray(robot.joints.upper_limits, jnp.float32)
    probs = [B.load_problems("plane")[i] for i in (5, 120, 300, 640, 950)]
    kw = dict(max_iterations=2000, step_size=1.0, num_new_samples=64, granularity=16,
              max_nodes=100_000, dynamic_domain=False, min_vals=lo, max_vals=hi,
              collision_checker="binary_coarse")

    def collision_free(path, boxes) -> bool:
        # Edges densified to 0.01 rad: the planner checks 16 waypoints per edge, not just nodes.
        pts = [path[:1]]
        for a, b in zip(path[:-1], path[1:]):
            n = max(int(np.ceil(np.linalg.norm(b - a) / 0.01)), 1)
            pts.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
        q = jnp.asarray(np.concatenate(pts), jnp.float32)
        d = float(jax.vmap(lambda x: coll.compute_self_collision_distance(robot, x))(q).min())
        if len(boxes):
            world = Box.from_center_and_half_lengths(boxes[:, :3], boxes[:, 12:15])
            d = min(d, float(jax.vmap(lambda x: coll.compute_world_collision_distance(robot, x, world))(q).min()))
        return d >= -1e-3

    for label, constrained in (("unconstrained", False), ("constrained", True)):
        cons = [[B.tsr_constraint(robot, p["tsr"])] if constrained else [] for p in probs]
        ctxs = [dict(ctx, world_boxes=p["boxes"]) for p in probs]
        starts = np.asarray([p["start"] for p in probs], np.float32)
        goals = np.asarray([[p["goal"]] for p in probs], np.float32)
        proj = P.ProjectionSettings(mode="cprrtc")
        base = P.prrtc_plan_batch(starts, goals, collision_context=ctxs, constraints=cons,
                                  projection=proj, **kw)
        ao = P.prrtc_plan_batch(starts, goals, collision_context=ctxs, constraints=cons,
                                projection=proj, ao_passes=8, **kw)
        simp = P.prrtc_plan_batch(starts, goals, collision_context=ctxs, constraints=cons,
                                  projection=proj, shortcut_rounds=1, **kw)
        ao_simp = P.prrtc_plan_batch(starts, goals, collision_context=ctxs, constraints=cons,
                                     projection=proj, ao_passes=8, shortcut_rounds=1, **kw)
        single = P.prrtc_plan(starts[0], goals[0], collision_context=ctxs[0], constraints=cons[0],
                              projection=proj, ao_passes=8, **kw)
        n_better = sum(r.cost < 0.99 * r0.cost for r0, r in zip(base, ao))
        for r, p, cs in zip(ao + simp + ao_simp, probs * 3, cons * 3):
            assert r.solved, label
            if constrained:
                v = B.validate_path(robot, coll, cs[0], r.path, p["boxes"], 1.01e-3, pen=False)
                assert v["on_manifold"] and v["collision_free"], (label, v)
            else:
                assert collision_free(r.path, p["boxes"]), label
            assert np.allclose(r.path[0], p["start"], atol=2e-2) and np.allclose(r.path[-1], p["goal"], atol=2e-2)
        assert single.solved
        assert n_better >= len(probs) // 2, (label, n_better)
        mean = lambda rs: np.mean([r.cost for r in rs])
        ms = lambda rs: np.mean([r.kernel_time_ms for r in rs])
        assert mean(simp) < mean(base) and mean(ao_simp) < mean(ao), label
        print(f"{label}: mean cost plain {mean(base):.3f}, simplified {mean(simp):.3f}, "
              f"AO-x {mean(ao):.3f} (lower on {n_better}/{len(probs)}), AO-x + simplified "
              f"{mean(ao_simp):.3f}; kernel ms {ms(base):.1f} / {ms(simp):.1f} / {ms(ao):.1f} / "
              f"{ms(ao_simp):.1f}; every path valid")


if __name__ == "__main__":
    main()
