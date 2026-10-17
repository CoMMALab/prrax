#!/usr/bin/env python3
"""Edge validation at VAMP's resolution: paths from every collision checker, unconstrained and
constrained, plain and simplified, are collision-free when re-checked at 1/64 rad (twice
VAMP's 1/32 rad), so no checker validates an edge by its waypoints alone.

    python test_edge_resolution.py
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
    probs = [p for p in B.load_problems("plane") if len(p["boxes"])][:6]
    self_d = jax.jit(jax.vmap(lambda x: coll.compute_self_collision_distance(robot, x)))

    def min_clearance(path, boxes) -> float:
        pts = [path[:1]]
        for a, b in zip(path[:-1], path[1:]):
            n = max(int(np.ceil(np.linalg.norm(b - a) * 64)), 1)
            pts.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
        q = jnp.asarray(np.concatenate(pts), jnp.float32)
        world = Box.from_center_and_half_lengths(boxes[:, :3], boxes[:, 12:15])
        wd = jax.vmap(lambda x: coll.compute_world_collision_distance(robot, x, world))(q)
        return min(float(self_d(q).min()), float(wd.min()))

    for checker in ("binary", "binary_coarse", "binary_coarse_coop"):
        for constrained in (False, True):
            for sr in (0, 1):
                worst = np.inf
                for p in probs:
                    r = P.prrtc_plan(
                        p["start"], np.asarray([p["goal"]]), collision_context=dict(ctx, world_boxes=p["boxes"]),
                        constraints=[B.tsr_constraint(robot, p["tsr"])] if constrained else [],
                        projection=P.ProjectionSettings(mode="cprrtc"), shortcut_rounds=sr,
                        max_iterations=2000, step_size=1.0, num_new_samples=64, granularity=16,
                        max_nodes=100_000, dynamic_domain=False, min_vals=lo, max_vals=hi,
                        collision_checker=checker)
                    assert r.solved, (checker, constrained, sr)
                    worst = min(worst, min_clearance(r.path, p["boxes"]))
                label = f"{checker}, {'constrained' if constrained else 'unconstrained'}, shortcut_rounds={sr}"
                assert worst >= -1e-3, f"{label}: clips through by {-worst:.4f} m"
                print(f"{label}: min clearance at 1/64 rad {worst:.4f} m")


if __name__ == "__main__":
    main()
