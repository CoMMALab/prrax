"""Unconstrained pRRTC on VAMP's bookshelf problem: single and batched solves are found and
collision-free at their waypoints under pyroffi's own model.

    pytest tests/test_prrtc.py
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from pyroffi.collision._obstacles import create_collision_environment

import benchmark_constrained as B
from prrax.jax import prrtc_plan, prrtc_plan_batch
from prrax.jax.utils import build_prrtc_collision_context, load_vamp_problem


def test_prrtc() -> None:
    robot, coll = B.load_panda()
    problem = load_vamp_problem(B.PANDA.parent, problem="bookshelf_tall", index=1)
    obstacles = create_collision_environment(problem)
    ctx = build_prrtc_collision_context(robot, coll, obstacles)
    lo = jnp.asarray(robot.joints.lower_limits, jnp.float32)
    hi = jnp.asarray(robot.joints.upper_limits, jnp.float32)
    start = np.asarray(problem["start"], np.float32)
    goal = np.asarray(problem["goals"][0], np.float32)
    kw = dict(max_iterations=5000, step_size=0.5, num_new_samples=64, dynamic_domain=False,
              min_vals=lo, max_vals=hi)

    def collision_free(path) -> bool:
        q = jnp.asarray(path, jnp.float32)
        d = min(float(np.min(coll.compute_self_collision_distance(robot, x))) for x in q)
        for obs in obstacles:
            d = min(d, min(float(np.min(coll.compute_world_collision_distance(robot, x, obs))) for x in q))
        return d > -1e-3   # spherized model: VAMP-valid roots can overlap by < 1 mm

    r = prrtc_plan(start, goal[None], collision_context=ctx, **kw)
    assert r.solved and collision_free(r.path)
    assert np.allclose(r.path[0], start, atol=1e-4) and np.allclose(r.path[-1], goal, atol=1e-4)

    rs = prrtc_plan_batch(np.stack([start] * 4), np.stack([goal[None]] * 4),
                          collision_context=[ctx] * 4, **kw)
    assert all(r.solved and collision_free(r.path) for r in rs)
