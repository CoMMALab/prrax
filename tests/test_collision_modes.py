"""Every binary collision mode (all on prrax's mantis-traced fused frames) gives the same verdict
as the flat binary check, on random configurations of the bookshelf problem.

    pytest tests/test_collision_modes.py
"""

from __future__ import annotations

import numpy as np
from pyroffi.collision._obstacles import create_collision_environment

import benchmark_constrained as B
from prrax.jax.prrtc import config_in_collision
from prrax.jax.utils import build_prrtc_collision_context, load_vamp_problem


def test_collision_modes_agree() -> None:
    robot, coll = B.load_panda()
    problem = load_vamp_problem(B.PANDA.parent, problem="bookshelf_tall", index=1)
    ctx = build_prrtc_collision_context(robot, coll, create_collision_environment(problem))
    lo, hi = np.asarray(robot.joints.lower_limits), np.asarray(robot.joints.upper_limits)
    q = np.random.default_rng(0).uniform(lo, hi, (4096, lo.size)).astype(np.float32)
    ref = config_in_collision(ctx, q, collision_checker="binary")
    assert 0.05 < ref.mean() < 0.95          # the sample hits both verdicts
    for mode in ("binary_coarse", "binary_coarse_coop"):
        assert (config_in_collision(ctx, q, collision_checker=mode) == ref).all(), mode
