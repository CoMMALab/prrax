"""McVAMP alongside the prrax demos, when McVAMP's Python module is installed (`pip install .` in
mcvamp/vamp; plain VAMP has no constraints and is ignored). Plans like McVAMP's own scripts
(benchmark_common.run_suite, viser_dynamic_obstacle_avoid.py): rrtc with constraints, a fresh
Halton sampler per problem, seeds projected first, then simplify. Runs on one CPU core.
"""

from __future__ import annotations

import numpy as np

try:
    import vamp
    AVAILABLE = hasattr(vamp, "ConstraintSettings")
except ImportError:
    vamp, AVAILABLE = None, False


class McVamp:
    """knobs: ConstraintSettings fields for planning (None: McVAMP's defaults)."""

    def __init__(self, range_: float, max_iterations: int, seed_iterations: int, knobs: dict | None = None):
        self.module, self.plan, self.plan_settings, self.simp_settings = (
            vamp.configure_robot_and_planner_with_kwargs("panda", "rrtc"))
        self.plan_settings.range = range_
        self.plan_settings.dynamic_domain = False
        self.plan_settings.max_iterations = max_iterations
        self.settings = vamp.ConstraintSettings()
        for k, v in (knobs or {}).items():
            setattr(self.settings, k, getattr(vamp.ProjMethod, v) if k == "method" else v)
        self.seed_settings = vamp.ConstraintSettings()
        self.seed_settings.method = self.settings.method
        self.seed_settings.descend_rate = self.settings.descend_rate
        self.seed_settings.max_iterations = seed_iterations

    def solve(self, start, goal, cuboids, tsr, spheres=(), attachment=None, simplify_min=0) -> dict:
        """cuboids [(center, half extents)], tsr (reference, pose, lower, upper), spheres [(center, r)],
        attachment [(position in the hand frame, radius)]. Returns status, and when solved plan_ms,
        simplify_ms, iterations, cost (of the simplified path) and path."""
        m = self.module
        env = vamp.Environment()
        for c, h in cuboids:
            env.add_cuboid(vamp.Cuboid(list(map(float, c)), [0.0, 0.0, 0.0], list(map(float, h))))
        for c, r in spheres:
            env.add_sphere(vamp.Sphere(list(map(float, c)), float(r)))
        if attachment:
            a = vamp.Attachment(np.identity(4), 0)
            a.add_spheres([vamp.Sphere(list(p), r) for p, r in attachment])
            env.attach(a)
        cons = [m.TaskSpaceConstraint(*[list(map(float, x)) for x in tsr])]
        sampler = m.halton()
        try:
            s = m.project(np.asarray(start, np.float32), cons, self.seed_settings)
            g = m.project(np.asarray(goal, np.float32), cons, self.seed_settings)
        except ValueError:
            return dict(status="not planned (seed projection failed)")
        if not (m.validate(s, env) and m.validate(g, env)):
            return dict(status="not planned (seed in collision)")
        r = self.plan(s, g, env, self.plan_settings, sampler, constraints=cons, constraint_settings=self.settings)
        if not r.solved:
            return dict(status=f"no path after {r.iterations} iterations", plan_ms=r.nanoseconds / 1e6)
        simple, simp_ms = r, 0.0
        if len(r.path) > simplify_min:
            simple = m.simplify(r.path, env, self.simp_settings, sampler, constraints=cons,
                                constraint_settings=self.settings)
            simp_ms = simple.nanoseconds / 1e6
        return dict(status="solved", plan_ms=r.nanoseconds / 1e6, simplify_ms=simp_ms,
                    iterations=int(r.iterations), cost=float(simple.path.cost()), path=simple.path.numpy())
