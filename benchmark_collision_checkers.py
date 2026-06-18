#!/usr/bin/env python3
"""
Benchmark: P-RRAX solve time with the binary vs. differentiable CUDA collision checker.

Both checkers run inside the same pRRTC planner kernel during edge validation;
they differ only in how a configuration's collision status is computed:

  * binary         — pRRTC-style early-exit boolean check. Returns at the first
                     penetrating obstacle, so it does the least work per config.
                     (mirrors pyroffi's ``collision_binary`` kernel)
  * differentiable — full smooth signed-distance-field sweep with no early exit:
                     every (robot sphere, obstacle) pair contributes a
                     ``colldist_from_sdf`` margin penalty, and a config is in
                     collision when the aggregate cost is negative.
                     (mirrors pyroffi's differentiable ``_collision_cuda_kernel``)

For each checker we run a warmup solve (pays JIT compile + FFI registration),
then ``--trials`` timed solves on the *same* start/goal/problem, and report wall
time (host dispatch + GPU sync) and the pure GPU planner-kernel time.

Usage:
    python benchmark_collision_checkers.py
    python benchmark_collision_checkers.py --vamp-problem bookshelf_tall --vamp-index 1 \
        --trials 30 --max-iterations 5000 --collision-margin 0.02
"""

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

try:
    import pyroffi as pk
    import yourdfpy
    from pyroffi.collision._obstacles import create_collision_environment
    from pyroffi.collision._robot_collision import RobotCollisionSpherized
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install pyroffi and yourdfpy")
    sys.exit(1)

# --- Load the cuda-rrtc JAX wrappers directly from source (mirrors test_prrtc.py) ---
try:
    _here = Path(__file__).resolve().parent
    prrtc_impl = _here / "cuda-rrtc" / "jax" / "prrtc.py"
    spec = importlib.util.spec_from_file_location("cuda_rrtc_prrtc", prrtc_impl)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {prrtc_impl}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prrtc_plan = mod.prrtc_plan

    utils_impl = _here / "cuda-rrtc" / "jax" / "utils.py"
    utils_spec = importlib.util.spec_from_file_location("cuda_rrtc_utils", utils_impl)
    if utils_spec is None or utils_spec.loader is None:
        raise ImportError(f"Failed to load module spec from {utils_impl}")
    utils_mod = importlib.util.module_from_spec(utils_spec)
    utils_spec.loader.exec_module(utils_mod)
    load_vamp_problem = utils_mod.load_vamp_problem
    build_prrtc_collision_context = utils_mod.build_prrtc_collision_context
    config_collision_report = utils_mod.config_collision_report
    timed_prrtc_solve = utils_mod.timed_prrtc_solve
except Exception as e:
    print(f"cuda-rrtc import error: {e}")
    print("Make sure the library is compiled: cd cuda-rrtc && bash build.sh")
    sys.exit(1)


# Resources ship inside the checked-out pyroffi alongside this repo.
RESOURCE_ROOT = Path(__file__).resolve().parent / "pyroffi" / "resources"
PANDA_URDF = RESOURCE_ROOT / "panda" / "panda_spherized.urdf"

# binary            — flat single-tier early-exit sweep (baseline)
# binary_coarse     — Phase A: coarse→fine hierarchy + per-link/self culling (per-thread)
# binary_coarse_coop— Phase B: + source-pRRTC cooperative lane-groups (shared FK, warp early-out)
# differentiable    — full smooth signed-distance sweep (context)
CHECKERS = ("binary", "binary_coarse", "binary_coarse_coop", "differentiable")


def summarize(label, wall_ms, kernel_ms, results):
    """Print a one-line-per-stat summary for one collision checker."""
    solved = sum(1 for r in results if r.solved)
    iters = [r.iterations for r in results]
    costs = [r.cost for r in results if r.solved and np.isfinite(r.cost)]

    def stats(xs):
        if not xs:
            return "n/a"
        return (
            f"mean={statistics.mean(xs):8.3f}  median={statistics.median(xs):8.3f}  "
            f"min={min(xs):8.3f}  max={max(xs):8.3f}"
        )

    iters_mean = statistics.mean(iters)
    kernel_mean = statistics.mean(kernel_ms) if kernel_ms else float("nan")
    # Per-iteration kernel cost is the clean collision-check metric: the planner is a
    # concurrent race so total iteration counts differ between checkers, but the cost of
    # one sample-loop pass (dominated by the collision check) is directly comparable.
    us_per_iter = (kernel_mean * 1000.0 / iters_mean) if iters_mean else float("nan")

    print(f"\n[{label}]  collision checker")
    print(f"  solved        : {solved}/{len(results)}")
    print(f"  wall time ms  : {stats(wall_ms)}")
    print(f"  kernel ms     : {stats(kernel_ms)}")
    print(f"  iterations    : mean={iters_mean:.1f}  min={min(iters)}  max={max(iters)}")
    print(f"  kernel us/iter: {us_per_iter:.2f}")
    if costs:
        print(f"  path cost     : mean={statistics.mean(costs):.4f}")
    return {
        "solved": solved,
        "wall_mean": statistics.mean(wall_ms) if wall_ms else float("nan"),
        "kernel_mean": kernel_mean,
        "iters_mean": iters_mean,
        "us_per_iter": us_per_iter,
    }


def run_checker(checker, plan_kwargs, trials, collision_margin):
    """Warm up, then run ``trials`` timed single-problem solves for one checker."""
    kwargs = dict(plan_kwargs)
    kwargs["collision_checker"] = checker
    kwargs["collision_margin"] = collision_margin

    # Warmup pays JIT compile + FFI registration; not counted in timings.
    _ = prrtc_plan(**kwargs)

    wall_ms, kernel_ms, results = [], [], []
    for _ in range(trials):
        result, elapsed_ms = timed_prrtc_solve(prrtc_plan, **kwargs)
        wall_ms.append(elapsed_ms)
        if result.kernel_time_ms is not None:
            kernel_ms.append(float(result.kernel_time_ms))
        results.append(result)
    return wall_ms, kernel_ms, results


def main():
    parser = argparse.ArgumentParser(
        description="Compare P-RRAX solve time: binary vs. differentiable collision checker"
    )
    parser.add_argument("--vamp-problem", default="bookshelf_tall", help="VAMP problem name")
    parser.add_argument("--vamp-index", type=int, default=1, help="VAMP problem index")
    parser.add_argument("--trials", type=int, default=25, help="Timed solves per checker")
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--num-new-samples", type=int, default=64)
    parser.add_argument(
        "--collision-margin",
        type=float,
        default=0.005,
        help="Safety margin (m) for the differentiable checker. Must stay below "
             "the start/goal clearance or the roots themselves get rejected.",
    )
    parser.add_argument("--no-jit-trace", action="store_false", dest="jit_trace", default=True)
    args = parser.parse_args()

    print("=" * 74)
    print("P-RRAX collision-checker benchmark — binary vs. differentiable")
    print("=" * 74)

    if not PANDA_URDF.exists():
        print(f"ERROR: URDF not found at {PANDA_URDF}")
        sys.exit(1)

    urdf = yourdfpy.URDF.load(str(PANDA_URDF))
    robot = pk.Robot.from_urdf(urdf)
    srdf_path = str(RESOURCE_ROOT / "panda" / "panda.srdf")
    robot_coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=srdf_path)
    n_act = robot.joints.num_actuated_joints

    vamp_problem = load_vamp_problem(
        RESOURCE_ROOT, problem=args.vamp_problem, index=args.vamp_index
    )
    if vamp_problem is None or "start" not in vamp_problem or "goals" not in vamp_problem:
        print(f"ERROR: VAMP problem {args.vamp_problem}[{args.vamp_index}] not usable")
        sys.exit(1)

    obstacles = create_collision_environment(vamp_problem)
    collision_context = build_prrtc_collision_context(robot, robot_coll, obstacles)

    n_fine = int(np.asarray(collision_context["sphere_radius"]).shape[0])
    n_coarse = int(np.asarray(collision_context["coarse_sphere_link_idx"]).shape[0])
    n_self = int(np.asarray(collision_context["self_pairs"]).shape[0])
    n_coarse_self = int(np.asarray(collision_context["coarse_self_pairs"]).shape[0])
    print(
        f"\nCoarse→fine model: world {n_fine}→{n_coarse} spheres "
        f"({n_fine / max(n_coarse, 1):.1f}x fewer), self {n_self}→{n_coarse_self} pairs "
        f"({n_self / max(n_coarse_self, 1):.1f}x fewer) on the coarse pre-pass."
    )

    lo = np.array(robot.joints.lower_limits)
    hi = np.array(robot.joints.upper_limits)
    start_config = jnp.array(vamp_problem["start"], dtype=jnp.float32)
    goal_config = jnp.array(vamp_problem["goals"][0], dtype=jnp.float32)

    start_report = config_collision_report(robot, robot_coll, start_config, obstacles)
    goal_report = config_collision_report(robot, robot_coll, goal_config, obstacles)
    print(
        f"\nProblem: {args.vamp_problem}[{args.vamp_index}]  "
        f"({len(obstacles)} obstacles, {n_act} joints)"
    )
    print(
        f"  start margin={start_report['min_margin']:.5f}  "
        f"goal margin={goal_report['min_margin']:.5f}"
    )
    if not (start_report["collision_free"] and goal_report["collision_free"]):
        print(
            "  WARNING: start/goal roots are not collision-free; the planner does "
            "not auto-repair roots, so solves may fail. Pick another --vamp-index."
        )

    # The differentiable checker inflates obstacles by collision_margin, so a
    # margin above the root clearance makes the start/goal themselves invalid and
    # the differentiable solve will (correctly) fail. Flag that up front.
    root_clearance = min(start_report["min_margin"], goal_report["min_margin"])
    if args.collision_margin >= root_clearance:
        print(
            f"  WARNING: collision_margin={args.collision_margin} >= root clearance "
            f"{root_clearance:.5f}. The differentiable checker will reject the "
            f"start/goal roots and fail to solve. Use a smaller --collision-margin."
        )

    print(
        f"\nConfig: trials={args.trials}  max_iterations={args.max_iterations}  "
        f"step_size={args.step_size}  num_new_samples={args.num_new_samples}  "
        f"collision_margin={args.collision_margin}"
    )

    plan_kwargs = dict(
        start_config=start_config,
        goal_configs=goal_config.reshape(1, -1),
        max_iterations=args.max_iterations,
        step_size=args.step_size,
        num_new_samples=args.num_new_samples,
        dynamic_domain=False,
        min_vals=jnp.array(lo, dtype=jnp.float32),
        max_vals=jnp.array(hi, dtype=jnp.float32),
        collision_context=collision_context,
        jit_trace=args.jit_trace,
    )

    summaries = {}
    for checker in CHECKERS:
        wall_ms, kernel_ms, results = run_checker(
            checker, plan_kwargs, args.trials, args.collision_margin
        )
        summaries[checker] = summarize(checker, wall_ms, kernel_ms, results)

    # --- Head-to-head ---
    # Compare per-iteration kernel cost (the per-config collision-check cost). Total time
    # is not comparable across checkers because the concurrent planner solves in a
    # different number of iterations depending on which block wins the race.
    print("\n" + "=" * 74)
    print("Per-iteration kernel cost (collision-check speed; lower is better)")
    print("=" * 74)
    base = summaries["binary"]["us_per_iter"]
    for checker in CHECKERS:
        s = summaries[checker]
        u = s["us_per_iter"]
        rel = (base / u) if (u and np.isfinite(u)) else float("nan")
        tag = "  (baseline)" if checker == "binary" else f"  ({rel:.2f}x vs binary)"
        print(f"  {checker:14s}: {u:7.2f} us/iter{tag}")

    ca = summaries["binary_coarse"]["us_per_iter"]
    cb = summaries["binary_coarse_coop"]["us_per_iter"]
    if base and np.isfinite(base):
        if ca and np.isfinite(ca):
            print(f"\nPhase A (coarse→fine vs flat binary):        {base / ca:.2f}x per check.")
        if cb and np.isfinite(cb):
            print(f"Phase B (coarse+cooperative vs flat binary): {base / cb:.2f}x per check.")
        if ca and cb and np.isfinite(ca) and np.isfinite(cb):
            print(f"Phase B vs Phase A:                          {ca / cb:.2f}x.")
    print("\nNote: iteration counts differ by checker due to the parallel race; that is "
          "expected and is why per-iteration cost is the comparison metric.")
    print("\nDone.")


if __name__ == "__main__":
    main()
