# P-RRAX: A reimplementation of pRRTC through the Jax FFI

`P-RRAX` is a CUDA-accelerated implementation of the Parallel Rapidly-exploring Random Tree (pRRTC) motion planning algorithm, designed for integration with PyRoFFI's robotics kinematics and collision checking framework. See the original pRRTC paper here: [pRRTC paper](https://arxiv.org/abs/2503.06757), and the original code at [pRRTC code](https://github.com/CoMMALab/pRRTC.git).

## Key Features

- **GPU-Accelerated**: one CUDA block per sample, one thread per edge waypoint
- **Trace-compiled per robot**: built through [GRiM](https://github.com/commalab/GRiM) with
  cricket2's straight-line FK and the robot's sphere model baked in as compile-time tables
- **Manifold-constrained planning**: Task Space Region constraints (several at once), with
  McVAMP's two-stage particle projection or cpRRTC's smoothed parallel projection
- **Batched Planning**: a batch of problems, each with its own world and constraints, in one call
- **Two-Tree Bidirectional Planning**: Uses start and goal trees for efficient search
- **Low-Discrepancy Sampling**: Halton sequence for better configuration space coverage
- **Memory-Efficient**: Structure-of-Arrays (SoA) layout for coalesced memory access

## Setup

prrax depends on [pyroffi-grim](../pyroffi-grim) (pyroffi with GRiM, its CUDA motion kernels,
and the cricket2 trace compiler), checked out next to this repository and linked as `pyroffi`:

```bash
conda activate prrax
pip install -e ../pyroffi-grim
bash ../pyroffi-grim/build_kernels/build_cricket.sh     # installs cricket2 (GRiM's trace compiler)
ln -sfn ../pyroffi-grim pyroffi
```

Tested against pyroffi-grim `2b2e916` / GRiM `cc13482`. The build goes through GRiM's private
`grim.motion._build` module (`cuda-rrtc/jax/_grim_build.py`), so pin those commits.

Prerequisites: CUDA toolkit (`nvcc`), Python 3.11+, jax/jaxlib with CUDA.

There is no separate build step: the planner is compiled per robot collision model (and per set of
constrained frames) on first use, cached on disk by GRiM (`~/.cache/grim/motion`, or
`GRIM_MOTION_CACHE`), and checked against pyroffi's FK when it loads. The first solve on a new robot
takes ~20 s; later processes load the cached build.

## Usage

### Collision-Aware Planning

```python
import numpy as np
import yourdfpy, pyroffi as pk
from pyroffi.collision._robot_collision import RobotCollisionSpherized
from cuda_rrtc.jax import prrtc_plan
from cuda_rrtc.jax.utils import build_prrtc_collision_context

urdf = yourdfpy.URDF.load("panda_spherized.urdf")
robot = pk.Robot.from_urdf(urdf)
coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path="panda.srdf")
ctx = build_prrtc_collision_context(robot, coll, obstacles)   # pyroffi Sphere/Capsule/Box/HalfSpace

result = prrtc_plan(start, goals, collision_context=ctx, step_size=0.5)
if result.solved:
    print(f"Found path with {len(result.path)} configurations")
```

The context's robot and sphere model (`sphere_*`, `self_pairs`, coarse groups) select the build;
its world geometry (`world_spheres` `(N, 4)`, `world_capsules` `(N, 7)`, `world_boxes` `(N, 15)`,
`world_halfspaces` `(N, 6)`) is a per-call input, so `dict(ctx, world_boxes=...)` plans in a new
world without rebuilding. `attached_spheres=(joint_frame, spheres)` adds a held object.

`prrtc_plan_batch(starts, goals, collision_context=[ctx_0, ...])` solves a batch, every problem
with its own world, on its own CUDA stream.

### Constrained Planning

```python
from cuda_rrtc.jax.prrtc import make_tsr_constraint, ProjectionSettings

# End effector held in the plane z = 0.24 (x, y free), orientation free.
plane = make_tsr_constraint(robot, "panda_grasptarget",
                            reference=[0, 1, 0, 0, 0.35, 0.65, 0.24],     # wTr [wxyz xyz]
                            lower=[-10, -10, -0.0025, -1000, -1000, -1000],
                            upper=[10, 10, 0.0025, 1000, 1000, 1000])
result = prrtc_plan(start, goals, collision_context=ctx, constraints=[plane], step_size=1.0,
                    projection=ProjectionSettings(mode="mcvamp"))     # or mode="cprrtc"
```

A TSR constraint (McVAMP's `TaskSpaceConstraint`) bounds `x = [t; log R]` of
`reference^-1 * T_ee * offset^-1`. Up to 4 constraints are satisfied together by cyclic projection.
Every edge is projected onto their manifold before its collision check and all of its projected
waypoints become tree nodes, so the returned path lies on the manifold at waypoint resolution
(`step_size / granularity`). Starts and goals off the manifold are projected onto it first.
`constraint_error(ctx, constraints, q)` evaluates (and projects) configurations. `projection.mode`:

- `"mcvamp"` (McVAMP, Iyer et al. 2026): every thread projects a perturbation of the steered
  configuration and the first to converge is the edge's end; the edge's waypoints are then
  projected independently and their gaps checked.
- `"cprrtc"` (cpRRTC, Hu et al. 2025): the edge's waypoints are projected jointly with a smoothing
  term, freezing the converged prefix; the end is free.

`benchmark_constrained.py` runs McVAMP's Panda problem sets (plane, line, maze) and
`compare_mcvamp.py` compares against McVAMP's own results; `test_constrained.py` checks the
constraint error, projection and paths against independent references.

### Asymptotically Optimal Planning (AO-x)

`ao_passes=k` (in `prrtc_plan` and `prrtc_plan_batch`, with or without constraints) adds k
optimizing passes after the first solution. The design follows the AO-x layer of MR. POP and
pAORRTC (Hauser & Zhou's AO-x meta-algorithm). Each pass replans from scratch for a path
cheaper than the best so far:
- samples come from the informed set (the prolate hyperspheroid between start and goal);
- each sample gets a random cost budget;
- nearest-neighbour search and connect only consider nodes that stay within that budget;
- candidates that cannot beat the best path are pruned.

The cheapest path is kept, and `kernel_time_ms` and `iterations` are summed over the passes.
`shortcut_rounds=r` runs r rounds of MR. POP's path simplifier on the GPU before the path is
returned. With `ao_passes`, each pass simplifies its path, so the next pass is bounded by the
simplified cost. Each round has three steps:
- try a direct shortcut from start to goal;
- find the shortest path through the shortcut graph between up to 64 evenly spaced anchor
  waypoints, checking all anchor pairs in parallel across the problem's blocks;
- run 3 sweeps of B-spline smoothing.

Every change is checked with the planner's own connect, using projected steps when the problem
is constrained.

`test_ao.py` checks both features.

### GPU Point-Cloud Planning (RoboGPU / OptiX)

For dense environment point clouds (e.g. RealSense captures), the in-kernel
analytic checker is impractical. `prrtc_plan_robogpu` instead validates edges
on the GPU's ray-tracing cores using PyRoFFI's
[`RoboGPUCollisionChecker`](https://github.com/commalab/pyroffi) (OptiX BVH over
the point cloud).

Because OptiX `optixTrace` cannot be called from inside an ordinary CUDA kernel,
this planner is **host-driven**: it runs batched RRT-Connect on the host and
validates each iteration's candidate edges with one batched robogpu call. This
trades the single-kernel-launch speed of `prrtc_plan` for true point-cloud
collision checking. (`prrtc_plan(..., collision_checker="robogpu")` is *not*
supported and raises an error pointing here.)

```python
import jax.numpy as jnp
import yourdfpy, pyroffi as pk
from pyroffi.collision import RobotCollisionSpherized, RoboGPUCollisionChecker
from cuda_rrtc.jax import prrtc_plan_robogpu

urdf  = yourdfpy.URDF.load("panda.urdf")
robot = pk.Robot.from_urdf(urdf)
coll  = RobotCollisionSpherized.from_urdf(urdf, srdf_path="panda.srdf")
checker = RoboGPUCollisionChecker(coll)        # build robogpu OptiX checker

result = prrtc_plan_robogpu(
    start_config=start,                        # (dim,)
    goal_configs=goal.reshape(1, -1),          # (num_goals, dim)
    robot=robot,
    robogpu_checker=checker,
    world_geom=world_geom,                     # optional analytic obstacles
    point_cloud=pc,                            # (Mp, 3) float32 environment cloud
    r_env=0.02,                                # env point sphere radius
    step_size=0.5,
    num_new_samples=128,
)
print(result.solved, result.cost)
```

Requires PyRoFFI's robogpu library to be built (`bash
build_kernels/build_robogpu_collision.sh`, needs the NVIDIA OptiX SDK 7.x).


## Architecture

### CUDA Kernels

- **CUDA/prrtc_planner.cu**: the planner (sample, nearest neighbor, extend, connect) and its FFI
  handler, plus FK / constraint entry points used to validate a build. A call is three launches:
  batch setup (roots, root projection), one persistent planning kernel for the whole batch, and
  path tracing. In the planning kernel every block works on one problem and, once it is solved,
  moves to the unsolved problem with the fewest blocks, so a batch's hardest problems get the
  whole GPU. Each edge waypoint is handled by 4 threads (as in pRRTC): they split its collision
  check, and all threads are steer particles during projection. Inputs arrive packed in one buffer.
- **CUDA/prrtc_constraint.cuh**: TSR error, cyclic LM projection, and the constrained extend step.
  The linear algebra and SO(3) math are GLASS thread-tier routines (`gemm`, `gemv`, `posv` via
  GRiM's `tier_posv`, `so3_left_jacobian_inv`, `axpy`/`nrm2`/`dot`) rather than hand-rolled loops.
- **jax/_grim_build.py**: the per-robot build: GRiM's robot header (traced FK, Jacobians) plus
  `prrax_gen.cuh` (the baked sphere model).

### Memory Layout

The tree uses a Structure-of-Arrays (SoA) layout for optimal memory coalescing:

```c
float* tree_configs;   // [dim, max_nodes]
int* parent_indices;   // [max_nodes]
```
