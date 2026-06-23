# P-RRAX: A reimplementation of pRRTC through the Jax FFI

`P-RRAX` is a CUDA-accelerated implementation of the Parallel Rapidly-exploring Random Tree (pRRTC) motion planning algorithm, designed for integration with PyRoFFI's robotics kinematics and collision checking framework. See the original pRRTC paper here: [pRRTC paper](https://arxiv.org/abs/2503.06757), and the original code at [pRRTC code](https://github.com/CoMMALab/pRRTC.git).

## Key Features

- **GPU-Accelerated**: Uses CUDA kernels for parallelism
- **JAX FFI Integration**: Exposes CUDA kernels through JAX's Foreign Function Interface
- **Batched Planning**: Supports `jax.vmap` for batched motion planning
- **Two-Tree Bidirectional Planning**: Uses start and goal trees for efficient search
- **Low-Discrepancy Sampling**: Halton sequence for better configuration space coverage
- **CUDA Graph Support**: Minimal kernel launch overhead via graph replay
- **Memory-Efficient**: Structure-of-Arrays (SoA) layout for coalesced memory access

## Setup

### PyRoFFI Dependency

Install PyRoFFI
```
git clone https://github.com/commalab/pyroffi
cd pyroffi
pip install -e .
pip install -r requirements.txt
```

### Prerequisites

- CUDA toolkit (11.0+)
- Python 3.11+
- PyRoFFI 

### Build

```bash
cd cuda-rrtc
bash build.sh
```

For debug builds:

```bash
bash build.sh --debug
```

This compiles the CUDA kernels into `_prrtc_planner_lib.so` in the current directory.

## Usage

### Basic Planning

```python
import jax.numpy as jnp
from cuda_rrtc.jax import prrtc_plan

# Define start and goal configurations
start = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
goals = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

# Plan path
result = prrtc_plan(
    start_config=start,
    goal_configs=goals,
    allow_unsafe_no_collision=True,  # quick geometric-only demo
    max_iterations=10000,
    step_size=0.5
)

if result.solved:
    print(f"Found path with {len(result.path)} configurations")
else:
    print("Planning failed")
```

### Collision-Aware Planning (PyRoFFI Tensors)

`prrtc_plan` is collision-aware by default and expects a `collision_context`
dictionary. CUDA-side FK + collision checks are applied during both extend
and connect.

Required keys:

- `fk_twists`: `(n_joints, 6)` float32
- `fk_parent_tf`: `(n_joints, 7)` float32, `[w, x, y, z, tx, ty, tz]`
- `fk_parent_idx`: `(n_joints,)` int32
- `fk_act_idx`: `(n_joints,)` int32
- `fk_mimic_mul`: `(n_joints,)` float32
- `fk_mimic_off`: `(n_joints,)` float32
- `fk_mimic_act_idx`: `(n_joints,)` int32
- `fk_topo_inv`: `(n_joints,)` int32
- `sphere_link_idx`: `(n_robot_spheres,)` int32
- `sphere_local`: `(n_robot_spheres, 3)` float32
- `sphere_radius`: `(n_robot_spheres,)` float32
- `world_spheres`: `(n_world_spheres, 4)` float32 (optional)

Additional optional world geometry keys:

- `world_capsules`: `(n_world_capsules, 7)` float32
- `world_boxes`: `(n_world_boxes, 15)` float32
- `world_halfspaces`: `(n_world_halfspaces, 6)` float32

Optional keys:

- `self_pairs`: `(n_pairs, 2)` int32 for active robot-sphere self-collision pairs


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

The implementation consists of modular CUDA kernels:

1. **prrtc_nearest_neighbor.cu**: Parallel nearest neighbor search with warp-level reductions
2. **prrtc_extend.cu**: Tree extension with step-size limiting
3. **prrtc_iteration.cu**: Single iteration of the RRTC algorithm
4. **prrtc_planner.cu**: Main planner with complete planning loop

### Memory Layout

The tree uses a Structure-of-Arrays (SoA) layout for optimal memory coalescing:

```c
float* tree_configs;   // [dim, max_nodes]
int* parent_indices;   // [max_nodes]
```

### JAX FFI Integration

The planner is exposed to JAX via a single FFI primitive:

```python
result = prrtc_plan(
    start_config,
    goal_configs,
    allow_unsafe_no_collision=True,
    max_iterations=10000,
    step_size=0.5,
)
```
