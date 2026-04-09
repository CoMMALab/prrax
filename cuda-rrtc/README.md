# cuda-rrtc: GPU-Accelerated Parallel RRTC Motion Planner

`cuda-rrtc` is a CUDA-accelerated implementation of the Parallel Rapidly-exploring Random Tree (pRRTC) motion planning algorithm, designed for integration with PyRoNot's robotics kinematics and collision checking framework.

## Directory Structure

```
cuda-rrtc/
├── CUDA/                    # CUDA kernels
│   ├── prrtc_helpers.cuh    # CUDA utilities and helpers
│   ├── prrtc_nearest_neighbor.cu
│   ├── prrtc_extend.cu
│   ├── prrtc_iteration.cu
│   └── prrtc_planner.cu
├── ffi/                     # FFI bindings
│   └── prrtc_ffi.cc
├── jax/                     # JAX interface
│   ├── __init__.py
│   └── prrtc.py
├── build.sh                 # Build script
├── __init__.py
└── README.md
```

## Key Features

- **GPU-Accelerated**: Uses CUDA kernels for massive parallelism
- **JAX FFI Integration**: Exposes CUDA kernels through JAX's Foreign Function Interface
- **Batched Planning**: Supports `jax.vmap` for batched motion planning
- **Two-Tree Bidirectional Planning**: Uses start and goal trees for efficient search
- **Low-Discrepancy Sampling**: Halton sequence for better configuration space coverage
- **CUDA Graph Support**: Minimal kernel launch overhead via graph replay
- **Memory-Efficient**: Structure-of-Arrays (SoA) layout for coalesced memory access

## Installation

### Prerequisites

- CUDA toolkit (11.0+)
- JAXLib >= 0.4.14
- Python 3.8+

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
import jax
import jax.numpy as jnp
from cuda_rrtc.jax import prrtc_plan

# Define start and goal configurations
start = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
goals = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

# Plan path
result = prrtc_plan(
    start_config=start,
    goal_configs=goals,
    max_iterations=10000,
    step_size=0.5
)

if result.solved:
    print(f"Found path with {len(result.path)} configurations")
else:
    print("Planning failed")
```

### Collision-Aware Planning (PyRoNot Tensors)

`prrtc_plan` is now collision-aware by default and expects a `collision_context`
dictionary. CUDA-side FK + sphere collision checks are applied during both extend
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
- `world_spheres`: `(n_world_spheres, 4)` float32

Optional keys:

- `self_pairs`: `(n_pairs, 2)` int32 for active robot-sphere self-collision pairs

To force geometric-only legacy behavior, set `allow_unsafe_no_collision=True`.

### Batched Planning with vmap

```python
import jax

# Generate batch of planning problems
starts = jax.random.uniform(key, (10, 7), minval=-2.0, maxval=2.0)
goals = jax.random.uniform(key, (10, 7), minval=-2.0, maxval=2.0)

# Vectorize planning across batch dimension
plan_fn = jax.jit(jax.vmap(prrtc_plan, in_axes=(0, 0, None, None)))
results = plan_fn(starts, goals, 10000, 0.5)

# Process results
for i, result in enumerate(results):
    if result.solved:
        print(f"Problem {i}: Found path with {len(result.path)} configurations")
```

### Using Lower-Level Primitives

```python
from cuda_rrtc.jax import prrtc_nearest_neighbor, prrtc_extend

# Find nearest neighbor
tree_configs = ...  # Shape: (dim, max_nodes)
query_configs = ...  # Shape: (batch, dim)
tree_size = ...  # Current number of nodes

distances, indices = prrtc_nearest_neighbor(tree_configs, query_configs, tree_size)

# Extend tree toward samples
nearest_indices = ...  # Shape: (batch,)
samples = ...  # Shape: (batch, dim)

new_configs, parent_indices, valid_flags = prrtc_extend(
    tree_configs, nearest_indices, samples, step_size=0.5
)
```

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
    max_iterations=10000,
    step_size=0.5,
)
```

JAX auto-vectorization via `vmap` is fully supported.

## Performance Considerations

- **Tree Size**: The planner supports trees with up to 1,000,000 nodes by default
- **Batch Size**: Use larger batch sizes (64-256) for better GPU utilization
- **Dimensionality**: Optimal for 7-14 DOF manipulators
- **Step Size**: Typical values are 0.5-2.0 meters/radians depending on configuration space scale

## API Reference

### `prrtc_plan`

```python
def prrtc_plan(
    start_config: Float[Array, "*batch dim"],
    goal_configs: Float[Array, "num_goals dim"],
    max_iterations: int = 10000,
    step_size: float = 0.5,
    num_new_samples: int = 64,
    min_vals: Optional[Float[Array, "dim"]] = None,
    max_vals: Optional[Float[Array, "dim"]] = None,
) -> PRRTCResult:
```

### `PRRTCResult` NamedTuple

```python
class PRRTCResult(NamedTuple):
    solved: bool
    path: Optional[Array]
    tree_a_size: int
    tree_b_size: int
    iterations: int
    cost: float
```

### `prrtc_nearest_neighbor`

```python
def prrtc_nearest_neighbor(
    tree_configs: Float[Array, "dim max_nodes"],
    query_configs: Float[Array, "batch dim"],
    tree_size: int,
) -> tuple[Float[Array, "batch"], Int[Array, "batch"]]:
```

### `prrtc_extend`

```python
def prrtc_extend(
    tree_configs: Float[Array, "dim max_nodes"],
    nearest_indices: Int[Array, "batch"],
    samples: Float[Array, "batch dim"],
    step_size: float,
) -> tuple[Float[Array, "batch dim"], Int[Array, "batch"], Int[Array, "batch"]]:
```

## Integration with PyRoNot

The implementation reuses PyRoNot's CUDA primitives for:

- Forward kinematics
- Collision checking
- Robot state representations

No modifications to PyRoNot are required.

## Troubleshooting

### Library Not Found

```
RuntimeError: pRRTC library not found at ...
```

**Solution**: Compile the library first with `bash build.sh` and ensure the `.so` file is in the `cuda-rrtc/` directory.

### CUDA Errors

```python
jax.ffi.FFILaunchError: CUDA error ...
```

**Solutions**:
- Ensure a CUDA-capable GPU is available
- Check CUDA driver version compatibility
- For debug builds, use `bash build.sh --debug` and Nsight Compute

## References

- Vectorized Accelerated Motion Planning (VAMP): https://github.com/robotlocomotion/vamp
- pRRTC: https://github.com/lyf44/pRRTC
- PyRoNot: https://github.com/rrax/pyronot

## License

MIT License
