# P-RRAX: A reimplementation of pRRTC through the Jax FFI

`P-RRAX` is a CUDA-accelerated implementation of the Parallel Rapidly-exploring Random Tree (pRRTC) motion planning algorithm, designed for integration with PyRoFFI's robotics kinematics and collision checking framework. This implementation builds on many pRRTC related work for a robust, batched, and accelerated point-to-point motion validator. 


## Setup

Prerequisites: an NVIDIA GPU, the CUDA toolkit (`nvcc`), Python 3.11+, and jax/jaxlib with CUDA.

prrax depends on [pyroffi](https://github.com/commalab/pyroffi) (v1.0, the GRiM version), which is not on
PyPI. pyroffi provides the robot model, collision spheres and GRiM's trace compiler, which generates the
robot-specific CUDA code the planner is built from (on first use, per robot).

```bash
conda activate prrax
git clone https://github.com/commalab/pyroffi ../pyroffi-grim
pip install -e ../pyroffi-grim
bash ../pyroffi-grim/build_kernels/build_cricket.sh   # GRiM's trace compiler
ln -sfn ../pyroffi-grim pyroffi                       # robot resources used by tests, demos, benchmarks
pip install -e ".[test]"                              # the `prrax` package (sources in cuda-rrtc/)
```

```python
from prrax.jax import prrtc_plan, prrtc_plan_batch
```

### Benchmark assets and baselines (optional)

| What | Needed for | How |
|---|---|---|
| `mcvamp` checkout | tests, the constrained demos and `benchmark/benchmark_constrained.py` (McVAMP's problem sets, maze, Panda model); files only, no build | `git submodule update --init mcvamp` |
| McVAMP Python module | live McVAMP comparison in `demos/demo_line_plane.py` and `demos/demo_maze.py`; without it they run prrax alone | `cd mcvamp/vamp && pip install . -C cmake.define.VAMP_ROBOTS="panda"` |
| `pRRTC` checkout | nothing: the original CUDA pRRTC, for reference | `git submodule update --init pRRTC` |

Demos and benchmarks also use `viser`, `trimesh`, `tqdm`, `pandas` and `loguru`:
`pip install viser trimesh tqdm pandas loguru`.

## Tests

```bash
pytest   # needs a GPU and the mcvamp checkout; the planner is built per robot on first use
```

## Demos

Each demo opens a viser viewer at the printed URL (`--visualize` where it is optional).

```bash
python demos/demo_prrtc.py                         # unconstrained pRRTC; tree and path viewer
python demos/batched_solve_demo.py                 # many problems in one batched solve
python demos/demo_line_plane.py --problems plane --visualize   # end effector on a plane (or --problems line)
python demos/demo_maze.py [--hard]                 # marker tip on the table through the maze; drag the red sphere
```

With the McVAMP module installed, `demo_line_plane.py` and `demo_maze.py` also solve every problem with McVAMP
(one CPU core) and print both times; `--no-mcvamp` turns that off. In the maze, McVAMP plans on its own model
(Panda with a pen, constraint on the hand), so the problems are close but not identical.

## Benchmarks

```bash
python benchmark/evaluate_mbm.py                   # MotionBenchMaker problems (pyroffi resources)
python benchmark/evaluate_mbm_batched.py           # the same, batched per problem set
python benchmark/benchmark_constrained.py --problems plane [--mode mcvamp|cprrtc]   # needs the mcvamp checkout
python benchmark/benchmark_collision_checkers.py   # the planner's collision checkers compared
```

Use `CUDA_VISIBLE_DEVICES` to pick a GPU and `XLA_PYTHON_CLIENT_PREALLOCATE=false` when sharing one.


## References

1. **pRRTC** — Chih H. Huang, Pranav Jadhav, Brian Plancher, and Zachary Kingston.  
   *pRRTC: GPU-Parallel RRT-Connect for Fast, Consistent, and Low-Cost Motion Planning.*  
   IEEE International Conference on Robotics and Automation (ICRA), 2026.  
   [Paper](https://arxiv.org/abs/2503.06757) | [Code](https://github.com/CoMMALab/pRRTC)

2. **pAORRTC** — Chih H. Huang et al.  
   *GPU-Parallel Asymptotically Optimal RRT-Connect.*  
   Workshop presentations, ICRA 2026.  
   [Project](https://commalab.org/papers/pRRTC/)

3. **MR. POP** — Chih H. Huang, Roy Xing, Brian Plancher, and Zachary Kingston.  
   *MR. POP: Multi-Robot Parallel Optimizing Planner for Almost-Surely Asymptotically Optimal Planning.*  
   arXiv preprint, 2026.  
   [Paper](https://arxiv.org/abs/2609.30644)

4. **cpRRTC** — Jiaming Hu, Jiawei Wang, and Henrik I. Christensen.  
   *cpRRTC: GPU-Parallel RRT-Connect for Constrained Motion Planning.*  
   RSS Workshop on Motion Planning and Control via Parallelization, 2025.  
   [Paper](https://arxiv.org/abs/2505.06791)

5. **McVAMP** — Shrutheesh R. Iyer, I-Chia Chang, Andrew Z. Liu, Yan Gu, and Zachary Kingston.  
   *Vectorizing Projection in Manifold-Constrained Motion Planning for Real-Time Whole-Body Control.*  
   IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2026.  
   [Paper](https://arxiv.org/abs/2604.13323) | [Project](https://commalab.org/papers/mcvamp/)

6. **Akima Spline** — Hiroshi Akima.  
   *A New Method of Interpolation and Smooth Curve Fitting Based on Local Procedures.*  
   Journal of the ACM, 17(4), 589–602, 1970.  
   [Wikipedia](https://en.wikipedia.org/wiki/Akima_spline) | [DOI](https://doi.org/10.1145/321607.321609)
