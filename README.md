# P-RRAX: A reimplementation of pRRTC through the Jax FFI

`P-RRAX` is a CUDA-accelerated implementation of the Parallel Rapidly-exploring Random Tree (pRRTC) motion planning algorithm, designed for integration with PyRoFFI's robotics kinematics and collision checking framework. This implementation builds on many pRRTC related work for a robust, batched, and accelerated point-to-point motion validator. 


## Setup

prrax depends on [pyroffi v1.0](../pyroffi-grim). PyRoFFI vendors a CUDA backend through GRiD and custom motion planning kernels as well as cricket2 (name is wip) which trace compiles CUDA code. 

```bash
conda activate prrax
pip install -e ../pyroffi-grim
bash ../pyroffi-grim/build_kernels/build_cricket.sh     # installs cricket2 (GRiM's trace compiler)
ln -sfn ../pyroffi-grim pyroffi
```


Prerequisites: CUDA toolkit (`nvcc`), Python 3.11+, jax/jaxlib with CUDA.


The following papers were used as reference material in the implementation of this repository:


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
