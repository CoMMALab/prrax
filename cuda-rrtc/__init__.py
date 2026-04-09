"""
cuda-rrtc: GPU-accelerated parallel RRTC motion planner for robotics.

This package provides a JAX FFI wrapper for pRRTC CUDA kernels that integrate
with PyRoNot's existing CUDA backend for robotics kinematics and collision checking.

Available modules:
  - cuda: CUDA kernels for nearest neighbor, extension, and planning
  - jax: JAX interface for easy integration with JAX computations

Example usage:

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
"""

__version__ = "0.1.0"
__author__ = "RRAX Team"
