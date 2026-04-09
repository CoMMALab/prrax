# pRRTC to JAX FFI Translation Review

## Scope

This report compares the original pRRTC implementation in `pRRTC/src/planning/pRRTC.cu` with the reimplementation in `cuda-rrtc` (`CUDA/*.cu`, `ffi/prrtc_ffi.cc`, `jax/prrtc.py`) and evaluates algorithmic equivalence.

## Executive Summary

The current `cuda-rrtc` JAX FFI implementation is **not algorithmically equivalent** to the original pRRTC kernel.

It keeps the broad two-tree RRT-Connect shape, but removes or simplifies several core mechanisms that are central to pRRTC behavior:

1. Robot/environment collision checking is removed from the planner path.
2. The original block-level parallel expansion model is replaced by mostly thread-0 serial logic.
3. Dynamic-domain logic and radius adaptation are removed.
4. Iteration semantics differ (single-kernel persistent loop vs host relaunch loop).
5. The JAX wrapper silently drops batch starts and solves only the first start state.

These differences are enough to explain why the reimplementation does not behave like original pRRTC and can fail unexpectedly in realistic motion-planning workloads.

## Translation Map (Original -> FFI)

### 1) Planner state and initialization

- Original:
  - Device globals for solved state, per-tree free indices, completed counts, path buffers, cost, reached goal index, and constant settings.
  - `__constant__ pRRTC_settings d_settings` drives behavior.
  - Environment is marshaled to device memory.
- FFI version:
  - Device globals reduced to solved/iteration counters.
  - Planner state largely held in output buffers (`tree_sizes`, `completed`, `connection_info`, etc.).
  - No robot/environment objects passed into planner kernel.

### 2) Sampling (Halton)

- Original:
  - Per-block Halton state initialized with shuffled prime bases (except block 0), optional skip iterations.
  - Samples are scaled through robot-specific bounds with `Robot::scale_cfg`.
- FFI version:
  - Shared static Halton state in planner kernel, deterministic primes, no per-block randomization.
  - Samples are scaled directly by `min_vals`/`max_vals` arrays.

### 3) Tree balancing policy

- Original:
  - Supports multiple balancing modes (`balance == 0/1/2`) and dynamic ratio logic.
- FFI version:
  - Simple strategy: expand smaller completed tree each sample.

### 4) Nearest-neighbor stage

- Original:
  - Per-block parallel reduction over candidate nodes.
- FFI version:
  - In planner kernel path, nearest-neighbor search runs inside thread 0 loops.
  - A separate NN kernel exists for primitive use, but main planner does not use it.

### 5) Extend and edge validation

- Original:
  - Extension is collision-validated using approximate then detailed FK + environment and self-collision checks along interpolated edge points.
- FFI version:
  - Main planner extends without any collision checking.
  - `prrtc_extend.cu` is geometric only and also has no collision checking.

### 6) Connect phase

- Original:
  - Connect-to-opposite-tree phase is also collision-validated at each incremental extension step.
- FFI version:
  - Connect phase extends directly to opposite nearest node in fixed number of interpolation steps, no collision checks.

### 7) Dynamic-domain adaptation

- Original:
  - Radius-based acceptance/rejection and shrink/grow updates around difficult regions.
- FFI version:
  - No dynamic-domain radii or adaptation logic.

### 8) Path extraction

- Original:
  - Device stores path segments and cumulative cost, plus reached goal index, then host reconstructs full path.
- FFI version:
  - Returns parent trees + connection metadata; Python traces back and computes cost.

### 9) Iteration model

- Original:
  - One persistent kernel launch (`num_new_configs` blocks) loops until solved/failed.
- FFI version:
  - Host-side loop launches `prrtc_planner_kernel` up to `max_iterations` times; each launch processes `num_new_samples` from thread 0.

## Findings: Non-Equivalence and Likely Failure Causes

### Critical 1: Collision checking removed from main planner

The original planner is collision-constrained at both extend and connect phases. The FFI planner accepts all geometric edges.

Impact:
- Produces paths that are invalid for constrained robots/scenes.
- In scenes where only narrow valid corridors exist, search behavior no longer matches pRRTC and may fail to find valid solutions even when pRRTC can.

### Critical 2: JAX wrapper discards batched starts

`prrtc_plan` reshapes start batch but always picks `start_flat[0]` before calling FFI.

Impact:
- Batched starts are silently ignored.
- Results can appear inconsistent or "not solving" for expected per-instance behavior.

### High 3: Parallelism model changed from pRRTC block-per-sample to serial thread-0 loop

The original relies on many blocks concurrently proposing nodes and exploring both trees. FFI planner processes samples serially in one block, thread 0.

Impact:
- Dramatically different exploration dynamics.
- Lower effective search breadth per wall-clock and iteration budget.

### High 4: Dynamic-domain logic removed

Original pRRTC adapts per-node expansion radii based on collision outcomes.

Impact:
- Less robust behavior near clutter and narrow passages.
- More wasted expansions into blocked regions.

### Medium 5: Iteration accounting and kernel relaunch semantics differ

Original `iter` corresponds to persistent-kernel loop cycles; FFI increments `iter_count` once per kernel launch while each launch may process many samples.

Impact:
- Parameter tuning and stopping behavior are not directly transferable.
- `max_iterations` in FFI is not equivalent to original pRRTC notion of iteration effort.

### Medium 6: Halton initialization differs

Original introduces per-block prime shuffling; FFI uses deterministic fixed sequence.

Impact:
- Can reduce sample diversity and alter convergence characteristics across runs.

## Code Pointers Used in This Review

- Original global state/settings/sampler: `pRRTC/src/planning/pRRTC.cu`
- Original balancing/collision/connect logic: `pRRTC/src/planning/pRRTC.cu`
- Original dynamic-domain update: `pRRTC/src/planning/pRRTC.cu`
- FFI planner kernel and host launcher: `cuda-rrtc/CUDA/prrtc_planner.cu`
- FFI iteration primitive: `cuda-rrtc/CUDA/prrtc_iteration.cu`
- FFI extend primitive: `cuda-rrtc/CUDA/prrtc_extend.cu`
- JAX wrapper batching/path reconstruction: `cuda-rrtc/jax/prrtc.py`

## Conclusion

Current `cuda-rrtc` is best described as a **simplified geometric two-tree planner exposed through JAX FFI**, not a faithful pRRTC translation.

The biggest reason it "doesn't seem to solve" in practical robotics scenarios is the mismatch between:

1. Original pRRTC's collision-aware, adaptive, highly parallel behavior.
2. FFI planner's collision-free geometric expansion with reduced parallel search dynamics and changed iteration semantics.

## Recommended Next Steps

1. Reintroduce collision checks into planner kernels using pyronot CUDA primitives for both extend and connect edge validation.
2. Restore pRRTC-style parallel expansion (block-per-sample or equivalent) before optimizing further.
3. Reintroduce dynamic-domain radius updates for parity with original behavior in clutter.
4. Fix JAX batching semantics so each start in batch is actually solved (via `vmap` wrapper or batched FFI contract).
5. Align iteration budgeting definitions with original pRRTC to make tuning portable.