# prrax constrained planning: results and findings

State as of commit `9ae098f` (branch `rtp_demo`). Raw benchmark records were kept locally (`bench_results/`, not committed).

## What prrax is now
- pRRTC (GPU RRT-Connect) built per robot through GRiM (`pyroffi-grim`): cricket2-traced FK, the
  sphere model baked as compile-time tables, GLASS for the projection math.
- Manifold constraints: TSR constraints (up to 4, cyclic LM projection), McVAMP two-stage
  (`mcvamp`) or cpRRTC smoothed (`cprrtc`) projection. Projection tolerance, waypoint gap and
  connect tolerance match McVAMP's (|d| < 1e-3); they were deliberately not loosened.
- One persistent kernel per call for the whole batch; 4 threads per edge waypoint (as pRRTC).
- Checks: `test_constrained.py` (TSR error vs JAX reference, projection, valid paths),
  `test_prrtc.py`, `benchmark_constrained.py --validate` (every path independently checked on the
  manifold and collision-free: no invalid path in any final run), `compare_mcvamp.py`.

## Benchmarks (McVAMP's Panda sets; McVAMP = 1 core i7-13700K, prrax = 1 RTX A5000)

Single solves, median wall ms (prrax kernel ms in brackets), cpRRTC mode:

| set (problems) | McVAMP solved / median | prrax solved / median | prrax faster on |
|---|---|---|---|
| plane (1000) | 1000 / 0.51 | 1000 / 7.4 (4.3) | 3% |
| line (766) | 758 / 0.15 | 766 / 5.4 (2.4) | 0% |
| plane + orientation | 1000 / 0.58 | 1000 / 8.6 (5.0) | 3% |
| line + orientation | 751 / 0.22 | 766 / 5.5 (2.7) | 1% |
| maze, pen attached (100) | 79 / 5.9 | 92 / 9.7 (6.9) | 31% (hardest: ~70%) |

Whole set in one batched dispatch vs McVAMP's summed planning time (solved problems only, so
McVAMP's failures are excluded and the comparison favours McVAMP):

| set | McVAMP | prrax batch | ratio (>1 = prrax faster) |
|---|---|---|---|
| plane | 3.48 s | 4.22 s | 0.82 |
| line | 0.70 s | 1.21 s | 0.58 |
| plane + orientation | 4.97 s | 4.62 s | 1.08 |
| line + orientation | 1.58 s | 1.63 s | 0.97 |
| maze | 4.75 s | 19.6 s (79 solved) | 0.24 |

Unconstrained MotionBenchMaker Panda (140 problems): 140/140, 1.8 ms mean kernel time.

Takeaways: prrax solves as many or more problems than McVAMP everywhere and wins on hard problems
(maze); McVAMP wins latency on easy problems by 10-40x. Batched, prrax is at rough parity on two
sets. Maze batches are held back by unsolved problems running the full budget and a 50k-node cap.

## Why McVAMP is faster on easy problems (measured)
- One LM projection step: 3.7 µs on one GPU thread (cycle-counted in-kernel). Split: cricket2 FK +
  Jacobian 0.85, TSR error + Jacobian transform 1.42, `Jx Jx^T` 0.6, 6x6 solve 0.85. McVAMP's step
  is ~0.3-0.5 µs (SIMD, 5.4 GHz out-of-order core). A GPU thread runs serial math 10-20x slower.
- Planner time split (in-kernel profile): collision checking 60-86% on plane/line, nearest-neighbour
  search 84% on the maze, projection 3-38%.
- Remaining host overhead ~2 ms per solve, ~1 ms of it a post-solve tail (in-flight collision check).

## What helped (kept)
| change | effect |
|---|---|
| persistent batch kernel, blocks move to unsolved problems | batch 1.5-2.8x faster |
| 4 threads per waypoint (split collision pairs) | MBM 6.7 -> 1.8 ms; constrained 1.2-1.4x |
| compile-time dim + one traced FK/J pass per LM step | projection 1.4-3.3x |
| early exit in projection; abort work once solved | post-solve tail 3 -> 1 ms |
| GPU path tracing, packed inputs, async 3-launch handler | host overhead ~5.5 -> ~2 ms |
| 256 blocks per single solve (64 per batched problem) | ~12% single, better p95 |

## Tried and rejected (measured slower or no gain; don't retry)
- McVAMP's endpoint-first collision check (+50% kernel: serializes one thread).
- Tangent-space steering (22% slower). Direct start->goal connect, dropping the connect progress
  rule (no gain). cricket2 warp/block tiers for FK (6-9x slower on a serial 7-DoF chain).
- GLASS for the projection math is 2-6% slower than hand-rolled; kept anyway (user prefers
  modular library code).

## Bugs fixed along the way (pre-existing)
- Base-link spheres (joint -1) were never collision-checked.
- Nearest-neighbour could read claimed-but-unwritten nodes (parent -1) -> paths not reaching the root.
  The fix costs ~15% on unconstrained runs; the old speed partly came from that race.
- Batch streams did not wait for their inputs; radii init covered only 32 goals.

## Open items / next steps
- Spatial index for nearest-neighbour search (dominates hard problems).
- Fused cricket2 trace of the whole TSR + LM step: needs upstream cricket2 support for per-call
  parameter inputs, min/max/select and atan2 (est. 2-3x on the step, ~1.6-2x kernel).
- Recursive waypoint subdivision (McVAMP) instead of failing wide-gap edges.

## Caveats for reading numbers
- Host times vary up to ~25% with CPU load (benchmarks were often run 4 at a time). Compare runs
  made side by side; kernel times are more stable.
- `kernel_time_ms` = GPU time from setup to solve (setup, root projection, planning).
- `nvidia-smi` reports "HW Power Brake Slowdown: Active" on all GPUs, but the braking counter is 0
  µs and clocks reach 1695 MHz under load; it did not affect results.
