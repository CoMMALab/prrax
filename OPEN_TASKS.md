# Open tasks

## Fused mantis FK + collision check in the planner

FK is already mantis-traced (GRiM `traced=True` → mantis thread tier). The open part is
collision: swap prrax's sphere loops for mantis's fused `sphere_collision`
(`GRiM/external/mantis/python/mantis/algorithms.py`), which benchmarks faster than GRiM's
traced FK + CC (panda B=1024: 0.18 vs 0.26 ms; baxter 0.25 vs 0.47 ms;
`GRiM/benchmarks/results/fkcc_*`).

Blockers: `sphere_collision` is boxes-only, its box count is compile-time (unrolled), and it
has no self-collision, margin, attached spheres, or coarse/early-exit pass.

Options:
1. Fast path: use it only for boxes-only problems with a fixed box count; keep prrax's
   checker otherwise.
2. Extend mantis's `sphere_collision` to runtime worlds (all primitives), self pairs and a
   margin, then replace the checker (edits vendored GRiM/mantis).

First A/B the planner on the MBM panda set to see if the microbenchmark gain survives.

Also: README/comments still say "cricket2" for the traced FK; it is mantis.

## Speedups seen in CRAB's SHRIMP RRT-Connect (`shrimp/cuda/rrtc.cuh`)

Same pRRTC algorithm; CRAB's edge is mostly overhead and the collision check, from reading the code
(not measured). Isolate with prrax's `kernel_time_ms`, then one knob at a time on the MBM set.

1. **Benchmark harness.** CRAB times `k_init` + `k_rrtc` with cudaEvents in a C++ exe, 8192 nodes per
   tree, epoch flags (no clearing). CRAB's `solve_prrax.py` times prrax by Python wall clock with
   `max_nodes=1e6`: big per-solve allocations, and `prrax_init_batch_kernel` resets 2e6 parents with one
   block per problem. Compare kernel ms to kernel ms; size `max_nodes` to the iteration budget; make the
   reset multi-block or epoch-based.
2. **Broad phase by default.** Default `binary` tests every fine sphere against every obstacle and every
   self pair. CRAB gates fine spheres with per-link bounding spheres, groups self pairs by link pair,
   orders distal links first, and votes `__any_sync` across the edge's waypoints for early exit.
   `binary_coarse` exists; make it (or a warp-voting version) the default.
3. **FK once per waypoint.** With `PRRAX_WP_LANES=4` each of the 4 lanes of a waypoint recomputes the
   full FK. Share it (as `binary_coarse_coop` does through shared memory) or give each lane a waypoint.
4. **Block shape and syncs.** CRAB: one warp per block, shuffle argmin, few barriers. prrax: 64 threads,
   shared-memory tree reduction, two `stopped()` checks (barriers + atomic), a `solved` atomic per edge,
   a per-iteration `atomicAdd` on the work counter, `check_aborted` polls in the sphere loops. Try a
   32-thread block with warp-shuffle NN, fewer polls.
5. **Waypoints per edge.** prrax always checks `granularity`=16 waypoints; CRAB uses `ceil(len*32)`, so
   short steer/connect hops check far fewer.
6. **Search settings (per-family effect, not a clear win).** prrax's benchmark runs balance mode 2 with
   dynamic domain (rejection after the NN scan); CRAB runs mode 1 without it. Baxter uses step 0.25 vs
   CRAB's 0.5. A/B `balance_mode=1, dynamic_domain=False`.

Not in CRAB's favour: prrax's SoA node layout coalesces the NN scan; CRAB's AoS does not.
