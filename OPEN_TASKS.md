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
