#!/usr/bin/env python3
"""Compare prrax constrained runs (benchmark_constrained.py) with McVAMP's own results.

Reads bench_results/mcvamp_<set>.json (McVAMP --results) and bench_results/prrax_<set>_<mode>.json,
prints solve rate and solve-time quantiles per set and per obstacle bin, and plots the CDF of
solve time per set (bench_results/compare_mcvamp.png). McVAMP runs on one CPU core, prrax on one
GPU; times are per-problem wall times (prrax: host-measured, kernel time in brackets).

    python compare_mcvamp.py [--sets plane line plane_ori line_ori maze]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "bench_results"
# Categorical slots 1-3 of the validated default palette (dataviz skill).
COLORS = {"McVAMP (1 CPU core)": "#2a78d6", "prrax mcvamp (GPU)": "#eb6834", "prrax cprrtc (GPU)": "#1baf7a"}


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def stats(recs, key="solve_time_ns"):
    ok = [r for r in recs if r["success"]]
    t = np.asarray([r[key] for r in ok], float) / 1e6
    return len(ok), len(recs), t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["plane", "line", "plane_ori", "line_ori", "maze"])
    args = ap.parse_args()

    runs = {}
    print(f"{'set':10} {'planner':22} {'solved':>11} {'median ms':>10} {'p95 ms':>9} {'kernel med':>11}")
    for s in args.sets:
        runs[s] = {"McVAMP (1 CPU core)": load(R / f"mcvamp_{s}.json"),
                   "prrax mcvamp (GPU)": load(R / f"prrax_{s}_mcvamp.json"),
                   "prrax cprrtc (GPU)": load(R / f"prrax_{s}_cprrtc.json")}
        for name, recs in runs[s].items():
            if recs is None:
                continue
            n, tot, t = stats(recs)
            k = stats(recs, "kernel_time_ns")[2] if "kernel_time_ns" in recs[0] else None
            print(f"{s:10} {name:22} {n:5}/{tot:<5} {np.median(t):10.2f} {np.percentile(t, 95):9.2f}"
                  + (f" {np.median(k):11.2f}" if k is not None else ""))
        batch = load(R / f"prrax_{s}_mcvamp_batch.json")
        if batch is not None:
            n, tot, _ = stats(batch)
            wall = max(r["solve_time_ns"] for r in batch) / 1e6
            print(f"{s:10} {'prrax batch (GPU)':22} {n:5}/{tot:<5}  whole set {wall:.0f} ms = {wall / tot:.3f} ms/problem")

    # Per obstacle bin, mcvamp mode vs McVAMP.
    for s in ("plane", "line"):
        m, p = runs.get(s, {}).get("McVAMP (1 CPU core)"), runs.get(s, {}).get("prrax mcvamp (GPU)")
        if not (m and p):
            continue
        print(f"\n{s} by obstacle count: solved and median ms, McVAMP | prrax")
        for b in sorted({r["num_cuboid_obstacles"] for r in m}):
            mb = [r for r in m if r["num_cuboid_obstacles"] == b]
            pb = [r for r in p if r["num_cuboid_obstacles"] == b]
            (mn, mt, mtt), (pn, pt, ptt) = stats(mb), stats(pb)
            print(f"  {b:4d}: {mn:4}/{mt:<4} {np.median(mtt) if mn else np.nan:7.2f} | "
                  f"{pn:4}/{pt:<4} {np.median(ptt) if pn else np.nan:7.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = [s for s in args.sets if any(runs[s].values())]
    fig, axes = plt.subplots(1, len(sets), figsize=(3.2 * len(sets), 3.0), sharey=True)
    for ax, s in zip(np.atleast_1d(axes), sets):
        for name, recs in runs[s].items():
            if recs is None:
                continue
            _, tot, t = stats(recs)
            t = np.sort(t)
            ax.step(t, np.arange(1, len(t) + 1) / tot, where="post", lw=2, color=COLORS[name], label=name)
        ax.set_xscale("log")
        ax.set_title(s, fontsize=10)
        ax.set_xlabel("solve time (ms)", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("fraction solved", fontsize=9)
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(R / "compare_mcvamp.png", dpi=150)
    print(f"\nCDF plot: {R / 'compare_mcvamp.png'}")


if __name__ == "__main__":
    main()
