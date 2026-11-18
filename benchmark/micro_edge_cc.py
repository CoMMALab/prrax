#!/usr/bin/env python3
"""Microbenchmark: one edge waypoint's validation, as the planner does it (mantis-traced FK of
the sphere frames, then runtime loops over the world with early exit: flat ``binary`` and
coarse-gated ``binary_coarse``) against the whole step traced by mantis (interpolation + FK +
every sphere/obstacle and self distance in straight-line code, no early exit; world counts
compile-time, worlds padded with out-of-reach obstacles).

One thread per waypoint, MBM worlds, random edges of the robot's step size at 32 waypoints/rad.
Times are cudaEvent medians of the kernel alone; verdicts are cross-checked.

    python benchmark/micro_edge_cc.py --robot panda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_mbm_batched as M  # noqa: E402
from prrax.jax import _grim_build, _mantis_algorithms as algs  # noqa: E402
from prrax.jax.prrtc import _pack_worlds  # noqa: E402

FAR = 1e4
DUMMY = {0: [FAR, FAR, FAR, 0.0], 1: [FAR, FAR, FAR, FAR, FAR, FAR, 0.0],
         2: [FAR, FAR, FAR, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 3: [0, 0, 1, 0, 0, -FAR]}

KERNELS = r"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include "robot.cuh"
#include "collision.cuh"
#include "prrax_gen.cuh"
#include "edge_traced.cuh"

struct World { const float *s, *c, *b, *h; int ns, nc, nb, nh; };

__device__ bool vs_world(const World& w, const float* p, float r) {
    for (int i = 0; i < w.ns; ++i) { const float* o = w.s + 4 * i;
        if (sphere_sphere_dist(p[0], p[1], p[2], r, o[0], o[1], o[2], o[3]) <= 0.f) return true; }
    for (int i = 0; i < w.nc; ++i) { const float* o = w.c + 7 * i;
        if (sphere_capsule_dist(p[0], p[1], p[2], r, o[0], o[1], o[2], o[3], o[4], o[5], o[6]) <= 0.f) return true; }
    for (int i = 0; i < w.nb; ++i) { const float* o = w.b + 15 * i;
        if (sphere_box_dist(p[0], p[1], p[2], r, o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8],
                            o[9], o[10], o[11], o[12], o[13], o[14]) <= 0.f) return true; }
    for (int i = 0; i < w.nh; ++i) { const float* o = w.h + 6 * i;
        if (sphere_halfspace_dist(p[0], p[1], p[2], r, o[0], o[1], o[2], o[3], o[4], o[5]) <= 0.f) return true; }
    return false;
}

__device__ bool self_hit(const float* X, int i, int j) {
    float a[3], b[3];
    prrax::fused_centre(X, i, a); prrax::fused_centre(X, j, b);
    return sphere_sphere_dist(a[0], a[1], a[2], prrax::kSphereRadius[i], b[0], b[1], b[2], prrax::kSphereRadius[j]) <= 0.f;
}

__device__ bool loop_flat(const float* q, const World& w) {
    float X[12 * (prrax::n_fused_frames + 1)];
    prrax::fused_frames(q, X);
    for (int s = 0; s < prrax::n_spheres; ++s) { float p[3]; prrax::fused_centre(X, s, p);
        if (vs_world(w, p, prrax::kSphereRadius[s])) return true; }
    for (int k = 0; k < prrax::n_self_pairs; ++k)
        if (self_hit(X, prrax::kSelfPairs[2 * k], prrax::kSelfPairs[2 * k + 1])) return true;
    return false;
}

__device__ bool loop_coarse(const float* q, const World& w) {
    float X[12 * (prrax::n_fused_frames + 1)];
    prrax::fused_frames(q, X);
    for (int k = 0; k < prrax::n_coarse; ++k) {
        float p[3]; prrax::fused_coarse_centre(X, k, p);
        if (!vs_world(w, p, prrax::kCoarseRadius[k])) continue;
        for (int s = prrax::kFineOffsets[k]; s < prrax::kFineOffsets[k + 1]; ++s) {
            float c[3]; prrax::fused_centre(X, s, c);
            if (vs_world(w, c, prrax::kSphereRadius[s])) return true;
        }
    }
    for (int p = 0; p < prrax::n_coarse_self_pairs; ++p) {
        const int ci = prrax::kCoarseSelfPairs[2 * p], cj = prrax::kCoarseSelfPairs[2 * p + 1];
        float a[3], b[3];
        prrax::fused_coarse_centre(X, ci, a); prrax::fused_coarse_centre(X, cj, b);
        if (sphere_sphere_dist(a[0], a[1], a[2], prrax::kCoarseRadius[ci], b[0], b[1], b[2], prrax::kCoarseRadius[cj]) > 0.f) continue;
        for (int q2 = prrax::kSelfPairRanges[2 * p]; q2 < prrax::kSelfPairRanges[2 * p + 1]; ++q2)
            if (self_hit(X, prrax::kSelfPairs[2 * q2], prrax::kSelfPairs[2 * q2 + 1])) return true;
    }
    return false;
}

// The coarse pre-pass traced (FK of the coarse frames + every coarse distance, branch-free), then
// the fine stage as loop_coarse, only behind a coarse margin <= 0 (fine FK computed lazily).
__device__ bool traced_coarse(const float* q, const float* ws, const float* wc, const float* wb, const float* wh,
                              const World& w) {
    constexpr int NC = prrax::n_coarse, NP = prrax::n_coarse_self_pairs;
    float qm[DIM], m[NC + NP];
    GATHERQ
    mantis::robots::prrax_edge::prrax_coarse_margins_thread<float>(qm, ws, wc, wb, wh, m);
    bool any = false;
    for (int k = 0; k < NC + NP; ++k) any |= m[k] <= 0.f;
    if (!any) return false;
    float X[12 * (prrax::n_fused_frames + 1)];
    prrax::fused_frames(q, X);
    for (int k = 0; k < NC; ++k) {
        if (m[k] > 0.f) continue;
        for (int s = prrax::kFineOffsets[k]; s < prrax::kFineOffsets[k + 1]; ++s) {
            float c[3]; prrax::fused_centre(X, s, c);
            if (vs_world(w, c, prrax::kSphereRadius[s])) return true;
        }
    }
    for (int p = 0; p < NP; ++p) {
        if (m[NC + p] > 0.f) continue;
        for (int q2 = prrax::kSelfPairRanges[2 * p]; q2 < prrax::kSelfPairRanges[2 * p + 1]; ++q2)
            if (self_hit(X, prrax::kSelfPairs[2 * q2], prrax::kSelfPairs[2 * q2 + 1])) return true;
    }
    return false;
}

// mode 0 flat loops, 1 coarse loops, 2 fully traced, 3 traced coarse pre-pass + fine loops. Thread = (edge, waypoint).
template <int mode>
__global__ void k_edges(int n_edges, int n_wp, const float* q0, const float* dq, const int* prob,
                        const float* ws, const float* wc, const float* wb, const float* wh, const int* counts,
                        int ms, int mc, int mb, int mh, int* hit) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_edges * n_wp) return;
    const int e = id / n_wp, p = prob[e];
    const float t = (id % n_wp + 1) / (float)n_wp;
    const float* a = q0 + e * DIM; const float* d = dq + e * DIM;
    bool h;
    if constexpr (mode == 2) {
        float qm[DIM], dm[DIM], m;
        GATHER
        mantis::robots::prrax_edge::prrax_edge_margin_thread<float>(qm, dm, &t, ws + p * ms * 4, wc + p * mc * 7,
                                                                    wb + p * mb * 15, wh + p * mh * 6, &m);
        h = m <= 0.f;
    } else {
        float q[DIM];
        for (int k = 0; k < DIM; ++k) q[k] = a[k] + t * d[k];
        World w{ws + p * ms * 4, wc + p * mc * 7, wb + p * mb * 15, wh + p * mh * 6,
                counts[4 * p], counts[4 * p + 1], counts[4 * p + 2], counts[4 * p + 3]};
        h = mode == 0 ? loop_flat(q, w) : mode == 1 ? loop_coarse(q, w)
            : traced_coarse(q, w.s, w.c, w.b, w.h, w);
    }
    hit[id] = h;
}


#ifdef COOP
namespace E = mantis::robots::prrax_edge;
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
// A cooperative tier (mantis warp / block): lane l of a warp serves problem l / (32 / P); every
// lane passes its own problem's pointers. Scratch in shared memory or in global memory.
__global__ void k_coop(int n, int n_wp, const float* q0m, const float* dqm, const float* tarr, const int* prob,
                       const float* ws, const float* wc, const float* wb, const float* wh,
                       int ms, int mc, int mb, int mh, float* marg, float* gscr) {
    extern __shared__ float sh[];
    float* tables = sh;
    E::CAT(COOP_FN, _load_tables)<float>(tables, threadIdx.x, blockDim.x);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, grp = lane / (32 / COOP_P);
#if COOP_BLOCK
    const int unit = blockIdx.x, id0 = blockIdx.x * COOP_P;
#else
    const int unit = blockIdx.x * (blockDim.x >> 5) + warp, id0 = unit * COOP_P;
#endif
#if COOP_SHARED
    float* s = sh + E::CAT(COOP_FN, _tables)<float>() + (COOP_BLOCK ? 0 : warp) * E::CAT(COOP_FN, _smem);
#else
    float* s = gscr + (size_t)unit * E::CAT(COOP_FN, _smem);
#endif
    const int id = min(id0 + grp, n - 1), e = id / n_wp, p = prob[e];
    E::COOP_FN<float>(q0m + e * DIM, dqm + e * DIM, tarr + id % n_wp, ws + p * ms * 4, wc + p * mc * 7,
                      wb + p * mb * 15, wh + p * mh * 6, marg + id, s, tables);
}
#endif

template <class T> static void rd(const char* f, std::vector<T>& v) {
    FILE* fp = fopen(f, "rb"); fseek(fp, 0, SEEK_END); v.resize(ftell(fp) / sizeof(T)); rewind(fp);
    if (fread(v.data(), sizeof(T), v.size(), fp) != v.size()) exit(3); fclose(fp);
}
template <class T> static T* up(const std::vector<T>& v) { T* d; cudaMalloc(&d, v.size() * sizeof(T) + 4);
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return d; }

int main(int argc, char** argv) {   // dir n_wp ms mc mb mh reps
    std::string D = argv[1]; const int n_wp = atoi(argv[2]), ms = atoi(argv[3]), mc = atoi(argv[4]),
        mb = atoi(argv[5]), mh = atoi(argv[6]), reps = atoi(argv[7]);
    std::vector<float> q0, dq, ws, wc, wb, wh; std::vector<int> prob, counts;
    rd((D + "/q0").c_str(), q0); rd((D + "/dq").c_str(), dq); rd((D + "/prob").c_str(), prob);
    rd((D + "/ws").c_str(), ws); rd((D + "/wc").c_str(), wc); rd((D + "/wb").c_str(), wb);
    rd((D + "/wh").c_str(), wh); rd((D + "/counts").c_str(), counts);
    const int n_edges = prob.size(), n = n_edges * n_wp;
    float *dq0 = up(q0), *ddq = up(dq), *dws = up(ws), *dwc = up(wc), *dwb = up(wb), *dwh = up(wh);
    int *dprob = up(prob), *dcounts = up(counts), *dhit; cudaMalloc(&dhit, n * 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<int> ref;
    for (int mode = 0; mode < 4; ++mode) {
        std::vector<float> ts;
        for (int r = 0; r < reps + 2; ++r) {
            cudaEventRecord(e0);
            auto k = mode == 0 ? k_edges<0> : mode == 1 ? k_edges<1> : mode == 2 ? k_edges<2> : k_edges<3>;
            k<<<(n + 127) / 128, 128>>>(n_edges, n_wp, dq0, ddq, dprob, dws, dwc, dwb, dwh, dcounts,
                                        ms, mc, mb, mh, dhit);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float t; cudaEventElapsedTime(&t, e0, e1); if (r >= 2) ts.push_back(t);
        }
        if (cudaGetLastError() != cudaSuccess) { printf("CUDA error\n"); return 2; }
        std::vector<int> h(n); cudaMemcpy(h.data(), dhit, n * 4, cudaMemcpyDeviceToHost);
        long hits = 0, diff = 0; for (int i = 0; i < n; ++i) { hits += h[i]; if (mode) diff += h[i] != ref[i]; }
        if (!mode) ref = h;
        std::sort(ts.begin(), ts.end());
        cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, mode == 0 ? k_edges<0> : mode == 1 ? k_edges<1> : mode == 2 ? k_edges<2> : k_edges<3>); const int regs = fa.numRegs;
        printf("%d %.4f %ld %ld %d %d\n", mode, ts[ts.size() / 2], hits, diff, n, regs);
#ifdef COOP
        break;   // cooperative build: the flat loops are the reference, then the tier
#endif
    }
#ifdef COOP
    {
        std::vector<float> q0m, dqm, tarr; rd((D + "/q0m").c_str(), q0m); rd((D + "/dqm").c_str(), dqm);
        rd((D + "/tarr").c_str(), tarr);
        float *dq0m = up(q0m), *ddqm = up(dqm), *dt = up(tarr), *dm, *gs = nullptr;
        cudaMalloc(&dm, n * 4);
        const int units_per_block = COOP_BLOCK ? 1 : COOP_WARPS, per_block = units_per_block * COOP_P;
        const int blocks = (n + per_block - 1) / per_block, threads = COOP_WARPS * 32;
        const size_t shm = (E::CAT(COOP_FN, _tables)<float>() + (COOP_SHARED ? units_per_block * E::CAT(COOP_FN, _smem) : 0)) * 4;
        if (!COOP_SHARED) cudaMalloc(&gs, (size_t)blocks * units_per_block * E::CAT(COOP_FN, _smem) * 4);
        cudaFuncSetAttribute(k_coop, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
        std::vector<float> ts;
        for (int r = 0; r < reps + 2; ++r) {
            cudaEventRecord(e0);
            k_coop<<<blocks, threads, shm>>>(n, n_wp, dq0m, ddqm, dt, dprob, dws, dwc, dwb, dwh, ms, mc, mb, mh, dm, gs);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float t; cudaEventElapsedTime(&t, e0, e1); if (r >= 2) ts.push_back(t);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { printf("CUDA error %s\n", cudaGetErrorString(err)); return 2; }
        std::vector<float> m(n); cudaMemcpy(m.data(), dm, n * 4, cudaMemcpyDeviceToHost);
        long hits = 0, diff = 0; for (int i = 0; i < n; ++i) { const int h = m[i] <= 0.f; hits += h; diff += h != ref[i]; }
        std::sort(ts.begin(), ts.end());
        cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, k_coop);
        printf("3 %.4f %ld %ld %d %d %zu\n", ts[ts.size() / 2], hits, diff, n, fa.numRegs, shm);
    }
#endif
    return 0;
}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--robot", default="panda", choices=["panda", "fetch", "baxter"])
    ap.add_argument("--problems", type=int, default=64, help="MBM problems (worlds), spread over the sets")
    ap.add_argument("--edges", type=int, default=256, help="random edges per problem")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--arch", default="sm_86")
    ap.add_argument("--tiers", default="warp:1:1:shared,warp:32:4:global,block:32:4:global",
                    help="cooperative mantis tiers of the fully traced step, comma separated "
                         "tier:P:warps per block:shared|global scratch ('' for thread tier only)")
    args = ap.parse_args()

    robot, coll, lo, hi = M.load_robot_models(args.robot)
    data = M.load_robot_dataset(args.robot)["problems"]
    valid = [p for ps in data.values() for p in ps if p.get("valid", False)]
    pick = [valid[i] for i in np.linspace(0, len(valid) - 1, args.problems).astype(int)]
    from pyroffi.collision._obstacles import create_collision_environment
    ctxs = [M.PRRTC_UTILS.build_prrtc_collision_context(robot, coll, create_collision_environment(p)) for p in pick]
    *worlds, counts = _pack_worlds(ctxs)
    worlds = [w.copy() for w in worlds]
    for k, w in enumerate(worlds):                       # padding: out of reach
        for i in range(len(ctxs)):
            w[i, counts[i, k]:] = DUMMY[k]
    mx = [w.shape[1] if counts[:, k].max() > 0 else 0 for k, w in enumerate(worlds)]

    from pyroffi import _grim
    _grim.ensure_importable()
    import mantis
    from grim.motion import _build
    ctx = ctxs[0]
    mr = _grim.motion_robot(robot)
    t = {k: np.asarray(ctx[k]) for k in _grim_build.ROBOT_KEYS}
    link = t["sphere_link_idx"].reshape(-1).astype(int)
    rows = tuple((mr.joint_names[j] if j >= 0 else None, tuple(map(float, c)), float(r))
                 for j, c, r in zip(link, t["sphere_local"].reshape(-1, 3), t["sphere_radius"].reshape(-1)))
    pairs = tuple(map(tuple, t["self_pairs"].reshape(-1, 2).astype(int)))
    pairs = tuple((i, j) for i, j in pairs if 0 <= i < len(rows) and 0 <= j < len(rows))

    tiers = [t for t in args.tiers.split(",") if t]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        urdf = tmp / "robot.urdf"
        urdf.write_text(_build.kinematic_urdf(mr.urdf_xml))
        model = mantis.Robot(urdf, name="prrax_edge")
        opts = dict(rows=rows, pairs=pairs, n_spheres=mx[0], n_capsules=mx[1], n_boxes=mx[2], n_halfspaces=mx[3])
        clink = t["coarse_sphere_link_idx"].reshape(-1).astype(int)
        crows = tuple((mr.joint_names[j] if j >= 0 else None, tuple(map(float, c)), float(r)) for j, c, r in
                      zip(clink, t["coarse_sphere_local"].reshape(-1, 3), t["coarse_sphere_radius"].reshape(-1)))
        cpairs = tuple(map(tuple, t["coarse_self_pairs"].reshape(-1, 2).astype(int)))
        copts = dict(opts, rows=crows, pairs=cpairs)
        def build(variant, flags=()):
            (tmp / "edge_traced.cuh").write_text(mantis.generate(model, {algs.edge_margin(): opts, algs.coarse_margins(): copts},
                                                                 variants=[mantis.Variant("thread")] + (
                                                                     [variant] if variant.tier != "thread" else []),
                                                                 name="prrax_edge"))
            exe = tmp / f"bench_{len(list(tmp.glob('bench_*')))}"
            subprocess.run(["nvcc", "-O3", f"-arch={args.arch}", "-std=c++17", "--extended-lambda",
                            "-DGRIM_SOLVE_N_BUCKETS(X)=X(1)", f"-DMAX_JOINTS={mr.n_joints}",
                            f"-DMAX_ACT={max(mr.n_act, 6)}", "-DMAX_EE=1", *flags, f"-I{tmp}",
                            *[f"-I{d}" for d in inc], str(tmp / "bench.cu"), "-o", str(exe)], check=True)
            return exe

        (tmp / "prrax_gen.cuh").write_text(_grim_build.prrax_header(ctx))
        (tmp / "grim_robot_gen.cuh").write_text(_build.robot_header(mr, _build.Problem(), True))
        gather = "".join(f"qm[{i}] = a[{mr.actuated_names.index(n)}]; dm[{i}] = d[{mr.actuated_names.index(n)}]; "
                         for i, n in enumerate(model.joint_names))
        gatherq = "".join(f"qm[{i}] = q[{mr.actuated_names.index(n)}]; " for i, n in enumerate(model.joint_names))
        (tmp / "bench.cu").write_text(KERNELS.replace("GATHERQ", gatherq).replace("GATHER", gather).replace("DIM", str(mr.n_act)))
        kdir = Path(_build.__file__).parent / "kernels"
        inc = sorted({str(p.parent) for p in kdir.rglob("*.cuh")})

        rng = np.random.default_rng(0)
        step = M.STEP_SIZE_BY_ROBOT[args.robot]
        lo_, hi_ = np.asarray(lo), np.asarray(hi)
        n_e = args.problems * args.edges
        q0 = rng.uniform(lo_, hi_, (n_e, lo_.size)).astype(np.float32)
        dirs = rng.normal(size=q0.shape)
        dq = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True) * step).astype(np.float32)
        prob = np.repeat(np.arange(args.problems), args.edges).astype(np.int32)
        for name, arr in (("q0", q0), ("dq", dq), ("prob", prob), ("ws", worlds[0]), ("wc", worlds[1]),
                          ("wb", worlds[2]), ("wh", worlds[3]), ("counts", counts.astype(np.int32))):
            np.ascontiguousarray(arr).tofile(tmp / name)
        n_wp = int(np.ceil(step * 32))
        perm = [mr.actuated_names.index(nm) for nm in model.joint_names]      # mantis's joint order
        np.ascontiguousarray(q0[:, perm]).tofile(tmp / "q0m")
        np.ascontiguousarray(dq[:, perm]).tofile(tmp / "dqm")
        ((np.arange(n_wp) + 1) / n_wp).astype(np.float32).tofile(tmp / "tarr")
        run = lambda exe: subprocess.run([str(exe), str(tmp), str(n_wp), *map(str, mx), str(args.reps)],
                                         check=True, capture_output=True, text=True).stdout.split("\n")
        out = run(build(mantis.Variant("thread")))
        coop = []
        for spec in tiers:
            tier, P, warps, where = spec.split(":")
            P, warps = int(P), int(warps)
            v = mantis.Variant(tier, 32 * warps if tier == "block" else 1, problems=P)
            flags = (f"-DCOOP", f"-DCOOP_FN=prrax_edge_margin_{tier}", f"-DCOOP_P={P}", f"-DCOOP_WARPS={warps}",
                     f"-DCOOP_BLOCK={int(tier == 'block')}", f"-DCOOP_SHARED={int(where == 'shared')}")
            try:
                lines = [l for l in run(build(v, flags)) if l.startswith("3 ")]
            except subprocess.CalledProcessError as e:
                print(f"{spec}: {(e.stdout or '')[-300:]} {(e.stderr or '')[-300:]}", file=sys.stderr)
                lines = []
            coop += [(spec, l) for l in lines] or [(spec, "failed")]

    print(f"{args.robot}: {args.problems} worlds (max spheres/capsules/boxes/halfspaces {mx}), "
          f"{n_e} edges x {n_wp} waypoints, {len(rows)} spheres, {len(pairs)} self pairs")
    names = ("loops, flat (binary)", "loops, coarse (binary_coarse)", "fully traced",
             "traced coarse pre-pass + fine loops")
    for line in filter(None, out):
        mode, ms, hits, diff, n, regs = line.split()
        print(f"  {names[int(mode)] + ', thread tier' if mode == '2' else names[int(mode)]:40s} {float(ms):8.3f} ms  "
              f"{float(ms) * 1e6 / int(n):7.1f} ns/waypoint  hits {int(hits) / int(n):.1%}  differing from flat {diff}  "
              f"regs {regs}")
    for spec, line in coop:
        if line == "failed":
            print(f"  fully traced, {spec:26s} failed"); continue
        _, ms, hits, diff, n, regs, shm = line.split()
        print(f"  fully traced, {spec:26s} {float(ms):8.3f} ms  {float(ms) * 1e6 / int(n):7.1f} ns/waypoint  "
              f"hits {int(hits) / int(n):.1%}  differing from flat {diff}  regs {regs}  smem {int(shm) // 1024} KB")


if __name__ == "__main__":
    main()
