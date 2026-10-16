// FLASK-RRTConnect on the GPU: pRRTC's bidirectional search over STATES (q, qd), every edge the
// closed-form FLASK BVP (flask.cuh). A batch of problems (start state -> goal state), `bpp`
// blocks each sharing the problem's two trees through atomics, as pRRTC does.
//
//   extend   sample a state (q uniform in the joint limits, qd uniform in +-vel_scale * vmax);
//            nearest node of the tree (|dq|^2 + vel_w |dqd|^2); q steered to within `step` of
//            it, the sampled qd kept; the edge validated in time order (start tree: near ->
//            new, goal tree: new -> near)
//   connect  the direct BVP from the new node to the other tree's nearest node
//
// Cost-to-come is the FLASK cost sum C(T*); the solution is the first connection (RRTConnect).
// Edge validation: W = blockDim / PRRAX_WP_LANES samples at t = (w + 1) T / W, limits and torque
// on lane 0 of each sample (PRRAX_FLASK_ID(q, qd, qdd, tau), defined by the includer, is the
// per-thread inverse dynamics; undefined: no torque check), then every sample's gap swept for
// collision at VAMP resolution by prrtc_gap_invalid, lanes splitting the collision pairs.
// Requires prrtc_planner.cu (CollisionContext, prrtc_gap_invalid) and flask.cuh.
#pragma once
#include "flask.cuh"

#define FLASK_S_MAX (2 * CONFIG_DIM_MAX)

struct FlaskLayout {
    float* states;     // [batch, 2 trees, S, max_nodes] SoA
    float* g;          // [batch, 2, max_nodes] cost-to-come (to the root of its tree)
    float* T;          // [batch, 2, max_nodes] duration of the edge to the parent
    int* par;          // [batch, 2, max_nodes]
    int* size;         // [batch, 2] allocated
    int* done;         // [batch, 2] published
    int* solved;       // [batch] 0 running, 1 solved, -1 out of nodes
    int* conn;         // [batch, 2] start-tree node, goal-tree node
    float* conn_c;     // [batch] bridge edge cost
    float* conn_T;     // [batch] bridge edge duration
    int* iters;        // [batch]
    const float* x0;   // [batch, S] start states
    const float* x1;   // [batch, S] goal states
    const float *w_sph, *w_cap, *w_box, *w_hs, *counts;   // worlds (prrax packing), counts [batch, 4]
    const float* cons;   // [batch, PRRAX_CONS_SIZE] TSRs every point of an edge must satisfy (nullptr: none)
    int batch, n, max_nodes, max_w[4];
    int world_per_problem;   // 1: world arrays are [batch, max_w, .]; 0: one world, counts select its prefix
};

struct FlaskSettings {
    flask::Limits lim;
    int max_iters, bpp;
    float step, vel_scale, vel_w;
    int collision_mode; float collision_margin;
};

__global__ void flask_init_kernel(FlaskLayout L) {
    const int p = blockIdx.x, S = 2 * L.n;
    const size_t mn = L.max_nodes;
    for (size_t i = threadIdx.x; i < 2 * mn; i += blockDim.x) L.par[p * 2 * mn + i] = -1;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int t = 0; t < 2; ++t) {
            const float* x = (t == 0 ? L.x0 : L.x1) + (size_t)p * S;
            float* st = L.states + ((size_t)p * 2 + t) * S * mn;
            for (int d = 0; d < S; ++d) st[d * mn] = x[d];
            L.g[(p * 2 + t) * mn] = 0.f;
            L.T[(p * 2 + t) * mn] = 0.f;
            L.size[p * 2 + t] = 1;
            L.done[p * 2 + t] = 1;
            L.par[(p * 2 + t) * mn] = 0;   // the root: published, its own parent
        }
        L.solved[p] = 0; L.iters[p] = 0; L.conn[2 * p] = L.conn[2 * p + 1] = -1;
        L.conn_c[p] = FLT_MAX; L.conn_T[p] = 0.f;
    }
}

// The FLASK edge xa -> xb (states in shared memory): T* and C(T*) into *T_out / *c_out, true if it
// violates a limit or collides. Collective over the block.
__device__ bool flask_edge_invalid(const float* xa, const float* xb, int n, const flask::Limits& lim,
                                   const CollisionContext& ctx, float* wp_sh, int* bad_sh, float* T_sh,
                                   float* c_sh, const float* cons) {
    // cons: the edge's TSRs (prrax packing; nullptr or CONS_N = 0: none), checked at every
    // collision sample of the curve within CONS_TOL (the curve is not projected: FLASK edges
    // stay closed-form, so a thin TSR admits only short ones)
    const bool tsr = cons && cons[CONS_N] > 0.f;
    const float tol2 = tsr ? cons[CONS_TOL] * cons[CONS_TOL] : 0.f;
    const int tid = threadIdx.x, W = blockDim.x / PRRAX_WP_LANES;
    if (tid == 0) {
        *bad_sh = ctx.solved && atomicAdd(const_cast<int*>(ctx.solved), 0) != 0;
        float c;
        *T_sh = flask::tstar(flask::terms(xa, xa + n, xb, xb + n, n), lim.rho, &c);
        *c_sh = c;
    }
    __syncthreads();
    if (*bad_sh) return true;
    const float T = *T_sh;
    const int slot = tid % W, part = tid / W;
    float c2[CONFIG_DIM_MAX], c3[CONFIG_DIM_MAX];
    if (T > 0.f) flask::coeffs(xa, xa + n, xb, xb + n, n, T, c2, c3);
    if (T > 0.f && tid == 0 && flask::bounds_violated(xa, xa + n, c2, c3, n, T, lim)) atomicOr(bad_sh, 1);
    if (T > 0.f && part == 0) {
        float q[CONFIG_DIM_MAX], v[CONFIG_DIM_MAX], a[CONFIG_DIM_MAX];
#ifdef PRRAX_FLASK_ID
        // torque (nonlinear in the state): sampled at t = 0 and the W sample times
        if (lim.tmax) {
            float tq[CONFIG_DIM_MAX];
            for (int r = (slot == 0 ? -1 : 0); r <= 0; ++r) {
                flask::eval(xa, xa + n, c2, c3, n, T * (slot + 1 + r) / W, q, v, a);
                PRRAX_FLASK_ID(q, v, a, tq);
                for (int k = 0; k < n; ++k) if (fabsf(tq[k]) > lim.tmax[k]) atomicOr(bad_sh, 1);
            }
        }
#endif
        flask::eval(xa, xa + n, c2, c3, n, T * (slot + 1) / W, q, v, a);
        for (int k = 0; k < n; ++k) wp_sh[slot * CONFIG_DIM_MAX + k] = q[k];
    }
    __syncthreads();
    if (*bad_sh) return true;
    // Collision along the CURVE: sample `slot` covers (t_w, t_{w+1}], split into as many pieces as
    // its chord needs at VAMP resolution (as prrtc_gap_invalid does for a line), each piece's end
    // evaluated on the cubic; lanes (`part`) split the collision pairs.
    if ((ctx.enabled || tsr) && T > 0.f) {
        const float* prev = slot == 0 ? xa : wp_sh + (slot - 1) * CONFIG_DIM_MAX;
        const float* cur = wp_sh + slot * CONFIG_DIM_MAX;
        float d2 = 0.f;
        for (int k = 0; k < n; ++k) { const float e = cur[k] - prev[k]; d2 += e * e; }
        const int m = min(PRRAX_GAP_MAX_SUB, max(1, static_cast<int>(ceilf(sqrtf(d2) * PRRAX_RESOLUTION))));
        float q[CONFIG_DIM_MAX], v[CONFIG_DIM_MAX], a[CONFIG_DIM_MAX];
        for (int s = 1; s <= m; ++s) {
            if (*(volatile int*)bad_sh) break;
            flask::eval(xa, xa + n, c2, c3, n, T * (slot + static_cast<float>(s) / m) / W, q, v, a);
            if (tsr && part == 0 && prrax_cons::tsr_error_all(cons, q) > tol2) { atomicOr(bad_sh, 1); break; }
            if (ctx.enabled && prrtc_config_in_collision(q, ctx, part)) { atomicOr(bad_sh, 1); break; }
        }
    }
    __syncthreads();
    return *bad_sh != 0;
}

// Nearest published node of tree `t` to state x (shared): its index (block-uniform).
__device__ int flask_nearest(const float* st, const int* par, int count, int mn, int n, const float* x, float vel_w,
                             float* sd, int* si) {
    const int tid = threadIdx.x;
    float best = FLT_MAX; int bi = 0;
    for (int i = tid; i < count; i += blockDim.x) {
        if (volatile_load(&par[i]) < 0) continue;
        float d2 = 0.f;
        for (int k = 0; k < n; ++k) { const float e = __ldcg(&st[k * mn + i]) - x[k]; d2 += e * e; }
        for (int k = n; k < 2 * n; ++k) { const float e = __ldcg(&st[k * mn + i]) - x[k]; d2 += vel_w * e * e; }
        if (d2 < best) { best = d2; bi = i; }
    }
    sd[tid] = best; si[tid] = bi;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sd[tid + s] < sd[tid]) { sd[tid] = sd[tid + s]; si[tid] = si[tid + s]; }
        __syncthreads();
    }
    const int r = si[0];
    __syncthreads();
    return r;
}

__global__ void flask_plan_kernel(FlaskLayout L, FlaskSettings S) {
    const int p = blockIdx.x / S.bpp, bid = blockIdx.x % S.bpp, tid = threadIdx.x;
    const int n = L.n, SD = 2 * n;
    const size_t mn = L.max_nodes;
    __shared__ CollisionContext ctx;
    __shared__ float x_sh[FLASK_S_MAX], near_sh[FLASK_S_MAX], wp_sh[PRRTC_BLOCK_THREADS_MAX * CONFIG_DIM_MAX];
    __shared__ float sd[PRRTC_BLOCK_THREADS_MAX];
    __shared__ int si[PRRTC_BLOCK_THREADS_MAX];
    __shared__ int bad_sh, node_sh, stop_sh;
    __shared__ float T_sh, c_sh, T1_sh, c1_sh;
    __shared__ unsigned rng;
    const float* cons = L.cons ? L.cons + (size_t)p * PRRAX_CONS_SIZE : nullptr;
    const bool constrained = cons && cons[CONS_N] > 0.f;
    if (tid == 0) {
        const size_t wp = L.world_per_problem ? p : 0;
        ctx.world_spheres = L.w_sph + wp * L.max_w[0] * 4;
        ctx.world_capsules = L.w_cap + wp * L.max_w[1] * 7;
        ctx.world_boxes = L.w_box + wp * L.max_w[2] * 15;
        ctx.world_halfspaces = L.w_hs + wp * L.max_w[3] * 6;
        const float* k = L.counts + p * 4;
        ctx.n_world_spheres = (int)k[0]; ctx.n_world_capsules = (int)k[1];
        ctx.n_world_boxes = (int)k[2]; ctx.n_world_halfspaces = (int)k[3];
        ctx.enabled = (prrax::n_spheres > 0 && (k[0] > 0.f || k[1] > 0.f || k[2] > 0.f || k[3] > 0.f ||
                                                prrax::n_self_pairs > 0)) ? 1 : 0;
        ctx.collision_mode = S.collision_mode;
        ctx.collision_margin = S.collision_margin;
        ctx.solved = L.solved + p;
        rng = (blockIdx.x + 1) * 2654435761u ^ 0x9e3779b9u;
    }
    __syncthreads();
    auto rand01 = [&]() -> float {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        return (static_cast<float>(rng >> 8) + 0.5f) * (1.0f / 16777216.0f);
    };
    float* st[2] = {L.states + (size_t)p * 2 * SD * mn, L.states + ((size_t)p * 2 + 1) * SD * mn};
    int* par[2] = {L.par + p * 2 * mn, L.par + (p * 2 + 1) * mn};
    float* g[2] = {L.g + p * 2 * mn, L.g + (p * 2 + 1) * mn};
    float* Te[2] = {L.T + p * 2 * mn, L.T + (p * 2 + 1) * mn};

    // The direct BVP start -> goal first (FLASK-RRTConnect's shortcut to the goal; with
    // max_iters = 1 the planner is a batched direct-edge validator)
    if (bid == 0) {
        if (tid < SD) { x_sh[tid] = st[0][tid * mn]; near_sh[tid] = st[1][tid * mn]; }
        __syncthreads();
        if (!flask_edge_invalid(x_sh, near_sh, n, S.lim, ctx, wp_sh, &bad_sh, &T1_sh, &c1_sh, cons) && tid == 0 &&
            atomicCAS(&L.solved[p], 0, 1) == 0) {
            L.conn[2 * p] = L.conn[2 * p + 1] = 0;
            L.conn_c[p] = c1_sh; L.conn_T[p] = T1_sh;
        }
        __syncthreads();
    }
    for (int it = 0; ; ++it) {
        if (tid == 0) {
            stop_sh = volatile_load(&L.solved[p]) != 0 || atomicAdd(&L.iters[p], 1) >= S.max_iters;
            if (!stop_sh)
                for (int k = 0; k < n; ++k) {
                    x_sh[k] = S.lim.lo[k] + rand01() * (S.lim.hi[k] - S.lim.lo[k]);
                    x_sh[n + k] = (2.f * rand01() - 1.f) * S.vel_scale * S.lim.vmax[k];
                }
            node_sh = 0;
        }
        __syncthreads();
        if (stop_sh) break;

        const int t = (bid + it) & 1, o = 1 - t;

        // extend
        const int near = flask_nearest(st[t], par[t], volatile_load(&L.done[p * 2 + t]), mn, n, x_sh, S.vel_w, sd, si);
        if (tid == 0) {
            float d2 = 0.f;
            for (int k = 0; k < SD; ++k) near_sh[k] = __ldcg(&st[t][k * mn + near]);
            for (int k = 0; k < n; ++k) { const float e = x_sh[k] - near_sh[k]; d2 += e * e; }
            const float d = sqrtf(d2);
            if (d > S.step)
                for (int k = 0; k < n; ++k) x_sh[k] = near_sh[k] + (x_sh[k] - near_sh[k]) * (S.step / d);
            // the steered sample onto the TSR (McVAMP's projection)
            if (constrained) node_sh = !prrax_cons::tsr_project(cons, x_sh, S.lim.lo, S.lim.hi);
        }
        __syncthreads();
        if (node_sh) continue;
        const bool bad = t == 0 ? flask_edge_invalid(near_sh, x_sh, n, S.lim, ctx, wp_sh, &bad_sh, &T_sh, &c_sh, cons)
                                : flask_edge_invalid(x_sh, near_sh, n, S.lim, ctx, wp_sh, &bad_sh, &T_sh, &c_sh, cons);
        if (bad) continue;
        if (tid == 0) {
            const int idx = atomicAdd(&L.size[p * 2 + t], 1);
            if (idx >= (int)mn) { atomicCAS(&L.solved[p], 0, -1); node_sh = -1; }
            else {
                node_sh = idx;
                for (int k = 0; k < SD; ++k) st[t][k * mn + idx] = x_sh[k];
                g[t][idx] = __ldcg(&g[t][near]) + c_sh;
                Te[t][idx] = T_sh;
                __threadfence();
                atomicExch(&par[t][idx], near);
                atomicAdd(&L.done[p * 2 + t], 1);
            }
        }
        __syncthreads();
        const int node = node_sh;
        if (node < 0) break;

        // connect: the direct BVP to the other tree's nearest node
        const int m = flask_nearest(st[o], par[o], volatile_load(&L.done[p * 2 + o]), mn, n, x_sh, S.vel_w, sd, si);
        if (tid == 0) for (int k = 0; k < SD; ++k) near_sh[k] = __ldcg(&st[o][k * mn + m]);
        __syncthreads();
        const bool bad1 = t == 0 ? flask_edge_invalid(x_sh, near_sh, n, S.lim, ctx, wp_sh, &bad_sh, &T1_sh, &c1_sh, cons)
                                 : flask_edge_invalid(near_sh, x_sh, n, S.lim, ctx, wp_sh, &bad_sh, &T1_sh, &c1_sh, cons);
        if (!bad1 && tid == 0 && atomicCAS(&L.solved[p], 0, 1) == 0) {
            L.conn[2 * p + 0] = t == 0 ? node : m;
            L.conn[2 * p + 1] = t == 0 ? m : node;
            L.conn_c[p] = c1_sh;
            L.conn_T[p] = T1_sh;
        }
        __syncthreads();
    }
}

// The solution's knots, start to goal: states [max_knots, S] and the duration of the edge INTO
// each knot (knot 0: 0). n_knots 0 if unsolved (or more than max_knots). One thread per problem.
__global__ void flask_path_kernel(FlaskLayout L, float* knots, float* dur, int* n_knots, float* cost, int max_knots) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= L.batch) return;
    n_knots[p] = 0; cost[p] = FLT_MAX;
    if (L.solved[p] != 1) return;
    const int SD = 2 * L.n;
    const size_t mn = L.max_nodes;
    const float* sa = L.states + (size_t)p * 2 * SD * mn;
    const float* sb = L.states + ((size_t)p * 2 + 1) * SD * mn;
    const int* pa = L.par + p * 2 * mn;
    const int* pb = L.par + (p * 2 + 1) * mn;
    const float* Ta = L.T + p * 2 * mn;
    const float* Tb = L.T + (p * 2 + 1) * mn;
    int na = 1, nb = 1;
    for (int i = L.conn[2 * p]; i != 0; i = pa[i]) ++na;
    for (int i = L.conn[2 * p + 1]; i != 0; i = pb[i]) ++nb;
    if (na + nb > max_knots) return;
    float* K = knots + (size_t)p * max_knots * SD;
    float* D = dur + (size_t)p * max_knots;
    // start tree, root first: walk back from the connection node, filling from the end
    int k = na - 1;
    for (int i = L.conn[2 * p]; ; i = pa[i], --k) {
        for (int d = 0; d < SD; ++d) K[k * SD + d] = sa[d * mn + i];
        D[k] = i == 0 ? 0.f : Ta[i];
        if (i == 0) break;
    }
    // goal tree, connection node first: each knot's incoming edge is the bridge or the edge
    // from its child (Tb of the previous node: goal-tree edges run node -> parent in time)
    k = na;
    float into = L.conn_T[p];
    for (int i = L.conn[2 * p + 1]; ; i = pb[i], ++k) {
        for (int d = 0; d < SD; ++d) K[k * SD + d] = sb[d * mn + i];
        D[k] = into;
        into = Tb[i];
        if (i == 0) break;
    }
    n_knots[p] = na + nb;
    cost[p] = L.g[p * 2 * mn + L.conn[2 * p]] + L.conn_c[p] + L.g[(p * 2 + 1) * mn + L.conn[2 * p + 1]];
}
