/**
 * Manifold constraints for the pRRTC planner: Task Space Region (TSR) constraints, their
 * projection, and the constrained extend step used by both tree extensions.
 *
 * A TSR constraint (Berenson et al.) bounds the pose of an end-effector frame:
 *     x = [t; log R] of  T_r = wTr^-1 * T_ee * rTe^-1,      d = x - clamp(x, lo, hi)
 * where T_ee is the world pose of the constrained frame (grim::robot::ee_joint(e)), wTr the
 * TSR reference frame in the world and rTe the end-effector offset (McVAMP's convention).
 * Several constraints are satisfied by cyclic projection (McVAMP): one damped LM step
 * (J^T (J J^T + lambda I)^-1 d, equal to McVAMP's OuterLM step) per constraint in turn, until
 * every |d| < tol. Jacobians come from GRiM: cricket2's traced ee_pose_jacobian for a single
 * end-effector, the geometric Jacobian of traced frame poses otherwise.
 *
 * Constrained extend from q_from toward q_target, run by all threads of a block, fills the
 * block's W = blockDim.x waypoints (on the manifold, gaps <= gap_factor * dist / W):
 *   mode 0 (McVAMP): each thread projects q_steer + eps_tid * v (eps ~ N(0, sigma^2), thread 0
 *       unperturbed); the lowest converged thread's result is q_proj. Then waypoint t projects
 *       lerp(q_from, q_proj, (t+1)/W) independently and the gaps are checked.
 *   mode 1 (cpRRTC): waypoints lerp(q_from, q_steer, (t+1)/W) are projected jointly with a
 *       smoothing term on the gap to the previous waypoint; a prefix counter `prog` freezes
 *       waypoints once they and all before them are valid; the end is free.
 * ponytail: no recursive subdivision of a segment whose gaps are too wide (McVAMP); it fails
 * the extension instead. Add it if hard constraints show a low extension success rate.
 *
 * Parameters (one problem): float[PRRAX_CONS_SIZE], see the CONS_* offsets; packed by
 * jax/prrtc.py:make_tsr_constraint/_pack_constraints.
 */
#pragma once

#include "robot.cuh"
#include "glass.cuh"         // GLASS: thread-tier BLAS / LAPACK / Lie-group ops
#include "glass_solve.cuh"   // GRiM: tier_posv over GLASS

// Threads per edge waypoint (pRRTC uses 4): blocks run PRRAX_WP_LANES * W threads for W waypoints;
// the lanes of a waypoint split its collision check and every thread is a stage-1 steer particle.
#ifndef PRRAX_WP_LANES
#define PRRAX_WP_LANES 4
#endif

#define PRRAX_CONS_MAX 4
#define PRRAX_CONS_HDR 12
#define PRRAX_CONS_STRIDE 28
#define PRRAX_CONS_SIZE (PRRAX_CONS_HDR + PRRAX_CONS_MAX * PRRAX_CONS_STRIDE)

// Header slots.
#define CONS_N 0           // number of constraints (0 = unconstrained)
#define CONS_MODE 1        // 0 McVAMP two-stage, 1 cpRRTC smoothed
#define CONS_TOL 2         // convergence: |d| < tol for every constraint
#define CONS_ITERS 3       // projection iteration cap
#define CONS_LAMBDA 4      // LM damping
#define CONS_ALPHA 5       // step rate
#define CONS_MAX_STEP 6    // a projection step longer than this diverged
#define CONS_SIGMA 7       // McVAMP: perturbation scale of the steer particles
#define CONS_GAP 8         // max waypoint gap = gap * dist / W
#define CONS_CONNECT_EPS 9 // connect: reached the target within this distance
// Per-constraint slots (at PRRAX_CONS_HDR + c * PRRAX_CONS_STRIDE).
#define CON_EE 0           // index into grim::robot::ee_joint
#define CON_RTE_INV 1      // rTe^-1 [wxyz xyz]
#define CON_WTR_INV 8      // wTr^-1 [wxyz xyz]
#define CON_LO 15          // lower bounds [6]
#define CON_HI 21          // upper bounds [6]

#ifndef CONFIG_DIM_MAX
#define CONFIG_DIM_MAX 16
#endif

namespace prrax_cons {

namespace R = grim::robot;
constexpr int n_ee_max = R::n_ee > 0 ? R::n_ee : 1;

// x of constraint `con` at end-effector pose T_ee; d the hinged error; returns |d|^2.
__device__ __forceinline__ float tsr_error(const float* __restrict__ con, const float* T_ee,
                                           float* __restrict__ T_obj, float* __restrict__ x,
                                           float* __restrict__ d)
{
    float T_r[7];
    const float ident[7] = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    se3_compose(T_ee, con + CON_RTE_INV, T_obj);
    se3_compose(con + CON_WTR_INV, T_obj, T_r);
    pose_residual(T_r, ident, x);   // [t; log R]
    float e2 = 0.f;
    for (int i = 0; i < 6; ++i) {
        d[i] = x[i] - fminf(fmaxf(x[i], con[CON_LO + i]), con[CON_HI + i]);
        e2 += d[i] * d[i];
    }
    return e2;
}


// Hinged error d of constraint `con` at q (returns |d|^2) and, if jac, Jx = dx/dq [6 x n_q].
__device__ float tsr_eval(const float* __restrict__ con, const float* __restrict__ cp,
                          const float* __restrict__ q, float* __restrict__ d, float* __restrict__ Jx,
                          bool jac)
{
    constexpr int dim = R::n_q;
    float T[R::n_frames * 7], r[6 * n_ee_max], J[6 * n_ee_max * (R::n_solved > 0 ? R::n_solved : 1)];
    float tgt[7 * n_ee_max];
    for (int e = 0; e < n_ee_max; ++e) {
        tgt[e * 7] = 1.f;
        for (int k = 1; k < 7; ++k) tgt[e * 7 + k] = 0.f;
    }
    const int e = static_cast<int>(con[CON_EE]);
#if defined(GRIM_TRACED_KINEMATICS) && !defined(GRIM_TRACED_FRAMES_ONLY)
    // One end-effector: cricket2's straight-line pose + Jacobian in a single pass.
    float qt[R::traced::n_q], Jt[6 * R::traced::n_q], T_ee_buf[7];
    R::gather_q(q, nullptr, qt);
    R::traced::ee_pose_jacobian(qt, T_ee_buf, Jt);
    R::scatter_jacobian(Jt, J);
    const float* T_ee = T_ee_buf;
    (void)T; (void)r; (void)tgt;
#else
    R::frame_poses(q, nullptr, T);
    R::residual_and_jacobian(q, nullptr, tgt, r, J);
    const float* T_ee = T + R::ee_joint(e) * 7;
#endif

    float T_obj[7], x[6];
    const float e2 = tsr_error(con, T_ee, T_obj, x, d);
    if (!jac) return e2;

    // Jx = d x / d q from the geometric Jacobian Je (rows: world linear velocity of the frame
    // origin, world angular velocity), with R = rotation of wTr^-1 and off = the offset point:
    //     Jx_t = R (Jv - [off]x Jw),      Jx_r = Jl^-1(log R_r) R Jw.
    // Row-major 3 x n_q blocks are column-major n_q x 3, so each block is one GLASS gemm
    // C^T = B^T M^T with the 3x3 factor M (column-major) transposed.
    constexpr uint32_t N = dim;
    const float* Je = J + e * 6 * R::n_solved;
    float off[3], Rm[9], S[9], M2[9], Jl[9], M3[9];
    for (int k = 0; k < 3; ++k) off[k] = T_obj[4 + k] - T_ee[4 + k];
    glass::thread::quat_to_rot<float, glass::QuatLayout::wxyz>(con + CON_WTR_INV, Rm);
    glass::thread::skew(off, S);
    glass::thread::gemm<float, 3, 3, 3>(-1.f, Rm, S, M2);                      // -R [off]x
    glass::thread::so3_left_jacobian_inv(x + 3, Jl);
    glass::thread::gemm<float, 3, 3, 3>(1.f, Jl, Rm, M3);                      // Jl^-1 R
    glass::thread::gemm<float, N, 3, 3, false, true>(1.f, Je, Rm, Jx);         // R Jv
    glass::thread::gemm<float, N, 3, 3, false, true>(1.f, Je + 3 * N, M2, 1.f, Jx);   // - R [off]x Jw
    glass::thread::gemm<float, N, 3, 3, false, true>(1.f, Je + 3 * N, M3, Jx + 3 * N);
    (void)cp;
    return e2;
}

// out = Jx^T (Jx Jx^T + lambda I)^-1 b for the 6-row, row-major Jx (= column-major Jx^T).
__device__ __forceinline__ void damped_pinv_apply(const float* Jx, const float* b, float lambda,
                                                  float* out)
{
    constexpr uint32_t N = R::n_q;
    float A[36], y[6];
    glass::thread::gemm<float, 6, 6, N, /*TRANSPOSE_A=*/true>(1.f, Jx, Jx, A);   // Jx Jx^T
    for (int i = 0; i < 6; ++i) { A[i * 6 + i] += lambda; y[i] = b[i]; }
    int fail = 0;
    grim::tier_posv<grim::Tier::Thread, float, 6>(A, y, &fail);
    glass::thread::gemv<float, 6, N, /*TRANSPOSE=*/true, /*ROW_MAJOR=*/true>(1.f, Jx, y, out);
}

// One LM step of constraint `con` at q (in place). Returns |d|^2 before the step and
// writes the step length to *step_len (0 when already satisfied).
__device__ float tsr_step(const float* __restrict__ con, const float* __restrict__ cp,
                          float* __restrict__ q, float* step_len)
{
    constexpr int dim = R::n_q;
    float d[6], Jx[6 * dim], g[dim];
    *step_len = 0.f;
    const float e2 = tsr_eval(con, cp, q, d, Jx, true);
    if (e2 < cp[CONS_TOL] * cp[CONS_TOL]) return e2;
    damped_pinv_apply(Jx, d, cp[CONS_LAMBDA], g);
    glass::thread::axpy<float, dim>(-cp[CONS_ALPHA], g, q);
    *step_len = cp[CONS_ALPHA] * glass::thread::nrm2<float, dim>(g);
    return e2;
}

// Project q onto the intersection of the constraints (in place, clamped to the joint limits).
// Early exit: gives up as soon as *stop < me (another thread's result makes this one moot).
__device__ bool tsr_project(const float* __restrict__ cp, float* __restrict__ q,
                            const float* __restrict__ lo, const float* __restrict__ hi,
                            const volatile int* stop = nullptr, int me = 0,
                            const volatile int* abort = nullptr)
{
    constexpr int dim = R::n_q;
    const int n = static_cast<int>(cp[CONS_N]);
    const int iters = static_cast<int>(cp[CONS_ITERS]);
    for (int it = 0; it <= iters; ++it) {
        if (stop && *stop < me) return false;
        if (abort && *abort != 0) return false;   // the problem is already solved
        bool moved = false;
        for (int c = 0; c < n; ++c) {
            float len;
            tsr_step(cp + PRRAX_CONS_HDR + c * PRRAX_CONS_STRIDE, cp, q, &len);
            if (len > cp[CONS_MAX_STEP]) return false;   // diverging
            moved |= len > 0.f;
        }
        if (!moved) return true;   // every constraint satisfied at this q
        if (it == iters) return false;
        for (int a = 0; a < dim; ++a) q[a] = fminf(fmaxf(q[a], lo[a]), hi[a]);
    }
    return false;
}

// |d|^2 summed over the constraints at q (no step).
__device__ float tsr_error_all(const float* __restrict__ cp, const float* __restrict__ q)
{
    float T[R::n_frames * 7];
    R::frame_poses(q, nullptr, T);
    float e2 = 0.f;
    for (int c = 0; c < static_cast<int>(cp[CONS_N]); ++c) {
        const float* con = cp + PRRAX_CONS_HDR + c * PRRAX_CONS_STRIDE;
        float T_obj[7], x[6], d[6];
        e2 += tsr_error(con, T + R::ee_joint(static_cast<int>(con[CON_EE])) * 7, T_obj, x, d);
    }
    return e2;
}

__device__ __forceinline__ float sq_dist(const float* a, const float* b)
{
    constexpr uint32_t N = R::n_q;
    float d[N];
    glass::thread::copy<float, N>(const_cast<float*>(a), d);   // GLASS takes non-const sources
    glass::thread::axpy<float, N>(-1.f, const_cast<float*>(b), d);
    return glass::thread::dot<float, N>(d, d);
}

// Counter-based standard normal for thread `tid` of block `bid` at call `iter` (Box-Muller
// over a hash; no RNG state to carry).
__device__ __forceinline__ float hash_normal(unsigned bid, unsigned iter, unsigned tid)
{
    unsigned h = bid * 0x9E3779B1u ^ (iter + 0x7F4A7C15u) * 0x85EBCA77u ^ tid * 0xC2B2AE3Du;
    auto mix = [](unsigned v) { v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16; return v; };
    const unsigned a = mix(h), b = mix(h ^ 0x68E31DA4u);
    const float u1 = (a + 1.f) * 2.3283064e-10f, u2 = b * 2.3283064e-10f;
    return sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);
}

/**
 * Constrained extend from q_from toward q_target (shared memory, n_q floats each). Every
 * thread of the block must call it. On success the W = blockDim.x waypoints (the last is
 * the new end) are in wp_sh [W * CONFIG_DIM_MAX] and true is returned (uniformly).
 */
__device__ bool constrained_extend(const float* __restrict__ cp, const float* q_from,
                                   const float* q_target, float step,
                                   const float* __restrict__ lo, const float* __restrict__ hi,
                                   float* wp_sh, int* flag_sh, unsigned bid, unsigned call,
                                   const int* solved)
{
    const volatile int* abort = solved;   // projections stop once the problem is solved
    if (threadIdx.x == 0) *flag_sh = *abort != 0;
    __syncthreads();
    if (*flag_sh) return false;   // block-uniform
    constexpr int dim = R::n_q;
    // W waypoints, handled by the first W threads; every thread is a stage-1 particle.
    const int tid = threadIdx.x, W = blockDim.x / PRRAX_WP_LANES, NP = blockDim.x;
    const bool lead = tid < W;
    const float dist_full = sqrtf(sq_dist(q_from, q_target));
    if (dist_full < 1e-8f) return false;
    const float dist = fminf(dist_full, step);
    const float max_gap = cp[CONS_GAP] * dist / W;
    float q[CONFIG_DIM_MAX];
    float* mine = wp_sh + tid * CONFIG_DIM_MAX;

    float v[CONFIG_DIM_MAX];   // steer direction (tangent-space steering measured slower)
    for (int k = 0; k < dim; ++k) v[k] = (q_target[k] - q_from[k]) / dist_full;

    if (static_cast<int>(cp[CONS_MODE]) == 0) {
        // Stage 1: perturbed steer particles; the lowest converged thread wins.
        if (tid == 0) *flag_sh = NP;
        __syncthreads();
        const float eps = tid == 0 ? 0.f : cp[CONS_SIGMA] * hash_normal(bid, call, tid);
        for (int k = 0; k < dim; ++k) q[k] = q_from[k] + (dist + eps) * v[k];
        // Early exit: a thread stops once a lower thread has converged (it would lose).
        const bool ok = tsr_project(cp, q, lo, hi, flag_sh, tid, abort);
        if (ok) atomicMin(flag_sh, tid);
        __syncthreads();
        const int winner = *flag_sh;
        if (winner == NP) return false;
        if (tid == winner) for (int k = 0; k < dim; ++k) wp_sh[(W - 1) * CONFIG_DIM_MAX + k] = q[k];
        __syncthreads();
        const float* q_proj = wp_sh + (W - 1) * CONFIG_DIM_MAX;
        if (sq_dist(q_proj, q_from) > 4.f * dist * dist) return false;   // uniform

        // (McVAMP collision-checks q_proj before projecting the waypoints; on the GPU that
        // serializes one thread's check ahead of the parallel stage and measured slower.)
        if (tid == 0) *flag_sh = W;
        __syncthreads();

        // Stage 2: project the waypoints between q_from and q_proj independently; every
        // thread stops at the first failure (flag -1).
        float qp[CONFIG_DIM_MAX];
        for (int k = 0; k < dim; ++k) qp[k] = q_proj[k];
        __syncthreads();
        if (lead) {
            const float t = static_cast<float>(tid + 1) / W;
            for (int k = 0; k < dim; ++k) q[k] = q_from[k] + t * (qp[k] - q_from[k]);
            const bool wp_ok = (tid == W - 1) || tsr_project(cp, q, lo, hi, flag_sh, 0, abort);
            for (int k = 0; k < dim; ++k) mine[k] = (tid == W - 1) ? qp[k] : q[k];
            if (!wp_ok) atomicMin(flag_sh, -1);
        }
        __syncthreads();
        if (lead) {
            const float* prev = tid == 0 ? q_from : wp_sh + (tid - 1) * CONFIG_DIM_MAX;
            if (*flag_sh >= 0 && sq_dist(mine, prev) > max_gap * max_gap) atomicMin(flag_sh, -1);
        }
        __syncthreads();
        return *flag_sh >= 0;
    }

    // cpRRTC: joint projection of the lerp to q_steer with smoothing, prefix-frozen by prog.
    __shared__ int prog_sh;
    __shared__ int valid_sh[64];
    if (lead)
        for (int k = 0; k < dim; ++k)
            mine[k] = q_from[k] + (static_cast<float>(tid + 1) / W) * dist * v[k];
    if (tid == 0) { prog_sh = -1; *flag_sh = 0; }
    __syncthreads();
    const int iters = static_cast<int>(cp[CONS_ITERS]);
    const float tol2 = cp[CONS_TOL] * cp[CONS_TOL];
    for (int it = 0; it < iters; ++it) {
        const int prog = prog_sh;
        bool diverged = false;
        if (lead && tid > prog) {
            const float* prev = tid == 0 ? q_from : wp_sh + (tid - 1) * CONFIG_DIM_MAX;
            for (int k = 0; k < dim; ++k) q[k] = mine[k];
            const float gap = sqrtf(sq_dist(q, prev));
            // Task: one cyclic LM sweep (also measures the error at q).
            float e2 = 0.f;
            for (int c = 0; c < static_cast<int>(cp[CONS_N]); ++c) {
                float len;
                e2 += tsr_step(cp + PRRAX_CONS_HDR + c * PRRAX_CONS_STRIDE, cp, q, &len);
                diverged |= len > cp[CONS_MAX_STEP];
            }
            valid_sh[tid] = (e2 < tol2 && gap <= max_gap) ? 1 : 0;
            // Smoothing: pull toward the previous waypoint by the gap's excess over max_gap.
            if (gap > max_gap) {
                const float s = cp[CONS_ALPHA] * (gap - max_gap) / gap;
                for (int k = 0; k < dim; ++k) q[k] -= s * (mine[k] - prev[k]);
            }
            for (int k = 0; k < dim; ++k) q[k] = fminf(fmaxf(q[k], lo[k]), hi[k]);
        }
        if (diverged) atomicOr(flag_sh, 1);
        __syncthreads();
        if (tid == 0) {
            int p = prog;
            while (p + 1 < W && valid_sh[p + 1]) ++p;
            prog_sh = p;
        }
        if (lead && tid > prog && !valid_sh[tid]) for (int k = 0; k < dim; ++k) mine[k] = q[k];
        __syncthreads();
        if (*flag_sh != 0) return false;
        if (prog_sh == W - 1) return true;
    }
    return false;
}

}  // namespace prrax_cons
