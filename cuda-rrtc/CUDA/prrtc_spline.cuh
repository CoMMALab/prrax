// C1 Akima smoothing of planned edge polylines (after cpRRTC + MR. POP), validated along the curve,
// with TSR-aware knot refinement.
//
// Per edge (one block): the polyline's vertices (duplicates merged) are interpolated per joint by
// Akima's local cubic Hermite slopes (Akima 1970: each knot's slope from its four neighbouring
// secants -- a knot only moves the curve near it, little overshoot, so the curve hugs the validated
// polyline), in chord-length parameter s (|dq/ds| ~ 1). The curve is C1 (a continuous tangent
// space), vertices and endpoints exact, its end tangents its own (Akima's end conditions).
//
// Validation: each Hermite piece gets ceil(|control polygon| * PRRAX_RESOLUTION) samples (the
// control polygon bounds the piece's length, so VAMP's resolution holds), and every sample passes
// the planner's own binary collision check (lanes splitting the pairs) and, when the edge is
// constrained, its TSRs (hinged error within CONS_TOL).
//
// Refinement: a curve that bulges out of the TSR between knots (a polyline's projected waypoints
// sit on the band's boundary) gets, for every piece that left it, the piece's midpoint projected
// onto the TSR (prrax's own projection) as a new knot, and is refitted and revalidated -- up to
// PRRAX_SPLINE_ROUNDS times. A collision, or a curve still outside the TSR after the last round,
// fails the edge (status < 0): with smoothing on, the caller rejects it.
// Requires prrtc_planner.cu (CollisionContext, prrtc_config_in_collision) and the constraints.
#pragma once

#define PRRAX_SPLINE_MAX_KNOTS 512
#define PRRAX_SPLINE_ROUNDS 6

struct SplineJob {
    const float* paths;        // [n, max_path, dim] polylines
    const int* lens;           // [n] vertex counts
    const int* solved;         // [n] the planner's verdict (1: a path to smooth)
    const float *lo, *hi;      // [dim] joint limits (the TSR projection clamps to them)
    const float *w_sph, *w_cap, *w_box, *w_hs, *counts, *cons;   // per-problem worlds (prrax packing)
    int max_w[4], max_path, dim, collision_mode;
    float collision_margin;
    float* dense;              // [n, max_dense, dim] out: the sampled curve
    int* dense_len;            // [n]
    int* status;               // [n] 1 valid curve; 0 no path; -1 over the knot / sample caps; -2 leaves
                               // the TSR (after refinement); -3 collides
    int max_dense;
};

__device__ __forceinline__ void hermite(const float* p0, const float* m0, const float* p1, const float* m1, float h,
                                        float u, int dim, float* q) {
    const float u2 = u * u, u3 = u2 * u;
    const float h00 = 2.f * u3 - 3.f * u2 + 1.f, h10 = u3 - 2.f * u2 + u, h01 = -2.f * u3 + 3.f * u2, h11 = u3 - u2;
    for (int d = 0; d < dim; ++d) q[d] = h00 * p0[d] + h10 * h * m0[d] + h01 * p1[d] + h11 * h * m1[d];
}

__global__ void prrax_spline_kernel(SplineJob J) {
    const int p = blockIdx.x, tid = threadIdx.x, dim = J.dim;
    constexpr int Q = grim::robot::n_q;                            // knot row stride
    __shared__ float X[PRRAX_SPLINE_MAX_KNOTS * Q];                // knots
    __shared__ float M[PRRAX_SPLINE_MAX_KNOTS * Q];                // slopes dq/ds
    __shared__ float S[PRRAX_SPLINE_MAX_KNOTS];                    // chord-length parameter
    __shared__ int off[PRRAX_SPLINE_MAX_KNOTS];                    // first sample of each piece
    __shared__ int out_piece[PRRAX_SPLINE_MAX_KNOTS];              // the piece left the TSR this round
    __shared__ int mk_sh, total_sh, coll_sh, tsr_sh, status_sh;
    __shared__ CollisionContext ctx;
    const float* cons = J.cons + (size_t)p * PRRAX_CONS_SIZE;
    const bool tsr = cons[CONS_N] > 0.f;
    const float tol2 = cons[CONS_TOL] * cons[CONS_TOL];
    if (tid == 0) {
        J.dense_len[p] = 0;
        status_sh = J.solved[p] == 1 && J.lens[p] >= 2 ? 1 : 0;
        ctx.world_spheres = J.w_sph + (size_t)p * J.max_w[0] * 4;
        ctx.world_capsules = J.w_cap + (size_t)p * J.max_w[1] * 7;
        ctx.world_boxes = J.w_box + (size_t)p * J.max_w[2] * 15;
        ctx.world_halfspaces = J.w_hs + (size_t)p * J.max_w[3] * 6;
        const float* k = J.counts + p * 4;
        ctx.n_world_spheres = (int)k[0]; ctx.n_world_capsules = (int)k[1];
        ctx.n_world_boxes = (int)k[2]; ctx.n_world_halfspaces = (int)k[3];
        ctx.enabled = (prrax::n_spheres > 0 && (k[0] > 0.f || k[1] > 0.f || k[2] > 0.f || k[3] > 0.f ||
                                                prrax::n_self_pairs > 0)) ? 1 : 0;
        ctx.collision_mode = J.collision_mode; ctx.collision_margin = J.collision_margin; ctx.solved = nullptr;

        // knots: the vertices with duplicates merged (a zero-length piece breaks the slopes), the
        // end vertex kept exactly
        const int m = J.lens[p];
        const float* path = J.paths + (size_t)p * J.max_path * dim;
        auto dist = [&](const float* u, const float* v) {
            float d2 = 0.f;
            for (int d = 0; d < dim; ++d) { const float e = u[d] - v[d]; d2 += e * e; }
            return sqrtf(d2);
        };
        int c = 0;
        for (int k = 0; status_sh == 1 && k < m; ++k) {
            const float* v = path + (size_t)k * dim;
            if (c > 0 && dist(v, X + (c - 1) * Q) < 1e-6f) {
                if (k == m - 1 && c > 1) for (int d = 0; d < dim; ++d) X[(c - 1) * Q + d] = v[d];   // keep the end exact
                continue;
            }
            if (c == PRRAX_SPLINE_MAX_KNOTS) { status_sh = -1; break; }
            for (int d = 0; d < dim; ++d) X[c * Q + d] = v[d];
            ++c;
        }
        mk_sh = c;
        if (status_sh == 1 && c < 2) status_sh = 0;
        J.status[p] = status_sh;
    }
    __syncthreads();
    if (status_sh != 1) return;

    float* out = J.dense + (size_t)p * J.max_dense * dim;
    const int lane = tid % PRRAX_WP_LANES, groups = blockDim.x / PRRAX_WP_LANES;
    for (int round = 0; ; ++round) {
        const int mk = mk_sh;
        if (tid == 0) {   // chord-length parameter
            S[0] = 0.f;
            for (int k = 1; k < mk; ++k) {
                float d2 = 0.f;
                for (int d = 0; d < dim; ++d) { const float e = X[k * Q + d] - X[(k - 1) * Q + d]; d2 += e * e; }
                S[k] = S[k - 1] + sqrtf(d2);
            }
        }
        __syncthreads();
        // Akima slopes, one joint per thread: secants d_k (k = 0..mk-2), extended by two on each side
        if (tid < dim) {
            const int d = tid, last = mk - 2;
            auto raw = [&](int k) { return (X[(k + 1) * Q + d] - X[k * Q + d]) / (S[k + 1] - S[k]); };
            const float d0 = raw(0), d1 = last >= 1 ? raw(1) : d0, dl = raw(last), dl1 = last >= 1 ? raw(last - 1) : dl;
            auto sec = [&](int k) -> float {
                if (k >= 0 && k <= last) return raw(k);
                if (k == -1) return 2.f * d0 - d1;
                if (k == -2) return 2.f * (2.f * d0 - d1) - d0;
                if (k == last + 1) return 2.f * dl - dl1;
                return 2.f * (2.f * dl - dl1) - dl;   // last + 2
            };
            for (int k = 0; k < mk; ++k) {
                const float w1 = fabsf(sec(k + 1) - sec(k)), w2 = fabsf(sec(k - 1) - sec(k - 2));
                M[k * Q + d] = (w1 + w2 > 1e-9f) ? (w1 * sec(k - 1) + w2 * sec(k)) / (w1 + w2) : 0.5f * (sec(k - 1) + sec(k));
            }
        }
        __syncthreads();
        // samples per piece, from its Bezier control polygon (an upper bound on the piece's length)
        if (tid == 0) {
            int t = 1;                                   // sample 0: the start vertex
            for (int k = 0; k + 1 < mk; ++k) {
                off[k] = t;
                out_piece[k] = 0;
                const float h = S[k + 1] - S[k];
                float e0 = 0.f, e1 = 0.f, e2 = 0.f;
                for (int d = 0; d < dim; ++d) {
                    const float a = X[k * Q + d] + h / 3.f * M[k * Q + d], b = X[(k + 1) * Q + d] - h / 3.f * M[(k + 1) * Q + d];
                    const float x0 = a - X[k * Q + d], x1 = b - a, x2 = X[(k + 1) * Q + d] - b;
                    e0 += x0 * x0; e1 += x1 * x1; e2 += x2 * x2;
                }
                t += max(1, static_cast<int>(ceilf((sqrtf(e0) + sqrtf(e1) + sqrtf(e2)) * PRRAX_RESOLUTION)));
            }
            off[mk - 1] = t;
            total_sh = t;
            coll_sh = 0; tsr_sh = 0;
            if (t > J.max_dense) status_sh = -1;
        }
        __syncthreads();
        if (status_sh != 1) break;
        // every sample, PRRAX_WP_LANES lanes each (they split the collision pairs): a collision ends
        // the edge; TSR violations mark their piece for refinement
        const int N = total_sh;
        for (int i = tid / PRRAX_WP_LANES; i < N; i += groups) {
            if (*(volatile int*)&coll_sh) break;
            float q[CONFIG_DIM_MAX];
            int k = 0;
            if (i == 0) {
                for (int d = 0; d < dim; ++d) q[d] = X[d];
            } else {
                int lo = 0, hi = mk - 2;                 // the last piece starting at or before i holds it
                while (lo < hi) { const int md = (lo + hi + 1) / 2; if (off[md] <= i) lo = md; else hi = md - 1; }
                k = lo;
                hermite(X + k * Q, M + k * Q, X + (k + 1) * Q, M + (k + 1) * Q, S[k + 1] - S[k],
                        static_cast<float>(i - off[k] + 1) / (off[k + 1] - off[k]), dim, q);
            }
            if (lane == 0) for (int d = 0; d < dim; ++d) out[(size_t)i * dim + d] = q[d];
            if (lane == 0 && tsr && prrax_cons::tsr_error_all(cons, q) > tol2) { out_piece[k] = 1; tsr_sh = 1; }
            if (prrtc_config_in_collision(q, ctx, lane)) { atomicOr(&coll_sh, 1); break; }
        }
        __syncthreads();
        if (tid == 0) {
            if (coll_sh) status_sh = -3;
            else if (!tsr_sh) { status_sh = 2; J.dense_len[p] = N; }          // valid
            else if (round + 1 >= PRRAX_SPLINE_ROUNDS) status_sh = -2;
            else {
                // refine: the projected midpoint of every piece that left the TSR becomes a knot
                int add = 0;
                for (int k = 0; k + 1 < mk; ++k) add += out_piece[k];
                if (mk + add > PRRAX_SPLINE_MAX_KNOTS) status_sh = -1;
                for (int k = mk - 2, top = mk - 1 + add; status_sh == 1 && k >= 0; --k) {
                    for (int d = 0; d < dim; ++d) X[top * Q + d] = X[(k + 1) * Q + d];   // knot k+1 moves up
                    --top;
                    if (out_piece[k]) {
                        float q[CONFIG_DIM_MAX];
                        hermite(X + k * Q, M + k * Q, X + (k + 1) * Q, M + (k + 1) * Q, S[k + 1] - S[k], 0.5f, dim, q);
                        if (!prrax_cons::tsr_project(cons, q, J.lo, J.hi)) { status_sh = -2; break; }
                        for (int d = 0; d < dim; ++d) X[top * Q + d] = q[d];
                        --top;
                    }
                }
                mk_sh = mk + add;
            }
        }
        __syncthreads();
        if (status_sh != 1) break;
    }
    if (tid == 0) J.status[p] = status_sh == 2 ? 1 : status_sh;
}
