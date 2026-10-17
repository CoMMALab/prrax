// C1 corner blending of shortcut paths (after MR. POP), validated along the blends.
//
// Per path (one block): duplicate vertices merged, every interior vertex v whose segments turn (by more than ~0.06 deg) gets
// its corner replaced by the quadratic Bezier A -> v -> C, A = v - r u_in, C = v + r u_out: tangent
// to both segments at A and C, so the path is C1 there (and only shorter: the blend lies inside
// the corner's triangle). r starts at min(r_max, half of either adjacent segment), so neighbouring
// blends never overlap. Each blend, its ends A and C included (in a TSR, MR. POP validated the
// segment's projected substeps, not its straight chord), is sampled at VAMP resolution and at most PRRAX_BLEND_MAX_TURN rad of turning per piece
// (so the returned waypoints, read as a polyline, turn by <= ~2 deg per step) and every sample passes
// the planner's binary collision check (lanes splitting the pairs) and the path's TSRs (hinged
// error within CONS_TOL). A failing corner's radius is halved, up to PRRAX_BLEND_ROUNDS times; a
// corner that still fails leaves the path non-C1 (status -2): the caller fails the path.
// Output: the vertices with each corner's A, blend samples, C in place of v.
// Requires prrtc_planner.cu (CollisionContext, prrtc_config_in_collision) and the constraints.
#pragma once

#define PRRAX_BLEND_ROUNDS 10
#define PRRAX_BLEND_MAX_TURN 0.035f   // rad (2 deg)

struct BlendJob {
    const float* paths;        // [n, max_path, dim] polylines
    const int* lens;           // [n]
    float* verts;              // [n, max_path, dim] scratch: the vertices with duplicates merged
    const float *w_sph, *w_cap, *w_box, *w_hs, *counts, *cons;   // per-problem worlds (prrax packing)
    int max_w[4], max_path, dim, collision_mode;
    float collision_margin, r_max;
    float* radius;             // [n, max_path] scratch: corner radii (0: no blend)
    int* fail;                 // [n, max_path] scratch: the corner failed this round
    float* out;                // [n, max_out, dim] out
    int* out_len;              // [n]
    int* status;               // [n] 1 C1; -1 over max_out; -2 a corner could not be blended
    int max_out;
};

// samples of a blend: VAMP resolution along its control polygon (2 r) and <= MAX_TURN per piece
__device__ __forceinline__ int blend_samples(float r, float turn) {
    return max(2, max(static_cast<int>(ceilf(2.f * r * PRRAX_RESOLUTION)),
                      static_cast<int>(ceilf(turn / PRRAX_BLEND_MAX_TURN))));
}

__device__ __forceinline__ void blend_point(const float* A, const float* v, const float* C, float t, int dim, float* q) {
    const float a = (1.f - t) * (1.f - t), b = 2.f * (1.f - t) * t, c = t * t;
    for (int d = 0; d < dim; ++d) q[d] = a * A[d] + b * v[d] + c * C[d];
}

__global__ void prrax_blend_kernel(BlendJob J) {
    const int p = blockIdx.x, tid = threadIdx.x, dim = J.dim;
    float* P = J.verts + (size_t)p * J.max_path * dim;
    __shared__ int m_sh;
    if (tid == 0) {   // merge duplicate vertices (a zero-length segment hides its corner)
        const float* in = J.paths + (size_t)p * J.max_path * dim;
        int c = 0;
        for (int k = 0; k < J.lens[p]; ++k) {
            float d2 = 0.f;
            if (c) for (int d = 0; d < dim; ++d) { const float e = in[k * dim + d] - P[(c - 1) * dim + d]; d2 += e * e; }
            if (c && d2 < 1e-12f) {
                if (k == J.lens[p] - 1) for (int d = 0; d < dim; ++d) P[(c - 1) * dim + d] = in[k * dim + d];   // exact end
                continue;
            }
            for (int d = 0; d < dim; ++d) P[c * dim + d] = in[k * dim + d];
            ++c;
        }
        m_sh = c;
    }
    __syncthreads();
    const int m = m_sh;
    float* R = J.radius + (size_t)p * J.max_path;
    int* F = J.fail + (size_t)p * J.max_path;
    __shared__ CollisionContext ctx;
    __shared__ int any_sh, status_sh;
    const float* cons = J.cons + (size_t)p * PRRAX_CONS_SIZE;
    const bool tsr = cons[CONS_N] > 0.f;
    const float tol2 = cons[CONS_TOL] * cons[CONS_TOL];
    // the corner's A and C (unit directions of its two segments, radius r)
    // the corner's A and C at radius r; returns its turning angle
    auto corner = [&](int k, float r, float* A, float* C) -> float {
        float li = 0.f, lo = 0.f, dot = 0.f;
        for (int d = 0; d < dim; ++d) {
            const float a = P[k * dim + d] - P[(k - 1) * dim + d], b = P[(k + 1) * dim + d] - P[k * dim + d];
            li += a * a; lo += b * b; dot += a * b;
        }
        li = sqrtf(li); lo = sqrtf(lo);
        const float turn = acosf(fminf(1.f, fmaxf(-1.f, dot / (li * lo))));
        for (int d = 0; d < dim; ++d) {
            A[d] = P[k * dim + d] - r * (P[k * dim + d] - P[(k - 1) * dim + d]) / li;
            C[d] = P[k * dim + d] + r * (P[(k + 1) * dim + d] - P[k * dim + d]) / lo;
        }
        return turn;
    };
    if (tid == 0) {
        status_sh = 1;
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
    }
    // initial radii: corners that turn get min(r_max, half of either segment)
    for (int k = tid; k < m; k += blockDim.x) {
        R[k] = 0.f; F[k] = 0;
        if (k == 0 || k == m - 1) continue;
        float li = 0.f, lo = 0.f, dot = 0.f;
        for (int d = 0; d < dim; ++d) {
            const float a = P[k * dim + d] - P[(k - 1) * dim + d], b = P[(k + 1) * dim + d] - P[k * dim + d];
            li += a * a; lo += b * b; dot += a * b;
        }
        if (dot / sqrtf(li * lo) > 1.f - 5e-7f) continue;   // straight through
        R[k] = fminf(J.r_max, 0.5f * fminf(sqrtf(li), sqrtf(lo)));
        F[k] = 1;   // to validate
    }
    __syncthreads();
    // validate the pending corners; a failing one halves its radius and is retried
    const int lane = tid % PRRAX_WP_LANES, groups = blockDim.x / PRRAX_WP_LANES;
    for (int round = 0; round <= PRRAX_BLEND_ROUNDS; ++round) {
        if (tid == 0) any_sh = 0;
        __syncthreads();
        for (int k = tid / PRRAX_WP_LANES; k < m; k += groups) {
            if (F[k] != 1) continue;   // group-uniform
            float A[CONFIG_DIM_MAX], C[CONFIG_DIM_MAX], q[CONFIG_DIM_MAX];
            const int n = blend_samples(R[k], corner(k, R[k], A, C));
            bool bad = false;
            for (int s = 0; s <= n && !bad; ++s) {   // A and C too: in a TSR the input chord itself was never checked
                blend_point(A, P + k * dim, C, static_cast<float>(s) / n, dim, q);
                if (lane == 0 && tsr && prrax_cons::tsr_error_all(cons, q) > tol2) bad = true;
                if (prrtc_config_in_collision(q, ctx, lane)) bad = true;
            }
            const int wl = tid & (WARP_SIZE - 1);   // this corner's lane group within the warp
            const unsigned mask = ((1u << PRRAX_WP_LANES) - 1u) << (wl / PRRAX_WP_LANES * PRRAX_WP_LANES);
            bad = __any_sync(mask, bad);
            if (lane == 0) {
                if (bad) { F[k] = 2; any_sh = 1; }   // failed this round
                else F[k] = 0;                       // validated
            }
        }
        __syncthreads();
        if (!any_sh) break;
        if (round == PRRAX_BLEND_ROUNDS) { if (tid == 0) status_sh = -2; break; }
        for (int k = tid; k < m; k += blockDim.x)
            if (F[k] == 2) { R[k] *= 0.5f; F[k] = 1; }
        __syncthreads();
    }
    __syncthreads();
    if (tid == 0) {
        int o = 0;
        float* out = J.out + (size_t)p * J.max_out * dim;
        auto put = [&](const float* q) {   // skips a point on the previous one (two blends meeting)
            if (o) {
                float d2 = 0.f;
                for (int d = 0; d < dim; ++d) { const float e = q[d] - out[(size_t)(o - 1) * dim + d]; d2 += e * e; }
                if (d2 < 1e-10f) return;   // within 1e-5 rad
            }
            if (o >= J.max_out) { status_sh = -1; return; }
            for (int d = 0; d < dim; ++d) out[(size_t)o * dim + d] = q[d];
            ++o;
        };
        if (status_sh == 1) {
            put(P);
            for (int k = 1; k + 1 < m && status_sh == 1; ++k) {
                if (R[k] <= 0.f) { put(P + k * dim); continue; }
                float A[CONFIG_DIM_MAX], C[CONFIG_DIM_MAX], q[CONFIG_DIM_MAX];
                const int n = blend_samples(R[k], corner(k, R[k], A, C));
                put(A);
                for (int s = 1; s < n; ++s) { blend_point(A, P + k * dim, C, static_cast<float>(s) / n, dim, q); put(q); }
                put(C);
            }
            put(P + (m - 1) * dim);
        }
        J.out_len[p] = status_sh == 1 ? o : 0;
        J.status[p] = status_sh;
    }
}
