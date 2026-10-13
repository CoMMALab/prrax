/**
 * pRRTC Main Planner Kernel with CUDA Graph Support.
 *
 * This kernel integrates all pRRTC components and supports repeated execution
 * via CUDA Graphs for minimal kernel launch overhead.
 *
 * Features:
 *   - Two-tree bidirectional planning (start and goal trees)
 *   - Parallel tree expansion
 *   - Dynamic tree balancing
 *   - CUDA Graph compatible for repeated calls
 *   - Batched planning support via JAX vmap
 */

#include "xla/ffi/api/ffi.h"
#include "prrtc_helpers.cuh"
#include "robot.cuh"          // GRiM: grim::robot::frame_poses (cricket2-traced FK)
#include "collision.cuh"      // GRiM: sphere/capsule/box/halfspace distances
#include "prrax_gen.cuh"      // prrax: baked sphere model (see jax/_grim_build.py)
#include "prrtc_constraint.cuh" // TSR constraints, projection, constrained extend
#include <float.h>
#include <algorithm>
#include <vector>
#include <math.h>

namespace ffi = xla::ffi;

// Constants
#ifndef CONFIG_DIM_MAX
#define CONFIG_DIM_MAX 16
#endif

#ifndef TREE_MAX_NODES
#define TREE_MAX_NODES 1000000
#endif

// Maximum threads per block. Granularity is passed at runtime; shared memory
// arrays are sized to this maximum. Must be a power of 2.
// Matches pRRTC's source default (4*granularity = 4*64 = 256 was its max; we cap at 64
// because collision checks dominate and registers become the bottleneck above that).
#define PRRTC_BLOCK_THREADS_MAX 64


__device__ __forceinline__ void prrtc_radius_shrink_atomic(
    float* radii,
    int idx,
    float dd_alpha,
    float dd_radius,
    float dd_min_radius
) {
    int* radius_i = reinterpret_cast<int*>(&radii[idx]);
    int old_i = atomicAdd(radius_i, 0);
    while (true) {
        const float old_r = __int_as_float(old_i);
        const float new_r = (old_r == FLT_MAX)
            ? dd_radius
            : fmaxf(old_r * (1.0f - dd_alpha), dd_min_radius);
        const int desired_i = __float_as_int(new_r);
        const int prev_i = atomicCAS(radius_i, old_i, desired_i);
        if (prev_i == old_i) break;
        old_i = prev_i;
    }
}

__device__ __forceinline__ void prrtc_radius_grow_atomic(
    float* radii,
    int idx,
    float dd_alpha
) {
    int* radius_i = reinterpret_cast<int*>(&radii[idx]);
    int old_i = atomicAdd(radius_i, 0);
    while (true) {
        const float old_r = __int_as_float(old_i);
        if (old_r == FLT_MAX) break;
        const float new_r = old_r * (1.0f + dd_alpha);
        const int desired_i = __float_as_int(new_r);
        const int prev_i = atomicCAS(radius_i, old_i, desired_i);
        if (prev_i == old_i) break;
        old_i = prev_i;
    }
}

__global__ void prrtc_fill_float_kernel(
    float* data,
    float value,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

// Initialize trees with start and goal configurations
__device__ __forceinline__ unsigned long long globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// One problem's setup: roots of both trees (start in tree A, goals in tree B), their
// dynamic-domain radii, the counters, and the start time of the solve. Replaces the host's
// per-problem copies (pageable memcpys are synchronous).
__global__ void prrtc_init_kernel(
    const float* __restrict__ start_config,     // [dim]
    const float* __restrict__ goal_configs,     // [num_goals, dim]
    float* __restrict__ tree_a_configs,         // [dim, max_nodes]
    float* __restrict__ tree_b_configs,         // [dim, max_nodes]
    int* __restrict__ tree_a_parents,           // [max_nodes]
    int* __restrict__ tree_b_parents,           // [max_nodes]
    float* __restrict__ tree_a_radii,
    float* __restrict__ tree_b_radii,
    int* __restrict__ tree_sizes,               // [2]
    int* __restrict__ completed,                // [2]
    int* __restrict__ iter_count,               // [1]
    int* __restrict__ connection_info,          // [3]
    int* __restrict__ solved,                   // [1]
    unsigned long long* __restrict__ t_start,   // [1]
    int num_goals,
    int dim,
    int max_nodes
) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        *t_start = globaltimer_ns();
        for (int d = 0; d < dim; d++) tree_a_configs[d * max_nodes] = start_config[d];
        tree_a_parents[0] = 0;
        tree_a_radii[0] = FLT_MAX;
        tree_sizes[0] = completed[0] = 1;
        tree_sizes[1] = completed[1] = num_goals;
        *iter_count = 0;
        connection_info[0] = connection_info[1] = connection_info[2] = -1;
        *solved = 0;
    }
    for (int g = tid; g < num_goals; g += blockDim.x) {
        for (int d = 0; d < dim; d++) tree_b_configs[d * max_nodes + g] = goal_configs[g * dim + d];
        tree_b_parents[g] = g;
        tree_b_radii[g] = FLT_MAX;
    }
}

// Halton sequence for low-discrepancy sampling
template<int BASE>
__device__ __forceinline__ float halton_next(int& state) {
    float fraction = 1.0f;
    float result = 0.0f;
    int n = state;

    while (n > 0) {
        fraction /= BASE;
        result += fraction * (n % BASE);
        n /= BASE;
    }

    state = state + 1;
    return result;
}

__device__ __forceinline__ float halton_next_runtime(int base, int& state) {
    float fraction = 1.0f;
    float result = 0.0f;
    int n = state;

    while (n > 0) {
        fraction /= static_cast<float>(base);
        result += fraction * static_cast<float>(n % base);
        n /= base;
    }

    state = state + 1;
    return result;
}

// World pose of joint frame j in T (n_frames x 7); links with no parent joint (j = -1, the
// robot base) sit at the identity.
__device__ __forceinline__ const float* frame_tf(const float* T, int j) {
    return j >= 0 ? T + j * 7 : kIdentityTf;
}

__device__ __forceinline__ int volatile_load(const int* p) { return *(const volatile int*)p; }

// out = base + t * delta (n_q joints), the waypoint at fraction t of an edge.
__device__ __forceinline__ void edge_point(const float* base, const float* delta, float t, float* out) {
    constexpr uint32_t N = grim::robot::n_q;
    glass::thread::copy<float, N>(const_cast<float*>(base), out);   // GLASS takes non-const sources
    glass::thread::axpy<float, N>(t, const_cast<float*>(delta), out);
}

struct CollisionContext {
    // World obstacles are per call; the robot model (FK tables or cricket2's trace, spheres,
    // self pairs, coarse groups) is baked into the build (grim::robot::, prrax::).
    const float* world_spheres;    // [n_world_spheres, 4]   (x,y,z,r)
    const float* world_capsules;   // [n_world_capsules, 7]  (x1,y1,z1,x2,y2,z2,r)
    const float* world_boxes;      // [n_world_boxes, 15]    (cx,cy,cz, a1x,a1y,a1z, a2x,a2y,a2z, a3x,a3y,a3z, hl1,hl2,hl3)
    const float* world_halfspaces; // [n_world_halfspaces, 6] (nx,ny,nz, px,py,pz)
    int n_world_spheres;
    int n_world_capsules;
    int n_world_boxes;
    int n_world_halfspaces;
    int enabled;
    // Collision-checker selection:
    //   0 = binary       — pRRTC-style early-exit boolean check. Returns at the first
    //                       penetrating obstacle; minimal work per config.
    //   1 = differentiable — sweeps every (robot sphere, obstacle) pair with no early exit
    //                       and accumulates colldist_from_sdf(d, margin); in collision when
    //                       the aggregate cost is negative (any sphere within the margin).
    //   2 = binary_coarse — coarse→fine hierarchical early-exit binary check: a coarse
    //                       bounding-sphere pre-pass gates the fine sweep per link.
    //   3 = binary_coarse_coop — mode 2 split over lane-groups of a block.
    int collision_mode;
    float collision_margin;
};

// Maximum robot links/groups the coarse per-link "dirty" mask supports.
#ifndef PRRTC_MAX_LINKS
#define PRRTC_MAX_LINKS 64
#endif

// Phase B cooperative collision check: lanes that cooperate on one waypoint config.
// Must divide the warp size (32) so groups never straddle a warp (for __syncwarp/__shfl).
#ifndef PRRTC_CC_LANES
#define PRRTC_CC_LANES 4
#endif
// Joint cap for the per-group shared FK buffer used by the cooperative path. Sized small
// (most manipulators ≤ this) to keep shared-memory use modest; falls back to the
// per-thread coarse check when a robot exceeds it.
#ifndef PRRTC_CC_MAX_JOINTS
#define PRRTC_CC_MAX_JOINTS 16
#endif

// Binary (early-exit) collision check — the original pRRTC SIMT semantics.
// FK transforms for the configuration are precomputed in T_world.
__device__ __forceinline__ bool prrtc_config_in_collision_binary(
    const CollisionContext& ctx,
    const float* __restrict__ T_world
) {
    // Robot-vs-world collision check against all primitive types.
    for (int rs = 0; rs < prrax::n_spheres; ++rs) {
        const int link_idx = prrax::kSphereJoint[rs];
        if (link_idx >= grim::robot::n_frames) continue;

        const float* T = frame_tf(T_world, link_idx);
        const float* local = &prrax::kSphereLocal[rs * 3];
        float world_pt[3];
        apply_se3_point(T, local, world_pt);
        const float rr = prrax::kSphereRadius[rs];
        const float sx = world_pt[0], sy = world_pt[1], sz = world_pt[2];

        for (int ws = 0; ws < ctx.n_world_spheres; ++ws) {
            const float* obs = &ctx.world_spheres[ws * 4];
            if (sphere_sphere_dist(sx, sy, sz, rr,
                                   obs[0], obs[1], obs[2], obs[3]) <= 0.0f)
                return true;
        }

        for (int wc = 0; wc < ctx.n_world_capsules; ++wc) {
            const float* cap = &ctx.world_capsules[wc * 7];
            if (sphere_capsule_dist(sx, sy, sz, rr,
                                    cap[0], cap[1], cap[2],
                                    cap[3], cap[4], cap[5], cap[6]) <= 0.0f)
                return true;
        }

        for (int wb = 0; wb < ctx.n_world_boxes; ++wb) {
            const float* box = &ctx.world_boxes[wb * 15];
            if (sphere_box_dist(sx, sy, sz, rr,
                                box[0], box[1], box[2],
                                box[3], box[4], box[5],
                                box[6], box[7], box[8],
                                box[9], box[10], box[11],
                                box[12], box[13], box[14]) <= 0.0f)
                return true;
        }

        for (int wh = 0; wh < ctx.n_world_halfspaces; ++wh) {
            const float* hs = &ctx.world_halfspaces[wh * 6];
            if (sphere_halfspace_dist(sx, sy, sz, rr,
                                      hs[0], hs[1], hs[2],
                                      hs[3], hs[4], hs[5]) <= 0.0f)
                return true;
        }
    }

    // Optional self-collision checks over active sphere-pair list.
    for (int p = 0; p < prrax::n_self_pairs; ++p) {
        const int i = prrax::kSelfPairs[p * 2 + 0];
        const int j = prrax::kSelfPairs[p * 2 + 1];
        if (i < 0 || j < 0 || i >= prrax::n_spheres || j >= prrax::n_spheres) continue;

        const int link_i = prrax::kSphereJoint[i];
        const int link_j = prrax::kSphereJoint[j];
        if (link_i >= grim::robot::n_frames || link_j >= grim::robot::n_frames) continue;

        float pi[3];
        float pj[3];
        apply_se3_point(frame_tf(T_world, link_i), &prrax::kSphereLocal[i * 3], pi);
        apply_se3_point(frame_tf(T_world, link_j), &prrax::kSphereLocal[j * 3], pj);
        if (sphere_sphere_dist(
                pi[0], pi[1], pi[2], prrax::kSphereRadius[i],
                pj[0], pj[1], pj[2], prrax::kSphereRadius[j]) <= 0.0f) {
            return true;
        }
    }

    return false;
}

// Differentiable (signed-distance-field) collision check. No early exit: every
// (robot sphere, obstacle) and self pair contributes a smooth margin penalty via
// colldist_from_sdf, exactly as pyroffi's differentiable collision kernel does.
// The configuration is reported in collision when the accumulated cost is
// negative — i.e. any geometry lies within `margin` of an obstacle.
__device__ __forceinline__ bool prrtc_config_in_collision_differentiable(
    const CollisionContext& ctx,
    const float* __restrict__ T_world
) {
    const float margin = ctx.collision_margin;
    float cost = 0.0f;

    for (int rs = 0; rs < prrax::n_spheres; ++rs) {
        const int link_idx = prrax::kSphereJoint[rs];
        if (link_idx >= grim::robot::n_frames) continue;

        const float* T = frame_tf(T_world, link_idx);
        const float* local = &prrax::kSphereLocal[rs * 3];
        float world_pt[3];
        apply_se3_point(T, local, world_pt);
        const float rr = prrax::kSphereRadius[rs];
        const float sx = world_pt[0], sy = world_pt[1], sz = world_pt[2];

        for (int ws = 0; ws < ctx.n_world_spheres; ++ws) {
            const float* obs = &ctx.world_spheres[ws * 4];
            const float d = sphere_sphere_dist(sx, sy, sz, rr,
                                               obs[0], obs[1], obs[2], obs[3]);
            cost += colldist_from_sdf(d, margin);
        }

        for (int wc = 0; wc < ctx.n_world_capsules; ++wc) {
            const float* cap = &ctx.world_capsules[wc * 7];
            const float d = sphere_capsule_dist(sx, sy, sz, rr,
                                                cap[0], cap[1], cap[2],
                                                cap[3], cap[4], cap[5], cap[6]);
            cost += colldist_from_sdf(d, margin);
        }

        for (int wb = 0; wb < ctx.n_world_boxes; ++wb) {
            const float* box = &ctx.world_boxes[wb * 15];
            const float d = sphere_box_dist(sx, sy, sz, rr,
                                            box[0], box[1], box[2],
                                            box[3], box[4], box[5],
                                            box[6], box[7], box[8],
                                            box[9], box[10], box[11],
                                            box[12], box[13], box[14]);
            cost += colldist_from_sdf(d, margin);
        }

        for (int wh = 0; wh < ctx.n_world_halfspaces; ++wh) {
            const float* hs = &ctx.world_halfspaces[wh * 6];
            const float d = sphere_halfspace_dist(sx, sy, sz, rr,
                                                  hs[0], hs[1], hs[2],
                                                  hs[3], hs[4], hs[5]);
            cost += colldist_from_sdf(d, margin);
        }
    }

    for (int p = 0; p < prrax::n_self_pairs; ++p) {
        const int i = prrax::kSelfPairs[p * 2 + 0];
        const int j = prrax::kSelfPairs[p * 2 + 1];
        if (i < 0 || j < 0 || i >= prrax::n_spheres || j >= prrax::n_spheres) continue;

        const int link_i = prrax::kSphereJoint[i];
        const int link_j = prrax::kSphereJoint[j];
        if (link_i >= grim::robot::n_frames || link_j >= grim::robot::n_frames) continue;

        float pi[3];
        float pj[3];
        apply_se3_point(frame_tf(T_world, link_i), &prrax::kSphereLocal[i * 3], pi);
        apply_se3_point(frame_tf(T_world, link_j), &prrax::kSphereLocal[j * 3], pj);
        const float d = sphere_sphere_dist(
            pi[0], pi[1], pi[2], prrax::kSphereRadius[i],
            pj[0], pj[1], pj[2], prrax::kSphereRadius[j]);
        cost += colldist_from_sdf(d, margin);
    }

    return cost < 0.0f;
}

// Binary signed-distance test of one world-frame robot sphere against all world obstacles.
// Early-exits (returns true) on the first penetrating obstacle.
__device__ __forceinline__ bool prrtc_sphere_vs_world(
    const CollisionContext& ctx, float sx, float sy, float sz, float rr
) {
    for (int ws = 0; ws < ctx.n_world_spheres; ++ws) {
        const float* obs = &ctx.world_spheres[ws * 4];
        if (sphere_sphere_dist(sx, sy, sz, rr, obs[0], obs[1], obs[2], obs[3]) <= 0.0f)
            return true;
    }
    for (int wc = 0; wc < ctx.n_world_capsules; ++wc) {
        const float* cap = &ctx.world_capsules[wc * 7];
        if (sphere_capsule_dist(sx, sy, sz, rr, cap[0], cap[1], cap[2],
                                cap[3], cap[4], cap[5], cap[6]) <= 0.0f)
            return true;
    }
    for (int wb = 0; wb < ctx.n_world_boxes; ++wb) {
        const float* box = &ctx.world_boxes[wb * 15];
        if (sphere_box_dist(sx, sy, sz, rr, box[0], box[1], box[2],
                            box[3], box[4], box[5], box[6], box[7], box[8],
                            box[9], box[10], box[11], box[12], box[13], box[14]) <= 0.0f)
            return true;
    }
    for (int wh = 0; wh < ctx.n_world_halfspaces; ++wh) {
        const float* hs = &ctx.world_halfspaces[wh * 6];
        if (sphere_halfspace_dist(sx, sy, sz, rr, hs[0], hs[1], hs[2],
                                  hs[3], hs[4], hs[5]) <= 0.0f)
            return true;
    }
    return false;
}

// Coarse→fine hierarchical binary check (source pRRTC style). A coarse bounding-sphere
// pre-pass proves most free configs clear without touching the fine model; only groups
// whose coarse sphere reports a possible hit are refined. Sound because each coarse sphere
// encloses its group's fine spheres, so this returns the same in-collision verdict as the
// flat binary check — just with far less work on the common (free) path.
__device__ __forceinline__ bool prrtc_config_in_collision_binary_coarse(
    const CollisionContext& ctx,
    const float* __restrict__ T_world
) {
    const int n_coarse = prrax::n_coarse;
    bool link_dirty[PRRTC_MAX_LINKS];
    for (int k = 0; k < n_coarse; ++k) link_dirty[k] = false;

    // --- Coarse world pre-pass: flag groups that might collide. ---
    bool any_dirty = false;
    for (int k = 0; k < n_coarse; ++k) {
        const int link_idx = prrax::kCoarseJoint[k];
        if (link_idx >= grim::robot::n_frames) continue;
        float wp[3];
        apply_se3_point(frame_tf(T_world, link_idx), &prrax::kCoarseLocal[k * 3], wp);
        if (prrtc_sphere_vs_world(ctx, wp[0], wp[1], wp[2], prrax::kCoarseRadius[k])) {
            link_dirty[k] = true;
            any_dirty = true;
        }
    }

    // --- Fine world refine: only the fine spheres of dirty groups. ---
    if (any_dirty) {
        for (int k = 0; k < n_coarse; ++k) {
            if (!link_dirty[k]) continue;
            const int start = prrax::kFineOffsets[k];
            const int end   = prrax::kFineOffsets[k + 1];
            for (int rs = start; rs < end; ++rs) {
                const int link_idx = prrax::kSphereJoint[rs];
                if (link_idx >= grim::robot::n_frames) continue;
                float wp[3];
                apply_se3_point(frame_tf(T_world, link_idx), &prrax::kSphereLocal[rs * 3], wp);
                if (prrtc_sphere_vs_world(ctx, wp[0], wp[1], wp[2], prrax::kSphereRadius[rs]))
                    return true;
            }
        }
    }

    // --- Self-collision: coarse-gated per link pair, refine only when coarse pair touches. ---
    for (int p = 0; p < prrax::n_coarse_self_pairs; ++p) {
        const int ck_i = prrax::kCoarseSelfPairs[p * 2 + 0];
        const int ck_j = prrax::kCoarseSelfPairs[p * 2 + 1];
        if (ck_i < 0 || ck_j < 0 || ck_i >= n_coarse || ck_j >= n_coarse) continue;
        const int cli = prrax::kCoarseJoint[ck_i];
        const int clj = prrax::kCoarseJoint[ck_j];
        if (cli >= grim::robot::n_frames || clj >= grim::robot::n_frames) continue;

        float ci[3], cj[3];
        apply_se3_point(frame_tf(T_world, cli), &prrax::kCoarseLocal[ck_i * 3], ci);
        apply_se3_point(frame_tf(T_world, clj), &prrax::kCoarseLocal[ck_j * 3], cj);
        if (sphere_sphere_dist(ci[0], ci[1], ci[2], prrax::kCoarseRadius[ck_i],
                               cj[0], cj[1], cj[2], prrax::kCoarseRadius[ck_j]) > 0.0f)
            continue;  // coarse pair clear ⇒ all its fine pairs clear

        const int start = prrax::kSelfPairRanges[p * 2 + 0];
        const int end   = prrax::kSelfPairRanges[p * 2 + 1];
        for (int q = start; q < end; ++q) {
            const int i = prrax::kSelfPairs[q * 2 + 0];
            const int j = prrax::kSelfPairs[q * 2 + 1];
            if (i < 0 || j < 0 || i >= prrax::n_spheres || j >= prrax::n_spheres) continue;
            const int link_i = prrax::kSphereJoint[i];
            const int link_j = prrax::kSphereJoint[j];
            if (link_i >= grim::robot::n_frames || link_j >= grim::robot::n_frames) continue;
            float pi[3], pj[3];
            apply_se3_point(frame_tf(T_world, link_i), &prrax::kSphereLocal[i * 3], pi);
            apply_se3_point(frame_tf(T_world, link_j), &prrax::kSphereLocal[j * 3], pj);
            if (sphere_sphere_dist(pi[0], pi[1], pi[2], prrax::kSphereRadius[i],
                                   pj[0], pj[1], pj[2], prrax::kSphereRadius[j]) <= 0.0f)
                return true;
        }
    }

    return false;
}

__device__ __forceinline__ bool prrtc_config_in_collision(
    const float* cfg,
    const CollisionContext& ctx
) {
    if (!ctx.enabled) return false;
    
    float T_world[grim::robot::n_frames * 7];
    grim::robot::frame_poses(cfg, nullptr, T_world);

    if (ctx.collision_mode == 1) {
        return prrtc_config_in_collision_differentiable(ctx, T_world);
    }
    if (ctx.collision_mode == 2 && prrax::coarse_enabled && prrax::n_coarse <= PRRTC_MAX_LINKS) {
        return prrtc_config_in_collision_binary_coarse(ctx, T_world);
    }
    return prrtc_config_in_collision_binary(ctx, T_world);
}

// Cooperative SIMT coarse→fine edge check (Phase B, source-pRRTC style). The block's
// threads form groups of PRRTC_CC_LANES lanes; each group validates one interpolated
// waypoint at a time (looping to cover all W waypoints of the edge). Lane 0 computes the
// shared FK transforms once per waypoint; the lanes then split the coarse spheres, fine
// spheres of "dirty" links, and self pairs among themselves and bail the whole group on
// the first hit via a warp-collective vote. Sets *any_collision_sh on any collision.
// Collective: every block thread must call this (it distributes waypoints across groups).
__device__ __forceinline__ void prrtc_edge_cc_cooperative(
    const float* __restrict__ base,
    const float* __restrict__ delta,
    int dim,
    int W,
    const CollisionContext& ctx,
    float* __restrict__ cc_group_T,   // shared scratch [n_groups * PRRTC_CC_MAX_JOINTS * 7]
    int* __restrict__ any_collision_sh,
    const float* __restrict__ waypoints = nullptr   // [W * CONFIG_DIM_MAX]; else base + t * delta
) {
    const int tid = threadIdx.x;
    const int lane = tid & (PRRTC_CC_LANES - 1);
    const int group = tid / PRRTC_CC_LANES;
    const int n_groups = blockDim.x / PRRTC_CC_LANES;
    if (n_groups == 0 || group >= n_groups) return;

    const int lane_in_warp = tid & (WARP_SIZE - 1);
    const unsigned group_mask =
        ((1u << PRRTC_CC_LANES) - 1u) << ((lane_in_warp / PRRTC_CC_LANES) * PRRTC_CC_LANES);
    const int group_lane0 = (lane_in_warp / PRRTC_CC_LANES) * PRRTC_CC_LANES;

    float* myT = &cc_group_T[group * PRRTC_CC_MAX_JOINTS * 7];

    for (int w = group; w < W; w += n_groups) {
        // Uniform early-out: lane 0 reads the shared flag, broadcasts to the group so all
        // lanes branch identically (keeps __syncwarp/__shfl well-formed).
        int stop = 0;
        if (lane == 0) stop = (atomicAdd(any_collision_sh, 0) != 0) ? 1 : 0;
        stop = __shfl_sync(group_mask, stop, group_lane0);
        if (stop) break;

        float interp[CONFIG_DIM_MAX];
        const float tf = static_cast<float>(w + 1) / static_cast<float>(W);
        if (waypoints) for (int d = 0; d < dim; ++d) interp[d] = waypoints[w * CONFIG_DIM_MAX + d];
        else edge_point(base, delta, tf, interp);

        if (lane == 0) {
            grim::robot::frame_poses(interp, nullptr, myT);
        }
        __syncwarp(group_mask);

        // Coarse pre-pass: lanes split coarse spheres, build a per-group dirty bitmask.
        unsigned long long dirty = 0ull;
        for (int k = lane; k < prrax::n_coarse; k += PRRTC_CC_LANES) {
            const int li = prrax::kCoarseJoint[k];
            if (li >= grim::robot::n_frames) continue;
            float wp[3];
            apply_se3_point(frame_tf(myT, li), &prrax::kCoarseLocal[k * 3], wp);
            if (prrtc_sphere_vs_world(ctx, wp[0], wp[1], wp[2], prrax::kCoarseRadius[k]))
                dirty |= (1ull << k);
        }
        for (int off = PRRTC_CC_LANES / 2; off > 0; off >>= 1)
            dirty |= __shfl_xor_sync(group_mask, dirty, off);

        // Fine world refine: lanes split the fine spheres of dirty groups, warp early-out.
        bool hit = false;
        for (int k = 0; k < prrax::n_coarse; ++k) {
            if (!((dirty >> k) & 1ull)) continue;
            const int s = prrax::kFineOffsets[k];
            const int e = prrax::kFineOffsets[k + 1];
            for (int rs = s + lane; rs < e; rs += PRRTC_CC_LANES) {
                const int li = prrax::kSphereJoint[rs];
                if (li >= grim::robot::n_frames) continue;
                float wp[3];
                apply_se3_point(frame_tf(myT, li), &prrax::kSphereLocal[rs * 3], wp);
                if (prrtc_sphere_vs_world(ctx, wp[0], wp[1], wp[2], prrax::kSphereRadius[rs])) {
                    hit = true;
                    break;
                }
            }
            if (__any_sync(group_mask, hit)) { hit = true; break; }
        }

        // Self-collision: lanes split coarse self pairs; refine fine pairs when coarse touches.
        if (!hit) {
            bool self_hit = false;
            for (int p = lane; p < prrax::n_coarse_self_pairs; p += PRRTC_CC_LANES) {
                const int ci = prrax::kCoarseSelfPairs[p * 2 + 0];
                const int cj = prrax::kCoarseSelfPairs[p * 2 + 1];
                if (ci < 0 || cj < 0 || ci >= prrax::n_coarse || cj >= prrax::n_coarse) continue;
                const int li = prrax::kCoarseJoint[ci];
                const int lj = prrax::kCoarseJoint[cj];
                if (li >= grim::robot::n_frames || lj >= grim::robot::n_frames) continue;
                float a[3], b[3];
                apply_se3_point(frame_tf(myT, li), &prrax::kCoarseLocal[ci * 3], a);
                apply_se3_point(frame_tf(myT, lj), &prrax::kCoarseLocal[cj * 3], b);
                if (sphere_sphere_dist(a[0], a[1], a[2], prrax::kCoarseRadius[ci],
                                       b[0], b[1], b[2], prrax::kCoarseRadius[cj]) > 0.0f)
                    continue;
                const int qs = prrax::kSelfPairRanges[p * 2 + 0];
                const int qe = prrax::kSelfPairRanges[p * 2 + 1];
                for (int q = qs; q < qe; ++q) {
                    const int i = prrax::kSelfPairs[q * 2 + 0];
                    const int j = prrax::kSelfPairs[q * 2 + 1];
                    if (i < 0 || j < 0 || i >= prrax::n_spheres || j >= prrax::n_spheres) continue;
                    const int lii = prrax::kSphereJoint[i];
                    const int ljj = prrax::kSphereJoint[j];
                    if (lii >= grim::robot::n_frames || ljj >= grim::robot::n_frames) continue;
                    float pi[3], pj[3];
                    apply_se3_point(frame_tf(myT, lii), &prrax::kSphereLocal[i * 3], pi);
                    apply_se3_point(frame_tf(myT, ljj), &prrax::kSphereLocal[j * 3], pj);
                    if (sphere_sphere_dist(pi[0], pi[1], pi[2], prrax::kSphereRadius[i],
                                           pj[0], pj[1], pj[2], prrax::kSphereRadius[j]) <= 0.0f) {
                        self_hit = true;
                        break;
                    }
                }
                if (self_hit) break;
            }
            hit = __any_sync(group_mask, self_hit);
        }

        if (hit && lane == 0) atomicOr(any_collision_sh, 1);
        __syncwarp(group_mask);
    }
}

// Main pRRTC planning kernel — aligned with pRRTC parallel semantics.
//
// Each CUDA block is an independent explorer (like pRRTC's num_new_configs blocks).
// The kernel is launched with gridDim.x == num_new_samples blocks, each running a
// persistent while-loop that processes ONE sample per iteration.
//
// Alignment with pRRTC:
//   - Per-block Halton state with LCG-shuffled prime bases (pRRTC uses curand shuffle)
//   - SIMT parallel nearest-neighbor search via shared-memory tree reduction
//   - Parallel edge collision check: each thread independently checks one interpolated
//     waypoint along the extend/connect edge, then atomicOr reduces to a pass/fail
//   - Atomic tree_sizes and completed counters (pRRTC uses device-global atomics)
//   - Per-block local iteration counter (pRRTC: per-block stack variable)
//   - Balance mode uses blockIdx.x ratio (pRRTC: blockIdx.x / num_new_configs ratio)
//
// Remaining gap vs pRRTC: collision model (sphere-only FK vs robot-specific approx+exact
// pipeline). The per-thread FK is fully independent (stack-allocated T_world), so all
// PRRTC_BLOCK_THREADS threads genuinely run collision checks in parallel.
__global__ void prrtc_planner_kernel(
    float* __restrict__ tree_a_configs,       // [dim, max_nodes] SoA - start tree
    float* __restrict__ tree_b_configs,       // [dim, max_nodes] SoA - goal tree
    int* __restrict__ tree_a_parents,         // [max_nodes]
    int* __restrict__ tree_b_parents,         // [max_nodes]
    float* __restrict__ tree_a_radii,         // [max_nodes]
    float* __restrict__ tree_b_radii,         // [max_nodes]
    float* __restrict__ edge_wps_a,           // [max_nodes, W-1, dim] constrained edges' interior waypoints
    float* __restrict__ edge_wps_b,
    const float* __restrict__ min_vals,       // [dim]
    const float* __restrict__ max_vals,       // [dim]
    int* __restrict__ tree_sizes,             // [2] atomic allocation counters
    int* __restrict__ completed,              // [2] write-completion counters
    int* __restrict__ iter_count,             // [1] output: max iterations reached
    int* __restrict__ connection_info,        // [3] [a_idx, b_idx, expand_tree_id]
    int* __restrict__ solved_out,             // [1] planner solved flag
    CollisionContext ctx_in,                  // world pointers; counts from world_counts
    const int* __restrict__ world_counts,     // [4] spheres, capsules, boxes, halfspaces
    int max_iterations,
    float step_size,
    int num_new_samples,                      // == gridDim.x
    int balance_mode,
    float tree_ratio,
    int dynamic_domain,
    float dd_alpha,
    float dd_radius,
    float dd_min_radius,
    int dim,
    int max_nodes,
    int granularity,
    const float* __restrict__ constraints     // [PRRAX_CONS_SIZE]; n = 0 => unconstrained
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ CollisionContext ctx_sh;
    if (tid == 0) {
        ctx_sh = ctx_in;
        ctx_sh.n_world_spheres = world_counts[0];
        ctx_sh.n_world_capsules = world_counts[1];
        ctx_sh.n_world_boxes = world_counts[2];
        ctx_sh.n_world_halfspaces = world_counts[3];
        ctx_sh.enabled = (prrax::n_spheres > 0 &&
            (world_counts[0] > 0 || world_counts[1] > 0 || world_counts[2] > 0 || world_counts[3] > 0 ||
             prrax::n_self_pairs > 0)) ? 1 : 0;
    }
    __syncthreads();
    const CollisionContext& collision_ctx = ctx_sh;

    // --- Shared memory ---
    // NN reduction arrays (one entry per thread in block; sized to max granularity)
    __shared__ float sdata[PRRTC_BLOCK_THREADS_MAX];
    __shared__ int   sindex_sh[PRRTC_BLOCK_THREADS_MAX];

    // Per-block Halton state (like pRRTC's per-block HaltonState)
    __shared__ int halton_st[CONFIG_DIM_MAX];
    __shared__ int prime_order[16];
    __shared__ int vamp_tree_id;

    // Sample and candidate configs
    __shared__ float sample_sh[CONFIG_DIM_MAX];
    __shared__ float cfg_candidate_sh[CONFIG_DIM_MAX];

    // Connect phase state
    __shared__ float curr_cfg_sh[CONFIG_DIM_MAX];   // current position stepping toward NN
    __shared__ float vec_sh[CONFIG_DIM_MAX];         // per-step vector toward connect NN

    // Scalar communication between tid==0 and all threads
    __shared__ int   t_tree_id_sh, o_tree_id_sh;
    __shared__ int   nearest_idx_sh;
    __shared__ bool  should_skip_sh;
    __shared__ int   new_idx_sh;
    __shared__ int   connect_nearest_idx_sh;
    __shared__ int   n_extensions_sh;
    __shared__ int   extension_parent_sh;
    __shared__ int   ext_idx_sh;
    __shared__ int   any_collision_sh;   // atomicOr target for per-thread CC results

    // Manifold constraints (prrtc_constraint.cuh): parameters, the projected waypoints of the
    // current edge, its start, and the connect target.
    __shared__ float cons_sh[PRRAX_CONS_SIZE];
    __shared__ float wp_sh[PRRTC_BLOCK_THREADS_MAX * CONFIG_DIM_MAX];
    __shared__ float near_sh[CONFIG_DIM_MAX];
    __shared__ float target_sh[CONFIG_DIM_MAX];
    __shared__ int   cflag_sh;
    __shared__ int   new_base_sh;
    for (int i = tid; i < PRRAX_CONS_SIZE; i += blockDim.x) cons_sh[i] = constraints[i];

    // Phase B cooperative-CC scratch: per-group shared FK transforms + shared edge endpoints.
    __shared__ float cc_group_T[(PRRTC_BLOCK_THREADS_MAX / PRRTC_CC_LANES) * PRRTC_CC_MAX_JOINTS * 7];
    __shared__ float cc_delta_sh[CONFIG_DIM_MAX];
    // Cooperative path is selected once: coarse model present, FK fits the shared buffer,
    // and the block is wide enough to form lane-groups. Uniform across the block.
    const bool cc_coop = (collision_ctx.collision_mode == 3 && prrax::coarse_enabled
                          && grim::robot::n_frames <= PRRTC_CC_MAX_JOINTS
                          && static_cast<int>(blockDim.x) >= PRRTC_CC_LANES);

    // Initialize per-block Halton with shuffled primes.
    // Block 0 uses canonical ordering; other blocks are shuffled.
    if (tid == 0) {
        int primes[16] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59};
        if (bid != 0) {
            // Fisher-Yates shuffle with a simple LCG seeded by block index
            unsigned int rng = bid * 1664525u + 1013904223u;
            for (int i = 15; i > 0; --i) {
                rng = rng * 1664525u + 1013904223u;
                const int j = static_cast<int>(rng % static_cast<unsigned int>(i + 1));
                const int tmp = primes[i]; primes[i] = primes[j]; primes[j] = tmp;
            }
        }
        for (int d = 0; d < 16; ++d) prime_order[d] = primes[d];
        for (int d = 0; d < CONFIG_DIM_MAX; ++d) {
            // pRRTC starts each dimension stream at its first Halton sample.
            halton_st[d] = 1;
        }
        vamp_tree_id = 0;
    }
    __syncthreads();
    const bool constrained = cons_sh[CONS_N] > 0.0f;
    const int W = blockDim.x;
    unsigned proj_calls = 0;   // per-block counter seeding the McVAMP steer perturbations

    // Collision check of the edge's W waypoints: base + (w+1)/W * delta, or the projected
    // waypoints in wp_sh for a constrained edge. Collective; true if any waypoint collides.
    auto edge_in_collision = [&](const float* base, const float* delta) -> bool {
        if (tid == 0) any_collision_sh = 0;
        __syncthreads();
        if (cc_coop) {
            prrtc_edge_cc_cooperative(base, delta, dim, W, collision_ctx, cc_group_T,
                                      &any_collision_sh, constrained ? wp_sh : nullptr);
        } else {
            float interp[CONFIG_DIM_MAX];
            const float t_frac = static_cast<float>(tid + 1) / static_cast<float>(W);
            if (constrained) for (int d = 0; d < dim; ++d) interp[d] = wp_sh[tid * CONFIG_DIM_MAX + d];
            else edge_point(base, delta, t_frac, interp);
            if (prrtc_config_in_collision(interp, collision_ctx)) atomicOr(&any_collision_sh, 1);
        }
        __syncthreads();
        return any_collision_sh != 0;
    };

    // Append an edge to t_tree from `parent`: one node at its end (cfg, or for a constrained
    // edge the last projected waypoint). A constrained edge's W-1 interior waypoints go to
    // edge_wps[node] (expanded into the path on the host), keeping the searched tree small.
    // Collective; returns the new node's index, or -1 when the tree is full.
    auto append_edge = [&](int t_tree, int parent, const float* cfg) -> int {
        float* cfgs = (t_tree == 0) ? tree_a_configs : tree_b_configs;
        int* parents = (t_tree == 0) ? tree_a_parents : tree_b_parents;
        float* radii = (t_tree == 0) ? tree_a_radii : tree_b_radii;
        float* wps = (t_tree == 0) ? edge_wps_a : edge_wps_b;
        if (tid == 0) {
            const int idx = atomicAdd(&tree_sizes[t_tree], 1);
            if (idx >= max_nodes) {
                atomicCAS(solved_out, 0, -1);
                new_base_sh = -1;
            } else {
                new_base_sh = idx;
                radii[idx] = FLT_MAX;
            }
        }
        __syncthreads();
        const int node = new_base_sh;
        if (node < 0) return -1;
        if (tid < dim)
            cfgs[tid * max_nodes + node] = constrained ? wp_sh[(W - 1) * CONFIG_DIM_MAX + tid] : cfg[tid];
        if (constrained)
            for (int i = tid; i < (W - 1) * dim; i += W)
                wps[((size_t)node * (W - 1) + i / dim) * dim + i % dim] = wp_sh[(i / dim) * CONFIG_DIM_MAX + i % dim];
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            // Publish last: other blocks' searches skip a node until its parent is set.
            atomicExch(&parents[node], parent);
            atomicAdd(&completed[t_tree], 1);
        }
        return node;
    };

    // Greedy connect from node from_idx (config from_cfg, shared) of t_tree to its nearest
    // node in the other tree: steps of step_size, each collision-checked and appended. A
    // constrained step is projected onto the manifold (McVAMP's connect: up to
    // 2 * ceil(d / step) + 2 steps, reached within connect_eps). Collective; true if it
    // connected (connection_info written).
    auto connect_from = [&](int t_tree, int from_idx, const float* from_cfg) -> bool {
        const int o_tree = 1 - t_tree;
        const float* o_cfgs = (o_tree == 0) ? tree_a_configs : tree_b_configs;
        // Fresh completed count for o_tree — more nodes may have been added by other blocks.
        const int comp_o_conn = atomicAdd(&completed[o_tree], 0);

        float local_min_sq2 = FLT_MAX;
        int   local_near2   = 0;
        const int* o_parents = (o_tree == 0) ? tree_a_parents : tree_b_parents;
        for (int i = tid; i < comp_o_conn; i += blockDim.x) {
            if (volatile_load(&o_parents[i]) < 0) continue;   // claimed, not yet written
            float dsq = 0.0f;
            for (int d = 0; d < dim; ++d) {
                const float diff = o_cfgs[d * max_nodes + i] - from_cfg[d];
                dsq += diff * diff;
            }
            if (dsq < local_min_sq2) { local_min_sq2 = dsq; local_near2 = i; }
        }
        sdata[tid]     = local_min_sq2;
        sindex_sh[tid] = local_near2;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && sdata[tid + s] < sdata[tid]) {
                sdata[tid]     = sdata[tid + s];
                sindex_sh[tid] = sindex_sh[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float cd         = sqrtf(sdata[0]);
            connect_nearest_idx_sh = sindex_sh[0];
            // n_extensions = ceil(dist / step_size); 0 means already touching (immediate connect)
            int n_ext = 0;
            if (step_size > 1e-8f && cd > 1e-8f) {
                n_ext = static_cast<int>(ceilf(cd / step_size));
            }
            // Constrained steps follow the manifold, so allow detours up to twice the count.
            n_extensions_sh     = constrained ? 2 * n_ext + 2 : n_ext;
            extension_parent_sh = from_idx;
            // Direction vector for one extension step (matches pRRTC's vec computation)
            for (int d = 0; d < dim; ++d) {
                curr_cfg_sh[d] = from_cfg[d];
                target_sh[d]   = o_cfgs[d * max_nodes + connect_nearest_idx_sh];
                vec_sh[d]      = (n_ext > 0)
                    ? (target_sh[d] - from_cfg[d]) / static_cast<float>(n_ext)
                    : 0.0f;
            }
        }
        __syncthreads();

        // Unconstrained: n_extensions fixed steps of vec. Constrained: project each step onto
        // the manifold; stop when it reaches the target or a step fails.
        bool extension_failed = false;
        for (int ext = 0; ext < n_extensions_sh; ++ext) {
            if (constrained) {
                const float eps = cons_sh[CONS_CONNECT_EPS];
                const float d_curr = prrax_cons::sq_dist(curr_cfg_sh, target_sh);
                if (d_curr < eps * eps) break;   // reached
                // A step that does not get closer abandons the connect: on the GPU a wandering
                // connect is slower than a fresh sample (measured).
                if (!prrax_cons::constrained_extend(cons_sh, curr_cfg_sh, target_sh, step_size,
                                                    min_vals, max_vals, wp_sh, &cflag_sh, bid,
                                                    proj_calls++) ||
                    prrax_cons::sq_dist(wp_sh + (W - 1) * CONFIG_DIM_MAX, target_sh) >= d_curr) {
                    extension_failed = true;
                    break;
                }
            }
            if (edge_in_collision(curr_cfg_sh, vec_sh)) { extension_failed = true; break; }

            // Advance (vec for unconstrained, the projected end for constrained) and append.
            __syncthreads();
            if (tid == 0) {
                for (int d = 0; d < dim; ++d)
                    curr_cfg_sh[d] = constrained ? wp_sh[(W - 1) * CONFIG_DIM_MAX + d]
                                                 : curr_cfg_sh[d] + vec_sh[d];
            }
            __syncthreads();
            const int eidx = append_edge(t_tree, extension_parent_sh, curr_cfg_sh);
            if (eidx < 0) { extension_failed = true; break; }
            if (tid == 0) extension_parent_sh = eidx;
            __syncthreads();
        }
        if (constrained && !extension_failed) {
            const float eps = cons_sh[CONS_CONNECT_EPS];
            extension_failed = prrax_cons::sq_dist(curr_cfg_sh, target_sh) >= eps * eps;
        }
        if (extension_failed) return false;

        if (atomicCAS(solved_out, 0, 1) == 0) {
            // extension_parent_sh = last node added in t_tree (at the connect point)
            // connect_nearest_idx_sh = NN in o_tree
            connection_info[t_tree] = extension_parent_sh;
            connection_info[o_tree] = connect_nearest_idx_sh;
            connection_info[2] = t_tree;
        }
        return true;
    };

    // Per-block iteration counter (like pRRTC's local `iter` stack variable)
    int local_iter = 0;

    while (true) {
        if (atomicAdd(solved_out, 0) != 0) return;

        // --- Iteration accounting (per block, like pRRTC) ---
        ++local_iter;
        if (tid == 0 && local_iter > max_iterations) {
            atomicCAS(solved_out, 0, -1);
            atomicMax(iter_count, local_iter);
        }
        __syncthreads();
        if (atomicAdd(solved_out, 0) != 0) return;

        // --- Sample one configuration and select tree side (tid == 0) ---
        if (tid == 0) {
            for (int d = 0; d < dim; ++d) {
                const int prime = prime_order[d % 16];
                const float val = halton_next_runtime(prime, halton_st[d]);
                sample_sh[d] = min_vals[d] + val * (max_vals[d] - min_vals[d]);
            }

            // Tree-side balance: uses bid / gridDim.x ratio, matching pRRTC's
            // per-block bid / num_new_configs assignment.
            const int size_a  = atomicAdd(&tree_sizes[0], 0);
            const int size_b  = atomicAdd(&tree_sizes[1], 0);
            const int total   = size_a + size_b;
            const int nblocks = static_cast<int>(gridDim.x);
            int t_tree_id = 0;

            if (balance_mode == 0 || local_iter == 1) {
                t_tree_id = (bid < (nblocks / 2)) ? 0 : 1;
            } else if (balance_mode == 1) {
                const int diff = (size_a >= size_b) ? (size_a - size_b) : (size_b - size_a);
                if (diff < static_cast<int>(1.5f * nblocks)) {
                    const float ratio = (total > 0) ? (static_cast<float>(size_a) / total) : 0.5f;
                    t_tree_id = (bid < static_cast<int>(nblocks * (1.0f - ratio))) ? 0 : 1;
                } else {
                    const float ratio = (total > 0) ? (static_cast<float>(size_a) / total) : 0.5f;
                    t_tree_id = (ratio < tree_ratio) ? 0 : 1;
                }
            } else if (balance_mode == 2) {
                const int o_id = 1 - vamp_tree_id;
                const int t_sz = (vamp_tree_id == 0) ? size_a : size_b;
                const int o_sz = (o_id       == 0) ? size_a : size_b;
                const float ratio = (t_sz > 0)
                    ? fabsf(static_cast<float>(t_sz - o_sz) / static_cast<float>(t_sz))
                    : 0.0f;
                if (ratio < tree_ratio) vamp_tree_id = 1 - vamp_tree_id;
                t_tree_id = vamp_tree_id;
            } else {
                t_tree_id = (size_a <= size_b) ? 0 : 1;
            }

            t_tree_id_sh = t_tree_id;
            o_tree_id_sh = 1 - t_tree_id;
        }
        __syncthreads();

        const int t_tree_id  = t_tree_id_sh;
        const int o_tree_id  = o_tree_id_sh;
        const float* t_cfgs  = (t_tree_id == 0) ? tree_a_configs : tree_b_configs;
        const float* o_cfgs  = (o_tree_id == 0) ? tree_a_configs : tree_b_configs;
        float* t_cfgs_w      = (t_tree_id == 0) ? tree_a_configs : tree_b_configs;
        int*   t_parents     = (t_tree_id == 0) ? tree_a_parents : tree_b_parents;
        float* t_radii       = (t_tree_id == 0) ? tree_a_radii   : tree_b_radii;

        // Read completed counts (nodes fully written) — safe search range.
        // Uses min(allocated, completed) pattern from pRRTC.
        const int comp_t = atomicAdd(&completed[t_tree_id], 0);
        const int comp_o = atomicAdd(&completed[o_tree_id], 0);
        if (comp_t <= 0 || comp_o <= 0) { __syncthreads(); continue; }

        // --- Parallel nearest-neighbor search in t_tree (pRRTC-style reduction) ---
        float local_min_sq = FLT_MAX;
        int   local_near   = 0;
        for (int i = tid; i < comp_t; i += blockDim.x) {
            if (volatile_load(&t_parents[i]) < 0) continue;   // claimed, not yet written
            float dsq = 0.0f;
            for (int d = 0; d < dim; ++d) {
                const float diff = t_cfgs[d * max_nodes + i] - sample_sh[d];
                dsq += diff * diff;
            }
            if (dsq < local_min_sq) { local_min_sq = dsq; local_near = i; }
        }
        sdata[tid]     = local_min_sq;
        sindex_sh[tid] = local_near;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && sdata[tid + s] < sdata[tid]) {
                sdata[tid]     = sdata[tid + s];
                sindex_sh[tid] = sindex_sh[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float nd  = sqrtf(sdata[0]);
            nearest_idx_sh  = sindex_sh[0];
            // Avoid over-pruning while trees are still tiny; otherwise one side can
            // stall at its root and appear as one-tree-only growth.
            should_skip_sh  = (dynamic_domain && t_radii[nearest_idx_sh] < nd);
            if (!should_skip_sh) {
                const float sc = (nd > 1e-8f) ? fminf(1.0f, step_size / nd) : 1.0f;
                for (int d = 0; d < dim; ++d) {
                    const float base    = t_cfgs[d * max_nodes + nearest_idx_sh];
                    cfg_candidate_sh[d] = base + (sample_sh[d] - base) * sc;
                }
            }
            any_collision_sh = 0;
        }
        __syncthreads();

        if (should_skip_sh) continue;

        // --- Extend toward the sample (edge collision check: one waypoint per thread) ---
        if (tid < dim) {
            near_sh[tid] = t_cfgs[tid * max_nodes + nearest_idx_sh];
            cc_delta_sh[tid] = cfg_candidate_sh[tid] - near_sh[tid];
        }
        __syncthreads();
        bool edge_ok = true;
        if (constrained) {
            edge_ok = prrax_cons::constrained_extend(cons_sh, near_sh, sample_sh, step_size,
                                                     min_vals, max_vals, wp_sh, &cflag_sh, bid,
                                                     proj_calls++);
            if (edge_ok && tid < dim) cfg_candidate_sh[tid] = wp_sh[(W - 1) * CONFIG_DIM_MAX + tid];
            __syncthreads();
        }
        if (edge_ok) edge_ok = !edge_in_collision(near_sh, cc_delta_sh);

        if (!edge_ok) {
            if (tid == 0 && dynamic_domain) {
                prrtc_radius_shrink_atomic(t_radii, nearest_idx_sh, dd_alpha, dd_radius, dd_min_radius);
            }
            __syncthreads();
            continue;
        }

        if (tid == 0 && dynamic_domain) prrtc_radius_grow_atomic(t_radii, nearest_idx_sh, dd_alpha);
        const int new_idx = append_edge(t_tree_id, nearest_idx_sh, cfg_candidate_sh);
        if (new_idx < 0) { if (atomicAdd(solved_out, 0) != 0) return; continue; }

        if (connect_from(t_tree_id, new_idx, cfg_candidate_sh)) {
            atomicMax(iter_count, local_iter);
            return;
        }

        if (atomicAdd(solved_out, 0) != 0) return;
    }
}


// Starts/goals off the constraint manifold are projected onto it (strong damping, many steps:
// seeds may sit near singularities); a root that does not project fails the problem.
__global__ void prrax_roots_kernel(const float* __restrict__ cons, float* __restrict__ ta_cfg,
                                   float* __restrict__ tb_cfg, const float* __restrict__ lo,
                                   const float* __restrict__ hi, int* __restrict__ solved,
                                   int num_goals, int dim, int max_nodes) {
    if (cons[CONS_N] <= 0.0f) return;
    float cp[PRRAX_CONS_SIZE];
    for (int i = 0; i < PRRAX_CONS_SIZE; ++i) cp[i] = cons[i];
    cp[CONS_LAMBDA] = fmaxf(cp[CONS_LAMBDA], 1e-2f);
    cp[CONS_ITERS] = 200.0f;
    cp[CONS_MAX_STEP] = 0.5f;
    for (int r = threadIdx.x; r < 1 + num_goals; r += blockDim.x) {
        float* cfg = (r == 0) ? ta_cfg : tb_cfg + (r - 1);
        float q[CONFIG_DIM_MAX];
        for (int d = 0; d < dim; ++d) q[d] = cfg[d * max_nodes];
        if (prrax_cons::tsr_error_all(cons, q) < cons[CONS_TOL] * cons[CONS_TOL]) continue;
        if (!prrax_cons::tsr_project(cp, q, lo, hi)) { atomicExch(solved, -1); continue; }
        for (int d = 0; d < dim; ++d) cfg[d * max_nodes] = q[d];
    }
}

// The solved path, start root -> goal root, with each constrained edge's interior waypoints
// (w1 per edge, 0 unconstrained) expanded: path [max_path, dim], path_len (-length if it
// does not fit, 0 unsolved).
__global__ void prrax_path_kernel(const int* __restrict__ solved, const int* __restrict__ conn,
                                  const float* __restrict__ ta_cfg, const float* __restrict__ tb_cfg,
                                  const int* __restrict__ ta_par, const int* __restrict__ tb_par,
                                  const float* __restrict__ wps_a, const float* __restrict__ wps_b,
                                  float* __restrict__ path, int* __restrict__ path_len,
                                  const unsigned long long* __restrict__ t_start,
                                  float* __restrict__ time_ms,
                                  int w1, int dim, int max_nodes, int max_path) {
    if (threadIdx.x != 0) return;
    *time_ms = static_cast<float>(globaltimer_ns() - *t_start) * 1e-6f;   // setup + roots + plan
    if (*solved != 1) { *path_len = 0; return; }
    auto depth = [&](const int* par, int n) { int k = 1; while (par[n] != n && par[n] >= 0) { n = par[n]; ++k; } return k; };
    const int la = depth(ta_par, conn[0]), lb = depth(tb_par, conn[1]);
    const int total = 1 + (la - 1) * (w1 + 1) + lb + (lb - 1) * w1;
    if (total > max_path) { *path_len = -total; return; }
    auto put = [&](int pos, const float* cfgs, int n) { for (int d = 0; d < dim; ++d) path[pos * dim + d] = cfgs[d * max_nodes + n]; };
    auto put_wp = [&](int pos, const float* wps, int n, int w) {
        for (int d = 0; d < dim; ++d) path[pos * dim + d] = wps[((size_t)n * w1 + w) * dim + d];
    };
    // Tree A, written back to front: node, preceded by the waypoints of its incoming edge.
    int pos = 1 + (la - 1) * (w1 + 1), n = conn[0];
    while (ta_par[n] != n && ta_par[n] >= 0) {
        put(--pos, ta_cfg, n);
        for (int w = w1 - 1; w >= 0; --w) put_wp(--pos, wps_a, n, w);
        n = ta_par[n];
    }
    put(0, ta_cfg, n);
    // Tree B, front to back: node, then its incoming edge's waypoints reversed, toward the root.
    pos = 1 + (la - 1) * (w1 + 1);
    n = conn[1];
    while (true) {
        put(pos++, tb_cfg, n);
        if (tb_par[n] == n || tb_par[n] < 0) break;
        for (int w = w1 - 1; w >= 0; --w) put_wp(pos++, wps_b, n, w);
        n = tb_par[n];
    }
    *path_len = total;
}

// Per-call scratch kept across calls (grow-only): dynamic-domain radii, streams, events.
// ponytail: one process-wide pool, not thread-safe; a mutex or per-device pools if the handler
// ever runs concurrently.
struct PlannerPool {
    float* radii = nullptr;
    size_t radii_size = 0;
    unsigned long long* t_start = nullptr;   // per-problem solve start (globaltimer)
    int t_size = 0;
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> starts, ends;
    cudaEvent_t inputs_ready = nullptr;
    cudaError_t reserve(int batch, size_t n_radii) {
        if (n_radii > radii_size) {
            cudaFree(radii);
            radii_size = 0;
            const cudaError_t e = cudaMalloc(&radii, sizeof(float) * n_radii);
            if (e != cudaSuccess) return e;
            radii_size = n_radii;
        }
        if (batch > t_size) {
            cudaFree(t_start);
            t_size = 0;
            const cudaError_t e = cudaMalloc(&t_start, sizeof(unsigned long long) * batch);
            if (e != cudaSuccess) return e;
            t_size = batch;
        }
        if (!inputs_ready) cudaEventCreateWithFlags(&inputs_ready, cudaEventDisableTiming);
        while (static_cast<int>(streams.size()) < batch) {
            cudaStream_t st; cudaEvent_t a, b;
            cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&a, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&b, cudaEventDisableTiming);
            streams.push_back(st); starts.push_back(a); ends.push_back(b);
        }
        return cudaSuccess;
    }
};
static PlannerPool g_pool;

// XLA FFI handler: a batch of independent problems, each with its own world, each launched on
// its own stream so they run concurrently. A single solve is a batch of one.
static ffi::Error PrrtcPlannerImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> start_configs,   // [batch, dim]
    ffi::Buffer<ffi::DataType::F32> goal_configs,    // [batch, num_goals, dim]
    ffi::Buffer<ffi::DataType::F32> min_vals,        // [dim]
    ffi::Buffer<ffi::DataType::F32> max_vals,        // [dim]
    ffi::Buffer<ffi::DataType::F32> world_spheres,    // [batch, max_ws, 4]
    ffi::Buffer<ffi::DataType::F32> world_capsules,   // [batch, max_wc, 7]
    ffi::Buffer<ffi::DataType::F32> world_boxes,      // [batch, max_wb, 15]
    ffi::Buffer<ffi::DataType::F32> world_halfspaces, // [batch, max_wh, 6]
    ffi::Buffer<ffi::DataType::S32> world_counts,     // [batch, 4] spheres, capsules, boxes, halfspaces
    ffi::Buffer<ffi::DataType::F32> constraints,      // [batch, PRRAX_CONS_SIZE] (prrtc_constraint.cuh)
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> tree_a_configs,   // [batch, dim, max_nodes]
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> tree_b_configs,   // [batch, dim, max_nodes]
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> tree_a_parents,   // [batch, max_nodes]
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> tree_b_parents,   // [batch, max_nodes]
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> tree_sizes,       // [batch, 2]
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> completed,        // [batch, 2]
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> iter_count,       // [batch, 1]
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> connection_info,  // [batch, 3]
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> solved_flag,      // [batch, 1]
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> kernel_time_ms,   // [batch]
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> edge_wps_a,       // [batch, max_nodes*(W-1)*dim], or [batch, 1] unconstrained
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> edge_wps_b,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> path,             // [batch, max_path, dim]
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> path_len,         // [batch]
    int max_iterations,
    float step_size,
    int num_new_samples,
    int balance_mode,
    float tree_ratio,
    int dynamic_domain,
    float dd_alpha,
    float dd_radius,
    float dd_min_radius,
    int max_nodes,
    int granularity,
    int collision_mode,
    float collision_margin
) {
    const int batch = static_cast<int>(start_configs.dimensions()[0]);
    const int dim = static_cast<int>(start_configs.dimensions()[1]);
    const int num_goals = static_cast<int>(goal_configs.dimensions()[1]);
    const size_t wps_size = static_cast<size_t>(edge_wps_a->dimensions()[1]);
    const int max_w[4] = {
        static_cast<int>(world_spheres.dimensions()[1]), static_cast<int>(world_capsules.dimensions()[1]),
        static_cast<int>(world_boxes.dimensions()[1]), static_cast<int>(world_halfspaces.dimensions()[1])};

    cudaError_t e = g_pool.reserve(batch, (size_t)2 * batch * max_nodes);
    if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    float* radii = g_pool.radii;   // dynamic-domain radii, [2, batch, max_nodes]
    unsigned long long* t_start = g_pool.t_start;
    const int w1 = (wps_size > 1) ? granularity - 1 : 0;
    const int max_path = static_cast<int>(path->dimensions()[1]);
    // Asynchronous: the per-problem streams start after XLA's stream (inputs written), and
    // XLA's stream waits on all of them; nothing blocks the host.
    cudaEventRecord(g_pool.inputs_ready, stream);

    for (int b = 0; b < batch; ++b) {
        cudaStream_t s = g_pool.streams[b];
        cudaStreamWaitEvent(s, g_pool.inputs_ready, 0);
        float* ta_cfg = tree_a_configs->typed_data() + (size_t)b * dim * max_nodes;
        float* tb_cfg = tree_b_configs->typed_data() + (size_t)b * dim * max_nodes;
        int* ta_par = tree_a_parents->typed_data() + (size_t)b * max_nodes;
        int* tb_par = tree_b_parents->typed_data() + (size_t)b * max_nodes;
        float* ta_rad = radii + (size_t)b * max_nodes;
        float* tb_rad = radii + (size_t)(batch + b) * max_nodes;
        float* wa = edge_wps_a->typed_data() + (size_t)b * wps_size;
        float* wb = edge_wps_b->typed_data() + (size_t)b * wps_size;
        const float* cons_b = constraints.typed_data() + (size_t)b * PRRAX_CONS_SIZE;
        int* solved_b = solved_flag->typed_data() + b;

        // A node is visible once its parent is published (append_edge): reset the parents.
        cudaMemsetAsync(ta_par, -1, sizeof(int) * max_nodes, s);
        cudaMemsetAsync(tb_par, -1, sizeof(int) * max_nodes, s);
        prrtc_init_kernel<<<1, 32, 0, s>>>(
            start_configs.typed_data() + b * dim, goal_configs.typed_data() + (size_t)b * num_goals * dim,
            ta_cfg, tb_cfg, ta_par, tb_par, ta_rad, tb_rad,
            tree_sizes->typed_data() + b * 2, completed->typed_data() + b * 2,
            iter_count->typed_data() + b, connection_info->typed_data() + b * 3, solved_b,
            t_start + b, num_goals, dim, max_nodes);
        prrax_roots_kernel<<<1, 32, 0, s>>>(cons_b, ta_cfg, tb_cfg, min_vals.typed_data(), max_vals.typed_data(),
                                            solved_b, num_goals, dim, max_nodes);

        CollisionContext ctx;
        ctx.world_spheres = world_spheres.typed_data() + (size_t)b * max_w[0] * 4;
        ctx.world_capsules = world_capsules.typed_data() + (size_t)b * max_w[1] * 7;
        ctx.world_boxes = world_boxes.typed_data() + (size_t)b * max_w[2] * 15;
        ctx.world_halfspaces = world_halfspaces.typed_data() + (size_t)b * max_w[3] * 6;
        ctx.collision_mode = collision_mode;
        ctx.collision_margin = collision_margin;

        prrtc_planner_kernel<<<num_new_samples, granularity, 0, s>>>(
            ta_cfg, tb_cfg, ta_par, tb_par, ta_rad, tb_rad, wa, wb,
            min_vals.typed_data(), max_vals.typed_data(),
            tree_sizes->typed_data() + b * 2, completed->typed_data() + b * 2,
            iter_count->typed_data() + b, connection_info->typed_data() + b * 3, solved_b,
            ctx, world_counts.typed_data() + b * 4, max_iterations, step_size, num_new_samples,
            balance_mode, tree_ratio, dynamic_domain, dd_alpha, dd_radius, dd_min_radius,
            dim, max_nodes, granularity, cons_b);
        prrax_path_kernel<<<1, 32, 0, s>>>(
            solved_b, connection_info->typed_data() + b * 3,
            ta_cfg, tb_cfg, ta_par, tb_par, wa, wb,
            path->typed_data() + (size_t)b * max_path * dim, path_len->typed_data() + b,
            t_start + b, kernel_time_ms->typed_data() + b,
            w1, dim, max_nodes, max_path);
        cudaEventRecord(g_pool.ends[b], s);
        cudaStreamWaitEvent(stream, g_pool.ends[b], 0);
    }
    e = cudaGetLastError();
    if (e != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PrrtcPlannerFfi, PrrtcPlannerImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // start_configs [batch, dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // goal_configs [batch, num_goals, dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // min_vals [dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // max_vals [dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres [batch, max_ws, 4]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules [batch, max_wc, 7]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes [batch, max_wb, 15]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces [batch, max_wh, 6]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // world_counts [batch, 4]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // constraints [batch, PRRAX_CONS_SIZE]
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // tree_a_configs
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // tree_b_configs
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_a_parents
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_b_parents
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_sizes
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // completed
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // iter_count
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // connection_info
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // solved_flag
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // kernel_time_ms
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // edge_wps_a
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // edge_wps_b
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // path
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // path_len
        .Attr<int>("max_iterations")
        .Attr<float>("step_size")
        .Attr<int>("num_new_samples")
        .Attr<int>("balance_mode")
        .Attr<float>("tree_ratio")
        .Attr<int>("dynamic_domain")
        .Attr<float>("dd_alpha")
        .Attr<float>("dd_radius")
        .Attr<float>("dd_min_radius")
        .Attr<int>("max_nodes")
        .Attr<int>("granularity")
        .Attr<int>("collision_mode")
        .Attr<float>("collision_margin")
);

// Frame poses of this build's FK at configurations q [B, dim] -> T [B, n_frames, 7]. Checks a
// traced build against the baked-table FK: ptxas has miscompiled large cricket2 traces before.
__global__ void prrax_fk_kernel(const float* __restrict__ q, float* __restrict__ T, int batch, int dim) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    grim::robot::frame_poses(q + b * dim, nullptr, T + (size_t)b * grim::robot::n_frames * 7);
}

static ffi::Error PrraxFkImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::F32> q,
                              ffi::Result<ffi::Buffer<ffi::DataType::F32>> T) {
    const int batch = static_cast<int>(q.dimensions()[0]);
    const int dim = static_cast<int>(q.dimensions()[1]);
    prrax_fk_kernel<<<(batch + 63) / 64, 64, 0, stream>>>(q.typed_data(), T->typed_data(), batch, dim);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? ffi::Error::Success() : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PrraxFkFfi, PrraxFkImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
);

// Constraint error and projection of this build at configurations q [B, dim]:
// err2 [B] = sum over constraints of |d|^2 at q, q_proj [B, dim] and ok [B] from tsr_project.
__global__ void prrax_tsr_kernel(const float* __restrict__ q, const float* __restrict__ cons,
                                 const float* __restrict__ lo, const float* __restrict__ hi,
                                 float* __restrict__ err2, float* __restrict__ q_proj,
                                 int* __restrict__ ok, int batch, int dim) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    float x[CONFIG_DIM_MAX];
    for (int k = 0; k < dim; ++k) x[k] = q[b * dim + k];
    err2[b] = prrax_cons::tsr_error_all(cons, x);
    ok[b] = prrax_cons::tsr_project(cons, x, lo, hi) ? 1 : 0;
    for (int k = 0; k < dim; ++k) q_proj[b * dim + k] = x[k];
}

static ffi::Error PrraxTsrImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::F32> q,
                               ffi::Buffer<ffi::DataType::F32> cons,
                               ffi::Buffer<ffi::DataType::F32> lo, ffi::Buffer<ffi::DataType::F32> hi,
                               ffi::Result<ffi::Buffer<ffi::DataType::F32>> err2,
                               ffi::Result<ffi::Buffer<ffi::DataType::F32>> q_proj,
                               ffi::Result<ffi::Buffer<ffi::DataType::S32>> ok) {
    const int batch = static_cast<int>(q.dimensions()[0]);
    const int dim = static_cast<int>(q.dimensions()[1]);
    prrax_tsr_kernel<<<(batch + 63) / 64, 64, 0, stream>>>(
        q.typed_data(), cons.typed_data(), lo.typed_data(), hi.typed_data(),
        err2->typed_data(), q_proj->typed_data(), ok->typed_data(), batch, dim);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? ffi::Error::Success() : ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PrraxTsrFfi, PrraxTsrImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // q [B, dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // constraints [PRRAX_CONS_SIZE]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // lo [dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // hi [dim]
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // err2 [B]
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // q_proj [B, dim]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()   // ok [B]
);
