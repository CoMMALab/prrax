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
#include "_collision_cuda_helpers.cuh"
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <float.h>
#include <algorithm>
#include <math.h>

namespace ffi = xla::ffi;

// Constants
#ifndef CONFIG_DIM_MAX
#define CONFIG_DIM_MAX 16
#endif

#ifndef PRRTC_MAX_JOINTS
#define PRRTC_MAX_JOINTS 64
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
__global__ void prrtc_init_kernel(
    const float* __restrict__ start_config,     // [dim]
    const float* __restrict__ goal_configs,     // [num_goals, dim]
    float* __restrict__ tree_a_configs,         // [dim, max_nodes]
    float* __restrict__ tree_b_configs,         // [dim, max_nodes]
    int* __restrict__ tree_a_parents,           // [max_nodes]
    int* __restrict__ tree_b_parents,           // [max_nodes]
    int num_goals,
    int dim,
    int max_nodes
) {
    const int tid = threadIdx.x;

    if (tid == 0 && blockIdx.x == 0) {
        // Copy start configuration to tree A (root node)
        for (int d = 0; d < dim; d++) {
            tree_a_configs[d * max_nodes] = start_config[d];
        }
        tree_a_parents[0] = 0;

        // Copy goal configurations to tree B
        for (int g = 0; g < num_goals; g++) {
            for (int d = 0; d < dim; d++) {
                tree_b_configs[d * max_nodes + g] = goal_configs[g * dim + d];
            }
            tree_b_parents[g] = g;
        }
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

struct CollisionContext {
    const float* twists;           // [n_joints, 6]
    const float* parent_tf;        // [n_joints, 7]
    const int* parent_idx;         // [n_joints]
    const int* act_idx;            // [n_joints]
    const float* mimic_mul;        // [n_joints]
    const float* mimic_off;        // [n_joints]
    const int* mimic_act_idx;      // [n_joints]
    const int* topo_inv;           // [n_joints]
    const int* sphere_link_idx;    // [n_robot_spheres]
    const float* sphere_local;     // [n_robot_spheres, 3]
    const float* sphere_radius;    // [n_robot_spheres]
    const float* world_spheres;    // [n_world_spheres, 4]   (x,y,z,r)
    const float* world_capsules;   // [n_world_capsules, 7]  (x1,y1,z1,x2,y2,z2,r)
    const float* world_boxes;      // [n_world_boxes, 15]    (cx,cy,cz, a1x,a1y,a1z, a2x,a2y,a2z, a3x,a3y,a3z, hl1,hl2,hl3)
    const float* world_halfspaces; // [n_world_halfspaces, 6] (nx,ny,nz, px,py,pz)
    const int* self_pairs;         // [n_self_pairs, 2]
    // Coarse→fine culling model (one coarse bounding sphere per robot link/group). Each
    // coarse sphere encloses the fine spheres in its group, so a coarse "clear" implies a
    // fine "clear" (sound). Used by collision_mode 2; empty/disabled ⇒ flat fallback.
    const int* coarse_sphere_link_idx;   // [n_coarse], FK frame index per coarse sphere
    const float* coarse_sphere_local;    // [n_coarse, 3]
    const float* coarse_sphere_radius;   // [n_coarse]
    const int* fine_link_offsets;        // [n_coarse+1], group k fine range [off[k],off[k+1])
    const int* coarse_self_pairs;        // [n_coarse_self_pairs, 2] coarse-sphere index pairs
    const int* self_pair_ranges;         // [n_coarse_self_pairs, 2] fine self_pairs [start,end)
    int n_joints;
    int n_act;
    int n_robot_spheres;
    int n_world_spheres;
    int n_world_capsules;
    int n_world_boxes;
    int n_world_halfspaces;
    int n_self_pairs;
    int n_coarse;
    int n_coarse_self_pairs;
    int enabled;
    int coarse_enabled;            // 1 ⇒ coarse arrays present and usable for mode 2
    // Collision-checker selection:
    //   0 = binary       — pRRTC-style early-exit boolean check (mirrors
    //                       pyroffi's collision_binary kernel). Returns at the
    //                       first penetrating obstacle; minimal work per config.
    //   1 = differentiable — sweeps every (robot sphere, obstacle) pair with no
    //                       early exit and accumulates the smooth signed-distance
    //                       cost colldist_from_sdf(d, margin) (mirrors pyroffi's
    //                       _collision_cuda_kernel signed-distance field). A config
    //                       is in collision when the aggregate cost is negative,
    //                       i.e. any sphere lies within `collision_margin`.
    //   2 = binary_coarse — coarse→fine hierarchical early-exit binary check
    //                       (mirrors source pRRTC): a coarse bounding-sphere pre-pass
    //                       proves most free configs clear and gates the fine sweep
    //                       per link. Falls back to mode 0 if coarse_enabled == 0.
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
    for (int rs = 0; rs < ctx.n_robot_spheres; ++rs) {
        const int link_idx = ctx.sphere_link_idx[rs];
        if (link_idx < 0 || link_idx >= ctx.n_joints) continue;

        const float* T = &T_world[link_idx * 7];
        const float* local = &ctx.sphere_local[rs * 3];
        float world_pt[3];
        apply_se3_point(T, local, world_pt);
        const float rr = ctx.sphere_radius[rs];
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
    for (int p = 0; p < ctx.n_self_pairs; ++p) {
        const int i = ctx.self_pairs[p * 2 + 0];
        const int j = ctx.self_pairs[p * 2 + 1];
        if (i < 0 || j < 0 || i >= ctx.n_robot_spheres || j >= ctx.n_robot_spheres) continue;

        const int link_i = ctx.sphere_link_idx[i];
        const int link_j = ctx.sphere_link_idx[j];
        if (link_i < 0 || link_j < 0 || link_i >= ctx.n_joints || link_j >= ctx.n_joints) continue;

        float pi[3];
        float pj[3];
        apply_se3_point(&T_world[link_i * 7], &ctx.sphere_local[i * 3], pi);
        apply_se3_point(&T_world[link_j * 7], &ctx.sphere_local[j * 3], pj);
        if (sphere_sphere_dist(
                pi[0], pi[1], pi[2], ctx.sphere_radius[i],
                pj[0], pj[1], pj[2], ctx.sphere_radius[j]) <= 0.0f) {
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

    for (int rs = 0; rs < ctx.n_robot_spheres; ++rs) {
        const int link_idx = ctx.sphere_link_idx[rs];
        if (link_idx < 0 || link_idx >= ctx.n_joints) continue;

        const float* T = &T_world[link_idx * 7];
        const float* local = &ctx.sphere_local[rs * 3];
        float world_pt[3];
        apply_se3_point(T, local, world_pt);
        const float rr = ctx.sphere_radius[rs];
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

    for (int p = 0; p < ctx.n_self_pairs; ++p) {
        const int i = ctx.self_pairs[p * 2 + 0];
        const int j = ctx.self_pairs[p * 2 + 1];
        if (i < 0 || j < 0 || i >= ctx.n_robot_spheres || j >= ctx.n_robot_spheres) continue;

        const int link_i = ctx.sphere_link_idx[i];
        const int link_j = ctx.sphere_link_idx[j];
        if (link_i < 0 || link_j < 0 || link_i >= ctx.n_joints || link_j >= ctx.n_joints) continue;

        float pi[3];
        float pj[3];
        apply_se3_point(&T_world[link_i * 7], &ctx.sphere_local[i * 3], pi);
        apply_se3_point(&T_world[link_j * 7], &ctx.sphere_local[j * 3], pj);
        const float d = sphere_sphere_dist(
            pi[0], pi[1], pi[2], ctx.sphere_radius[i],
            pj[0], pj[1], pj[2], ctx.sphere_radius[j]);
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
    const int n_coarse = ctx.n_coarse;
    bool link_dirty[PRRTC_MAX_LINKS];
    for (int k = 0; k < n_coarse; ++k) link_dirty[k] = false;

    // --- Coarse world pre-pass: flag groups that might collide. ---
    bool any_dirty = false;
    for (int k = 0; k < n_coarse; ++k) {
        const int link_idx = ctx.coarse_sphere_link_idx[k];
        if (link_idx < 0 || link_idx >= ctx.n_joints) continue;
        float wp[3];
        apply_se3_point(&T_world[link_idx * 7], &ctx.coarse_sphere_local[k * 3], wp);
        if (prrtc_sphere_vs_world(ctx, wp[0], wp[1], wp[2], ctx.coarse_sphere_radius[k])) {
            link_dirty[k] = true;
            any_dirty = true;
        }
    }

    // --- Fine world refine: only the fine spheres of dirty groups. ---
    if (any_dirty) {
        for (int k = 0; k < n_coarse; ++k) {
            if (!link_dirty[k]) continue;
            const int start = ctx.fine_link_offsets[k];
            const int end   = ctx.fine_link_offsets[k + 1];
            for (int rs = start; rs < end; ++rs) {
                const int link_idx = ctx.sphere_link_idx[rs];
                if (link_idx < 0 || link_idx >= ctx.n_joints) continue;
                float wp[3];
                apply_se3_point(&T_world[link_idx * 7], &ctx.sphere_local[rs * 3], wp);
                if (prrtc_sphere_vs_world(ctx, wp[0], wp[1], wp[2], ctx.sphere_radius[rs]))
                    return true;
            }
        }
    }

    // --- Self-collision: coarse-gated per link pair, refine only when coarse pair touches. ---
    for (int p = 0; p < ctx.n_coarse_self_pairs; ++p) {
        const int ck_i = ctx.coarse_self_pairs[p * 2 + 0];
        const int ck_j = ctx.coarse_self_pairs[p * 2 + 1];
        if (ck_i < 0 || ck_j < 0 || ck_i >= n_coarse || ck_j >= n_coarse) continue;
        const int cli = ctx.coarse_sphere_link_idx[ck_i];
        const int clj = ctx.coarse_sphere_link_idx[ck_j];
        if (cli < 0 || clj < 0 || cli >= ctx.n_joints || clj >= ctx.n_joints) continue;

        float ci[3], cj[3];
        apply_se3_point(&T_world[cli * 7], &ctx.coarse_sphere_local[ck_i * 3], ci);
        apply_se3_point(&T_world[clj * 7], &ctx.coarse_sphere_local[ck_j * 3], cj);
        if (sphere_sphere_dist(ci[0], ci[1], ci[2], ctx.coarse_sphere_radius[ck_i],
                               cj[0], cj[1], cj[2], ctx.coarse_sphere_radius[ck_j]) > 0.0f)
            continue;  // coarse pair clear ⇒ all its fine pairs clear

        const int start = ctx.self_pair_ranges[p * 2 + 0];
        const int end   = ctx.self_pair_ranges[p * 2 + 1];
        for (int q = start; q < end; ++q) {
            const int i = ctx.self_pairs[q * 2 + 0];
            const int j = ctx.self_pairs[q * 2 + 1];
            if (i < 0 || j < 0 || i >= ctx.n_robot_spheres || j >= ctx.n_robot_spheres) continue;
            const int link_i = ctx.sphere_link_idx[i];
            const int link_j = ctx.sphere_link_idx[j];
            if (link_i < 0 || link_j < 0 || link_i >= ctx.n_joints || link_j >= ctx.n_joints) continue;
            float pi[3], pj[3];
            apply_se3_point(&T_world[link_i * 7], &ctx.sphere_local[i * 3], pi);
            apply_se3_point(&T_world[link_j * 7], &ctx.sphere_local[j * 3], pj);
            if (sphere_sphere_dist(pi[0], pi[1], pi[2], ctx.sphere_radius[i],
                                   pj[0], pj[1], pj[2], ctx.sphere_radius[j]) <= 0.0f)
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
    if (ctx.n_joints <= 0 || ctx.n_joints > PRRTC_MAX_JOINTS) return false;

    float T_world[PRRTC_MAX_JOINTS * 7];
    fk_single(
        cfg,
        ctx.twists,
        ctx.parent_tf,
        ctx.parent_idx,
        ctx.act_idx,
        ctx.mimic_mul,
        ctx.mimic_off,
        ctx.mimic_act_idx,
        ctx.topo_inv,
        T_world,
        ctx.n_joints,
        ctx.n_act
    );

    if (ctx.collision_mode == 1) {
        return prrtc_config_in_collision_differentiable(ctx, T_world);
    }
    if (ctx.collision_mode == 2 && ctx.coarse_enabled && ctx.n_coarse <= PRRTC_MAX_LINKS) {
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
    int* __restrict__ any_collision_sh
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
        for (int d = 0; d < dim; ++d) interp[d] = base[d] + tf * delta[d];

        if (lane == 0) {
            fk_single(interp, ctx.twists, ctx.parent_tf, ctx.parent_idx, ctx.act_idx,
                      ctx.mimic_mul, ctx.mimic_off, ctx.mimic_act_idx, ctx.topo_inv,
                      myT, ctx.n_joints, ctx.n_act);
        }
        __syncwarp(group_mask);

        // Coarse pre-pass: lanes split coarse spheres, build a per-group dirty bitmask.
        unsigned long long dirty = 0ull;
        for (int k = lane; k < ctx.n_coarse; k += PRRTC_CC_LANES) {
            const int li = ctx.coarse_sphere_link_idx[k];
            if (li < 0 || li >= ctx.n_joints) continue;
            float wp[3];
            apply_se3_point(&myT[li * 7], &ctx.coarse_sphere_local[k * 3], wp);
            if (prrtc_sphere_vs_world(ctx, wp[0], wp[1], wp[2], ctx.coarse_sphere_radius[k]))
                dirty |= (1ull << k);
        }
        for (int off = PRRTC_CC_LANES / 2; off > 0; off >>= 1)
            dirty |= __shfl_xor_sync(group_mask, dirty, off);

        // Fine world refine: lanes split the fine spheres of dirty groups, warp early-out.
        bool hit = false;
        for (int k = 0; k < ctx.n_coarse; ++k) {
            if (!((dirty >> k) & 1ull)) continue;
            const int s = ctx.fine_link_offsets[k];
            const int e = ctx.fine_link_offsets[k + 1];
            for (int rs = s + lane; rs < e; rs += PRRTC_CC_LANES) {
                const int li = ctx.sphere_link_idx[rs];
                if (li < 0 || li >= ctx.n_joints) continue;
                float wp[3];
                apply_se3_point(&myT[li * 7], &ctx.sphere_local[rs * 3], wp);
                if (prrtc_sphere_vs_world(ctx, wp[0], wp[1], wp[2], ctx.sphere_radius[rs])) {
                    hit = true;
                    break;
                }
            }
            if (__any_sync(group_mask, hit)) { hit = true; break; }
        }

        // Self-collision: lanes split coarse self pairs; refine fine pairs when coarse touches.
        if (!hit) {
            bool self_hit = false;
            for (int p = lane; p < ctx.n_coarse_self_pairs; p += PRRTC_CC_LANES) {
                const int ci = ctx.coarse_self_pairs[p * 2 + 0];
                const int cj = ctx.coarse_self_pairs[p * 2 + 1];
                if (ci < 0 || cj < 0 || ci >= ctx.n_coarse || cj >= ctx.n_coarse) continue;
                const int li = ctx.coarse_sphere_link_idx[ci];
                const int lj = ctx.coarse_sphere_link_idx[cj];
                if (li < 0 || lj < 0 || li >= ctx.n_joints || lj >= ctx.n_joints) continue;
                float a[3], b[3];
                apply_se3_point(&myT[li * 7], &ctx.coarse_sphere_local[ci * 3], a);
                apply_se3_point(&myT[lj * 7], &ctx.coarse_sphere_local[cj * 3], b);
                if (sphere_sphere_dist(a[0], a[1], a[2], ctx.coarse_sphere_radius[ci],
                                       b[0], b[1], b[2], ctx.coarse_sphere_radius[cj]) > 0.0f)
                    continue;
                const int qs = ctx.self_pair_ranges[p * 2 + 0];
                const int qe = ctx.self_pair_ranges[p * 2 + 1];
                for (int q = qs; q < qe; ++q) {
                    const int i = ctx.self_pairs[q * 2 + 0];
                    const int j = ctx.self_pairs[q * 2 + 1];
                    if (i < 0 || j < 0 || i >= ctx.n_robot_spheres || j >= ctx.n_robot_spheres) continue;
                    const int lii = ctx.sphere_link_idx[i];
                    const int ljj = ctx.sphere_link_idx[j];
                    if (lii < 0 || ljj < 0 || lii >= ctx.n_joints || ljj >= ctx.n_joints) continue;
                    float pi[3], pj[3];
                    apply_se3_point(&myT[lii * 7], &ctx.sphere_local[i * 3], pi);
                    apply_se3_point(&myT[ljj * 7], &ctx.sphere_local[j * 3], pj);
                    if (sphere_sphere_dist(pi[0], pi[1], pi[2], ctx.sphere_radius[i],
                                           pj[0], pj[1], pj[2], ctx.sphere_radius[j]) <= 0.0f) {
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
    const float* __restrict__ min_vals,       // [dim]
    const float* __restrict__ max_vals,       // [dim]
    int* __restrict__ tree_sizes,             // [2] atomic allocation counters
    int* __restrict__ completed,              // [2] write-completion counters
    int* __restrict__ iter_count,             // [1] output: max iterations reached
    int* __restrict__ connection_info,        // [3] [a_idx, b_idx, expand_tree_id]
    int* __restrict__ solved_out,             // [1] planner solved flag
    CollisionContext collision_ctx,
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
    int granularity
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

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

    // Phase B cooperative-CC scratch: per-group shared FK transforms + shared edge endpoints.
    __shared__ float cc_group_T[(PRRTC_BLOCK_THREADS_MAX / PRRTC_CC_LANES) * PRRTC_CC_MAX_JOINTS * 7];
    __shared__ float cc_base_sh[CONFIG_DIM_MAX];
    __shared__ float cc_delta_sh[CONFIG_DIM_MAX];
    // Cooperative path is selected once: coarse model present, FK fits the shared buffer,
    // and the block is wide enough to form lane-groups. Uniform across the block.
    const bool cc_coop = (collision_ctx.collision_mode == 3 && collision_ctx.coarse_enabled
                          && collision_ctx.n_joints <= PRRTC_CC_MAX_JOINTS
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

        // --- Parallel edge collision check ---
        // Each thread independently evaluates one waypoint along the extend edge
        // at t = (tid+1)/blockDim.x, so the last thread checks the endpoint.
        // This mirrors pRRTC's granularity-parallel FK+CC across all block threads.
        // T_world inside prrtc_config_in_collision is stack-allocated, so concurrent
        // calls from all threads are fully independent.
        if (cc_coop) {
            // Cooperative path validates all blockDim.x waypoints with lane-groups.
            if (tid < dim) {
                cc_base_sh[tid] = t_cfgs[tid * max_nodes + nearest_idx_sh];
                cc_delta_sh[tid] = cfg_candidate_sh[tid] - cc_base_sh[tid];
            }
            __syncthreads();
            prrtc_edge_cc_cooperative(cc_base_sh, cc_delta_sh, dim, blockDim.x,
                                      collision_ctx, cc_group_T, &any_collision_sh);
        } else {
            const float t_frac = static_cast<float>(tid + 1) / static_cast<float>(blockDim.x);
            float interp[CONFIG_DIM_MAX];
            for (int d = 0; d < dim; ++d) {
                const float base = t_cfgs[d * max_nodes + nearest_idx_sh];
                interp[d] = base + t_frac * (cfg_candidate_sh[d] - base);
            }
            if (prrtc_config_in_collision(interp, collision_ctx)) {
                atomicOr(&any_collision_sh, 1);
            }
        }
        __syncthreads();

        if (any_collision_sh != 0) {
            if (tid == 0 && dynamic_domain) {
                prrtc_radius_shrink_atomic(
                    t_radii,
                    nearest_idx_sh,
                    dd_alpha,
                    dd_radius,
                    dd_min_radius
                );
            }
            __syncthreads();
            continue;
        }

        // --- Claim a slot and write new node (parallel over dim dimensions) ---
        if (tid == 0) {
            const int idx = atomicAdd(&tree_sizes[t_tree_id], 1);
            if (idx >= max_nodes) {
                atomicCAS(solved_out, 0, -1);
                new_idx_sh = -1;
            } else {
                new_idx_sh = idx;
                t_parents[idx] = nearest_idx_sh;
                t_radii[idx]   = FLT_MAX;
                if (dynamic_domain) {
                    prrtc_radius_grow_atomic(t_radii, nearest_idx_sh, dd_alpha);
                }
            }
        }
        __syncthreads();

        if (new_idx_sh < 0) { if (atomicAdd(solved_out, 0) != 0) return; continue; }

        // Parallel write: each thread writes one dimension (like pRRTC's tid < dim writes)
        if (tid < dim) t_cfgs_w[tid * max_nodes + new_idx_sh] = cfg_candidate_sh[tid];
        if (tid == 0) atomicAdd(&completed[t_tree_id], 1);
        __syncthreads();

        // --- Connect phase: parallel NN in o_tree ---
        // Fresh completed count for o_tree — more nodes may have been added by other blocks.
        const int comp_o_conn = atomicAdd(&completed[o_tree_id], 0);

        float local_min_sq2 = FLT_MAX;
        int   local_near2   = 0;
        for (int i = tid; i < comp_o_conn; i += blockDim.x) {
            float dsq = 0.0f;
            for (int d = 0; d < dim; ++d) {
                const float diff = o_cfgs[d * max_nodes + i] - cfg_candidate_sh[d];
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
            n_extensions_sh     = n_ext;
            extension_parent_sh = new_idx_sh;
            // Direction vector for one extension step (matches pRRTC's vec computation)
            for (int d = 0; d < dim; ++d) {
                curr_cfg_sh[d] = cfg_candidate_sh[d];
                vec_sh[d]      = (n_ext > 0)
                    ? (o_cfgs[d * max_nodes + connect_nearest_idx_sh] - cfg_candidate_sh[d])
                      / static_cast<float>(n_ext)
                    : 0.0f;
            }
        }
        __syncthreads();

        // --- Extension loop: step toward connect NN, collision-checking each sub-edge ---
        bool extension_failed = false;
        for (int ext = 0; ext < n_extensions_sh; ++ext) {
            // Parallel CC: each thread checks one interpolation point on the sub-edge
            // [curr_cfg, curr_cfg + vec], matching pRRTC's granularity-parallel CC
            // within each extension step.
            if (tid == 0) any_collision_sh = 0;
            __syncthreads();

            if (cc_coop) {
                prrtc_edge_cc_cooperative(curr_cfg_sh, vec_sh, dim, blockDim.x,
                                          collision_ctx, cc_group_T, &any_collision_sh);
            } else {
                const float t_frac = static_cast<float>(tid + 1) / static_cast<float>(blockDim.x);
                float interp[CONFIG_DIM_MAX];
                for (int d = 0; d < dim; ++d)
                    interp[d] = curr_cfg_sh[d] + t_frac * vec_sh[d];
                if (prrtc_config_in_collision(interp, collision_ctx))
                    atomicOr(&any_collision_sh, 1);
            }
            __syncthreads();

            if (any_collision_sh != 0) { extension_failed = true; break; }

            // Claim slot and write extension node
            if (tid == 0) {
                const int eidx = atomicAdd(&tree_sizes[t_tree_id], 1);
                if (eidx >= max_nodes) {
                    atomicCAS(solved_out, 0, -1);
                    ext_idx_sh = -1;
                } else {
                    ext_idx_sh = eidx;
                    t_parents[eidx] = extension_parent_sh;
                    t_radii[eidx]   = FLT_MAX;
                    extension_parent_sh = eidx;
                    // Advance position by one step
                    for (int d = 0; d < dim; ++d) curr_cfg_sh[d] += vec_sh[d];
                }
            }
            __syncthreads();

            if (ext_idx_sh < 0) { extension_failed = true; break; }

            // Parallel write of extension node config
            if (tid < dim) t_cfgs_w[tid * max_nodes + ext_idx_sh] = curr_cfg_sh[tid];
            if (tid == 0) atomicAdd(&completed[t_tree_id], 1);
            __syncthreads();
        }

        // --- Connection found ---
        if (!extension_failed) {
            if (atomicCAS(solved_out, 0, 1) == 0) {
                // extension_parent_sh = last node added in t_tree (at the connect point)
                // connect_nearest_idx_sh = NN in o_tree
                if (t_tree_id == 0) {
                    connection_info[0] = extension_parent_sh;
                    connection_info[1] = connect_nearest_idx_sh;
                } else {
                    connection_info[0] = connect_nearest_idx_sh;
                    connection_info[1] = extension_parent_sh;
                }
                connection_info[2] = t_tree_id;
                solved_out[0]      = 1;
            }
            atomicMax(iter_count, local_iter);
            return;
        }

        if (atomicAdd(solved_out, 0) != 0) return;
    }
}

// XLA FFI handler for pRRTC planner
static ffi::Error PrrtcPlannerImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> start_config,    // [dim]
    ffi::Buffer<ffi::DataType::F32> goal_configs,    // [num_goals, dim]
    ffi::Buffer<ffi::DataType::F32> min_vals,        // [dim]
    ffi::Buffer<ffi::DataType::F32> max_vals,        // [dim]
    ffi::Buffer<ffi::DataType::F32> fk_twists,       // [n_joints, 6]
    ffi::Buffer<ffi::DataType::F32> fk_parent_tf,    // [n_joints, 7]
    ffi::Buffer<ffi::DataType::S32> fk_parent_idx,   // [n_joints]
    ffi::Buffer<ffi::DataType::S32> fk_act_idx,      // [n_joints]
    ffi::Buffer<ffi::DataType::F32> fk_mimic_mul,    // [n_joints]
    ffi::Buffer<ffi::DataType::F32> fk_mimic_off,    // [n_joints]
    ffi::Buffer<ffi::DataType::S32> fk_mimic_act_idx,// [n_joints]
    ffi::Buffer<ffi::DataType::S32> fk_topo_inv,     // [n_joints]
    ffi::Buffer<ffi::DataType::S32> sphere_link_idx, // [n_robot_spheres]
    ffi::Buffer<ffi::DataType::F32> sphere_local,    // [n_robot_spheres, 3]
    ffi::Buffer<ffi::DataType::F32> sphere_radius,   // [n_robot_spheres]
    ffi::Buffer<ffi::DataType::F32> world_spheres,    // [n_world_spheres, 4]
    ffi::Buffer<ffi::DataType::F32> world_capsules,  // [n_world_capsules, 7]
    ffi::Buffer<ffi::DataType::F32> world_boxes,     // [n_world_boxes, 15]
    ffi::Buffer<ffi::DataType::F32> world_halfspaces,// [n_world_halfspaces, 6]
    ffi::Buffer<ffi::DataType::S32> self_pairs,      // [n_self_pairs, 2]
    ffi::Buffer<ffi::DataType::S32> coarse_sphere_link_idx,  // [n_coarse]
    ffi::Buffer<ffi::DataType::F32> coarse_sphere_local,     // [n_coarse, 3]
    ffi::Buffer<ffi::DataType::F32> coarse_sphere_radius,    // [n_coarse]
    ffi::Buffer<ffi::DataType::S32> fine_link_offsets,       // [n_coarse+1]
    ffi::Buffer<ffi::DataType::S32> coarse_self_pairs,       // [n_coarse_self_pairs, 2]
    ffi::Buffer<ffi::DataType::S32> self_pair_ranges,        // [n_coarse_self_pairs, 2]
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> tree_a_configs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> tree_b_configs,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> tree_a_parents,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> tree_b_parents,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> tree_sizes,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> completed,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> iter_count,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> connection_info,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> solved_flag,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> kernel_time_ms,
    int max_iterations,
    float step_size,
    int num_new_samples,
    int balance_mode,
    float tree_ratio,
    int dynamic_domain,
    float dd_alpha,
    float dd_radius,
    float dd_min_radius,
    int dim,
    int max_nodes,
    int granularity,
    int collision_mode,
    float collision_margin
) {
    const int num_goals = static_cast<int>(goal_configs.dimensions()[0]);
    const int n_joints = static_cast<int>(fk_parent_idx.dimensions().size() == 0 ? 0 : fk_parent_idx.dimensions()[0]);
    const int n_robot_spheres = static_cast<int>(sphere_link_idx.dimensions().size() == 0 ? 0 : sphere_link_idx.dimensions()[0]);
    const int n_world_spheres = static_cast<int>(world_spheres.dimensions().size() == 0 ? 0 : world_spheres.dimensions()[0]);
    const int n_world_capsules = static_cast<int>(world_capsules.dimensions().size() == 0 ? 0 : world_capsules.dimensions()[0]);
    const int n_world_boxes = static_cast<int>(world_boxes.dimensions().size() == 0 ? 0 : world_boxes.dimensions()[0]);
    const int n_world_halfspaces = static_cast<int>(world_halfspaces.dimensions().size() == 0 ? 0 : world_halfspaces.dimensions()[0]);
    const int n_self_pairs = static_cast<int>(self_pairs.dimensions().size() == 0 ? 0 : self_pairs.dimensions()[0]);

    CollisionContext collision_ctx;
    collision_ctx.twists = fk_twists.typed_data();
    collision_ctx.parent_tf = fk_parent_tf.typed_data();
    collision_ctx.parent_idx = fk_parent_idx.typed_data();
    collision_ctx.act_idx = fk_act_idx.typed_data();
    collision_ctx.mimic_mul = fk_mimic_mul.typed_data();
    collision_ctx.mimic_off = fk_mimic_off.typed_data();
    collision_ctx.mimic_act_idx = fk_mimic_act_idx.typed_data();
    collision_ctx.topo_inv = fk_topo_inv.typed_data();
    collision_ctx.sphere_link_idx = sphere_link_idx.typed_data();
    collision_ctx.sphere_local = sphere_local.typed_data();
    collision_ctx.sphere_radius = sphere_radius.typed_data();
    collision_ctx.world_spheres = world_spheres.typed_data();
    collision_ctx.world_capsules = world_capsules.typed_data();
    collision_ctx.world_boxes = world_boxes.typed_data();
    collision_ctx.world_halfspaces = world_halfspaces.typed_data();
    collision_ctx.self_pairs = self_pairs.typed_data();
    collision_ctx.n_joints = n_joints;
    collision_ctx.n_act = dim;
    collision_ctx.n_robot_spheres = n_robot_spheres;
    collision_ctx.n_world_spheres = n_world_spheres;
    collision_ctx.n_world_capsules = n_world_capsules;
    collision_ctx.n_world_boxes = n_world_boxes;
    collision_ctx.n_world_halfspaces = n_world_halfspaces;
    collision_ctx.n_self_pairs = n_self_pairs;
    collision_ctx.enabled = (n_joints > 0 && n_robot_spheres > 0 &&
        (n_world_spheres > 0 || n_world_capsules > 0 || n_world_boxes > 0 || n_world_halfspaces > 0 || n_self_pairs > 0)) ? 1 : 0;
    collision_ctx.collision_mode = collision_mode;
    collision_ctx.collision_margin = collision_margin;
    {
        const int n_coarse = (coarse_sphere_link_idx.dimensions().size() == 0)
            ? 0 : static_cast<int>(coarse_sphere_link_idx.dimensions()[0]);
        const int n_coarse_self_pairs = (coarse_self_pairs.dimensions().size() == 0)
            ? 0 : static_cast<int>(coarse_self_pairs.dimensions()[0]);
        collision_ctx.coarse_sphere_link_idx = coarse_sphere_link_idx.typed_data();
        collision_ctx.coarse_sphere_local    = coarse_sphere_local.typed_data();
        collision_ctx.coarse_sphere_radius   = coarse_sphere_radius.typed_data();
        collision_ctx.fine_link_offsets      = fine_link_offsets.typed_data();
        collision_ctx.coarse_self_pairs      = coarse_self_pairs.typed_data();
        collision_ctx.self_pair_ranges       = self_pair_ranges.typed_data();
        collision_ctx.n_coarse               = n_coarse;
        collision_ctx.n_coarse_self_pairs    = n_coarse_self_pairs;
        collision_ctx.coarse_enabled = (n_coarse > 0 && n_coarse <= PRRTC_MAX_LINKS) ? 1 : 0;
    }

    float* tree_a_radii = nullptr;
    float* tree_b_radii = nullptr;

    // Initialize trees to zero
    cudaError_t e = cudaMemsetAsync(tree_a_configs->typed_data(), 0,
                                      sizeof(float) * dim * max_nodes, stream);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    e = cudaMemsetAsync(tree_b_configs->typed_data(), 0,
                        sizeof(float) * dim * max_nodes, stream);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    e = cudaMemsetAsync(tree_a_parents->typed_data(), -1,
                        sizeof(int) * max_nodes, stream);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    e = cudaMemsetAsync(tree_b_parents->typed_data(), -1,
                        sizeof(int) * max_nodes, stream);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    // Initialize counters
    int init_sizes[2] = {1, num_goals};
    e = cudaMemcpyAsync(tree_sizes->typed_data(), init_sizes, sizeof(int) * 2,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    int init_completed[2] = {1, num_goals};
    e = cudaMemcpyAsync(completed->typed_data(), init_completed, sizeof(int) * 2,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    int zero = 0;
    e = cudaMemcpyAsync(iter_count->typed_data(), &zero, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    int init_connection[3] = {-1, -1, -1};
    e = cudaMemcpyAsync(connection_info->typed_data(), init_connection, sizeof(int) * 3,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    e = cudaMemcpyAsync(solved_flag->typed_data(), &zero, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    e = cudaMalloc(&tree_a_radii, sizeof(float) * max_nodes);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    e = cudaMalloc(&tree_b_radii, sizeof(float) * max_nodes);
    if (e != cudaSuccess) {
        cudaFree(tree_a_radii);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    int threads_fill = 16;
    int blocks_fill = (max_nodes + threads_fill - 1) / threads_fill;
    prrtc_fill_float_kernel<<<blocks_fill, threads_fill, 0, stream>>>(tree_a_radii, FLT_MAX, max_nodes);
    prrtc_fill_float_kernel<<<blocks_fill, threads_fill, 0, stream>>>(tree_b_radii, FLT_MAX, max_nodes);

    e = cudaGetLastError();
    if (e != cudaSuccess) {
        cudaFree(tree_a_radii);
        cudaFree(tree_b_radii);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    // Initialize trees with start and goal configs
    prrtc_init_kernel<<<1, 32, 0, stream>>>(
        start_config.typed_data(),
        goal_configs.typed_data(),
        tree_a_configs->typed_data(),
        tree_b_configs->typed_data(),
        tree_a_parents->typed_data(),
        tree_b_parents->typed_data(),
        num_goals, dim, max_nodes
    );

    e = cudaGetLastError();
    if (e != cudaSuccess) {
        cudaFree(tree_a_radii);
        cudaFree(tree_b_radii);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    // Launch one block per sample (num_new_samples blocks), each block is an
    // independent explorer — matching pRRTC's num_new_configs-block launch pattern.
    // Block size = granularity (must be power of 2, capped at PRRTC_BLOCK_THREADS_MAX).
    // All shared memory is statically declared; no dynamic shared memory needed.
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_end = nullptr;
    e = cudaEventCreate(&kernel_start);
    if (e != cudaSuccess) {
        cudaFree(tree_a_radii);
        cudaFree(tree_b_radii);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    e = cudaEventCreate(&kernel_end);
    if (e != cudaSuccess) {
        cudaEventDestroy(kernel_start);
        cudaFree(tree_a_radii);
        cudaFree(tree_b_radii);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    e = cudaEventRecord(kernel_start, stream);
    if (e != cudaSuccess) {
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaFree(tree_a_radii);
        cudaFree(tree_b_radii);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    prrtc_planner_kernel<<<num_new_samples, granularity, 0, stream>>>(
        tree_a_configs->typed_data(),
        tree_b_configs->typed_data(),
        tree_a_parents->typed_data(),
        tree_b_parents->typed_data(),
        tree_a_radii,
        tree_b_radii,
        min_vals.typed_data(),
        max_vals.typed_data(),
        tree_sizes->typed_data(),
        completed->typed_data(),
        iter_count->typed_data(),
        connection_info->typed_data(),
        solved_flag->typed_data(),
        collision_ctx,
        max_iterations,
        step_size,
        num_new_samples,
        balance_mode,
        tree_ratio,
        dynamic_domain,
        dd_alpha,
        dd_radius,
        dd_min_radius,
        dim,
        max_nodes,
        granularity
    );

    e = cudaGetLastError();
    if (e != cudaSuccess) {
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaFree(tree_a_radii);
        cudaFree(tree_b_radii);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    e = cudaEventRecord(kernel_end, stream);
    if (e != cudaSuccess) {
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaFree(tree_a_radii);
        cudaFree(tree_b_radii);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    e = cudaEventSynchronize(kernel_end);
    if (e != cudaSuccess) {
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaFree(tree_a_radii);
        cudaFree(tree_b_radii);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    float elapsed_ms = 0.0f;
    e = cudaEventElapsedTime(&elapsed_ms, kernel_start, kernel_end);
    if (e != cudaSuccess) {
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaFree(tree_a_radii);
        cudaFree(tree_b_radii);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    e = cudaMemcpyAsync(
        kernel_time_ms->typed_data(),
        &elapsed_ms,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream
    );
    if (e != cudaSuccess) {
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
        cudaFree(tree_a_radii);
        cudaFree(tree_b_radii);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_end);

    cudaFree(tree_a_radii);
    cudaFree(tree_b_radii);

    return ffi::Error::Success();
}

// XLA FFI handler for pRRTC batch planner.
// Accepts start_configs [batch, dim] and launches each problem on its own
// child CUDA stream so all batch elements run concurrently on the GPU.
// Results have a leading batch dimension: tree_*_configs [batch, dim, max_nodes],
// tree_*_parents [batch, max_nodes], tree_sizes/completed [batch, 2],
// iter_count/solved_flag [batch, 1], connection_info [batch, 3].
static ffi::Error PrrtcPlannerBatchImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> start_configs,   // [batch, dim]
    ffi::Buffer<ffi::DataType::F32> goal_configs,    // [num_goals, dim]
    ffi::Buffer<ffi::DataType::F32> min_vals,        // [dim]
    ffi::Buffer<ffi::DataType::F32> max_vals,        // [dim]
    ffi::Buffer<ffi::DataType::F32> fk_twists,
    ffi::Buffer<ffi::DataType::F32> fk_parent_tf,
    ffi::Buffer<ffi::DataType::S32> fk_parent_idx,
    ffi::Buffer<ffi::DataType::S32> fk_act_idx,
    ffi::Buffer<ffi::DataType::F32> fk_mimic_mul,
    ffi::Buffer<ffi::DataType::F32> fk_mimic_off,
    ffi::Buffer<ffi::DataType::S32> fk_mimic_act_idx,
    ffi::Buffer<ffi::DataType::S32> fk_topo_inv,
    ffi::Buffer<ffi::DataType::S32> sphere_link_idx,
    ffi::Buffer<ffi::DataType::F32> sphere_local,
    ffi::Buffer<ffi::DataType::F32> sphere_radius,
    ffi::Buffer<ffi::DataType::F32> world_spheres,
    ffi::Buffer<ffi::DataType::F32> world_capsules,
    ffi::Buffer<ffi::DataType::F32> world_boxes,
    ffi::Buffer<ffi::DataType::F32> world_halfspaces,
    ffi::Buffer<ffi::DataType::S32> self_pairs,
    ffi::Buffer<ffi::DataType::S32> coarse_sphere_link_idx,
    ffi::Buffer<ffi::DataType::F32> coarse_sphere_local,
    ffi::Buffer<ffi::DataType::F32> coarse_sphere_radius,
    ffi::Buffer<ffi::DataType::S32> fine_link_offsets,
    ffi::Buffer<ffi::DataType::S32> coarse_self_pairs,
    ffi::Buffer<ffi::DataType::S32> self_pair_ranges,
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
    int max_iterations,
    float step_size,
    int num_new_samples,
    int balance_mode,
    float tree_ratio,
    int dynamic_domain,
    float dd_alpha,
    float dd_radius,
    float dd_min_radius,
    int dim,
    int max_nodes,
    int granularity,
    int collision_mode,
    float collision_margin
) {
    const int batch     = static_cast<int>(start_configs.dimensions()[0]);
    // goal_configs is [batch, num_goals, dim]; each problem has its own goal slice
    const int num_goals = static_cast<int>(goal_configs.dimensions()[1]);
    const int n_joints        = static_cast<int>(fk_parent_idx.dimensions().size() == 0 ? 0 : fk_parent_idx.dimensions()[0]);
    const int n_robot_spheres = static_cast<int>(sphere_link_idx.dimensions().size() == 0 ? 0 : sphere_link_idx.dimensions()[0]);
    const int n_world_spheres    = static_cast<int>(world_spheres.dimensions().size()    == 0 ? 0 : world_spheres.dimensions()[0]);
    const int n_world_capsules   = static_cast<int>(world_capsules.dimensions().size()   == 0 ? 0 : world_capsules.dimensions()[0]);
    const int n_world_boxes      = static_cast<int>(world_boxes.dimensions().size()      == 0 ? 0 : world_boxes.dimensions()[0]);
    const int n_world_halfspaces = static_cast<int>(world_halfspaces.dimensions().size() == 0 ? 0 : world_halfspaces.dimensions()[0]);
    const int n_self_pairs       = static_cast<int>(self_pairs.dimensions().size()       == 0 ? 0 : self_pairs.dimensions()[0]);

    CollisionContext collision_ctx;
    collision_ctx.twists        = fk_twists.typed_data();
    collision_ctx.parent_tf     = fk_parent_tf.typed_data();
    collision_ctx.parent_idx    = fk_parent_idx.typed_data();
    collision_ctx.act_idx       = fk_act_idx.typed_data();
    collision_ctx.mimic_mul     = fk_mimic_mul.typed_data();
    collision_ctx.mimic_off     = fk_mimic_off.typed_data();
    collision_ctx.mimic_act_idx = fk_mimic_act_idx.typed_data();
    collision_ctx.topo_inv      = fk_topo_inv.typed_data();
    collision_ctx.sphere_link_idx   = sphere_link_idx.typed_data();
    collision_ctx.sphere_local      = sphere_local.typed_data();
    collision_ctx.sphere_radius     = sphere_radius.typed_data();
    collision_ctx.world_spheres     = world_spheres.typed_data();
    collision_ctx.world_capsules    = world_capsules.typed_data();
    collision_ctx.world_boxes       = world_boxes.typed_data();
    collision_ctx.world_halfspaces  = world_halfspaces.typed_data();
    collision_ctx.self_pairs        = self_pairs.typed_data();
    collision_ctx.n_joints          = n_joints;
    collision_ctx.n_act             = dim;
    collision_ctx.n_robot_spheres   = n_robot_spheres;
    collision_ctx.n_world_spheres   = n_world_spheres;
    collision_ctx.n_world_capsules  = n_world_capsules;
    collision_ctx.n_world_boxes     = n_world_boxes;
    collision_ctx.n_world_halfspaces = n_world_halfspaces;
    collision_ctx.n_self_pairs      = n_self_pairs;
    collision_ctx.enabled = (n_joints > 0 && n_robot_spheres > 0 &&
        (n_world_spheres > 0 || n_world_capsules > 0 || n_world_boxes > 0 ||
         n_world_halfspaces > 0 || n_self_pairs > 0)) ? 1 : 0;
    collision_ctx.collision_mode = collision_mode;
    collision_ctx.collision_margin = collision_margin;
    {
        const int n_coarse = (coarse_sphere_link_idx.dimensions().size() == 0)
            ? 0 : static_cast<int>(coarse_sphere_link_idx.dimensions()[0]);
        const int n_coarse_self_pairs = (coarse_self_pairs.dimensions().size() == 0)
            ? 0 : static_cast<int>(coarse_self_pairs.dimensions()[0]);
        collision_ctx.coarse_sphere_link_idx = coarse_sphere_link_idx.typed_data();
        collision_ctx.coarse_sphere_local    = coarse_sphere_local.typed_data();
        collision_ctx.coarse_sphere_radius   = coarse_sphere_radius.typed_data();
        collision_ctx.fine_link_offsets      = fine_link_offsets.typed_data();
        collision_ctx.coarse_self_pairs      = coarse_self_pairs.typed_data();
        collision_ctx.self_pair_ranges       = self_pair_ranges.typed_data();
        collision_ctx.n_coarse               = n_coarse;
        collision_ctx.n_coarse_self_pairs    = n_coarse_self_pairs;
        collision_ctx.coarse_enabled = (n_coarse > 0 && n_coarse <= PRRTC_MAX_LINKS) ? 1 : 0;
    }

    // Allocate radii arrays for all problems in one shot
    float* all_ta_radii = nullptr;
    float* all_tb_radii = nullptr;
    cudaError_t e = cudaMalloc(&all_ta_radii, sizeof(float) * max_nodes * batch);
    if (e != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    e = cudaMalloc(&all_tb_radii, sizeof(float) * max_nodes * batch);
    if (e != cudaSuccess) {
        cudaFree(all_ta_radii);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    // One child stream per problem — these run concurrently on the GPU
    cudaStream_t* child_streams = new cudaStream_t[batch];
    for (int b = 0; b < batch; ++b)
        cudaStreamCreateWithFlags(&child_streams[b], cudaStreamNonBlocking);
    cudaEvent_t* kernel_start_events = new cudaEvent_t[batch];
    cudaEvent_t* kernel_end_events = new cudaEvent_t[batch];
    for (int b = 0; b < batch; ++b) {
        cudaEventCreate(&kernel_start_events[b]);
        cudaEventCreate(&kernel_end_events[b]);
    }

    const int threads_fill = 16;
    const int blocks_fill  = (max_nodes + threads_fill - 1) / threads_fill;

    // Constants written once; cudaMemcpyAsync with pageable host memory is
    // synchronous from the host side, so local-variable addresses are safe.
    const int init_sizes[2]   = {1, num_goals};
    const int init_conn[3]    = {-1, -1, -1};
    const int zero            = 0;

    for (int b = 0; b < batch; ++b) {
        cudaStream_t s = child_streams[b];

        float* start_b   = start_configs.typed_data()        + b * dim;
        float* ta_cfg_b  = tree_a_configs->typed_data()      + (size_t)b * dim * max_nodes;
        float* tb_cfg_b  = tree_b_configs->typed_data()      + (size_t)b * dim * max_nodes;
        int*   ta_par_b  = tree_a_parents->typed_data()      + b * max_nodes;
        int*   tb_par_b  = tree_b_parents->typed_data()      + b * max_nodes;
        int*   tsizes_b  = tree_sizes->typed_data()          + b * 2;
        int*   comp_b    = completed->typed_data()           + b * 2;
        int*   iter_b    = iter_count->typed_data()          + b * 1;
        int*   conn_b    = connection_info->typed_data()     + b * 3;
        int*   solved_b  = solved_flag->typed_data()         + b * 1;
        float* ta_rad_b  = all_ta_radii                      + b * max_nodes;
        float* tb_rad_b  = all_tb_radii                      + b * max_nodes;

        cudaMemsetAsync(ta_cfg_b, 0,  sizeof(float) * dim * max_nodes, s);
        cudaMemsetAsync(tb_cfg_b, 0,  sizeof(float) * dim * max_nodes, s);
        cudaMemsetAsync(ta_par_b, -1, sizeof(int)   * max_nodes,       s);
        cudaMemsetAsync(tb_par_b, -1, sizeof(int)   * max_nodes,       s);

        cudaMemcpyAsync(tsizes_b, init_sizes, sizeof(int) * 2, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(comp_b,   init_sizes, sizeof(int) * 2, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(iter_b,   &zero,      sizeof(int),     cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(conn_b,   init_conn,  sizeof(int) * 3, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(solved_b, &zero,      sizeof(int),     cudaMemcpyHostToDevice, s);

        prrtc_fill_float_kernel<<<blocks_fill, threads_fill, 0, s>>>(ta_rad_b, FLT_MAX, max_nodes);
        prrtc_fill_float_kernel<<<blocks_fill, threads_fill, 0, s>>>(tb_rad_b, FLT_MAX, max_nodes);

        const float* goals_b = goal_configs.typed_data() + b * num_goals * dim;
        prrtc_init_kernel<<<1, 32, 0, s>>>(
            start_b, goals_b,
            ta_cfg_b, tb_cfg_b, ta_par_b, tb_par_b,
            num_goals, dim, max_nodes
        );

        cudaEventRecord(kernel_start_events[b], s);
        prrtc_planner_kernel<<<num_new_samples, granularity, 0, s>>>(
            ta_cfg_b, tb_cfg_b, ta_par_b, tb_par_b,
            ta_rad_b, tb_rad_b,
            min_vals.typed_data(), max_vals.typed_data(),
            tsizes_b, comp_b, iter_b, conn_b, solved_b,
            collision_ctx,
            max_iterations, step_size, num_new_samples,
            balance_mode, tree_ratio, dynamic_domain,
            dd_alpha, dd_radius, dd_min_radius,
            dim, max_nodes, granularity
        );
        cudaEventRecord(kernel_end_events[b], s);
    }

    float* host_kernel_times = new float[batch];
    for (int b = 0; b < batch; ++b) {
        cudaEventSynchronize(kernel_end_events[b]);
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, kernel_start_events[b], kernel_end_events[b]);
        host_kernel_times[b] = elapsed_ms;
    }
    cudaMemcpyAsync(
        kernel_time_ms->typed_data(),
        host_kernel_times,
        sizeof(float) * batch,
        cudaMemcpyHostToDevice,
        stream
    );
    delete[] host_kernel_times;

    // Record an event on each child stream, then make the XLA stream wait on
    // all of them before returning — this is the standard CUDA pattern for
    // fan-out/fan-in across non-blocking child streams.
    cudaEvent_t* events = new cudaEvent_t[batch];
    for (int b = 0; b < batch; ++b) {
        cudaEventCreateWithFlags(&events[b], cudaEventDisableTiming);
        cudaEventRecord(events[b], child_streams[b]);
        cudaStreamWaitEvent(stream, events[b], 0);
    }

    for (int b = 0; b < batch; ++b) {
        cudaEventDestroy(events[b]);
        cudaEventDestroy(kernel_start_events[b]);
        cudaEventDestroy(kernel_end_events[b]);
        cudaStreamDestroy(child_streams[b]);
    }
    delete[] events;
    delete[] kernel_start_events;
    delete[] kernel_end_events;
    delete[] child_streams;
    cudaFree(all_ta_radii);
    cudaFree(all_tb_radii);

    return ffi::Error::Success();
}

// XLA FFI handler for pRRTC batch planner with per-problem collision contexts.
// World geometry/self-pairs are provided as padded batched tensors with per-
// problem count vectors to avoid Python-side fanout by context.
static ffi::Error PrrtcPlannerBatchCtxImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> start_configs,   // [batch, dim]
    ffi::Buffer<ffi::DataType::F32> goal_configs,    // [batch, num_goals, dim]
    ffi::Buffer<ffi::DataType::F32> min_vals,        // [dim]
    ffi::Buffer<ffi::DataType::F32> max_vals,        // [dim]
    ffi::Buffer<ffi::DataType::F32> fk_twists,
    ffi::Buffer<ffi::DataType::F32> fk_parent_tf,
    ffi::Buffer<ffi::DataType::S32> fk_parent_idx,
    ffi::Buffer<ffi::DataType::S32> fk_act_idx,
    ffi::Buffer<ffi::DataType::F32> fk_mimic_mul,
    ffi::Buffer<ffi::DataType::F32> fk_mimic_off,
    ffi::Buffer<ffi::DataType::S32> fk_mimic_act_idx,
    ffi::Buffer<ffi::DataType::S32> fk_topo_inv,
    ffi::Buffer<ffi::DataType::S32> sphere_link_idx,
    ffi::Buffer<ffi::DataType::F32> sphere_local,
    ffi::Buffer<ffi::DataType::F32> sphere_radius,
    ffi::Buffer<ffi::DataType::F32> world_spheres_batched,    // [batch, max_ws, 4]
    ffi::Buffer<ffi::DataType::F32> world_capsules_batched,   // [batch, max_wc, 7]
    ffi::Buffer<ffi::DataType::F32> world_boxes_batched,      // [batch, max_wb, 15]
    ffi::Buffer<ffi::DataType::F32> world_halfspaces_batched, // [batch, max_wh, 6]
    ffi::Buffer<ffi::DataType::S32> self_pairs_batched,       // [batch, max_sp, 2]
    ffi::Buffer<ffi::DataType::S32> world_spheres_count,      // [batch]
    ffi::Buffer<ffi::DataType::S32> world_capsules_count,     // [batch]
    ffi::Buffer<ffi::DataType::S32> world_boxes_count,        // [batch]
    ffi::Buffer<ffi::DataType::S32> world_halfspaces_count,   // [batch]
    ffi::Buffer<ffi::DataType::S32> self_pairs_count,         // [batch]
    ffi::Buffer<ffi::DataType::S32> coarse_sphere_link_idx,   // [n_coarse] (shared)
    ffi::Buffer<ffi::DataType::F32> coarse_sphere_local,      // [n_coarse, 3] (shared)
    ffi::Buffer<ffi::DataType::F32> coarse_sphere_radius,     // [n_coarse] (shared)
    ffi::Buffer<ffi::DataType::S32> fine_link_offsets,        // [n_coarse+1] (shared)
    ffi::Buffer<ffi::DataType::S32> coarse_self_pairs,        // [n_coarse_self_pairs, 2] (shared)
    ffi::Buffer<ffi::DataType::S32> self_pair_ranges,         // [n_coarse_self_pairs, 2] (shared)
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
    int max_iterations,
    float step_size,
    int num_new_samples,
    int balance_mode,
    float tree_ratio,
    int dynamic_domain,
    float dd_alpha,
    float dd_radius,
    float dd_min_radius,
    int dim,
    int max_nodes,
    int granularity,
    int collision_mode,
    float collision_margin
) {
    const int batch = static_cast<int>(start_configs.dimensions()[0]);
    const int num_goals = static_cast<int>(goal_configs.dimensions()[1]);
    const int n_joints = static_cast<int>(fk_parent_idx.dimensions().size() == 0 ? 0 : fk_parent_idx.dimensions()[0]);
    const int n_robot_spheres = static_cast<int>(sphere_link_idx.dimensions().size() == 0 ? 0 : sphere_link_idx.dimensions()[0]);

    const int max_world_spheres = static_cast<int>(world_spheres_batched.dimensions().size() < 2 ? 0 : world_spheres_batched.dimensions()[1]);
    const int max_world_capsules = static_cast<int>(world_capsules_batched.dimensions().size() < 2 ? 0 : world_capsules_batched.dimensions()[1]);
    const int max_world_boxes = static_cast<int>(world_boxes_batched.dimensions().size() < 2 ? 0 : world_boxes_batched.dimensions()[1]);
    const int max_world_halfspaces = static_cast<int>(world_halfspaces_batched.dimensions().size() < 2 ? 0 : world_halfspaces_batched.dimensions()[1]);
    const int max_self_pairs = static_cast<int>(self_pairs_batched.dimensions().size() < 2 ? 0 : self_pairs_batched.dimensions()[1]);

    int* host_world_spheres_count = new int[batch];
    int* host_world_capsules_count = new int[batch];
    int* host_world_boxes_count = new int[batch];
    int* host_world_halfspaces_count = new int[batch];
    int* host_self_pairs_count = new int[batch];
    cudaError_t e = cudaMemcpy(host_world_spheres_count, world_spheres_count.typed_data(), sizeof(int) * batch, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        delete[] host_world_spheres_count;
        delete[] host_world_capsules_count;
        delete[] host_world_boxes_count;
        delete[] host_world_halfspaces_count;
        delete[] host_self_pairs_count;
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    e = cudaMemcpy(host_world_capsules_count, world_capsules_count.typed_data(), sizeof(int) * batch, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        delete[] host_world_spheres_count;
        delete[] host_world_capsules_count;
        delete[] host_world_boxes_count;
        delete[] host_world_halfspaces_count;
        delete[] host_self_pairs_count;
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    e = cudaMemcpy(host_world_boxes_count, world_boxes_count.typed_data(), sizeof(int) * batch, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        delete[] host_world_spheres_count;
        delete[] host_world_capsules_count;
        delete[] host_world_boxes_count;
        delete[] host_world_halfspaces_count;
        delete[] host_self_pairs_count;
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    e = cudaMemcpy(host_world_halfspaces_count, world_halfspaces_count.typed_data(), sizeof(int) * batch, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        delete[] host_world_spheres_count;
        delete[] host_world_capsules_count;
        delete[] host_world_boxes_count;
        delete[] host_world_halfspaces_count;
        delete[] host_self_pairs_count;
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    e = cudaMemcpy(host_self_pairs_count, self_pairs_count.typed_data(), sizeof(int) * batch, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        delete[] host_world_spheres_count;
        delete[] host_world_capsules_count;
        delete[] host_world_boxes_count;
        delete[] host_world_halfspaces_count;
        delete[] host_self_pairs_count;
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    float* all_ta_radii = nullptr;
    float* all_tb_radii = nullptr;
    e = cudaMalloc(&all_ta_radii, sizeof(float) * max_nodes * batch);
    if (e != cudaSuccess) {
        delete[] host_world_spheres_count;
        delete[] host_world_capsules_count;
        delete[] host_world_boxes_count;
        delete[] host_world_halfspaces_count;
        delete[] host_self_pairs_count;
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    e = cudaMalloc(&all_tb_radii, sizeof(float) * max_nodes * batch);
    if (e != cudaSuccess) {
        delete[] host_world_spheres_count;
        delete[] host_world_capsules_count;
        delete[] host_world_boxes_count;
        delete[] host_world_halfspaces_count;
        delete[] host_self_pairs_count;
        cudaFree(all_ta_radii);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    cudaStream_t* child_streams = new cudaStream_t[batch];
    for (int b = 0; b < batch; ++b)
        cudaStreamCreateWithFlags(&child_streams[b], cudaStreamNonBlocking);
    cudaEvent_t* kernel_start_events = new cudaEvent_t[batch];
    cudaEvent_t* kernel_end_events = new cudaEvent_t[batch];
    for (int b = 0; b < batch; ++b) {
        cudaEventCreate(&kernel_start_events[b]);
        cudaEventCreate(&kernel_end_events[b]);
    }

    const int threads_fill = 16;
    const int blocks_fill = (max_nodes + threads_fill - 1) / threads_fill;

    const int init_sizes[2] = {1, num_goals};
    const int init_conn[3] = {-1, -1, -1};
    const int zero = 0;

    for (int b = 0; b < batch; ++b) {
        cudaStream_t s = child_streams[b];

        float* start_b = start_configs.typed_data() + b * dim;
        float* ta_cfg_b = tree_a_configs->typed_data() + (size_t)b * dim * max_nodes;
        float* tb_cfg_b = tree_b_configs->typed_data() + (size_t)b * dim * max_nodes;
        int* ta_par_b = tree_a_parents->typed_data() + b * max_nodes;
        int* tb_par_b = tree_b_parents->typed_data() + b * max_nodes;
        int* tsizes_b = tree_sizes->typed_data() + b * 2;
        int* comp_b = completed->typed_data() + b * 2;
        int* iter_b = iter_count->typed_data() + b * 1;
        int* conn_b = connection_info->typed_data() + b * 3;
        int* solved_b = solved_flag->typed_data() + b * 1;
        float* ta_rad_b = all_ta_radii + b * max_nodes;
        float* tb_rad_b = all_tb_radii + b * max_nodes;

        cudaMemsetAsync(ta_cfg_b, 0, sizeof(float) * dim * max_nodes, s);
        cudaMemsetAsync(tb_cfg_b, 0, sizeof(float) * dim * max_nodes, s);
        cudaMemsetAsync(ta_par_b, -1, sizeof(int) * max_nodes, s);
        cudaMemsetAsync(tb_par_b, -1, sizeof(int) * max_nodes, s);

        cudaMemcpyAsync(tsizes_b, init_sizes, sizeof(int) * 2, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(comp_b, init_sizes, sizeof(int) * 2, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(iter_b, &zero, sizeof(int), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(conn_b, init_conn, sizeof(int) * 3, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(solved_b, &zero, sizeof(int), cudaMemcpyHostToDevice, s);

        prrtc_fill_float_kernel<<<blocks_fill, threads_fill, 0, s>>>(ta_rad_b, FLT_MAX, max_nodes);
        prrtc_fill_float_kernel<<<blocks_fill, threads_fill, 0, s>>>(tb_rad_b, FLT_MAX, max_nodes);

        const float* goals_b = goal_configs.typed_data() + b * num_goals * dim;
        prrtc_init_kernel<<<1, 32, 0, s>>>(
            start_b, goals_b,
            ta_cfg_b, tb_cfg_b, ta_par_b, tb_par_b,
            num_goals, dim, max_nodes
        );

        CollisionContext collision_ctx;
        collision_ctx.twists = fk_twists.typed_data();
        collision_ctx.parent_tf = fk_parent_tf.typed_data();
        collision_ctx.parent_idx = fk_parent_idx.typed_data();
        collision_ctx.act_idx = fk_act_idx.typed_data();
        collision_ctx.mimic_mul = fk_mimic_mul.typed_data();
        collision_ctx.mimic_off = fk_mimic_off.typed_data();
        collision_ctx.mimic_act_idx = fk_mimic_act_idx.typed_data();
        collision_ctx.topo_inv = fk_topo_inv.typed_data();
        collision_ctx.sphere_link_idx = sphere_link_idx.typed_data();
        collision_ctx.sphere_local = sphere_local.typed_data();
        collision_ctx.sphere_radius = sphere_radius.typed_data();
        collision_ctx.world_spheres = world_spheres_batched.typed_data() + (size_t)b * max_world_spheres * 4;
        collision_ctx.world_capsules = world_capsules_batched.typed_data() + (size_t)b * max_world_capsules * 7;
        collision_ctx.world_boxes = world_boxes_batched.typed_data() + (size_t)b * max_world_boxes * 15;
        collision_ctx.world_halfspaces = world_halfspaces_batched.typed_data() + (size_t)b * max_world_halfspaces * 6;
        collision_ctx.self_pairs = self_pairs_batched.typed_data() + (size_t)b * max_self_pairs * 2;

        collision_ctx.n_joints = n_joints;
        collision_ctx.n_act = dim;
        collision_ctx.n_robot_spheres = n_robot_spheres;
        collision_ctx.n_world_spheres = host_world_spheres_count[b];
        collision_ctx.n_world_capsules = host_world_capsules_count[b];
        collision_ctx.n_world_boxes = host_world_boxes_count[b];
        collision_ctx.n_world_halfspaces = host_world_halfspaces_count[b];
        collision_ctx.n_self_pairs = host_self_pairs_count[b];
        collision_ctx.enabled = (n_joints > 0 && n_robot_spheres > 0 &&
            (collision_ctx.n_world_spheres > 0 || collision_ctx.n_world_capsules > 0 ||
             collision_ctx.n_world_boxes > 0 || collision_ctx.n_world_halfspaces > 0 ||
             collision_ctx.n_self_pairs > 0)) ? 1 : 0;
        collision_ctx.collision_mode = collision_mode;
        collision_ctx.collision_margin = collision_margin;
        {
            const int n_coarse = (coarse_sphere_link_idx.dimensions().size() == 0)
                ? 0 : static_cast<int>(coarse_sphere_link_idx.dimensions()[0]);
            const int n_coarse_self_pairs = (coarse_self_pairs.dimensions().size() == 0)
                ? 0 : static_cast<int>(coarse_self_pairs.dimensions()[0]);
            collision_ctx.coarse_sphere_link_idx = coarse_sphere_link_idx.typed_data();
            collision_ctx.coarse_sphere_local    = coarse_sphere_local.typed_data();
            collision_ctx.coarse_sphere_radius   = coarse_sphere_radius.typed_data();
            collision_ctx.fine_link_offsets      = fine_link_offsets.typed_data();
            collision_ctx.coarse_self_pairs      = coarse_self_pairs.typed_data();
            collision_ctx.self_pair_ranges       = self_pair_ranges.typed_data();
            collision_ctx.n_coarse               = n_coarse;
            collision_ctx.n_coarse_self_pairs    = n_coarse_self_pairs;
            collision_ctx.coarse_enabled = (n_coarse > 0 && n_coarse <= PRRTC_MAX_LINKS) ? 1 : 0;
        }

        cudaEventRecord(kernel_start_events[b], s);
        prrtc_planner_kernel<<<num_new_samples, granularity, 0, s>>>(
            ta_cfg_b, tb_cfg_b, ta_par_b, tb_par_b,
            ta_rad_b, tb_rad_b,
            min_vals.typed_data(), max_vals.typed_data(),
            tsizes_b, comp_b, iter_b, conn_b, solved_b,
            collision_ctx,
            max_iterations, step_size, num_new_samples,
            balance_mode, tree_ratio, dynamic_domain,
            dd_alpha, dd_radius, dd_min_radius,
            dim, max_nodes, granularity
        );
        cudaEventRecord(kernel_end_events[b], s);
    }

    float* host_kernel_times = new float[batch];
    for (int b = 0; b < batch; ++b) {
        cudaEventSynchronize(kernel_end_events[b]);
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, kernel_start_events[b], kernel_end_events[b]);
        host_kernel_times[b] = elapsed_ms;
    }
    cudaMemcpyAsync(
        kernel_time_ms->typed_data(),
        host_kernel_times,
        sizeof(float) * batch,
        cudaMemcpyHostToDevice,
        stream
    );
    delete[] host_kernel_times;

    cudaEvent_t* events = new cudaEvent_t[batch];
    for (int b = 0; b < batch; ++b) {
        cudaEventCreateWithFlags(&events[b], cudaEventDisableTiming);
        cudaEventRecord(events[b], child_streams[b]);
        cudaStreamWaitEvent(stream, events[b], 0);
    }

    for (int b = 0; b < batch; ++b) {
        cudaEventDestroy(events[b]);
        cudaEventDestroy(kernel_start_events[b]);
        cudaEventDestroy(kernel_end_events[b]);
        cudaStreamDestroy(child_streams[b]);
    }
    delete[] events;
    delete[] kernel_start_events;
    delete[] kernel_end_events;
    delete[] child_streams;
    delete[] host_world_spheres_count;
    delete[] host_world_capsules_count;
    delete[] host_world_boxes_count;
    delete[] host_world_halfspaces_count;
    delete[] host_self_pairs_count;
    cudaFree(all_ta_radii);
    cudaFree(all_tb_radii);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PrrtcPlannerBatchCtxFfi, PrrtcPlannerBatchCtxImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // start_configs [batch, dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // goal_configs [batch, num_goals, dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // min_vals [dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // max_vals [dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // fk_twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // fk_parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fk_parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fk_act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // fk_mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // fk_mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fk_mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fk_topo_inv
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // sphere_link_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // sphere_local
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // sphere_radius
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres_batched [batch, max_ws, 4]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules_batched [batch, max_wc, 7]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes_batched [batch, max_wb, 15]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces_batched [batch, max_wh, 6]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // self_pairs_batched [batch, max_sp, 2]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // world_spheres_count [batch]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // world_capsules_count [batch]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // world_boxes_count [batch]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // world_halfspaces_count [batch]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // self_pairs_count [batch]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // coarse_sphere_link_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // coarse_sphere_local
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // coarse_sphere_radius
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fine_link_offsets
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // coarse_self_pairs
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // self_pair_ranges
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // tree_a_configs [batch, dim, max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // tree_b_configs [batch, dim, max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_a_parents [batch, max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_b_parents [batch, max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_sizes [batch, 2]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // completed [batch, 2]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // iter_count [batch, 1]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // connection_info [batch, 3]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // solved_flag [batch, 1]
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // kernel_time_ms [batch]
        .Attr<int>("max_iterations")
        .Attr<float>("step_size")
        .Attr<int>("num_new_samples")
        .Attr<int>("balance_mode")
        .Attr<float>("tree_ratio")
        .Attr<int>("dynamic_domain")
        .Attr<float>("dd_alpha")
        .Attr<float>("dd_radius")
        .Attr<float>("dd_min_radius")
        .Attr<int>("dim")
        .Attr<int>("max_nodes")
        .Attr<int>("granularity")
        .Attr<int>("collision_mode")
        .Attr<float>("collision_margin")
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PrrtcPlannerBatchFfi, PrrtcPlannerBatchImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // start_configs [batch, dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // goal_configs [batch, num_goals, dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // min_vals [dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // max_vals [dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // fk_twists
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // fk_parent_tf
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fk_parent_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fk_act_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // fk_mimic_mul
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // fk_mimic_off
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fk_mimic_act_idx
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fk_topo_inv
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // sphere_link_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // sphere_local
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // sphere_radius
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // self_pairs
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // coarse_sphere_link_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // coarse_sphere_local
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // coarse_sphere_radius
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fine_link_offsets
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // coarse_self_pairs
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // self_pair_ranges
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // tree_a_configs [batch, dim, max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // tree_b_configs [batch, dim, max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_a_parents [batch, max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_b_parents [batch, max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_sizes [batch, 2]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // completed [batch, 2]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // iter_count [batch, 1]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // connection_info [batch, 3]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // solved_flag [batch, 1]
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // kernel_time_ms [batch]
        .Attr<int>("max_iterations")
        .Attr<float>("step_size")
        .Attr<int>("num_new_samples")
        .Attr<int>("balance_mode")
        .Attr<float>("tree_ratio")
        .Attr<int>("dynamic_domain")
        .Attr<float>("dd_alpha")
        .Attr<float>("dd_radius")
        .Attr<float>("dd_min_radius")
        .Attr<int>("dim")
        .Attr<int>("max_nodes")
        .Attr<int>("granularity")
        .Attr<int>("collision_mode")
        .Attr<float>("collision_margin")
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PrrtcPlannerFfi, PrrtcPlannerImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // start_config [dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // goal_configs [num_goals, dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // min_vals [dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // max_vals [dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // fk_twists [n_joints, 6]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // fk_parent_tf [n_joints, 7]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fk_parent_idx [n_joints]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fk_act_idx [n_joints]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // fk_mimic_mul [n_joints]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // fk_mimic_off [n_joints]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fk_mimic_act_idx [n_joints]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fk_topo_inv [n_joints]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // sphere_link_idx [n_robot_spheres]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // sphere_local [n_robot_spheres, 3]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // sphere_radius [n_robot_spheres]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_spheres [n_world_spheres, 4]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_capsules [n_world_capsules, 7]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_boxes [n_world_boxes, 15]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // world_halfspaces [n_world_halfspaces, 6]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // self_pairs [n_self_pairs, 2]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // coarse_sphere_link_idx
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // coarse_sphere_local
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // coarse_sphere_radius
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // fine_link_offsets
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // coarse_self_pairs
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // self_pair_ranges
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // tree_a_configs [dim, max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // tree_b_configs [dim, max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_a_parents [max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_b_parents [max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_sizes [2]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // completed [2]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // iter_count [1]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // connection_info [3]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // solved_flag [1]
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // kernel_time_ms [1]
        .Attr<int>("max_iterations")
        .Attr<float>("step_size")
        .Attr<int>("num_new_samples")
        .Attr<int>("balance_mode")
        .Attr<float>("tree_ratio")
        .Attr<int>("dynamic_domain")
        .Attr<float>("dd_alpha")
        .Attr<float>("dd_radius")
        .Attr<float>("dd_min_radius")
        .Attr<int>("dim")
        .Attr<int>("max_nodes")
        .Attr<int>("granularity")
        .Attr<int>("collision_mode")
        .Attr<float>("collision_margin")
);
