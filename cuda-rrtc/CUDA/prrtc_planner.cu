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

// Threads per block: each block = one parallel explorer (pRRTC uses 4*granularity).
// With source defaults (granularity=64), this is 256 threads per block.
#define PRRTC_BLOCK_THREADS 256

// Global state
__device__ int d_prrtc_solved = 0;
__device__ int d_prrtc_iterations = 0;

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
        d_prrtc_solved = 0;
        d_prrtc_iterations = 0;

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
    int n_joints;
    int n_act;
    int n_robot_spheres;
    int n_world_spheres;
    int n_world_capsules;
    int n_world_boxes;
    int n_world_halfspaces;
    int n_self_pairs;
    int enabled;
};

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
    int max_nodes
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // --- Shared memory ---
    // NN reduction arrays (one entry per thread in block)
    __shared__ float sdata[PRRTC_BLOCK_THREADS];
    __shared__ int   sindex_sh[PRRTC_BLOCK_THREADS];

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
        if (atomicAdd(&d_prrtc_solved, 0) != 0) return;

        // --- Iteration accounting (per block, like pRRTC) ---
        ++local_iter;
        if (tid == 0 && local_iter > max_iterations) {
            atomicCAS(&d_prrtc_solved, 0, -1);
            atomicMax(iter_count, local_iter);
        }
        __syncthreads();
        if (atomicAdd(&d_prrtc_solved, 0) != 0) return;

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
        {
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
                atomicCAS(&d_prrtc_solved, 0, -1);
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

        if (new_idx_sh < 0) { if (atomicAdd(&d_prrtc_solved, 0) != 0) return; continue; }

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

            {
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
                    atomicCAS(&d_prrtc_solved, 0, -1);
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
            if (atomicCAS(&d_prrtc_solved, 0, 1) == 0) {
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

        if (atomicAdd(&d_prrtc_solved, 0) != 0) return;
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
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> tree_a_configs,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> tree_b_configs,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> tree_a_parents,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> tree_b_parents,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> tree_sizes,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> completed,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> iter_count,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> connection_info,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> solved_flag,
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
    int max_nodes
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

    int threads_fill = 256;
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
    // All shared memory is statically declared; no dynamic shared memory needed.
    prrtc_planner_kernel<<<num_new_samples, PRRTC_BLOCK_THREADS, 0, stream>>>(
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
        max_nodes
    );

    e = cudaGetLastError();
    if (e != cudaSuccess) {
        cudaFree(tree_a_radii);
        cudaFree(tree_b_radii);
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }

    cudaFree(tree_a_radii);
    cudaFree(tree_b_radii);

    return ffi::Error::Success();
}

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
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // tree_a_configs [dim, max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // tree_b_configs [dim, max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_a_parents [max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_b_parents [max_nodes]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // tree_sizes [2]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // completed [2]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // iter_count [1]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // connection_info [3]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // solved_flag [1]
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
);
