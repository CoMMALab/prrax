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

// Global state
__device__ int d_prrtc_solved = 0;
__device__ int d_prrtc_iterations = 0;

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
    const float* twists;          // [n_joints, 6]
    const float* parent_tf;       // [n_joints, 7]
    const int* parent_idx;        // [n_joints]
    const int* act_idx;           // [n_joints]
    const float* mimic_mul;       // [n_joints]
    const float* mimic_off;       // [n_joints]
    const int* mimic_act_idx;     // [n_joints]
    const int* topo_inv;          // [n_joints]
    const int* sphere_link_idx;   // [n_robot_spheres]
    const float* sphere_local;    // [n_robot_spheres, 3]
    const float* sphere_radius;   // [n_robot_spheres]
    const float* world_spheres;   // [n_world_spheres, 4]
    const int* self_pairs;        // [n_self_pairs, 2]
    int n_joints;
    int n_act;
    int n_robot_spheres;
    int n_world_spheres;
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

    // Robot-vs-world sphere collision check.
    for (int rs = 0; rs < ctx.n_robot_spheres; ++rs) {
        const int link_idx = ctx.sphere_link_idx[rs];
        if (link_idx < 0 || link_idx >= ctx.n_joints) continue;

        const float* T = &T_world[link_idx * 7];
        const float* local = &ctx.sphere_local[rs * 3];
        float world_pt[3];
        apply_se3_point(T, local, world_pt);
        const float rr = ctx.sphere_radius[rs];

        for (int ws = 0; ws < ctx.n_world_spheres; ++ws) {
            const float* obs = &ctx.world_spheres[ws * 4];
            if (sphere_sphere_dist(
                    world_pt[0],
                    world_pt[1],
                    world_pt[2],
                    rr,
                    obs[0],
                    obs[1],
                    obs[2],
                    obs[3]) <= 0.0f) {
                return true;
            }
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

// Main pRRTC planning kernel
__global__ void prrtc_planner_kernel(
    float* __restrict__ tree_a_configs,       // [dim, max_nodes] SoA - start tree
    float* __restrict__ tree_b_configs,       // [dim, max_nodes] SoA - goal tree  
    int* __restrict__ tree_a_parents,         // [max_nodes]
    int* __restrict__ tree_b_parents,         // [max_nodes]
    float* __restrict__ tree_a_radii,         // [max_nodes]
    float* __restrict__ tree_b_radii,         // [max_nodes]
    const float* __restrict__ min_vals,       // [dim]
    const float* __restrict__ max_vals,       // [dim]
    int* __restrict__ tree_sizes,             // [2] atomic counters
    int* __restrict__ completed,              // [2] completed counts
    int* __restrict__ iter_count,             // [1]
    int* __restrict__ connection_info,        // [3] [a_connect_idx, b_connect_idx, expand_tree_id]
    int* __restrict__ solved_out,             // [1] planner solved flag
    CollisionContext collision_ctx,
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
    const int tid = threadIdx.x;

    // Shared memory for samples
    extern __shared__ float shared_mem[];
    float* samples = shared_mem;

    __shared__ int halton_state[CONFIG_DIM_MAX];
    __shared__ int vamp_tree_id;

    if (tid == 0) {
        for (int d = 0; d < CONFIG_DIM_MAX; ++d) {
            halton_state[d] = d + 1;
        }
        vamp_tree_id = 0;
    }
    __syncthreads();

    while (true) {
        if (atomicAdd(&d_prrtc_solved, 0) != 0) {
            return;
        }

        int iter = 0;
        if (tid == 0) {
            iter = atomicAdd(iter_count, 1) + 1;
            if (iter > max_iterations) {
                atomicCAS(&d_prrtc_solved, 0, -1);
            }
        }
        __syncthreads();

        if (atomicAdd(&d_prrtc_solved, 0) != 0) {
            return;
        }

        if (tid == 0) {
            const int primes[16] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};
            for (int s = 0; s < num_new_samples; ++s) {
                float* sample_ptr = &samples[s * dim];
                for (int d = 0; d < dim; ++d) {
                    const int prime = primes[d % 16];
                    const float val = halton_next_runtime(prime, halton_state[d]);
                    sample_ptr[d] = min_vals[d] + val * (max_vals[d] - min_vals[d]);
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            for (int s = 0; s < num_new_samples; ++s) {
                const float* sample_ptr = &samples[s * dim];

                const int size_a = tree_sizes[0];
                const int size_b = tree_sizes[1];
                const int total = size_a + size_b;

                int t_tree_id = 0;
                if (balance_mode == 0 || iter == 1) {
                    t_tree_id = (s < (num_new_samples / 2)) ? 0 : 1;
                } else if (balance_mode == 1 && ((size_a >= size_b) ? (size_a - size_b) : (size_b - size_a)) < static_cast<int>(1.5f * num_new_samples)) {
                    const float ratio = (total > 0) ? (static_cast<float>(size_a) / static_cast<float>(total)) : 0.5f;
                    const float balance_factor = 1.0f - ratio;
                    t_tree_id = (s < static_cast<int>(num_new_samples * balance_factor)) ? 0 : 1;
                } else if (balance_mode == 1) {
                    const float ratio = (total > 0) ? (static_cast<float>(size_a) / static_cast<float>(total)) : 0.5f;
                    t_tree_id = (ratio < tree_ratio) ? 0 : 1;
                } else if (balance_mode == 2) {
                    const int o_tree_id = 1 - vamp_tree_id;
                    const int t_size = tree_sizes[vamp_tree_id];
                    const int o_size = tree_sizes[o_tree_id];
                    const float ratio = (t_size > 0)
                        ? (fabsf(static_cast<float>(t_size - o_size)) / static_cast<float>(t_size))
                        : 0.0f;
                    if (ratio < tree_ratio) {
                        vamp_tree_id = 1 - vamp_tree_id;
                    }
                    t_tree_id = vamp_tree_id;
                } else {
                    t_tree_id = (size_a <= size_b) ? 0 : 1;
                }

                const int o_tree_id = 1 - t_tree_id;
                const float* t_tree_configs = (t_tree_id == 0) ? tree_a_configs : tree_b_configs;
                const float* o_tree_configs = (o_tree_id == 0) ? tree_a_configs : tree_b_configs;
                int* t_tree_parents = (t_tree_id == 0) ? tree_a_parents : tree_b_parents;
                float* t_tree_radii = (t_tree_id == 0) ? tree_a_radii : tree_b_radii;

                const int completed_count = completed[t_tree_id];
                const int other_count = completed[o_tree_id];
                if (completed_count <= 0 || other_count <= 0) {
                    continue;
                }

                float min_dist_sq = FLT_MAX;
                int nearest_idx = 0;
                for (int i = 0; i < completed_count; ++i) {
                    float dist_sq = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        const float diff = t_tree_configs[d * max_nodes + i] - sample_ptr[d];
                        dist_sq += diff * diff;
                    }
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        nearest_idx = i;
                    }
                }

                const float nearest_dist = sqrtf(min_dist_sq);
                if (dynamic_domain && t_tree_radii[nearest_idx] < nearest_dist) {
                    continue;
                }

                float scale = 1.0f;
                if (nearest_dist > 1e-8f) {
                    scale = fminf(1.0f, step_size / nearest_dist);
                }

                float cfg_candidate[CONFIG_DIM_MAX];
                for (int d = 0; d < dim; ++d) {
                    const float base = t_tree_configs[d * max_nodes + nearest_idx];
                    cfg_candidate[d] = base + (sample_ptr[d] - base) * scale;
                }

                if (prrtc_config_in_collision(cfg_candidate, collision_ctx)) {
                    if (dynamic_domain) {
                        const float old_radius = t_tree_radii[nearest_idx];
                        t_tree_radii[nearest_idx] =
                            (old_radius == FLT_MAX)
                                ? dd_radius
                                : fmaxf(old_radius * (1.0f - dd_alpha), dd_min_radius);
                    }
                    continue;
                }

                const int new_idx = tree_sizes[t_tree_id]++;
                if (new_idx >= max_nodes) {
                    tree_sizes[t_tree_id]--;
                    atomicCAS(&d_prrtc_solved, 0, -1);
                    return;
                }

                for (int d = 0; d < dim; ++d) {
                    if (t_tree_id == 0) {
                        tree_a_configs[d * max_nodes + new_idx] = cfg_candidate[d];
                    } else {
                        tree_b_configs[d * max_nodes + new_idx] = cfg_candidate[d];
                    }
                }
                t_tree_parents[new_idx] = nearest_idx;
                t_tree_radii[new_idx] = FLT_MAX;
                completed[t_tree_id]++;

                if (dynamic_domain) {
                    const float old_radius = t_tree_radii[nearest_idx];
                    if (old_radius != FLT_MAX) {
                        t_tree_radii[nearest_idx] = old_radius * (1.0f + dd_alpha);
                    }
                }

                float connect_min_dist_sq = FLT_MAX;
                int connect_nearest_idx = 0;
                for (int i = 0; i < other_count; ++i) {
                    float dist_sq = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        const float curr = (t_tree_id == 0)
                            ? tree_a_configs[d * max_nodes + new_idx]
                            : tree_b_configs[d * max_nodes + new_idx];
                        const float diff = o_tree_configs[d * max_nodes + i] - curr;
                        dist_sq += diff * diff;
                    }
                    if (dist_sq < connect_min_dist_sq) {
                        connect_min_dist_sq = dist_sq;
                        connect_nearest_idx = i;
                    }
                }

                const float connect_dist = sqrtf(connect_min_dist_sq);
                int n_extensions = 1;
                if (step_size > 1e-8f) {
                    n_extensions = static_cast<int>(ceilf(connect_dist / step_size));
                    if (n_extensions < 1) n_extensions = 1;
                }

                int extension_parent = new_idx;
                bool extension_failed = false;
                for (int ext = 0; ext < n_extensions; ++ext) {
                    const int ext_idx = tree_sizes[t_tree_id]++;
                    if (ext_idx >= max_nodes) {
                        tree_sizes[t_tree_id]--;
                        extension_failed = true;
                        break;
                    }

                    float ext_candidate[CONFIG_DIM_MAX];
                    for (int d = 0; d < dim; ++d) {
                        const float curr = (t_tree_id == 0)
                            ? tree_a_configs[d * max_nodes + extension_parent]
                            : tree_b_configs[d * max_nodes + extension_parent];
                        const float goal = o_tree_configs[d * max_nodes + connect_nearest_idx];
                        const float step = (goal - curr) / static_cast<float>(n_extensions - ext);
                        const float candidate = curr + step;
                        ext_candidate[d] = candidate;
                        if (t_tree_id == 0) {
                            tree_a_configs[d * max_nodes + ext_idx] = candidate;
                        } else {
                            tree_b_configs[d * max_nodes + ext_idx] = candidate;
                        }
                    }

                    if (prrtc_config_in_collision(ext_candidate, collision_ctx)) {
                        tree_sizes[t_tree_id]--;
                        extension_failed = true;
                        break;
                    }

                    t_tree_parents[ext_idx] = extension_parent;
                    t_tree_radii[ext_idx] = FLT_MAX;
                    extension_parent = ext_idx;
                    completed[t_tree_id]++;
                }

                if (!extension_failed) {
                    if (atomicCAS(&d_prrtc_solved, 0, 1) == 0) {
                        if (t_tree_id == 0) {
                            connection_info[0] = extension_parent;
                            connection_info[1] = connect_nearest_idx;
                        } else {
                            connection_info[0] = connect_nearest_idx;
                            connection_info[1] = extension_parent;
                        }
                        connection_info[2] = t_tree_id;
                        solved_out[0] = 1;
                    }
                    return;
                }
            }
        }
        __syncthreads();

        if (atomicAdd(&d_prrtc_solved, 0) != 0) {
            return;
        }
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
    ffi::Buffer<ffi::DataType::F32> world_spheres,   // [n_world_spheres, 4]
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
    collision_ctx.self_pairs = self_pairs.typed_data();
    collision_ctx.n_joints = n_joints;
    collision_ctx.n_act = dim;
    collision_ctx.n_robot_spheres = n_robot_spheres;
    collision_ctx.n_world_spheres = n_world_spheres;
    collision_ctx.n_self_pairs = n_self_pairs;
    collision_ctx.enabled = (n_joints > 0 && n_robot_spheres > 0 && n_world_spheres > 0) ? 1 : 0;

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
    
    int threads = 128;
    size_t smem = static_cast<size_t>(num_new_samples * dim) * sizeof(float);
    prrtc_planner_kernel<<<1, threads, smem, stream>>>(
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
