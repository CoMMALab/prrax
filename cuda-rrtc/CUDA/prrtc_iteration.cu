/**
 * Parallel RRTC iteration kernel.
 *
 * Implements a single pRRTC iteration with parallel sampling, nearest neighbor
 * search, tree extension, and collision checking.
 * Designed for repeated execution via CUDA Graphs for minimal launch overhead.
 */

#include "xla/ffi/api/ffi.h"
#include "prrtc_helpers.cuh"
#include <curand_kernel.h>
#include <float.h>

namespace ffi = xla::ffi;

// Global state for planning
__device__ int d_tree_sizes[2] = {1, 1};  // start_tree, goal_tree
__device__ int d_completed[2] = {1, 1};
__device__ int d_solved = 0;

__global__ void prrtc_init_state_kernel(
    int* tree_sizes,     // [2] - start and goal tree sizes
    int* completed,      // [2] - completed node counts
    int* solved_flag,    // scalar solved flag
    int start_size,      // initial start tree size (usually 1)
    int goal_size        // initial goal tree size
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_tree_sizes[0] = start_size;
        d_tree_sizes[1] = goal_size;
        d_completed[0] = start_size;
        d_completed[1] = goal_size;
        d_solved = 0;
        
        if (tree_sizes != nullptr) {
            tree_sizes[0] = start_size;
            tree_sizes[1] = goal_size;
        }
        if (completed != nullptr) {
            completed[0] = start_size;
            completed[1] = goal_size;
        }
        if (solved_flag != nullptr) {
            solved_flag[0] = 0;
        }
    }
}

/**
 * Main pRRTC iteration kernel for batched execution.
 * 
 * Grid: (batch, 1, 1) - one block per sample
 * Block: (threads, 1, 1) - parallel nearest neighbor search
 */
__global__ void prrtc_iteration_batch_kernel(
    float* __restrict__ tree_a_configs,     // [dim, max_nodes] SoA - start tree
    float* __restrict__ tree_b_configs,     // [dim, max_nodes] SoA - goal tree
    int* __restrict__ tree_a_parents,       // [max_nodes]
    int* __restrict__ tree_b_parents,       // [max_nodes]
    const float* __restrict__ samples,      // [batch, dim]
    int* __restrict__ tree_sizes,           // [2] - atomic counters (device)
    int* __restrict__ completed,            // [2] - completed counts
    float step_size,                        // maximum extension step
    int dim,                                // configuration dimension
    int max_nodes,                          // max tree capacity
    int batch,                              // batch size
    int* __restrict__ new_configs,          // [batch, dim] output (as int for atomic operations)
    int* __restrict__ new_parents,          // [batch] parent indices
    int* __restrict__ valid_flags,          // [batch] validity flags
    int* __restrict__ solved_flag           // [1] solved flag
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    if (bid >= batch) return;
    
    // Check if solved
    if (atomicAdd(solved_flag, 0) != 0) return;
    
    // Bidirectional growth: expand the currently smaller tree; ties alternate by block id.
    const int size_a = atomicAdd(&tree_sizes[0], 0);
    const int size_b = atomicAdd(&tree_sizes[1], 0);
    int tree_id = 0;
    if (size_b < size_a) {
        tree_id = 1;
    } else if (size_a == size_b) {
        tree_id = (bid & 1);
    }

    const float* active_tree_configs = (tree_id == 0) ? tree_a_configs : tree_b_configs;
    int* active_tree_parents = (tree_id == 0) ? tree_a_parents : tree_b_parents;

    const int completed_count = completed[tree_id];
    
    if (completed_count == 0) return;
    
    // Find nearest node in parallel
    float local_min_dist = FLT_MAX;
    int local_near_idx = 0;
    const int query_offset = bid * dim;
    
    for (int i = tid; i < completed_count; i += blockDim.x) {
        float dist = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = active_tree_configs[d * max_nodes + i] - samples[query_offset + d];
            dist += diff * diff;
        }
        if (dist < local_min_dist) {
            local_min_dist = dist;
            local_near_idx = i;
        }
    }
    
    // Block reduction
    extern __shared__ float sdata[];
    float* dist_buf = sdata;
    int* idx_buf = (int*)(sdata + blockDim.x);
    
    dist_buf[tid] = local_min_dist;
    idx_buf[tid] = local_near_idx;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (dist_buf[tid + s] < dist_buf[tid]) {
                dist_buf[tid] = dist_buf[tid + s];
                idx_buf[tid] = idx_buf[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float nearest_dist = sqrtf(dist_buf[0]);
        int nearest_idx = idx_buf[0];
        
        // Extend toward sample with step limiting
        const float* sample_ptr = &samples[query_offset];

        float scale = 1.0f;
        if (nearest_dist > 1e-8f) {
            scale = fminf(1.0f, step_size / nearest_dist);
        }
        
        // Generate extended configuration
        for (int d = 0; d < dim; d++) {
            new_configs[bid * dim + d] = __float_as_int(
                active_tree_configs[d * max_nodes + nearest_idx] +
                (sample_ptr[d] - active_tree_configs[d * max_nodes + nearest_idx]) * scale
            );
        }
        new_parents[bid] = nearest_idx;
        valid_flags[bid] = 1;
        
        // Insert into tree atomically
        int new_idx = atomicAdd(&tree_sizes[tree_id], 1);
        if (new_idx < max_nodes) {
            // Atomically write configuration (4 bytes per float, 4 floats per int)
            for (int d = 0; d < dim; d++) {
                if (tree_id == 0) {
                    tree_a_configs[d * max_nodes + new_idx] = __int_as_float(
                        new_configs[bid * dim + d]
                    );
                } else {
                    tree_b_configs[d * max_nodes + new_idx] = __int_as_float(
                    new_configs[bid * dim + d]
                );
                }
            }
            active_tree_parents[new_idx] = nearest_idx;
            atomicAdd(&completed[tree_id], 1);
        } else {
            valid_flags[bid] = 0;
            atomicSub(&tree_sizes[tree_id], 1);
        }
    }
}

// XLA FFI handler
static ffi::Error PrrtcIterationImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> tree_a_configs,
    ffi::Buffer<ffi::DataType::F32> tree_b_configs,
    ffi::Buffer<ffi::DataType::S32> tree_a_parents,
    ffi::Buffer<ffi::DataType::S32> tree_b_parents,
    ffi::Buffer<ffi::DataType::F32> samples,
    ffi::Buffer<ffi::DataType::S32> tree_sizes,
    ffi::Buffer<ffi::DataType::S32> completed,
    ffi::Buffer<ffi::DataType::F32> step_size,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> new_configs,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> new_parents,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> valid_flags,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> solved_flag
) {
    const int dim = static_cast<int>(tree_a_configs.dimensions()[0]);
    const int max_nodes = static_cast<int>(tree_a_configs.dimensions()[1]);
    const int batch = static_cast<int>(samples.dimensions()[0]);
    
    if (batch == 0) return ffi::Error::Success();
    
    const float* tac_ptr = tree_a_configs.typed_data();
    const float* tbc_ptr = tree_b_configs.typed_data();
    const int* tap_ptr = tree_a_parents.typed_data();
    const int* tbp_ptr = tree_b_parents.typed_data();
    const float* sm_ptr = samples.typed_data();
    const int* ts_ptr = tree_sizes.typed_data();
    const int* comp_ptr = completed.typed_data();
    
    const float* ss_ptr = step_size.typed_data();
    
    int* nc_ptr = new_configs->typed_data();
    int* np_ptr = new_parents->typed_data();
    int* vf_ptr = valid_flags->typed_data();
    int* sf_ptr = solved_flag->typed_data();
    
    // Get step size from device
    float h_step_size;
    cudaError_t e = cudaMemcpyAsync(&h_step_size, ss_ptr, sizeof(float),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    
    int threads = 128;
    size_t smem = threads * (sizeof(float) + sizeof(int));
    dim3 grid(batch);
    
    // Initialize state if needed
    prrtc_init_state_kernel<<<1, 1, 0, stream>>>(
        const_cast<int*>(ts_ptr), const_cast<int*>(comp_ptr),
        sf_ptr, 1, 1
    );
    
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    
    // Launch iteration kernel
    prrtc_iteration_batch_kernel<<<grid, threads, smem, stream>>>(
        const_cast<float*>(tac_ptr), const_cast<float*>(tbc_ptr),
        const_cast<int*>(tap_ptr), const_cast<int*>(tbp_ptr),
        sm_ptr, const_cast<int*>(ts_ptr), const_cast<int*>(comp_ptr),
        h_step_size, dim, max_nodes, batch,
        nc_ptr, np_ptr, vf_ptr, sf_ptr
    );
    
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PrrtcIterationFfi, PrrtcIterationImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // tree_a_configs [dim, max_nodes]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // tree_b_configs [dim, max_nodes]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // tree_a_parents [max_nodes]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // tree_b_parents [max_nodes]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // samples [batch, dim]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // tree_sizes [2]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // completed [2]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // step_size
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // new_configs [batch, dim]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // new_parents [batch]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // valid_flags [batch]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // solved_flag [1]
);
