/**
 * Helper header for pRRTC CUDA kernels.
 * Defines constants and utilities used across multiple kernel files.
 */

#pragma once

// Maximum configuration space dimension supported
#ifndef CONFIG_DIM_MAX
#define CONFIG_DIM_MAX 16
#endif

// Maximum tree size
#ifndef TREE_MAX_NODES
#define TREE_MAX_NODES 1000000
#endif

// Warp size constant
#define WARP_SIZE 32

// Thread block size for parallel operations
#define BLOCK_SIZE 256

// Collision check result constants
#define COLLISION_FREE 0
#define COLLISION_OCCURS 1

/**
 * Compute squared L2 distance between two configurations.
 */
__device__ __forceinline__ float dist_sq_to_config(
    const float* __restrict__ tree_configs,
    int node_idx,
    int dim,
    int max_nodes,
    const float* __restrict__ query_ptr
) {
    float dist = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = tree_configs[d * max_nodes + node_idx] - query_ptr[d];
        dist += diff * diff;
    }
    return dist;
}

/**
 * Warp-level reduction for finding minimum.
 */
__device__ __forceinline__ float warp_reduce_min_float(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fminf(val, other);
    }
    return val;
}

/**
 * Warp-level reduction for finding minimum with index.
 */
__device__ __forceinline__ void warp_reduce_min_with_index(float& val, int& index) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, index, offset);
        if (other_val < val) {
            val = other_val;
            index = other_idx;
        }
    }
}

/**
 * Block-level reduction for minimum.
 */
__device__ __forceinline__ void block_reduce_min_float(float& val) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    val = sdata[0];
}

/**
 * Clamp value to [min, max] range.
 */
__device__ __forceinline__ float clamp_val(float x, float min_val, float max_val) {
    return fmaxf(min_val, fminf(x, max_val));
}

/**
 * Warp-level voting for boolean predicates.
 */
__device__ __forceinline__ bool warp_any(bool pred) {
    return __any_sync(0xffffffff, pred);
}

/**
 * Thread-safe atomic add for float (via integer casting).
 */
__device__ __forceinline__ float atomicAddFloat(float* addr, float val) {
    return atomicAdd((float*)addr, val);
}
