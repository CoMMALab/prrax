/**
 * CUDA kernel for parallel nearest neighbor search in pRRTC.
 *
 * Implements warp-efficient reduction for finding the nearest node in the 
 * configuration space tree to a given candidate configuration.
 *
 * Memory layout: Structure-of-Arrays (SoA)
 */

#include "xla/ffi/api/ffi.h"
#include "prrtc_helpers.cuh"
#include <float.h>

namespace ffi = xla::ffi;

/**
 * Batched nearest neighbor kernel with parallel reduction.
 * Grid: (batch, 1, 1) - one block per query
 * Block: (threads, 1, 1) - parallel search within tree
 */
__global__ void prrtc_nearest_neighbor_kernel(
    const float* __restrict__ tree_configs,  // [dim, max_nodes] SoA
    const float* __restrict__ query_configs, // [batch, dim]
    int tree_size,                            // current number of nodes
    int dim,                                  // configuration dimension
    int max_nodes,                            // max tree capacity
    int* __restrict__ nearest_indices,      // [batch] output indices
    float* __restrict__ nearest_dists        // [batch] output distances
) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;  // batch index
    
    if (bid >= gridDim.x) return;
    
    const int query_offset = bid * dim;
    const float* query = query_configs + query_offset;
    
    float local_min_dist = FLT_MAX;
    int local_near_idx = 0;
    
    // Strided parallel search across tree nodes
    for (int i = tid; i < tree_size; i += blockDim.x) {
        float dist = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = tree_configs[d * max_nodes + i] - query[d];
            dist += diff * diff;
        }
        if (dist < local_min_dist) {
            local_min_dist = dist;
            local_near_idx = i;
        }
    }
    
    // Block-level reduction
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
        nearest_dists[bid] = sqrtf(dist_buf[0]);
        nearest_indices[bid] = idx_buf[0];
    }
}

// XLA FFI handler
static ffi::Error PrrtcNearestNeighborImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> tree_configs,
    ffi::Buffer<ffi::DataType::F32> query_configs,
    ffi::Buffer<ffi::DataType::S32> tree_size_buf,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> indices,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> dists
) {
    const int dim = static_cast<int>(tree_configs.dimensions()[0]);
    const int max_nodes = static_cast<int>(tree_configs.dimensions()[1]);
    const int batch = static_cast<int>(query_configs.dimensions()[0]);
    
    if (batch == 0) return ffi::Error::Success();
    
    const float* tc_ptr = tree_configs.typed_data();
    const float* qc_ptr = query_configs.typed_data();
    int* idx_ptr = indices->typed_data();
    float* dist_ptr = dists->typed_data();
    
    int h_tree_size;
    cudaError_t e = cudaMemcpyAsync(
        &h_tree_size, tree_size_buf.typed_data(), sizeof(int),
        cudaMemcpyDeviceToHost, stream
    );
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    
    if (h_tree_size > max_nodes) h_tree_size = max_nodes;
    if (h_tree_size < 1) h_tree_size = 1;
    
    int threads = 256;
    size_t smem = threads * (sizeof(float) + sizeof(int));
    dim3 grid(batch);
    
    prrtc_nearest_neighbor_kernel<<<grid, threads, smem, stream>>>(
        tc_ptr, qc_ptr, h_tree_size, dim, max_nodes, idx_ptr, dist_ptr
    );
    
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PrrtcNearestNeighborFfi, PrrtcNearestNeighborImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
);
