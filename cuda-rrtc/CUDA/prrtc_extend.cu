/**
 * CUDA kernel for tree extension in pRRTC.
 *
 * Extends the tree from the nearest node toward a sampled configuration.
 * Performs step-size limiting for valid motion.
 */

#include "xla/ffi/api/ffi.h"
#include "prrtc_helpers.cuh"
#include <float.h>

namespace ffi = xla::ffi;

__global__ void prrtc_extend_kernel(
    const float* __restrict__ tree_configs,    // [dim, max_nodes] SoA
    const int* __restrict__ nearest_indices,   // [batch]
    const float* __restrict__ samples,         // [batch, dim]
    float step_size,                            // maximum extension step
    int dim,                                    // configuration dimension
    int max_nodes,                              // max tree capacity
    float* __restrict__ new_configs,           // [batch, dim] output
    int* __restrict__ parent_indices,          // [batch] output
    int* __restrict__ valid_flags              // [batch] output
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    if (bid >= gridDim.x) return;
    
    const int nearest_idx = nearest_indices[bid];
    
    if (tid == 0) {
        // Copy sample to new config
        for (int d = 0; d < dim; d++) {
            new_configs[bid * dim + d] = samples[bid * dim + d];
        }
        parent_indices[bid] = nearest_idx;
        valid_flags[bid] = 1;
    }
    __syncthreads();
    
    if (tid == 0) {
        // Compute direction from nearest to sample
        float nearest_vals[CONFIG_DIM_MAX];
        #pragma unroll
        for (int d = 0; d < dim; d++) {
            nearest_vals[d] = tree_configs[d * max_nodes + nearest_idx];
        }
        
        float sample_vals[CONFIG_DIM_MAX];
        #pragma unroll
        for (int d = 0; d < dim; d++) {
            sample_vals[d] = samples[bid * dim + d];
        }
        
        // Compute distance to sample
        float dist_sq = 0.0f;
        #pragma unroll
        for (int d = 0; d < dim; d++) {
            float diff = sample_vals[d] - nearest_vals[d];
            dist_sq += diff * diff;
        }
        
        float dist = sqrtf(dist_sq);
        float scale = fminf(1.0f, step_size / dist);
        
        // Generate extended configuration
        #pragma unroll
        for (int d = 0; d < dim; d++) {
            new_configs[bid * dim + d] = nearest_vals[d] + (sample_vals[d] - nearest_vals[d]) * scale;
        }
    }
}

// XLA FFI handler
static ffi::Error PrrtcExtendImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> tree_configs,
    ffi::Buffer<ffi::DataType::S32> nearest_indices,
    ffi::Buffer<ffi::DataType::F32> samples,
    ffi::Buffer<ffi::DataType::F32> step_size,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> new_configs,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> parent_indices,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> valid_flags
) {
    const int dim = static_cast<int>(tree_configs.dimensions()[0]);
    const int max_nodes = static_cast<int>(tree_configs.dimensions()[1]);
    const int batch = static_cast<int>(nearest_indices.dimensions()[0]);
    
    if (batch == 0) return ffi::Error::Success();
    
    const float* tc_ptr = tree_configs.typed_data();
    const int* ni_ptr = nearest_indices.typed_data();
    const float* sm_ptr = samples.typed_data();
    
    float* nc_ptr = new_configs->typed_data();
    int* pi_ptr = parent_indices->typed_data();
    int* vf_ptr = valid_flags->typed_data();
    
    // Get step size from device
    float h_step_size;
    cudaError_t e = cudaMemcpyAsync(&h_step_size, step_size.typed_data(), sizeof(float),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    
    dim3 block(128);
    dim3 grid(batch);
    
    prrtc_extend_kernel<<<grid, block, 0, stream>>>(
        tc_ptr, ni_ptr, sm_ptr, h_step_size, dim, max_nodes,
        nc_ptr, pi_ptr, vf_ptr
    );
    
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(e));
    }
    
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PrrtcExtendFfi, PrrtcExtendImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // tree_configs [dim, max_nodes]
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // nearest_indices [batch]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // samples [batch, dim]
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // step_size
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // new_configs [batch, dim]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // parent_indices [batch]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // valid_flags [batch]
);
