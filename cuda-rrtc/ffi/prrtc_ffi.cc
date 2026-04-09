/**
 * XLA FFI bindings for pRRTC CUDA kernels.
 * 
 * This file provides C++ FFI handlers that can be called from JAX.
 * The handlers wrap the CUDA kernels for:
 *   - Nearest neighbor search
 *   - Tree extension
 *   - Iteration planning
 *   - Full planning
 */

#include <cuda_runtime.h>
#include <xla/ffi/api/ffi.h>

// Forward declare FFI handlers defined in .cu files
extern "C" {

// From prrtc_nearest_neighbor.cu
XLA_FFI_REGISTER_HANDLER_SYMBOL(PrrtcNearestNeighborFfi);

// From prrtc_extend.cu
XLA_FFI_REGISTER_HANDLER_SYMBOL(PrrtcExtendFfi);

// From prrtc_iteration.cu
XLA_FFI_REGISTER_HANDLER_SYMBOL(PrrtcIterationFfi);

// From prrtc_planner.cu
XLA_FFI_REGISTER_HANDLER_SYMBOL(PrrtcPlannerFfi);

}  // extern "C"

