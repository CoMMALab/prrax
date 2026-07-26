#!/usr/bin/env bash
# Build _prrtc_planner_lib.so from pRRTC CUDA kernels for cuda-rrtc.
#
# Usage:
#   bash build.sh
#   bash build.sh --debug
#
# Requirements:
#   - nvcc (CUDA toolkit)
#   - jaxlib >= 0.4.14 installed
#     (provides the xla/ffi/api/ffi.h headers)

set -euo pipefail

DEBUG=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      DEBUG=1
      shift
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Output library
OUT="${SCRIPT_DIR}/_prrtc_planner_lib.so"

# pyroffi CUDA kernel headers (collision + FK)
PYROFFI_CUDA_INC="${SCRIPT_DIR}/../pyroffi/src/pyroffi/cuda_kernels"
if [ ! -f "${PYROFFI_CUDA_INC}/_fk_cuda_helpers.cuh" ]; then
  echo "ERROR: pyroffi cuda_kernels not found at ${PYROFFI_CUDA_INC}"
  echo "Make sure pyroffi is checked out alongside cuda-rrtc."
  exit 1
fi

# Source kernel files in CUDA subdirectory
SRC_DIR="${SCRIPT_DIR}/CUDA"
SOURCES=(
    "${SRC_DIR}/prrtc_nearest_neighbor.cu"
    "${SRC_DIR}/prrtc_extend.cu"
    "${SRC_DIR}/prrtc_iteration.cu"
    "${SRC_DIR}/prrtc_planner.cu"
)

# Locate the jaxlib include directory that ships xla/ffi/api/ffi.h.
# Use the active env's interpreter: a bare `python3` can resolve to a system
# python ahead of the conda env on PATH, which won't have jaxlib.
PYTHON="${PYTHON:-python}"
JAXLIB_INC="$("${PYTHON}" -c "
import os, sys
try:
    import jaxlib
    print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))
except Exception as e:
    sys.stderr.write(f'Error: {e}\n')
    sys.exit(1)
")"

if [ ! -f "${JAXLIB_INC}/xla/ffi/api/ffi.h" ]; then
  echo "ERROR: xla/ffi/api/ffi.h not found under ${JAXLIB_INC}"
  echo "Make sure jaxlib >= 0.4.14 is installed in your Python environment."
  exit 1
fi

# GPU architecture flag.
GPU_ARCH="${GPU_ARCH:--arch=native}"

NVCC_OPT="-O3"
if [ "${DEBUG}" -eq 1 ]; then
  NVCC_OPT="-O0 -G -lineinfo"
  echo "Building in DEBUG mode (with -G for Nsight Compute)..."
fi

# Compile all CUDA sources into shared library
nvcc \
  ${NVCC_OPT} \
  -std=c++17 \
  ${GPU_ARCH} \
  --shared \
  --compiler-options "-fPIC" \
  -I"${JAXLIB_INC}" \
  -I"${PYROFFI_CUDA_INC}" \
  -o "${OUT}" \
  "${SOURCES[@]}"

echo "Built: ${OUT}"
