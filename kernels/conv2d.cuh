// ============================================================================
// conv2d.cuh — declarations for the 2D convolution kernels.
// Definitions live in conv2d.cu.
// NOTE: conv2d_constant and conv2d_shared read the filter from the file-local
// __constant__ c_filter (loaded via cudaMemcpyToSymbol in conv2d.cu), so they
// take NO filter pointer. conv2d_naive and conv2d_filter take the filter as an
// argument.
// ============================================================================
#pragma once

// naive: filter passed as a device pointer
__global__ void conv2d_naive(const float *input, const float *filter,
                             float *output, int H, int W, int FH, int FW);

// constant-memory: filter comes from c_filter (no filter arg)
__global__ void conv2d_constant(const float *input, float *output,
                                int H, int W, int FH, int FW);

// shared-memory tiled: filter comes from c_filter (no filter arg).
// Launch needs a dynamic shared-mem size of
//   (blockDim.x + FW - 1) * (blockDim.y + FH - 1) * sizeof(float)
__global__ void conv2d_shared(const float *input, float *output,
                              int H, int W, int FH, int FW);

// depthwise: one filter per channel, filters passed as a device pointer
__global__ void conv2d_filter(const float *input, const float *filters,
                              float *output, int H, int W, int C, int FH, int FW);

__global__ void conv2d_backward_filter(const float *dOut, const float *input, float *dFilter, int H, int W, int FH, int FW);

__global__ void conv2d_backward_input (const float *dOut, const float *filter, float *dInput, int H, int W, int FH, int FW);

__global__ void conv2d_wrapper(const float *dOut, const float *filter, float *dInput, int H, int W, int FH, int FW);
