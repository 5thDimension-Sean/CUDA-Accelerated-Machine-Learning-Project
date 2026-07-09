// ============================================================================
// batchnorm.cuh — declarations for the BatchNorm kernels + wrappers.
// Definitions live in batchnorm.cu.
// ============================================================================
#pragma once

// --- kernels (device) ---
__global__ void mean_kernel(float *x, float *mean, int N);
__global__ void variance_kernel(float *x, float mean, float *variance, int N);
__global__ void batchNormForward(const float *d_x, float *d_y,
                                 float mean, float variance,
                                 float gamma, float beta,
                                 float epsilon, int N);

// per-channel version for the network layer (gamma/beta/mean/variance are
// length-C vectors, passed as pointers; [C, H, W] layout)
__global__ void batchNormForwardPerChannel(const float *d_x, float *d_y,
                                           const float *mean, const float *variance,
                                           const float *gamma, const float *beta,
                                           float epsilon, int H, int W, int C);

// --- host wrappers (malloc + copy + launch + copy-back + free) ---
void meanWrapKernel(float *matrix, float *mean, int N);
void varianceWrapKernel(float *matrix, float mean, float *variance, int N);
void batchNormWrapKernel(float *matrix, float *matriy, float mean, float variance,
                         float gamma, float beta, float epsilon, int N);
