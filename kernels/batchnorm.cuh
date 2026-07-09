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

// --- host wrappers (malloc + copy + launch + copy-back + free) ---
void meanWrapKernel(float *matrix, float *mean, int N);
void varianceWrapKernel(float *matrix, float mean, float *variance, int N);
void batchNormWrapKernel(float *matrix, float *matriy, float mean, float variance,
                         float gamma, float beta, float epsilon, int N);
