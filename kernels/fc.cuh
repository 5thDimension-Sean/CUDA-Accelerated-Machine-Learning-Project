#ifndef NETWORK_FC_CUH
#define NETWORK_FC_CUH

#include <cuda_runtime.h>

__global__ void fc_forward_kernel(const float* X, const float *W, const float *b, float *Y, int batch, int in, int out);


void fc_forward(const float* X, const float *W, const float *b, float *Y, int batch, int in, int out);

#endif