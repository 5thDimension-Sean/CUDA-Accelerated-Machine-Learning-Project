#ifndef NETWORK_FC_CUH
#define NETWORK_FC_CUH

#include <cuda_runtime.h>

__global__ void fc_forward_kernel(const float* X, const float *W, const float *b, float *Y, int batch, int in, int out);

__global__ void fc_backward_weights_kernel(const float *dY, const float*X, float*dW, int batch, int in, int out);

__global__ void fc_backward_bias_kernel(const float *dY, float*db, int batch, int out);

__global__ void fc_backward_input_kernel(const float *dY, const float *W, float *dX, int batch, int in, int out);

void fc_forward(const float* X, const float *W, const float *b, float *Y, int batch, int in, int out);

void fc_backward(const float *dY, const float *X, const float *W,
                 float *dW, float *db, float *dX, int batch, int in, int out);
#endif