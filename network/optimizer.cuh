#ifndef NETWORK_OPTIMER_CUH
#define NETWORK_OPTIMZER_CUH

#include <cuda_runtime.h>


__global__ void sgd_kernel(float *weights, const float *grad, float lr, int n);

__global__ void momentum_kernel(float *weights, const float *grad, float *velocity, float lr, float beta, int n);

void sgd(float *weights, const float *grad, float lr, int n);

void momentum(float *weights, const float *grad, float *velocity, float lr, float beta, int n);

#endif 
