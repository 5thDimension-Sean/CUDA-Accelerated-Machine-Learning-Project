#include "optimizer.cuh"
#include <cmath>
#include <cstdio>


//optimizer.cu

__global__ void sgd_kernel(float *weights, const float *grad, float lr, int n){}

__global__ void momentum_kernel(float *weights, const float *grad, float *velocity, float lr, float beta, int n){}


void sgd(float *weights, const float *grad, float lr, int n){}

void momentum(float *weights, const float *grad, float *velocity, float lr, float beta, int n){}