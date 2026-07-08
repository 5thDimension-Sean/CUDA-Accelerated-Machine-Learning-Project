// ============================================================================
// batchnorm.cu — Week 5
// Goal: Implement Batch Normalization forward + backward in CUDA.
//
// Forward pass:
//   1. Compute mean across the batch (parallel reduction)
//   2. Compute variance across the batch (parallel reduction)
//   3. Normalize: x_hat = (x - mean) / sqrt(variance + epsilon)
//   4. Scale and shift: y = gamma * x_hat + beta
//
// Backward pass requires gradients w.r.t. gamma, beta, and input x.
// This is the hardest backward pass you'll write — derive it on paper first.
//
// Resources:
//   - Original BatchNorm paper: arxiv.org/abs/1502.03167
//   - The backward pass derivation: kevinzakka.github.io/2016/09/14/batch_normalization
// ============================================================================
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
//#include <torch/torch.h>
#include <iostream>

// TODO — Week 5: implement here
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

const int N = 5; 
dim3 block(5);
dim3 grid(1);
//3 kernels mean, variance, normalize + scale

__global__ void mean_kernel(float *x, float *mean, int N) {
    float sum = 0.0f;
    for(int i = 0; i<N; i++){
        sum += x[i];
    }
    *mean = sum / N;
}

void meanWrapKernel(float *matrix, float *x, float *mean, int N){
    float *d_x, *d_mean;
    size_t bytes_data = N * sizeof(float);
    size_t bytes_stats = 1 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_x, bytes_data));
    CUDA_CHECK(cudaMalloc(&d_mean, bytes_stats));
    CUDA_CHECK(cudaMemcpy(d_x, matrix, bytes_data, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_mean, 0, bytes_stats));
    mean_kernel<<<grid, block>>>(d_x, d_mean, N);
    cudaFree(d_x);
    cudaFree(d_mean);
}

__global__ void variance_kernel(float *x, float *mean, float *variance, int N){

}

__global__ void batchNormForward(float *x, float *y, float *mean, float *variance, float gamma, float beta, float epsilon, int N){

}



int main(){
    float arrX[N] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};

}