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

const int N = 8; 
dim3 block(8);
dim3 grid(1);
//3 kernels mean, variance, normalize + scale

__global__ void mean_kernel(float *x, float *mean, int N) {
    __shared__ float sdata[8]; //shared memory for the block
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    //load data into shared memory
    sdata[tid] = (idx < N) ? x[idx] : 0.0f; //handle out of bounds
    __syncthreads(); //make sure all threads have loaded their data
    for(int i = blockDim.x/2;i > 0; i/=2){
        if(tid<i){
            sdata[tid]  += sdata[tid+i];
        }
        __syncthreads(); //wait for all threads to finish adding before next round
    }
    if(tid == 0)
    *mean = sdata[0]/N; //store the mean in global memory
}

void meanWrapKernel(float *matrix, float *mean, int N){
    float *d_x, *d_mean;
    size_t bytes_data = N * sizeof(float);
    size_t bytes_stats = 1 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_x, bytes_data));
    CUDA_CHECK(cudaMalloc(&d_mean, bytes_stats));
    CUDA_CHECK(cudaMemcpy(d_x, matrix, bytes_data, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_mean, 0, bytes_stats));
    mean_kernel<<<grid, block>>>(d_x, d_mean, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(mean, d_mean, bytes_stats, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_mean));
}

__global__ void variance_kernel(float *x, float mean, float *variance, int N){
    __shared__ float sdata[8]; // Shared memory for the block
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    if (idx < N) {
        float diff = x[idx] - mean;
        sdata[tid] = diff * diff; // Move squaring up into the load
    } else {
        sdata[tid] = 0.0f;        // Handle out of bounds
    }
    __syncthreads(); // Make sure all threads have loaded/squared their data

    for(int i = blockDim.x / 2; i > 0; i /= 2){
        if(tid < i){
            sdata[tid] += sdata[tid + i]; // Just plain addition
        }
        __syncthreads(); // Wait for all threads to finish adding
    }
    
    // Divide by N at the end
    if(tid == 0) {
        *variance = sdata[0] / N; // Store the variance in global memory
    }
}
void varianceWrapKernel(float *matrix, float mean, float *variance, int N){
    float *d_x, *d_variance;
    size_t bytes_data = N * sizeof(float);
    size_t bytes_stats = 1 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_x, bytes_data));
    CUDA_CHECK(cudaMalloc(&d_variance, bytes_stats));
    CUDA_CHECK(cudaMemcpy(d_x, matrix, bytes_data, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_variance, 0, bytes_stats));
    variance_kernel<<<grid, block>>>(d_x, mean,  d_variance, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(variance, d_variance, bytes_stats, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_variance));
}
__global__ void batchNormForward(
    const float *d_x, 
    float *d_y, 
    float mean, 
    float variance, 
    float gamma, 
    float beta, 
    float epsilon, 
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // calculate x_hat: (x - mean) / sqrt(var + epsilon)
        float x_hat = (d_x[idx] - mean) / sqrtf(variance + epsilon);
        
        // calculate y: gamma * x_hat + beta
        d_y[idx] = gamma * x_hat + beta;
    }
}
//Mean, variance, gamma, beta, epsilon, and N are scalar
void batchNormWrapKernel(float *matrix, float *matriy, float mean, float variance,
                         float gamma, float beta, float epsilon, int N) {
    float *d_x, *d_y;
    size_t bytes_data = N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_x, bytes_data));
    CUDA_CHECK(cudaMalloc(&d_y, bytes_data));
    CUDA_CHECK(cudaMemcpy(d_x, matrix, bytes_data, cudaMemcpyHostToDevice));
    batchNormForward<<<grid, block>>>(d_x, d_y, mean, variance, gamma, beta, epsilon, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(matriy, d_y, bytes_data, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

int main() {
    float arrX[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    float mean = 0.0f, variance = 0.0f;
    float y[N];

    float gamma = 1.0f, beta = 0.0f, epsilon = 1e-5f;

    meanWrapKernel(arrX, &mean, N);
    varianceWrapKernel(arrX, mean, &variance, N);
    batchNormWrapKernel(arrX, y, mean, variance, gamma, beta, epsilon, N);

    printf("mean = %.4f\n", mean);          // expect 4.5000
    printf("variance = %.4f\n", variance);  // expect 5.2500
    printf("output: ");
    for (int i = 0; i < N; i++) printf("%.4f ", y[i]);
    printf("\n");

    return 0;
}