#include "optimizer.cuh"
#include "../kernels/common.cuh"
#include <cmath>
#include <cstdio>

//optimizer.cu

__global__ void sgd_kernel(float *weights, const float *grad, float lr, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        weights[idx] -= lr * grad[idx];
    }
}

__global__ void momentum_kernel(float *weights, const float *grad, float *velocity, float lr, float beta, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        velocity[idx] = beta * velocity[idx] + grad[idx];
        weights[idx] -= lr * velocity[idx];
    }
}


void sgd(float *weights, const float *grad, float lr, int n){
    size_t bytes = n * sizeof(float);
    float *d_weights, *d_grad;
    CUDA_CHECK(cudaMalloc(&d_weights, bytes));
    CUDA_CHECK(cudaMalloc(&d_grad, bytes));
    CUDA_CHECK(cudaMemcpy(d_weights, weights, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad, grad, bytes, cudaMemcpyHostToDevice));
    dim3 block(256);
    dim3 grid ((n+block.x-1)/block.x);
    sgd_kernel<<<grid, block>>>(d_weights, d_grad, lr, n);
    CUDA_CHECK(cudaMemcpy(weights, d_weights, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_grad));
}

void momentum(float *weights, const float *grad, float *velocity, float lr, float beta, int n){
    size_t bytes = n * sizeof(float);
    float *d_weights, *d_grad, *d_velocity;
    CUDA_CHECK(cudaMalloc(&d_weights, bytes));
    CUDA_CHECK(cudaMalloc(&d_grad, bytes));
    CUDA_CHECK(cudaMalloc(&d_velocity, bytes));
    CUDA_CHECK(cudaMemcpy(d_weights, weights, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad, grad, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velocity, velocity, bytes, cudaMemcpyHostToDevice));
    dim3 block(256);
    dim3 grid ((n+block.x-1)/block.x);
    momentum_kernel<<<grid, block>>>(d_weights, d_grad, d_velocity, lr, beta, n);
    CUDA_CHECK(cudaMemcpy(weights, d_weights, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velocity, d_velocity, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_velocity));
}

#ifndef BUILD_AS_LIBRARY
int main(){
    //n is num of weights
    const int n = 4;
    float weights[n] = {1.0f, 2.0f, 3.0f, 4.0f};
    float grad[n] = {0.1f, 0.2f, 0.3f, 0.4f};
    float velocity[n] = {0.0f, 0.0f, 0.0f, 0.0f};
    float lr = 0.1f;
    float beta = 0.9f;
    sgd(weights, grad, lr, n);
    printf("Weights after SGD: ");
    for(int i = 0; i < n; ++i){
        printf("%.4f ", weights[i]);
    }
    // Reset weights for momentum test
    for(int i = 0; i < n; ++i){
        weights[i] = (float)(i + 1.0f);
    }
    momentum(weights, grad, velocity, lr, beta, n);
    printf("\nWeights after Momentum: ");
    for(int i = 0; i < n; ++i){
        printf("%.4f ", weights[i]);
    }
    momentum(weights, grad, velocity, lr, beta, n);
    printf("\nWeights after Momentum step 2: ");
    for(int i = 0; i < n; ++i){
        printf("%.4f ", weights[i]);
    }
    
    return 0;
}
#endif