// ============================================================================
// activations.cu — Week 5
// Goal: Implement neural network activation functions as CUDA kernels.
//
// Functions to implement:
//   - ReLU:       f(x) = max(0, x)         backward: 1 if x > 0 else 0
//   - Leaky ReLU: f(x) = max(0.01x, x)     backward: 1 if x > 0 else 0.01
//   - Sigmoid:    f(x) = 1 / (1 + e^-x)   backward: f(x) * (1 - f(x))
//   - Softmax:    f(x_i) = e^x_i / sum(e^x) (requires parallel reduction)
//
// Each activation needs TWO kernels: forward pass + backward pass (gradient).
// The backward pass is used in Week 6 (backpropagation).
//
// Verify against PyTorch:
//   import torch
//   x = torch.tensor([...])
//   torch.nn.functional.relu(x)
// ============================================================================
#include "common.cuh"
#include "activations.cuh"
#include <cmath>
//#include <torch/torch.h>
#include <iostream>

static dim3 block(5, 1);
static dim3 grid(1, 1);

__global__ void sigmoidActivation(float *z_matrix, float *activation_matrix, int width, int height, bool isForward) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // X coordinate (Width)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Y coordinate (Height)
    if (row < height && col < width) {
        int index = row * width + col;
    if(isForward){
        //sigmoid formula
        activation_matrix[index] = 1.0f / (1.0f + expf(-z_matrix[index]));
    }else{
        //backward or derivative of sigmoid formula
        //activation_matrix[index] = expf(z_matrix[index])/(pow(1.0f + expf(z_matrix[index]), 2)); Slow due to pow 
        float s = activation_matrix[index];

        activation_matrix[index] = s * (1.0f - s);
    }
    }
}

__global__ void reLuActivation(float *z_matrix, float *activation_matrix, int width, int height, bool isForward) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // X coordinate (Width)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Y coordinate (Height)
    if (row < height && col < width) {
        int index = row * width + col;
    if(isForward){
        //sigmoid formula
        activation_matrix[index] = max(0.0f,  z_matrix[index]);
    }else{
        //backward or derivative of sigmoid formula
        //activation_matrix[index] = expf(z_matrix[index])/(pow(1.0f + expf(z_matrix[index]), 2)); Slow due to pow 

        activation_matrix[index] = (z_matrix[index] > 0.0f) ? 1.0f : 0.0f;
    }
    }
}

__global__ void leakyreLuActivation(float *z_matrix, float *activation_matrix, int width, int height, bool isForward) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // X coordinate (Width)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Y coordinate (Height)
    if (row < height && col < width) {
        int index = row * width + col;
    if(isForward){
        //leakRelu formula
        activation_matrix[index] = max(0.01f*z_matrix[index], z_matrix[index]);
    }else{
        //backward or derivative of leaky relu formula

        activation_matrix[index] = (z_matrix[index] > 0.0f) ? 1.0f : 0.01f;
    }
    }
}

__global__ void softMaxActivation(float *z_matrix, float *activation_matrix, int width, int height, bool isForward) {
      int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float row_sum;

    if (row < height && col < width) {
        int index = row * width + col;
        //missing max subtraction
        if (isForward) {
            float val = z_matrix[index];
            float exp_val = expf(val);
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                row_sum = 0.0f;
            }
            __syncthreads();
            atomicAdd(&row_sum, exp_val);
            __syncthreads();
            activation_matrix[index] = exp_val / row_sum;
        } else {
            // Backward pass placeholder
        }
    }
}

//condensed wrapper function
void wrapperKernel(float *host_z_matrix, float *host_activation_matrix, int width, int height, bool isForward, ActivationType type) {
    if (!isForward && type == ActivationType::Softmax) {
        fprintf(stderr, "Error: Softmax does not support a backward pass kernel!\n");
        return;
    }

    int arrSize = width * height;
    size_t bytes = arrSize * sizeof(float);

    float *device_z_matrix, *device_activation_matrix;
    CUDA_CHECK(cudaMalloc(&device_z_matrix, bytes));
    CUDA_CHECK(cudaMalloc(&device_activation_matrix, bytes));

    if (isForward) {
        CUDA_CHECK(cudaMemcpy(device_z_matrix, host_z_matrix, bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(device_activation_matrix, host_activation_matrix, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_z_matrix, host_z_matrix, bytes, cudaMemcpyHostToDevice));
    }

    if (type == ActivationType::ReLU) {
        reLuActivation<<<grid, block>>>(device_z_matrix, device_activation_matrix, width, height, isForward);
    } 
    else if (type == ActivationType::LRELU) {
        leakyreLuActivation<<<grid, block>>>(device_z_matrix, device_activation_matrix, width, height, isForward);
    } 
    else if (type == ActivationType::Sigmoid) {
        sigmoidActivation<<<grid, block>>>(device_z_matrix, device_activation_matrix, width, height, isForward);
    } 
    else if (type == ActivationType::Softmax) {
        softMaxActivation<<<grid, block>>>(device_z_matrix, device_activation_matrix, width, height, isForward);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(host_activation_matrix, device_activation_matrix, bytes, cudaMemcpyDeviceToHost));

    printf("Activation execution (Type: %d, IsForward: %s) output:\n", (int)type, isForward ? "True" : "False");
    for (int i = 0; i < arrSize; i++) {
        printf("%.4f ", host_activation_matrix[i]);
    }
    printf("\n\n");

    cudaFree(device_z_matrix);
    cudaFree(device_activation_matrix);
}



#ifndef BUILD_AS_LIBRARY
int main(){
  const int arrSize = 5;
    
    float host_z_values[arrSize] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float host_activations[arrSize] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    printf("--- Case 1: ReLU Forward ---\n");
    wrapperKernel(host_z_values, host_activations, arrSize, 1, true, ActivationType::ReLU);

    printf("--- Case 2: ReLU Backward ---\n");
    wrapperKernel(host_z_values, host_activations, arrSize, 1, false, ActivationType::ReLU);

    for(int i = 0; i < arrSize; i++) host_activations[i] = 0.0f;
    printf("--- Case 3: Leaky ReLU Forward ---\n");
    wrapperKernel(host_z_values, host_activations, arrSize, 1, true, ActivationType::LRELU);

    printf("--- Case 4: Leaky ReLU Backward ---\n");
    wrapperKernel(host_z_values, host_activations, arrSize, 1, false, ActivationType::LRELU);

    for(int i = 0; i < arrSize; i++) host_activations[i] = 0.0f;
    printf("--- Case 5: Sigmoid Forward ---\n");
    wrapperKernel(host_z_values, host_activations, arrSize, 1, true, ActivationType::Sigmoid);

    printf("--- Case 6: Sigmoid Backward ---\n");
    wrapperKernel(host_z_values, host_activations, arrSize, 1, false, ActivationType::Sigmoid);

    for(int i = 0; i < arrSize; i++) host_activations[i] = 0.0f;
    printf("--- Case 7: Softmax Forward ---\n");
    wrapperKernel(host_z_values, host_activations, arrSize, 1, true, ActivationType::Softmax);

    printf("--- Case 8: Softmax Backward (Expected Error Block) ---\n");
    wrapperKernel(host_z_values, host_activations, arrSize, 1, false, ActivationType::Softmax);

    return 0;
}
#endif