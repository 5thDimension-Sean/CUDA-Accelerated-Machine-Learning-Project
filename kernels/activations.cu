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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

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


// TODO — Week 5: implement here
void sigMoid(float *z_matrix, float *activation_matrix, int width, int height) {
    const int arrSize = 5;
    float host_z_values[arrSize] = {1., 2., 3., 4., 5.};
    float host_activations[arrSize] = {0.};

    const size_t bytes_z_values = arrSize * sizeof(float);
    const size_t bytes_activations = arrSize * sizeof(float);

    float *device_z_values, *device_activations;

    CUDA_CHECK(cudaMalloc(&device_z_values, bytes_z_values));
    CUDA_CHECK(cudaMalloc(&device_activations, bytes_activations));

    CUDA_CHECK(cudaMemcpy(device_z_values, host_z_values, bytes_z_values, cudaMemcpyHostToDevice));
    //sigmoidActivation<< <1, arrSize>> > (device_z_values, device_activations, 256, 256, true);
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


void ReLu(float *z_matrix, float *activation_matrix, int width, int height) {
    const int arrSize = 5;
    float host_z_values[arrSize] = {1., 2., 3., 4., 5.};
    float host_activations[arrSize] = {0.};

    const size_t bytes_z_values = arrSize * sizeof(float);
    const size_t bytes_activations = arrSize * sizeof(float);

    float *device_z_values, *device_activations;

    CUDA_CHECK(cudaMalloc(&device_z_values, bytes_z_values));
    CUDA_CHECK(cudaMalloc(&device_activations, bytes_activations));

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

void leakyRelu(float *z_matrix, float *activation_matrix, int width, int height) {
    const int arrSize = 5;
    float host_z_values[arrSize] = {1., 2., 3., 4., 5.};
    float host_activations[arrSize] = {0.};

    const size_t bytes_z_values = arrSize * sizeof(float);
    const size_t bytes_activations = arrSize * sizeof(float);

    float *device_z_values, *device_activations;

    CUDA_CHECK(cudaMalloc(&device_z_values, bytes_z_values));
    CUDA_CHECK(cudaMalloc(&device_activations, bytes_activations));

}

__global__ void softMaxActivation(float *z_matrix, float *activation_matrix, int width, int height, bool isForward) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    
    if (row < height && col < width) {
        int index = row * width + col;
        
        if (isForward) {
            float sum = 0.0f;
            int total_elements = width * height;
            for (int i = 0; i < total_elements; i++) {
                sum += expf(z_matrix[i]);
            }
            activation_matrix[index] = expf(z_matrix[index]) / sum;
        } else {
            // Backward pass placeholder
        }
    }
}

void softMax(float *z_matrix, float *activation_matrix, int width, int height) {
    const int arrSize = 5;
    float host_z_values[arrSize] = {1., 2., 3., 4., 5.};
    float host_activations[arrSize] = {0.};

    const size_t bytes_z_values = arrSize * sizeof(float);
    const size_t bytes_activations = arrSize * sizeof(float);

    float *device_z_values, *device_activations;

    CUDA_CHECK(cudaMalloc(&device_z_values, bytes_z_values));
    CUDA_CHECK(cudaMalloc(&device_activations, bytes_activations));

}