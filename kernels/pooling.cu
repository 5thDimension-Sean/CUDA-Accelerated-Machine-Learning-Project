// ============================================================================
// pooling.cu — Week 5
// Goal: Implement 2D max pooling in CUDA.
//
// Max pooling: slide a window over the input, output the maximum value.
//   Input:  [H x W x C]
//   Output: [H/stride x W/stride x C]
//
// Backward pass: gradient flows only through the element that was the max.
// You need to save which element WAS the max during forward (called a "mask").
//1
// Also implement: average pooling (output = mean of window, not max).
// ============================================================================

// TODO — Week 5: implement here

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


int P = 2; //pool window size
int S = 2; //stride
int H = 4; //input dim
int W = 4; //input dim
//4x4 input
dim3 block(16, 16);
int out_H = (H - P) / S + 1;
int out_W = (W - P) / S + 1;
dim3 grid((out_W + block.x - 1) / block.x, (out_H + block.y - 1) / block.y);

__global__ void maxPool2D(const float *input, float *output, int H, int W, int out_H, int out_W, int P, int S){
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < out_W && out_y < out_H) {
        float max_val = -INFINITY;
        for (int i = 0; i < P; ++i) {
            for (int j = 0; j < P; ++j) {
                int in_x = out_x * S + j;
                int in_y = out_y * S + i;
                if (in_x < W && in_y < H) {
                    float val = input[in_y * W + in_x];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
        output[out_y * out_W + out_x] = max_val;
    }
}

void maxPoolWrapKernel(float *h_input, float *h_output, int H, int W, int P, int S) {
    float *d_input, *d_output;
    size_t bytes_in  = H * W * sizeof(float);
    size_t bytes_out = out_H * out_W * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, bytes_in));
    CUDA_CHECK(cudaMalloc(&d_output, bytes_out));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes_in, cudaMemcpyHostToDevice));
    maxPool2D<<<grid, block>>>(d_input, d_output, H, W, out_H, out_W, P, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}


int main(){
    float h_input[16] = {
     1,  3,  2,  9,
     8,  4,  7,  5,
    -1, -3,  6,  0,
    -5, -2, -4,  1
};
    float h_output[4];

    maxPoolWrapKernel(h_input, h_output, H, W, P, S);
    for(int i = 0; i < out_H; i++){
        for(int j = 0; j < out_W; j++){
            printf("%.4f ", h_output[i * out_W + j]);
        }
        printf("\n");
    }
    return 0;
}