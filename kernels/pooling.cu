// ============================================================================
// pooling.cu — Week 5
// Goal: Implement 2D max pooling in CUDA.
// ============================================================================
#include "common.cuh"
#include <cmath>
#include <iostream>


static dim3 block(16, 16);
//First out_H, out_w, and grid are static. Therefore, they need to be changed inside the wrapper from the arguments.
__global__ void maxPool2D(const float *input, float *output, int *argmax, int H, int W, int out_H, int out_W, int P, int S, int C) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if (out_x >= out_W || out_y >= out_H || c >= C) return;
        const float *in_c = input + c*(H*W);
        float max_val = -INFINITY; int max_idx = -1;
        for (int i=0;i<P;++i) for (int j=0;j<P;++j){
            int in_y = out_y*S+i, in_x = out_x*S+j;
            if (in_y<H && in_x<W){
                float v = in_c[in_y*W + in_x];
                if (v>max_val){ max_val=v; max_idx = c*(H*W) + in_y*W + in_x; }  // GLOBAL
            }
        }
        int o = c*(out_H*out_W) + out_y*out_W + out_x;
        output[o] = max_val;  
        argmax[o] = max_idx;     
}

__global__ void backMaxPool2D(const float *dOut, const int *argmax, float *dInput, int out_H, int out_W) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < out_W && out_y < out_H) {
        int out_offset = out_y * out_W + out_x;
        int target_in_idx = argmax[out_offset];
        if (target_in_idx != -1) {
            dInput[target_in_idx] = dOut[out_offset];
        }
    }
}

void maxPoolWrapKernel(float *h_input, float *h_output, int *argmax, int H, int W, int P, int S, int C) {
    float *d_input, *d_output;
    int *d_argmax;

    int out_H = (H - P) / S + 1;
    int out_W = (W - P) / S + 1;
    dim3 grid((out_W + block.x - 1) / block.x, (out_H + block.y - 1) / block.y, C);
    size_t bytes_in = H * W * sizeof(float);
    size_t bytes_out = out_H * out_W * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, C *bytes_in));
    CUDA_CHECK(cudaMalloc(&d_output, C * bytes_out));
    CUDA_CHECK(cudaMalloc(&d_argmax, C *out_H * out_W * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes_in, cudaMemcpyHostToDevice));

    maxPool2D<<<grid, block>>>(d_input, d_output, d_argmax, H, W, out_H, out_W, P, S, C);
    CUDA_CHECK(cudaGetLastError()); 

    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(argmax, d_argmax, out_H * out_W * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_argmax));
}


void backMaxPoolWrapKernel(float *h_dOut, float *h_dInput, int *h_argmax, int H, int W, int P, int S) {
    float *d_dOut, *d_dInput;
    int *d_argmax;
    static int out_H = (H - P) / S + 1;
    static int out_W = (W - P) / S + 1;
    static dim3 grid((out_W + block.x - 1) / block.x, (out_H + block.y - 1) / block.y);
    size_t bytes_out = out_H * out_W * sizeof(float);
    size_t bytes_in = H * W * sizeof(float);
    size_t bytes_argmax = out_H * out_W * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_dOut, bytes_out));
    CUDA_CHECK(cudaMalloc(&d_dInput, bytes_in));
    CUDA_CHECK(cudaMalloc(&d_argmax, bytes_argmax));


    CUDA_CHECK(cudaMemcpy(d_dOut, h_dOut, bytes_out, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_argmax, h_argmax, bytes_argmax, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_dInput, 0, bytes_in));

    backMaxPool2D<<<grid, block>>>(d_dOut, d_argmax, d_dInput, out_H, out_W);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_dInput, d_dInput, bytes_in, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dOut));
    CUDA_CHECK(cudaFree(d_dInput));
    CUDA_CHECK(cudaFree(d_argmax));
}

#ifndef BUILD_AS_LIBRARY
int main() {
    // Forward Pass Elements
    float h_input[16] = { 
         1,  3,  2,  9, 
         8,  4,  7,  5, 
        -1, -3,  6,  0, 
        -5, -2, -4,  1 
    };
    float h_output[4];
    int argmax[4];
    int C = 2;
    std::cout << "--- FORWARD PASS ---" << std::endl;
    maxPoolWrapKernel(h_input, h_output, argmax, H, W, P, S, C);
    
    for (int i = 0; i < out_H; i++) {
        for (int j = 0; j < out_W; j++) {
            int offset = i * out_W + j;
            printf("Val: %.1f (Argmax Idx: %d) | ", h_output[offset], argmax[offset]);
        }
        printf("\n");
    }

    float h_dOut[4] = {1.0f, 2.0f, 3.0f, 4.0f}; 
    float h_dInput[16];                       

    std::cout << "\n--- BACKWARD PASS (VJP) ---" << std::endl;
    backMaxPoolWrapKernel(h_dOut, h_dInput, argmax, H, W, P, S);

    std::cout << "Resulting h_dInput Grid:" << std::endl;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            printf("%4.1f ", h_dInput[i * W + j]);
        }
        printf("\n");
    }

    return 0;
}
#endif