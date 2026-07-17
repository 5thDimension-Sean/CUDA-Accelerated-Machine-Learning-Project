// ============================================================================
// conv2d.cu — Week 4
// Goal: Implement 2D convolution in CUDA.
//
// You will build this in stages:
//   Stage 1 — Naive:          one thread per output pixel, reads from global memory
//   Stage 2 — Constant memory: move the filter kernel to __constant__ memory
//   Stage 3 — Shared memory:  load input tiles + halo cells into shared memory
//
// Test cases to verify correctness:
//   - Identity filter [[0,0,0],[0,1,0],[0,0,0]] → output = input
//   - Blur filter (all 1/9)                     → gaussian blur
//   - Sobel filter                              → edge detection
//
// Resources:
//   - CUDA Programming Guide, Ch. 3 (constant memory)
//   - "GPU Gems" Ch. 39 (convolution optimization)
// ============================================================================

// TODO — Week 4: implement here
//4 parameters k f s p 1 
#include "common.cuh"
#include <cmath>
#include <cstdio>
#define MAX_FILTER_SIZE 49
__constant__ float c_filter[MAX_FILTER_SIZE];
  __global__ void conv2d_naive(
      const float* input,
      const float* filter,
      float* output,
      int H, int W,
      int FH, int FW
  ){
      int outH = H - FH + 1;
      int outW = W - FW + 1;
    //output height/width
      int out_x = blockIdx.x * blockDim.x + threadIdx.x;
      int out_y = blockIdx.y * blockDim.y + threadIdx.y;

      if (out_x >= outW || out_y >= outH) return;

      float sum = 0.0f;
      for (int fy = 0; fy < FH; ++fy) {
          for (int fx = 0; fx < FW; ++fx) {
              sum += input[(out_y + fy) * W + (out_x + fx)] * filter[fy * FW + fx];
          }
      }

      output[out_y * outW + out_x] = sum;
  }
  //The main difference between naive and this is that this uses a different filter, but with the 
  __global__ void conv2d_constant(
    const float* input,
      float* output,
      int H, int W,
      int FH, int FW){
        int outH = H - FH + 1;
        int outW = W - FW + 1;
        //output height/width
        int out_x = blockIdx.x * blockDim.x + threadIdx.x;
        int out_y = blockIdx.y * blockDim.y + threadIdx.y;
        if (out_x >= outW || out_y >= outH) return;
        float sum = 0.0f;
        for (int fy = 0; fy < FH; ++fy) {
          for (int fx = 0; fx < FW; ++fx) {
              sum += input[(out_y + fy) * W + (out_x + fx)] * c_filter[fy * FW + fx];
          }
      }

      output[out_y * outW + out_x] = sum;

    }


__global__ void conv2d_shared(
    const float* input,
    float* output,
    int H, int W,
    int FH, int FW
) {
    extern __shared__ float shared_input[];

    int outH = H - FH + 1;
    int outW = W - FW + 1;

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tile_width = blockDim.x + FW - 1;
    int tile_height = blockDim.y + FH - 1;

    int shared_x = threadIdx.x;
    int shared_y = threadIdx.y;

    for (int fy = 0; fy < FH; ++fy) {
        for (int fx = 0; fx < FW; ++fx) {
            int global_x = out_x + fx;
            int global_y = out_y + fy;
            if (global_x < W && global_y < H) {
                shared_input[(shared_y + fy) * tile_width + (shared_x + fx)] =
                    input[global_y * W + global_x];
            } else {
                shared_input[(shared_y + fy) * tile_width + (shared_x + fx)] = 0.0f; // Zero padding
            }
        }
    }

    __syncthreads();

    if (out_x >= outW || out_y >= outH) return;

    float sum = 0.0f;
    for (int fy = 0; fy < FH; ++fy) {
        for (int fx = 0; fx < FW; ++fx) {
            sum += shared_input[(shared_y + fy) * tile_width + (shared_x + fx)] * c_filter[fy * FW + fx];
        }
    }

    output[out_y * outW + out_x] = sum;
}


__global__ void conv2d_filter(const float* input,
    const float* filters,
    float* output,
    int H, int W,
    int C,
    int FH, int FW){
      int outH = H - FH + 1;
      int outW = W - FW + 1;
    //output height/width
      int out_x = blockIdx.x * blockDim.x + threadIdx.x;
      int out_y = blockIdx.y * blockDim.y + threadIdx.y;
      int out_z = blockIdx.z * blockDim.z + threadIdx.z;
      if (out_x >= outW || out_y >= outH || out_z >= C) return;

      float sum = 0.0f;
      for (int fy = 0; fy < FH; ++fy) {
          for (int fx = 0; fx < FW; ++fx) {
              sum += input[out_z * H * W + (out_y + fy) * W + (out_x + fx)] * filters[out_z * FH * FW + fy * FW + fx];
          }
      }

      output[out_z * outH * outW + out_y * outW + out_x] = sum;
  }


__global__ void conv2d_backward_filter(const float *dOut, const float *input,
                                       float *dFilter, int H, int W, int FH, int FW){
    int outH = H - FH + 1;
    int outW = W - FW + 1;

    // one thread per FILTER element, not per output pixel
    int fx = blockIdx.x * blockDim.x + threadIdx.x;   // filter column
    int fy = blockIdx.y * blockDim.y + threadIdx.y;   // filter row
    if (fx >= FW || fy >= FH) return;

    // sum over ALL output positions
    float sum = 0.0f;
    for (int oy = 0; oy < outH; ++oy) {
        for (int ox = 0; ox < outW; ++ox) {
            sum += dOut[oy * outW + ox] * input[(oy + fy) * W + (ox + fx)];
        }
    }
    dFilter[fy * FW + fx] = sum;
}

__global__ void conv2d_backward_input(const float *dOut, const float *filter,
                                      float *dInput, int H, int W, int FH, int FW){
    int outH = H - FH + 1;
    int outW = W - FW + 1;

    // one thread per INPUT pixel (gather)
    int ix = blockIdx.x * blockDim.x + threadIdx.x;   // input column
    int iy = blockIdx.y * blockDim.y + threadIdx.y;   // input row
    if (ix >= W || iy >= H) return;

    float sum = 0.0f;
    for (int fy = 0; fy < FH; ++fy) {
        for (int fx = 0; fx < FW; ++fx) {
            int oy = iy - fy;      // which output pixel used this input via filter[fy][fx]
            int ox = ix - fx;
            if (oy >= 0 && oy < outH && ox >= 0 && ox < outW) {
                sum += dOut[oy * outW + ox] * filter[fy * FW + fx];
            }
        }
    }
    dInput[iy * W + ix] = sum;
}
void conv2d_backward(const float *dOut, const float *input, const float *filter,
                     float *dFilter, float *dInput, int H, int W, int FH, int FW) {
    int outH = H - FH + 1, outW = W - FW + 1;

    size_t bytes_dOut   = outH * outW * sizeof(float);
    size_t bytes_input  = H * W       * sizeof(float);   // also dInput's si
    size_t bytes_input  = H * W       * sizeof(float);   // also dInput's size
    size_t bytes_filter = FH * FW     * sizeof(float);   // also dFilter's size

    float *d_dOut, *d_input, *d_filter, *d_dFilter, *d_dInput;
    CUDA_CHECK(cudaMalloc(&d_dOut,    bytes_dOut));
    CUDA_CHECK(cudaMalloc(&d_input,   bytes_input));
    CUDA_CHECK(cudaMalloc(&d_filter,  bytes_filter));
    CUDA_CHECK(cudaMalloc(&d_dFilter, bytes_filter));   // output
    CUDA_CHECK(cudaMalloc(&d_dInput,  bytes_input));    // output

    // inputs go UP (H2D)
    CUDA_CHECK(cudaMemcpy(d_dOut,   dOut,   bytes_dOut,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input,  input,  bytes_input,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, filter, bytes_filter, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 filter_grid((FW + 15) / 16, (FH + 15) / 16);
    dim3 input_grid ((W  + 15) / 16, (H  + 15) / 16);

    conv2d_backward_filter<<<filter_grid, block>>>(d_dOut, d_input,  d_dFilter, H, W, FH, FW);
    conv2d_backward_input <<<input_grid,  block>>>(d_dOut, d_filter, d_dInput,  H, W, FH, FW);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // outputs come DOWN (D2H)
    CUDA_CHECK(cudaMemcpy(dFilter, d_dFilter, bytes_filter, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dInput,  d_dInput,  bytes_input,  cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dOut));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_dFilter));
    CUDA_CHECK(cudaFree(d_dInput));
}
