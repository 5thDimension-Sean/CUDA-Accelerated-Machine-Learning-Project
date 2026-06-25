#include <cstdio>
#include <cstdlib>
#include "../kernels/conv2d.cu"
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)



void cpu_baseline(
      const float* input,
      const float* filter,
      float* output,
      int H, int W,
      int FH, int FW
  ){
      int outH = H - FH + 1;
      int outW = W - FW + 1;

      for (int out_y = 0; out_y < outH; ++out_y) {
          for (int out_x = 0; out_x < outW; ++out_x) {
              float sum = 0.0f;
              for (int fy = 0; fy < FH; ++fy) {
                  for (int fx = 0; fx < FW; ++fx) {
                      sum += input[(out_y + fy) * W + (out_x + fx)] * filter[fy * FW + fx];
                  }
              }
              output[out_y * outW + out_x] = sum;
          }
      }
  }
const int H = 256;
const int W = 256; 
const int FH = 3;
const int FW = 3;

int main(){
        const int outH = H - FH + 1;
        const int outW = W - FW + 1;
        dim3 blockDim(32, 32);
        dim3 gridDim((outW + 31) / 32, (outH + 31) / 32);

        float* h_input;
        float* h_filter;
        float* h_output;
        float* d_input;
        float* d_filter;
        float* d_output;

        CUDA_CHECK(cudaMallocHost(&h_input,  H    * W    * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_filter, FH   * FW   * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_output, outH * outW * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_input,  H    * W    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_filter, FH   * FW   * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, outH * outW * sizeof(float)));
        
        for (int i = 0; i < H * W; i++) {
            h_input[i] = 1.0f;
        }
        for (int i = 0; i < outH * outW; i++) {
            h_output[i] = 0.0f;
        }
        h_filter[0] = 0.0f; h_filter[1] = 0.0f; h_filter[2] = 0.0f;
        h_filter[3] = 0.0f; h_filter[4] = 1.0f; h_filter[5] = 0.0f;
        h_filter[6] = 0.0f; h_filter[7] = 0.0f; h_filter[8] = 0.0f;
        CUDA_CHECK(cudaMemcpy(d_input, h_input, H * W * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_filter, h_filter, FH * FW * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_output, 0, outH * outW * sizeof(float)));
        
        double cpu_ms = 0.0;
        const int N_RUNS = 2;
        double cpu_total_ms = 0.0;
        for (int r = 0; r < N_RUNS; r++) {
            auto cpu_start = std::chrono::high_resolution_clock::now();
            cpu_baseline(h_input, h_filter, h_output, H, W, FH, FW);
            auto cpu_end = std::chrono::high_resolution_clock::now();
            cpu_total_ms += std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        }
        cpu_ms = cpu_total_ms / N_RUNS;

        const int N_RUNS_GPU = 10;
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        float naiveTotal = 0.0f;
        for (int r = 0; r < N_RUNS_GPU; r++) {
            CUDA_CHECK(cudaMemset(d_output, 0, outH * outW * sizeof(float)));
            CUDA_CHECK(cudaEventRecord(start));
            conv2d_naive<<<gridDim, blockDim>>>(d_input, d_filter, d_output, H, W, FH, FW);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            naiveTotal += ms;
        }
        CUDA_CHECK(cudaMemcpy(h_output, d_output, outH * outW * sizeof(float), cudaMemcpyDeviceToHost));

        bool correct = true;
        for (int i = 0; i < outH * outW; i++) {
            if (fabsf(h_output[i] - 1.0f) > 1e-5f) {
                printf("MISMATCH at %d: got %.4f\n", i, h_output[i]);
                correct = false;
                break;
            }
        }
        printf("Correct: %s\n", correct ? "YES" : "NO");
        float naiveMs = naiveTotal / N_RUNS_GPU;

        cudaMemcpyToSymbol(c_filter, h_filter, FH * FW * sizeof(float));
        float constantTotal = 0.0f;
        for (int r = 0; r < N_RUNS_GPU; r++) {
            CUDA_CHECK(cudaMemset(d_output, 0, outH * outW * sizeof(float)));
            CUDA_CHECK(cudaEventRecord(start));
            conv2d_constant<<<gridDim, blockDim>>>(d_input, d_output, H, W, FH, FW);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            constantTotal += ms;
        }
        CUDA_CHECK(cudaMemcpy(h_output, d_output, outH * outW * sizeof(float), cudaMemcpyDeviceToHost));

        correct = true;
        for (int i = 0; i < outH * outW; i++) {
            if (fabsf(h_output[i] - 1.0f) > 1e-5f) {
                printf("MISMATCH at %d: got %.4f\n", i, h_output[i]);
                correct = false;
                break;
            }
        }
        float constantMs = constantTotal / N_RUNS_GPU;
        printf("Correct: %s\n", correct ? "YES" : "NO");
        printf("--- Naive Conv2D ---\n");
        printf("GPU time: %.3f ms\n", naiveMs);
        printf("--- CPU Baseline ---\n");
        printf("CPU time: %.3f ms\n", cpu_ms);
        printf("--- Constant Conv2D ---\n");
        printf("GPU time: %.3f ms\n", constantMs);
        printf("--- Comparison ---\n");
        printf("Naive speedup over cpu = %.2fx\n", cpu_ms / naiveMs);
        printf("Constant / CPU = %.2fx\n", cpu_ms / constantMs);
        printf("Constant speed up over naive = %.2fx\n", naiveMs/constantMs);


        CUDA_CHECK(cudaFreeHost(h_input));
        CUDA_CHECK(cudaFreeHost(h_filter));
        CUDA_CHECK(cudaFreeHost(h_output));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_filter));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return 0;
}

    