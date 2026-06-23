#include <cstdio>
#include <cstdlib>
#include "../kernels/matmul.cu"
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

void cpu_baseline(const float* A, const float* B, float* C, int N){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            float sum = 0.0f;
            for(int k = 0; k < N; k++){
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(){
    #define N 1024
    #define Threads 256
    dim3 blockDim(32, 32);
    dim3 gridDim((N + 31) / 32, (N + 31) / 32);


    float* h_A, *h_B, *h_C;
    float* d_A, *d_B, *d_C;
    for(int i = 0; i < N*N; i++){
      h_A[i] = 1.0f;
      h_B[i] = 1.0f;
      h_C[i] = 0.0f;
  }

    CUDA_CHECK(cudaMallocHost((void**)&h_A, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_B, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_C, N * sizeof(float)));
    const int N_RUNS = 2;  // average over 2 runs for time sake
    //CPU Baseline
    float cpu_sum = 0.0f;
    double cpu_total_ms = 0.0;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));


    // input data
    cudaMemset(d_A, 0, N*N*sizeof(float));
    cudaMemset(d_B, 0, N*N*sizeof(float));


    for (int r = 0; r < N_RUNS; r++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_baseline(h_A, h_B, h_C, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_total_ms += std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    }
    double cpu_ms = cpu_total_ms / N_RUNS;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));


    // input data
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_C, 0, N*N*sizeof(float)));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);   // wait for GPU to actually finish
    float naiveMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&naiveMs, start, stop));
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    // input data
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_C, 0, N*N*sizeof(float)));
    cudaEventRecord(start);
    matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);   // wait for GPU to actually finish
    float tiledMs = 0.0f;
    cudaEventElapsedTime(&tiledMs, start, stop);

    printf("Naive Matmul: %f ms\n", naiveMs);
    printf("Tiled Matmul: %f ms\n", tiledMs);
    printf("CPU baseline: %f ms\n", cpu_ms);
    printf("Comparison: Tiled vs CPU: %f x\n", cpu_ms / tiledMs);
    printf("Comparison: Tiled vs Naive: %f x\n", naiveMs / tiledMs);
    printf("Comparison: Naive vs CPU: %f x\n", cpu_ms / naiveMs);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
