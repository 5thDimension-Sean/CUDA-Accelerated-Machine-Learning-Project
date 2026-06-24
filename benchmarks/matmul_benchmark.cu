#include <cstdio>
#include <cstdlib>
#include "../kernels/matmul.cu"
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 2048

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

void cpu_baseline(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    dim3 blockDim(32, 32);
    dim3 gridDim((N + 31) / 32, (N + 31) / 32);

    // allocate host memory
    float* h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost((void**)&h_A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_C, N * N * sizeof(float)));

    // fill with data
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }

    // allocate device memory
    float* d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(float)));

    // copy input to GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice));

    // CPU baseline
    const int N_RUNS = 2;
    double cpu_total_ms = 0.0;
    for (int r = 0; r < N_RUNS; r++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_baseline(h_A, h_B, h_C, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_total_ms += std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    }
    double cpu_ms = cpu_total_ms / N_RUNS;

    const int N_RUNS_GPU = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // naive matmul
    float naiveTotal = 0.0f;
    for (int r = 0; r < N_RUNS_GPU; r++) {
        CUDA_CHECK(cudaMemset(d_C, 0, N * N * sizeof(float)));
        cudaEventRecord(start);
        matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        naiveTotal += ms;
    }
    float naiveMs = naiveTotal / N_RUNS_GPU;

    // tiled matmul
    float tiledTotal = 0.0f;
    for (int r = 0; r < N_RUNS_GPU; r++) {
        CUDA_CHECK(cudaMemset(d_C, 0, N * N * sizeof(float)));
        cudaEventRecord(start);
        matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        tiledTotal += ms;
    }
    float tiledMs = tiledTotal / N_RUNS_GPU;

    //cuBlas
    float cublasMs = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    for (int r = 0; r < N_RUNS_GPU; r++) {
        CUDA_CHECK(cudaMemset(d_C, 0, N * N * sizeof(float)));
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        cublasMs += ms;
    }
    cublasMs /= N_RUNS_GPU;
    cublasDestroy(handle);
    // print results
    printf("--- Naive Matmul ---\n");
    printf("GPU time: %.3f ms\n", naiveMs);
    printf("--- Tiled Matmul ---\n");
    printf("GPU time: %.3f ms\n", tiledMs);
    printf("--- CPU Baseline ---\n");
    printf("CPU time: %.3f ms\n", cpu_ms);
    printf("--- cuBlas Matmul ---\n");
    printf("GPU time: %.3f ms\n", cublasMs);
    printf("--- Comparison ---\n");
    printf("Tiled vs CPU:   %.2fx\n", cpu_ms / tiledMs);
    printf("Tiled vs Naive: %.2fx\n", naiveMs / tiledMs);
    printf("Naive vs CPU:   %.2fx\n", cpu_ms / naiveMs);
    printf("cuBlas vs CPU:  %.2fx\n", cpu_ms / cublasMs);
    printf("cuBlas vs Tiled: %.2fx\n", tiledMs / cublasMs);
    printf("cuBlas vs Naive: %.2fx\n", naiveMs / cublasMs);

    // free
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
