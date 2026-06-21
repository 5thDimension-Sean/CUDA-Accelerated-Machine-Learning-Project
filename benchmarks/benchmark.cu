#include <cstdio>
#include "../kernels/reduction.cu"
#include <chrono>
#include <cuda_runtime.h>

__global__ void reduceNaive(const float* input, float* output, int n);

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

bool verify(const float* cpu, const float* gpu, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(cpu[i] - gpu[i]) > 1e-5f) {
            printf("MISMATCH at index %d: CPU=%.6f GPU=%.6f\n", i, cpu[i], gpu[i]);
            return false;
        }
    }
    return true;
}

void vector_add_cpu(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}
//This will print both the Naive and Shared reduction's CPU & GPU and then compare the speed up of both
int main() {
    const int N = 1<<24;
    #define Threads 256
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));
    //cpu memory allocation
    float* h_A   = (float*)malloc(N * sizeof(float));
    float* h_B   = (float*)malloc(N * sizeof(float));
    float* h_C   = (float*)malloc(N * sizeof(float));   // GPU result lands here
    float* h_ref = (float*)malloc(N * sizeof(float));   // CPU reference
    //gpu memory allocation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    // ── Copy input data from CPU → GPU ───────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

        // ── Timing config ────────────────────────────────────────────────────────
    const int N_RUNS = 10;  // average over 10 runs for stable numbers
    //CPU Baseline
    double cpu_total_ms = 0.0;
    for (int r = 0; r < N_RUNS; r++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        vector_add_cpu(h_A, h_B, h_ref, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_total_ms += std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    }
    double cpu_ms = cpu_total_ms / N_RUNS;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));
    // Verifying
    bool correct = verify(h_ref, h_C, N);
    //Naive Reduction
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + Threads - 1) / Threads;
    cudaEventRecord(start);
    reduce_naive<<<blocks, Threads>>>(d_A, d_result, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);   // wait for GPU to actually finish

    float naiveMs = 0.0f;
    cudaEventElapsedTime(&naiveMs, start, stop);
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    //Shared Reduction
    cudaEventRecord(start);
    reduce_shared<<<blocks, Threads>>>(d_A, d_result, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);   // wait for GPU to actually finish

    float sharedMs = 0.0f;
    cudaEventElapsedTime(&sharedMs, start, stop);
    cudaEventElapsedTime(&naiveMs, start, stop);
    //Comparison to CPU with Naive, Shared, and Shared vs Naive
    float naiveSpeedup = cpu_ms / naiveMs;
    float sharedSpeedup = cpu_ms / sharedMs;
    float sharedVsNaiveSpeedup = naiveMs / sharedMs;
    //Final prints
    printf("--- Naive Reduction ---\n");
    printf("GPU time: %.3f ms\n: ", naiveMs);

    printf("--- Shared Reduction ---\n");
    printf("GPU time: %.3f ms\n: ", sharedMs);

    printf("--- CPU baseline ---\n");
    printf("  CPU time : %.3f ms\n", cpu_ms);


    printf("--- Comparison ---\n");
    printf("Naive speedup vs CPU: %.2fx\n", naiveSpeedup);
    printf("Shared speedup vs CPU: %.2fx\n", sharedSpeedup);
    printf("Shared speedup vs Naive: %.2fx\n", sharedVsNaiveSpeedup);

    //free
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}