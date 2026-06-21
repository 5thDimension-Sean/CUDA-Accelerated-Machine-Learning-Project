#include <cstdio>
#include "../kernels/reduction.cu"
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

float cpu_reduce(const float* A, int n) {
      float sum = 0.0f;
      for (int i = 0; i < n; i++) sum += A[i];
      return sum;
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
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.001f;
    }
    //gpu memory allocation
    float *d_A;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));

    // input data
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));

        //Timings
    const int N_RUNS = 10;  // average over 10 runs for stable numbers
    //CPU Baseline
    float cpu_sum = 0.0f;
    double cpu_total_ms = 0.0;
    for (int r = 0; r < N_RUNS; r++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_sum = cpu_reduce(h_A, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_total_ms += std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    }
    double cpu_ms = cpu_total_ms / N_RUNS;
    cudaMemset(d_result, 0, sizeof(float));
    //Naive Reduction
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + Threads - 1) / Threads;
    cudaEventRecord(start);
    reduce_naive<<<blocks, Threads>>>(d_A, d_result, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);   // wait for GPU to actually finish
    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    bool correct = fabsf(cpu_sum - h_result) < 1.0f;
    printf("Correct: %s\n", correct ? "YES" : "NO");
    float naiveMs = 0.0f;
    cudaEventElapsedTime(&naiveMs, start, stop);
    cudaMemset(d_result, 0, sizeof(float));

    //Shared Reduction
    cudaEventRecord(start);
    reduce_shared<<<blocks, Threads, Threads * sizeof(float)>>>(d_A, d_result, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);   // wait for GPU to actually finish

    float sharedMs = 0.0f;
    cudaEventElapsedTime(&sharedMs, start, stop);
    //Comparison to CPU with Naive, Shared, and Shared vs Naive
    float naiveSpeedup = cpu_ms / naiveMs;
    float sharedSpeedup = cpu_ms / sharedMs;
    float sharedVsNaiveSpeedup = naiveMs / sharedMs;
    //Final prints
    printf("--- Naive Reduction ---\n");
    printf("GPU time: %.3f ms:\n ", naiveMs);

    printf("--- Shared Reduction ---\n");
    printf("GPU time: %.3f ms:\n ", sharedMs);

    printf("--- CPU baseline ---\n");
    printf("  CPU time : %.3f ms:\n", cpu_ms);


    printf("--- Comparison ---\n");
    printf("Naive speedup vs CPU: %.2fx\n", naiveSpeedup);
    printf("Shared speedup vs CPU: %.2fx\n", sharedSpeedup);
    printf("Shared speedup vs Naive: %.2fx\n", sharedVsNaiveSpeedup);

    //free
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_A);
    return 0;
}