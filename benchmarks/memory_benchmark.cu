#include <cstdio>
#include "../kernels/memory_coalescing.cu"
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

void cpu_copy(const float* A, float* B, int n) {
      for (int i = 0; i < n; i++) B[i] = A[i];
}

//Prints cpu baseline, coalesced and uncoalesced read and the comparisons
int main(){
    const int N = 1<<24;
    #define Threads 256
    float* d_B;
    cudaMalloc(&d_B, N * sizeof(float));
    float* h_B = (float*)malloc(N * sizeof(float));
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
    double cpu_total_ms = 0.0;
    for (int r = 0; r < N_RUNS; r++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_copy(h_A, h_B, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_total_ms += std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    }
    double cpu_ms = cpu_total_ms / N_RUNS;
    //coalesced read
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + Threads - 1) / Threads;
    cudaEventRecord(start);
    coalesced_read<<<blocks, Threads>>>(d_A, d_B, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);   // wait for GPU to actually finish
    float coalescedMs = 0.0f;
    cudaEventElapsedTime(&coalescedMs, start, stop);

    //uncoalesced read
    cudaEventRecord(start);
    uncoalesced_read<<<blocks, Threads>>>(d_A, d_B, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);   // wait for GPU to actually finish
    float uncoalescedMs = 0.0f;
    cudaEventElapsedTime(&uncoalescedMs, start, stop);
    //Comparison to CPU with Naive, Shared, and Shared vs Naive
    float coalescedSpeedup = cpu_ms / coalescedMs;
    float uncoalescedSpeedup = cpu_ms / uncoalescedMs;
    float uncoalescedVsCoalescedSpeedup = uncoalescedMs / coalescedMs;
    //Final prints
    printf("--- Coalesced Read---\n");
    printf("GPU time: %.3f ms:\n ", coalescedMs);

    printf("--- Uncoalesced Read ---\n");
    printf("GPU time: %.3f ms:\n ", uncoalescedMs);

    printf("--- CPU baseline ---\n");
    printf("  CPU time : %.3f ms:\n", cpu_ms);


    printf("--- Comparison ---\n");
    printf("Coalesced speedup vs CPU: %.2fx\n", coalescedSpeedup);
    printf("Uncoalesced speedup vs CPU: %.2fx\n", uncoalescedSpeedup);
    printf("Uncoalesced speedup vs Coalesced: %.2fx\n", uncoalescedVsCoalescedSpeedup);
    //free
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_A);
    free(h_B);
    return 0;
}