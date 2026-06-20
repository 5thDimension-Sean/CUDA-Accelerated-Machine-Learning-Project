// ============================================================================
// vector_add.cu — Week 1
// Goal: Add two vectors A + B = C on the GPU, time it, compare to CPU.
//
// Concepts covered:
//   - CUDA kernel syntax (__global__, threadIdx, blockIdx, blockDim)
//   - Memory management (cudaMalloc, cudaMemcpy, cudaFree)
//   - Error checking (CUDA_CHECK macro)
//   - Timing with cudaEvent_t
//   - CPU vs GPU result verification
// ============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

// ── Error checking macro ─────────────────────────────────────────────────────
// Wrap every CUDA call in this. It prints exactly where and why things break.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ── Constants ────────────────────────────────────────────────────────────────
const int N = 1 << 24; //global constant = 16 million elements

#define THREADS   1024        // threads per block (must be multiple of 32 — why? warps)

// ── GPU Kernel ───────────────────────────────────────────────────────────────
// __global__ means: runs on GPU, called from CPU
// Each thread computes ONE element of the output — that's the parallel part
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int n) {
    // Which element does THIS thread handle?
    // blockIdx.x  = which block we're in
    // blockDim.x  = how many threads per block (256)
    // threadIdx.x = which thread within the block (0–255)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: don't go out of bounds if N isn't divisible by THREADS
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// ── CPU Reference ────────────────────────────────────────────────────────────
// We'll compare GPU output against this to verify correctness
void vector_add_cpu(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

// ── Verify Results ───────────────────────────────────────────────────────────
bool verify(const float* cpu, const float* gpu, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(cpu[i] - gpu[i]) > 1e-5f) {
            printf("MISMATCH at index %d: CPU=%.6f GPU=%.6f\n", i, cpu[i], gpu[i]);
            return false;
        }
    }
    return true;
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main() {
    printf("Vector Addition: N = %d elements (%.1f MB per vector)\n\n",
           N, N * sizeof(float) / 1e6f);

    // ── Allocate host (CPU) memory ───────────────────────────────────────────
    float* h_A   = (float*)malloc(N * sizeof(float));
    float* h_B   = (float*)malloc(N * sizeof(float));
    float* h_C   = (float*)malloc(N * sizeof(float));   // GPU result lands here
    float* h_ref = (float*)malloc(N * sizeof(float));   // CPU reference

    // Fill A and B with test data
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.001f;
        h_B[i] = (float)i * 0.002f;
    }

    // ── Allocate device (GPU) memory ─────────────────────────────────────────
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    // ── Copy input data from CPU → GPU ───────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

        // ── Timing config ────────────────────────────────────────────────────────
    const int N_RUNS = 10;  // average over 10 runs for stable numbers

    // ── GPU timing (averaged) ────────────────────────────────────────────────
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int blocks = (N + THREADS - 1) / THREADS;
    printf("Launching kernel: %d blocks * %d threads = %d total threads\n\n",
           blocks, THREADS, blocks * THREADS);

    // Warmup run (GPU clocks need to ramp up — don't count this one)
    vector_add_kernel<<<blocks, THREADS>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_total_ms = 0.0f;
    for (int r = 0; r < N_RUNS; r++) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(start));
        vector_add_kernel<<<blocks, THREADS>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        gpu_total_ms += ms;
    }
    float gpu_ms = gpu_total_ms / N_RUNS;
    CUDA_CHECK(cudaGetLastError());

    // Copy result back once after timing
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // ── CPU timing (averaged) ────────────────────────────────────────────────
    double cpu_total_ms = 0.0;
    for (int r = 0; r < N_RUNS; r++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        vector_add_cpu(h_A, h_B, h_ref, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_total_ms += std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    }
    double cpu_ms = cpu_total_ms / N_RUNS;

    // ── Verify GPU result matches CPU ────────────────────────────────────────
    bool correct = verify(h_ref, h_C, N);

    // ── Print results ────────────────────────────────────────────────────────
    printf("Results (averaged over %d runs):\n", N_RUNS);
    printf("  GPU time : %.3f ms\n", gpu_ms);
    printf("  CPU time : %.3f ms\n", cpu_ms);
    printf("  Speedup  : %.2fx\n", cpu_ms / gpu_ms);
    printf("  Correct  : %s\n\n", correct ? "YES" : "NO");

    printf("Memory bandwidth (GPU): %.1f GB/s\n",
           (3.0f * N * sizeof(float)) / (gpu_ms * 1e6f));
    // ── Cleanup ──────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_A); free(h_B); free(h_C); free(h_ref);

    return 0;
}

// ── How to build and run ─────────────────────────────────────────────────────
// From the project root:
//   mkdir build && cd build
//   cmake ..
//   cmake --build . --config Release
//   ./bin/vector_add          (Linux/Mac)
//   .\bin\Release\vector_add  (Windows)
//
// Expected output (numbers will vary by GPU):
//   GPU time : ~1.5 ms
//   CPU time : ~30 ms
//   Speedup  : ~20x
//   Correct  : YES 
