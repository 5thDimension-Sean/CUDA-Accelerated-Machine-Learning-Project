#include <cstdio>
#include "../kernels/streams.cu"
#include <chrono>
#include <cuda_runtime.h>


//run sync and run async. async will be the one using streams
const int N = 1<<24;
#define Threads 256
int blocks = (N + Threads - 1) / Threads;
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

void run_sync(const float* d_A, float* d_B, int n){
    int blocks = (n + Threads - 1) / Threads;
    stream_kernel<<<blocks, Threads>>>(d_A, d_B, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
void run_with_streams(const float* d_A, float* d_B, int n, int num_streams, cudaStream_t* streams){
    int stream_size = (n + num_streams - 1) / num_streams;
    for(int i = 0; i < num_streams; i++){
        int offset = i * stream_size;
        int current_size = min(stream_size, n - offset);
        stream_kernel<<<(current_size + Threads - 1) / Threads, Threads, 0, streams[i]>>>(d_A + offset, d_B + offset, current_size);
    }
    for(int i = 0; i < num_streams; i++){
          cudaStreamSynchronize(streams[i]);
      }
}

int main(){
    float* d_A;
    float* d_B;
    float* h_A   = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.001f;
    }
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));


    // input data
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    int num_streams = 4; // default number of streams
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    run_sync(d_A, d_B, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);   // wait for GPU to actually finish
    float syncMs = 0.0f;
    cudaEventElapsedTime(&syncMs, start, stop);

    cudaStream_t* streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    for(int i = 0; i < num_streams; i++){
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    cudaEventRecord(start);
    run_with_streams(d_A, d_B, N, num_streams, streams);
    cudaEventRecord(stop);
 
    cudaEventSynchronize(stop);   // wait for GPU to actually finish
    float asyncMs = 0.0f;
    cudaEventElapsedTime(&asyncMs, start, stop);
    for(int i = 0; i < num_streams; i++)
      cudaStreamDestroy(streams[i]);

    float compared = syncMs/asyncMs;
    printf("Synchronous version: %.3f ms\n", syncMs);
    printf("Asynchronous version with streams: %.3f ms\n", asyncMs);
    printf("Streams speedup vs Sync: %.2fx\n", compared);
    printf("Number of streams used: %d\n", num_streams);
    free(h_A);
    free(streams);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}