#include "common.cuh"


/*

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_size = 32;
    float sum = 0.0f;
    __shared__ float A_tile[32][32];
    __shared__ float B_tile[32][32];
    for(int i = 0; i < N; i += tile_size){
      A_tile[threadIdx.y][threadIdx.x] = (row < N && (i + threadIdx.x) < N) ? A[row * N + (i + threadIdx.x)] : 0.0f;
      B_tile[threadIdx.y][threadIdx.x] = ((i + threadIdx.y) < N && col < N) ? B[(i + threadIdx.y) * N + col] : 0.0f;
      __syncthreads();
      for(int j = 0; j < tile_size; j++)
          sum += A_tile[threadIdx.y][j] * B_tile[j][threadIdx.x];
      __syncthreads();
    }
    if(row < N && col < N)
        C[row * N + col] = sum;

}
*/

__global__ void sgd_kernel(float *weights, const float *grad, float lr, int n){

}

void fc_forward(const float* X, const float *W, const float *b, float *Y, int batch, int in, int out){
    
}