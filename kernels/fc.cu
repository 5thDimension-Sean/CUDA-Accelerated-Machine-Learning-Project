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


#ifndef BUILD_AS_LIBRARY
int main(){
    //n is num of weights
    const int batch = 2, in = 2, out = 3;
    float X[batch * in] = {1.0f, 2.0f, 3.0f, 4.0f};
    float W[in * out] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    float b[out] = {0.0f, 0.0f, 0.0f};
    float Y[batch * out]; //filled by fc forward
    fc_forward(X, W, b, Y, batch, in, out);
    printf("Output after FC: ");
    for(int i = 0; i < batch * out; ++i){
        printf("%.4f ", Y[i]);
    }
    return 0;
}
#endif