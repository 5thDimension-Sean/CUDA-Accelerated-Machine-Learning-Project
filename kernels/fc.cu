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

__global__ void fc_forward_kernel(const float* X, const float *W, const float *b, float *Y, int batch, int in, int out){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output index

    if(row < batch && col < out){
        float sum = 0.0f;
        for(int i = 0; i < in; ++i){
            sum += X[row * in + i] * W[col*in + i];
        }
        Y[row * out + col] = sum + b[col];
    }
}

__global__ void fc_backward_weights_kernel(const float *dY, const float*X, float*dW, int batch, int in, int out){

}

__global__ void fc_backward_bias_kernel(const float *dY, float*db, int batch, int out){

}


__global__ void fc_backward_input_kernel(const float *dY, const float *W, float *dX, int batch, int in, int out){

}

void fc_forward(const float* X, const float *W, const float *b, float *Y, int batch, int in, int out){
    size_t bytes_X = batch * in  * sizeof(float);
    size_t bytes_W = out   * in  * sizeof(float);
    size_t bytes_b = out * sizeof(float);
    size_t bytes_Y = batch * out * sizeof(float);
    float *d_X, *d_W, *d_b, *d_Y;
    CUDA_CHECK(cudaMalloc(&d_X, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_W, bytes_W));
    CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
    CUDA_CHECK(cudaMalloc(&d_Y, bytes_Y));
    CUDA_CHECK(cudaMemcpy(d_X, X, bytes_X, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W, bytes_W, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes_b, cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((out + block.x - 1) / block.x, (batch + block.y - 1) / block.y);
    fc_forward_kernel<<<grid, block>>>(d_X, d_W, d_b, d_Y, batch, in, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(Y, d_Y, bytes_Y, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_Y));
}

void fc_backward(const float *dY, const float *X, const float *W, float *dW, float *db, float *dX, int batch, int in, int out){
    
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