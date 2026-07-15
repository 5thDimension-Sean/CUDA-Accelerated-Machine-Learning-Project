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
    
    int row = blockIdx.x * blockDim.x + threadIdx.x; // batch index
    int col = blockIdx.y * blockDim.y + threadIdx.y; // output index

    if(row < batch && col < out){
        float sum = 0.0f;
        for(int i = 0; i < in; ++i){
            sum += X[row * in + i] * W[col*in + i];
        }
        Y[row * out + col] = sum + b[col];
    }
}

__global__ void fc_backward_weights_kernel(const float *dY, const float*X, float*dW, int batch, int in, int out){
    int row = blockIdx.y * blockDim.y + threadIdx.y; // output index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // input index

    if(row < out && col < in){
        float sum = 0.0f;
        for(int i = 0; i < batch; ++i){
            sum += dY[i * out + row] * X[i * in + col];
        }
        dW[row * in + col] = sum;
    }
}

__global__ void fc_backward_bias_kernel(const float *dY, float*dB, int batch, int out){
    int row = blockIdx.x * blockDim.x + threadIdx.x; // output index

    if(row < out){
        float sum = 0.0f;
        for(int i = 0; i < batch; ++i){
            sum += dY[i * out + row];
        }
        dB[row] = sum;
    }
}


__global__ void fc_backward_input_kernel(const float *dY, const float *W, float *dX, int batch, int in, int out){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < batch && col < in){
        float sum = 0;
        for(int i = 0; i < out; ++i){
            sum += dY[row * out + i] * W[i * in + col];
        }
        dX[row * in + col] = sum; 
    }
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
    dim3 grid((batch + block.x - 1) / block.x, (out + block.y - 1) / block.y);
    fc_forward_kernel<<<grid, block>>>(d_X, d_W, d_b, d_Y, batch, in, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(Y, d_Y, bytes_Y, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_Y));
}

void fc_backward(const float *dY, const float *X, const float *W, float *dW, float *dB, float *dX, int batch, int in, int out){
    size_t bytes_X = batch * in  * sizeof(float);
    size_t bytes_W = out   * in  * sizeof(float);
    size_t d_bytes_W = out   * in  * sizeof(float);
    size_t d_bytes_b = out * sizeof(float);
    size_t d_bytes_Y = batch * out * sizeof(float);
    size_t d_bytes_X = batch*in*sizeof(float);
    float *d_X, *d_W, *dd_B, *dd_Y, *dd_W, *dd_X;
    CUDA_CHECK(cudaMalloc(&d_X, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_W, bytes_W));
    CUDA_CHECK(cudaMalloc(&dd_B, d_bytes_b));
    CUDA_CHECK(cudaMalloc(&dd_Y, d_bytes_Y));
    CUDA_CHECK(cudaMalloc(&dd_W, d_bytes_W));
    CUDA_CHECK(cudaMalloc(&dd_X, d_bytes_X));
    CUDA_CHECK(cudaMemcpy(d_X, X, bytes_X, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W, bytes_W, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd_Y, dY, d_bytes_Y, cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 weights_grid((in + 15) / 16, (out + 15) / 16);
    dim3 bias_grid((out + 15) / 16);
    dim3 input_grid((batch + 15) / 16, (in + 15) / 16);

    fc_backward_weights_kernel<<<weights_grid, block>>>(dd_Y, d_X, dd_W, batch, in, out);
    fc_backward_bias_kernel<<<bias_grid, 16>>>(dd_Y, dd_B, batch, out);
    fc_backward_input_kernel<<<input_grid, block>>>(dd_Y, d_W, dd_X, batch, in, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dW, dd_W, d_bytes_W, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dB, dd_B, d_bytes_b, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dX, dd_X, d_bytes_X, cudaMemcpyDeviceToHost));
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(dd_B);
    cudaFree(dd_Y);
    cudaFree(dd_W);
    cudaFree(dd_X);

}

#ifndef BUILD_AS_LIBRARY
int main(){
    //n is num of weights
    const int batch = 2, in = 2, out = 3;
    float X[batch * in] = {1.0f, 2.0f, 3.0f, 4.0f};
    float W[in * out] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    float b[out] = {0.0f, 0.0f, 0.0f};
    float Y[batch * out]; //filled by fc forward
    float dW[out*in] = {0};
    float dB[out] = {0};
    float dX[batch*in] = {0};
    float dY[batch * out] = {1, 2, 3, 4, 5, 6};

    fc_forward(X, W, b, Y, batch, in, out);
    printf("Output after FC: ");
    for(int i = 0; i < batch * out; ++i){
        printf("%.4f ", Y[i]);
    }
    fc_backward(dY, X, W, dW, dB, dX, batch, in, out);
    printf("Backwards FC: ");
    for(int i = 0; i < out*in; ++i){
        printf("dW[%d]: %.4f ", i, dW[i]);
    }
    printf("\n");
    for(int i = 0; i < out; ++i){
        printf("dB[%d]: %.4f ", i, dB[i]);
    }
    printf("\n");
    for(int i = 0; i < batch*in; ++i){
        printf("dX[%d]: %.4f ", i, dX[i]);
    }
    printf("\n");
    return 0;
}
#endif