//instead of IDX it would be row and col meaning it has x and y
__global__ void matmul_naive(const float* A, const float* B, float* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i =0; i < N; i++){
        C[row * N + col] += A[row * N + i] * B[i * N + col];
    }
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
}

