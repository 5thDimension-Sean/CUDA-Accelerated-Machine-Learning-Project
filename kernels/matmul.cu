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
    int tile_size = N/32;
    for(int i = 0; i<N; i+=tile_size){
        __shared__ float A_tile[32][32];
        __shared__ float B_tile[32][32];
        A_tile[threadIdx.y][threadIdx.x] = A[row * N + (i + threadIdx.x)];
        B_tile[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * N + col];
        __syncthreads();
        for(int j = 0; j<tile_size; j++){
            C[row * N + col] += A_tile[threadIdx.y][j] * B_tile[j][threadIdx.x];
        }
        __syncthreads();
    } 
}

