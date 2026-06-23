//instead of IDX it would be row and col meaning it has x and y
__global__ void matmul_naive(const float* A, const float* B, float* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for(int i =0; i < N && row < N && col < N; i++){
        sum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_size = 32;
    float sum = 0.0f;
    __shared__ float A_tile[32][32];
    __shared__ float B_tile[32][32];
    for(int i = 0; i<N && row < N && col < N; i+=tile_size){
        A_tile[threadIdx.y][threadIdx.x] = A[row * N + (i + threadIdx.x)];
        B_tile[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * N + col];
        __syncthreads();
        for(int j = 0; j<tile_size; j++){
            sum += A_tile[threadIdx.y][j] * B_tile[j][threadIdx.x];
        }
        __syncthreads();
    } 
    C[row * N + col] = sum;
}

