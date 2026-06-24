//instead of IDX it would be row and col meaning it has x and y
__global__ void matmul_naive(const float* A, const float* B, float* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if(row < N && col < N){
        for(int i =0; i < N; i++){
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

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

__global__ void float4_vectorized(const float* A, const float* B, float* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_size = 128;
    float sum = 0.0f;
    __shared__ float A_tile[32][128];
    __shared__ float B_tile[32][128];
    for(int i = 0; i < N; i += tile_size){
        if(i+threadIdx.x * 4 < N && row < N && i+threadIdx.y < N && col*4 < N){
            float4 a4 = reinterpret_cast<const float4*>(A)[(row * N + i + threadIdx.x * 4) / 4];
            A_tile[threadIdx.y][threadIdx.x * 4 + 0] = a4.x;
            A_tile[threadIdx.y][threadIdx.x * 4 + 1] = a4.y;
            A_tile[threadIdx.y][threadIdx.x * 4 + 2] = a4.z;
            A_tile[threadIdx.y][threadIdx.x * 4 + 3] = a4.w;
            float4 b4 = reinterpret_cast<const float4*>(B)[((i + threadIdx.y) * N + col * 4) / 4];
            B_tile[threadIdx.y][threadIdx.x * 4 + 0] = b4.x;
            B_tile[threadIdx.y][threadIdx.x * 4 + 1] = b4.y;
            B_tile[threadIdx.y][threadIdx.x * 4 + 2] = b4.z;
            B_tile[threadIdx.y][threadIdx.x * 4 + 3] = b4.w;
        }else{
            A_tile[threadIdx.y][threadIdx.x * 4 + 0] = 0.0f;
            A_tile[threadIdx.y][threadIdx.x * 4 + 1] = 0.0f;
            A_tile[threadIdx.y][threadIdx.x * 4 + 2] = 0.0f;
            A_tile[threadIdx.y][threadIdx.x * 4 + 3] = 0.0f;
            B_tile[threadIdx.y][threadIdx.x * 4 + 0] = 0.0f;
            B_tile[threadIdx.y][threadIdx.x * 4 + 1] =  0.0f;
            B_tile[threadIdx.y][threadIdx.x * 4 + 2] = 0.0f;
            B_tile[threadIdx.y][threadIdx.x * 4 + 3] = 0.0f;    
        }
            __syncthreads();
            for(int j = 0; j < tile_size; j++)
                sum += A_tile[threadIdx.y][j] * B_tile[j][threadIdx.x];
            __syncthreads();
        }
    
    if(row < N && col < N)
        C[row * N + col] = sum;
}

