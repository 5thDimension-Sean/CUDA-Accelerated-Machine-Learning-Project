//This file will be using memory coalescing It will have 2 versions coalesced vs uncoalesced. 

__global__ void coalesced_read(float* A, float* B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        B[idx] = A[idx];
    }
}

  __global__ void uncoalesced_read(float* A, float* B, int n) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int lane = idx % 32;
      int warp = idx / 32;
      int strided_idx = lane * (n / 32) + warp;
      if(idx < n && strided_idx < n){
          B[idx] = A[strided_idx];
      }
  }
