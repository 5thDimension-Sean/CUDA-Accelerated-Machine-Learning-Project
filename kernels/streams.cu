//This will use streams to overlap data transfer and computation. It will have 2 versions, one with streams and one without and then compare the speed up of both.


__global__ void stream_kernel(const float* A, float* B, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        B[idx] = A[idx] * 2.0f;
    }
}
