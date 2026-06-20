//2 versions one will be naive and the other will be shared memory tree reduction
/*
 * A = input array of the gpu
 * result = a single float on the gpu and is initialized to 0 before the kernel runs
 * n = number of elements 
*/
/*
blockIdx.x = block index
blockIdx.x = position within block
blockDim.x = Threads per block
*/

//this will add up all of the elements one at a time... very slow and not efficent at all
__global__ void reduce_naive(const float*A, float* result, int n){
    //it would be block num * block elemtn num + offset within the block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        atomicAdd(result, A[idx]);
    }
}
//faster as it moves through shared memory by takin O(log n) rounds rather than O(n)
__global__ void reduce_shared(const float*A, float* result, int n){
    extern __shared__ float sdata[]; //shared memory for the block
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    //load data into shared memory
    sdata[tid] = (idx < n) ? A[idx] : 0.0f; //handle out of bounds
    __syncthreads(); //make sure all threads have loaded their data
    for(int i = blockDim.x/2;i > 0; i/=2){
        if(tid<i){
            sdata[tid]  += sdata[tid+i];
            __syncthreads(); //wait for all threads to finish adding before next round
            if(tid == 0){
                atomicAdd(result, sdata[0]); //add the block's sum to the global result
            }
        }
    }
}