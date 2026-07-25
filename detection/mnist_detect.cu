#include "common.cuh"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>

__global__ void conv2d_mc_forward(const float*, const float*, const float*, float*, int,int,int,int,int,int);
__global__ void maxPool2D(const float*, float*, int*, int,int,int,int,int,int,int);
__global__ void backMaxPool2D(const float*, const int*, float*, int,int,int);
__global__ void conv2d_mc_backward_bias   (const float*, float*, int,int,int);
__global__ void conv2d_mc_backward_weights (const float*, const float*, float*, int,int,int,int,int,int);
__global__ void conv2d_mc_backward_input   (const float*, const float*, float*, int,int,int,int,int,int);
__global__ void fc_forward_kernel         (const float*, const float*, const float*, float*, int,int,int);
__global__ void fc_backward_weights_kernel(const float*, const float*, float*, int,int,int);
__global__ void fc_backward_bias_kernel   (const float*, float*, int,int);
__global__ void fc_backward_input_kernel  (const float*, const float*, float*, int,int,int);
__global__ void sgd_kernel(float*, const float*, float, int);

__global__ void relu_forward(const float* in, float* out, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] > 0.f ? in[i] : 0.f;
}

__global__ void relu_backward(const float* dOut, const float* preact, float* dIn, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) dIn[i] = preact[i] > 0.f ? dOut[i] : 0.f;   // mask by pre-activation
}


__global__ void detection_loss_grad(const float* pred, const float* target, float* dY, float* loss, int S, int C, float lam_coord, float lam_noobj){

}

struct Net {
    float *conv1_f, *conv1_b;
    float *conv2_f, *conv2_b;
    float *conv3_f, *conv3_b;
    float *fc_W,    *fc_b;
};

struct Grads {
    float *conv1_f, *conv1_b;
    float *conv2_f, *conv2_b;
    float *conv3_f, *conv3_b;
    float *fc_W,    *fc_b;
};

struct Acts {
    float *conv1_out, *relu1_out, *pool1_out; int *argmax1;
    float *conv2_out, *relu2_out, *pool2_out; int *argmax2;
    float *conv3_out, *relu3_out, *pool3_out; int *argmax3;
    float *logits, *probs;
    float *preds;
};

struct Back {           
    float *dY;                     
    float *d_pool2, *d_pool3;                   
    float *d_relu2, *d_conv2_out;  
    float *d_pool1;               
    float *d_relu1, *d_conv1_out;      
    float *d_image_grad;      
    float *d_relu3, *d_conv3_out;          
};

void forward(float* d_image, Net net, Acts a){

}

void backward(float* d_image, const float* d_target, const Net& net, const Acts& a, const Grads& g, const Back& bp, float* d_loss){

}

void update(Net net, Grads g, int lr){

}

int main(){
    Net net; Grads g; Acts a; Back bp;
    CUDA_CHECK(cudaMalloc(&bp.dY,          240   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_pool3,     1152  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_relu3,     4608  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_conv3_out, 4608  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_pool2,     3136  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_relu2,     13456 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_conv2_out, 13456 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_pool1,     7688  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_relu1,     30752 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_conv1_out, 30752 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_image_grad, 4096 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&net.conv1_f, 72     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv1_b, 8      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv2_f, 1152   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv2_b, 16     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv3_f, 4608   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv3_b, 32     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.fc_W,    276480 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv3_b, 32     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.fc_W,    276480 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.fc_b,    240    * sizeof(float)));

    // ---- grads (Grads g) — same sizes as weights ----
    CUDA_CHECK(cudaMalloc(&g.conv1_f, 72     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv1_b, 8      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv2_f, 1152   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv2_b, 16     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv3_f, 4608   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv3_b, 32     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.fc_W,    276480 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.fc_b,    240    * sizeof(float)));

    // ---- activations (Acts a) ----
    CUDA_CHECK(cudaMalloc(&a.conv1_out, 30752 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.relu1_out, 30752 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.pool1_out, 7688  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.argmax1,   7688  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&a.conv2_out, 13456 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.relu2_out, 13456 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.pool2_out, 3136  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.argmax2,   3136  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&a.conv3_out, 4608  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.relu3_out, 4608  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.pool3_out, 1152  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.argmax3,   1152  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&a.preds,     240   * sizeof(float)));

    // ---- loss + dataset (uploaded once) ----
    int N = 10000;
    int EPOCHS = 20;
    float lr = 0.001f;
    float *d_loss; CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    float *d_X;    CUDA_CHECK(cudaMalloc(&d_X, (size_t)N*4096 * sizeof(float)));
    float *d_T;    CUDA_CHECK(cudaMalloc(&d_T, (size_t)N*240  * sizeof(float)));
    float *d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));

    const int N = 10000;
    const int EPOCHS = 10;
    float lr = 0.001f;
    return 0;
}