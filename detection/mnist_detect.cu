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
    float *fc_W,    *fc_b;
};

struct Grads {
    float *conv1_f, *conv1_b;
    float *conv2_f, *conv2_b;
    float *fc_W,    *fc_b;
};

struct Acts {
    float *conv1_out, *relu1_out, *pool1_out; int *argmax1;
    float *conv2_out, *relu2_out, *pool2_out; int *argmax2;
    float *logits, *probs;
};

struct Back {           
    float *dY;                     
    float *d_pool2;                   
    float *d_relu2, *d_conv2_out;  
    float *d_pool1;               
    float *d_relu1, *d_conv1_out;      
    float *d_image_grad;            
};

void forward(float* d_image, Net net, Acts a){

}

void backward(float* d_image, const float* d_target, const Net& net, const Acts& a, const Grads& g, const Back& bp, float* d_loss){

}

void update(Net net, Grads g, int lr){

}

int main(){

    return 0;
}