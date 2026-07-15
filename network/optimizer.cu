#include "optimizer.cuh"
#include <cmath>
#include <cstdio>


//optimizer.cu

__global__ void sgd_kernel(float *weights, const float *grad, float lr, int n){

}

__global__ void momentum_kernel(float *weights, const float *grad, float *velocity, float lr, float beta, int n){

}


void sgd(float *weights, const float *grad, float lr, int n){

}

void momentum(float *weights, const float *grad, float *velocity, float lr, float beta, int n){

}

int main(){
    //n is num of weights
    const int n = 4;
    float weights[n] = {1.0f, 2.0f, 3.0f, 4.0f};
    float grad[n] = {0.1f, 0.2f, 0.3f, 0.4f};
    float velocity[n] = {0.0f, 0.0f, 0.0f, 0.0f};
    float lr = 0.1f;
    float beta = 0.9f;
    sgd(weights, grad, lr, n);
    printf("Weights after SGD: ");
    for(int i = 0; i < n; ++i){
        printf("%.4f ", weights[i]);
    }
    // Reset weights for momentum test
    for(int i = 0; i < n; ++i){
        weights[i] = (float)(i + 1.0f);
    }
    momentum(weights, grad, velocity, lr, beta, n);
    printf("\nWeights after Momentum: ");
    for(int i = 0; i < n; ++i){
        printf("%.4f ", weights[i]);
    }
    momentum(weights, grad, velocity, lr, beta, n);
    printf("\nWeights after Momentum step 2: ");
    for(int i = 0; i < n; ++i){
        printf("%.4f ", weights[i]);
    }
    
    return 0;
}