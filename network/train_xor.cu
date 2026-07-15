#include "fc.cu"
#include "activations.cu"
#include "loss.cu"
#include "optimizer.cu"
#include "common.cuh"


int main(){
    float W1[4*2] = {
    0.5f, -0.3f,
   -0.4f,  0.8f,
    0.2f,  0.6f,
   -0.7f,  0.1f
    };
    float b1[4] = {0, 0, 0, 0};

    float W2[1*4] = { 0.3f, -0.6f, 0.4f, -0.2f };
    float b2[1]   = { 0 };
    return 0;
}