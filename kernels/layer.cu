#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <iostream>
#include "batchnorm.cu"
#include "activations.cu"
#include "pooling.cu"

// TODO — Week 5: implement here
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

enum LayerType {
    CONV, 
    BATCHNORM,
    RELU, 
    FC, 
};

struct Layer {
    LayerType type;
    int in_H, in_W, in_C;      // input dimensions
    int out_H, out_W, out_C;   // output dimensions (computed at setup)
    float *d_output;           // this layer's output, allocated once, reused each forward
    float *d_weights;          // CONV/FC only; null for RELU/POOL/BATCHNORM
    float *d_bias;             // CONV/FC/BATCHNORM; null for RELU/POOL
    float *d_input;            // cached input pointer, needed by backward
    float *d_grad_weights;     // gradient w.r.t. weights
    float *d_grad_bias;        // gradient w.r.t. bias
    int P, S;                  // POOL: window size, stride
    float epsilon;             // BATCHNORM
};

void layer_setup(Layer *layer){

}

float* layer_forward(Layer *layer, float *d_input){
    switch (layer->type) {
        case CONV:

}

void layer_free(Layer *Layer){

}