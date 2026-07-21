// ============================================================================
// layer.cuh — Week 5
// Layer abstraction: bundles a layer's device buffers + shape + type so the
// forward pass can be chained device-to-device (no host round-trips between
// layers). Definitions go in network/layer.cu.
// ============================================================================
#pragma once

#include "../kernels/common.cuh"
#include "../kernels/activations.cuh"
#include "../kernels/batchnorm.cuh"
#include "../kernels/pooling.cuh"
#include "../kernels/conv2d.cuh"
#include "../kernels/matmul.cuh"

enum class LayerType {
    SIGMOID,
    BATCHNORM,
    RELU,
    POOL,
    SF, 
    FC, 
    CONV, 
    LRELU
};

struct Layer {
    LayerType type;
    int in_H, in_W, in_C;      // input dimensions
    int out_H, out_W, out_C;   // output dimensions (computed in layer_setup)
    float *d_output;           // this layer's output; allocated once, reused each forward
    float *d_weights;          // CONV/FC only; nullptr for RELU/POOL/BATCHNORM
    float *d_bias;             // CONV/FC/BATCHNORM; nullptr for RELU/POOL
    float *d_input;            // cached input pointer, needed by backward
    float *d_grad_weights;     // gradient w.r.t. weights
    float *d_grad_bias;        // gradient w.r.t. bias
    int P, S;                  // POOL: window size, stride
    float epsilon;             // BATCHNORM
    int filter_H, filter_W;     // CONV: filter dimensions
    int num_filters;           // CONV: number of filters
    float *d_grad_input;       // gradient w.r.t. input; produced by backward, handed to the previous layer
    int *argmax;               // POOL: winning input indices from forward, needed by backward
    int *C;
};
void layer_setup(Layer *layer);
float *layer_forward(Layer *layer, float *d_input);
void layer_free(Layer *layer);
