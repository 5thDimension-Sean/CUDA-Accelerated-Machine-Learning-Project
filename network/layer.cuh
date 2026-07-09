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

// which kind of layer this is — drives the dispatch in layer_forward()
enum class LayerType {
    CONV,
    BATCHNORM,
    RELU,
    POOL,
    FC
};

struct Layer {
    LayerType type;

    // --- shape info ---
    int in_H, in_W, in_C;      // input dimensions
    int out_H, out_W, out_C;   // output dimensions (computed in layer_setup)

    // --- device-resident buffers (all float*, all on GPU) ---
    float *d_output;           // this layer's output; allocated once, reused each forward
    float *d_weights;          // CONV/FC only; nullptr for RELU/POOL/BATCHNORM
    float *d_bias;             // CONV/FC/BATCHNORM; nullptr for RELU/POOL

    // --- for Week 6 backprop (set to nullptr for now, unused this week) ---
    float *d_input;            // cached input pointer, needed by backward
    float *d_grad_weights;     // gradient w.r.t. weights
    float *d_grad_bias;        // gradient w.r.t. bias

    // --- layer-specific params (only some used per type) ---
    int P, S;                  // POOL: window size, stride
    float epsilon;             // BATCHNORM
    // gamma/beta for batchnorm can live in d_weights/d_bias
};

// allocate d_output (+ weights/bias if the type needs them) on the device and
// compute out_H/out_W/out_C from the input dims + params. Call once per layer.
void layer_setup(Layer *layer);

// core call: take a device input pointer, run this layer's kernel into
// d_output, return d_output. NO malloc/copy/free inside — pure launch.
float *layer_forward(Layer *layer, float *d_input);

// free every device buffer this layer owns. Call once at teardown.
void layer_free(Layer *layer);
