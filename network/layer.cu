// ============================================================================
// layer.cu — Week 5
// Implements the Layer abstraction declared in layer.cuh.
// layer.cuh already pulls in common.cuh + the three kernel headers, so every
// kernel (reLuActivation, maxPool2D, batchNormForward, ...) and CUDA_CHECK
// are visible here. Don't re-include or re-define them.
// ============================================================================
#include "layer.cuh"

// ----------------------------------------------------------------------------
// layer_setup: compute output dims from input dims + params, then allocate
// d_output (and weights/bias if this type needs them). Null the backprop ptrs.
// ----------------------------------------------------------------------------
void layer_setup(Layer *layer) {
    // TODO:
    //  - switch on layer->type to set out_H / out_W / out_C:
    //      RELU / BATCHNORM: out dims == in dims (element-wise)
    //      POOL:             out_H = (in_H - P)/S + 1, out_W likewise, out_C == in_C
    //      CONV / FC:        depends on weights (later)
    //  - cudaMalloc d_output for (out_H * out_W * out_C) floats
    //  - cudaMalloc d_weights / d_bias only for CONV / FC / BATCHNORM
    //  - layer->d_input = layer->d_grad_weights = layer->d_grad_bias = nullptr;
}

// ----------------------------------------------------------------------------
// layer_forward: run this layer's kernel on d_input into d_output.
// Pure launch — NO malloc / memcpy / free. Returns d_output so calls chain.
// ----------------------------------------------------------------------------
float *layer_forward(Layer *layer, float *d_input) {
    layer->d_input = d_input;   // cache for the backward pass (Week 6)

    // TODO: build grid/block from this layer's output dims, then launch:
    switch (layer->type) {
        case LayerType::CONV:
            // TODO: launch conv kernel
            break;
        case LayerType::BATCHNORM:
            // TODO: launch batchNormForward into layer->d_output
            break;
        case LayerType::RELU:
            // reLuActivation<<<grid, block>>>(d_input, layer->d_output,
            //                                 width, height, /*isForward=*/true);
            break;
        case LayerType::POOL:
            // maxPool2D<<<grid, block>>>(d_input, layer->d_output,
            //                            layer->in_H, layer->in_W,
            //                            layer->out_H, layer->out_W,
            //                            layer->P, layer->S);
            break;
        case LayerType::FC:
            // TODO: launch matmul + bias
            break;
    }

    CUDA_CHECK(cudaGetLastError());
    return layer->d_output;
}

// ----------------------------------------------------------------------------
// layer_free: release every device buffer this layer owns.
// ----------------------------------------------------------------------------
void layer_free(Layer *layer) {
    // guard each with `if (ptr)` since some are nullptr depending on type
    if (layer->d_output)       cudaFree(layer->d_output);
    if (layer->d_weights)      cudaFree(layer->d_weights);
    if (layer->d_bias)         cudaFree(layer->d_bias);
    if (layer->d_grad_weights) cudaFree(layer->d_grad_weights);
    if (layer->d_grad_bias)    cudaFree(layer->d_grad_bias);
}
