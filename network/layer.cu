// ============================================================================
// layer.cu — Week 5
// Implements the Layer abstraction declared in layer.cuh.
// layer.cuh already pulls in common.cuh + the three kernel headers, so every
// kernel (reLuActivation, maxPool2D, batchNormForward, ...) and CUDA_CHECK
// are visible here. Don't re-include or re-define them.
// ============================================================================
#include "layer.cuh"
void layer_setup(Layer *layer) {
    switch (layer->type) {
        case LayerType::RELU:
        case LayerType::BATCHNORM:
            layer->out_H = layer->in_H;
            layer->out_W = layer->in_W;
            layer->out_C = layer->in_C;
            break;

        case LayerType::POOL:
            layer->out_H = (layer->in_H - layer->P) / layer->S + 1;
            layer->out_W = (layer->in_W - layer->P) / layer->S + 1;
            layer->out_C = layer->in_C;
            break;

        case LayerType::CONV:
            layer->out_H = layer->in_H - layer->filter_H + 1;
            layer->out_W = layer->in_W - layer->filter_W + 1;
            layer->out_C = layer->num_filters;
            break;
        case LayerType::FC:
            break;
    }
    size_t out_size = layer->out_H * layer->out_W * layer->out_C * sizeof(float);
    cudaMalloc((void**)&layer->d_output, out_size);
    if (layer->type == LayerType::CONV || layer->type == LayerType::FC || layer->type == LayerType::BATCHNORM) {
        size_t weight_size = layer->out_C * layer->in_C * layer->filter_H * layer->filter_W;  // for CONV
        size_t bias_size = layer->out_C * sizeof(float); 

        cudaMalloc((void**)&layer->d_weights, weight_size);
        cudaMalloc((void**)&layer->d_bias, bias_size);
    } else {
        layer->d_weights = nullptr;
        layer->d_bias = nullptr;
    }
    layer->d_input = nullptr;
    layer->d_grad_weights = nullptr;
    layer->d_grad_bias = nullptr;
}
float *layer_forward(Layer *layer, float *d_input) {
    layer->d_input = d_input;   // cache for the backward pass (Week 6)
    dim3 block(16, 16);
    dim3 grid((layer->out_W + block.x - 1) / block.x, (layer->out_H + block.y - 1) / block.y);

    switch (layer->type) {
        case LayerType::SIGMOID:
            sigmoidActivation<<<grid, block>>>(d_input, layer->d_output, layer->in_W, layer->in_H, /*isForward=*/true);
            break;
        case LayerType::BATCHNORM:
            batchNormForward<<<grid, block>>>(d_input, layer->d_output, layer->d_weights, layer->d_bias, layer->in_H, layer->in_W, layer->in_C, layer->epsilon);
            break;
        case LayerType::RELU:
            reLuActivation<<<grid, block>>>(d_input, layer->d_output, layer->in_W, layer->in_H, /*isForward=*/true);
            break;
        case LayerType::POOL:
            maxPool2D<<<grid, block>>>(d_input, layer->d_output, layer->in_H, layer->in_W, layer->out_H, layer->out_W, layer->P, layer->S);
            break;
        case LayerType::SF:
            softMaxActivation<<<grid, block>>>(d_input, layer->d_output, layer->in_W, layer->in_H, /*isForward=*/true);
            break;
        case LayerType::FC:
            matmul_tiled<<<grid, block>>>(d_input, layer->d_weights, layer->d_output, layer->in_W);
            break;
        case LayerType::CONV:
            conv2d_shared<<<grid, block>>>(d_input, layer->d_output, layer->in_H, layer->in_W, layer->filter_H, layer->filter_W);
            break;
        case LayerType::LRELU:
            leakyreLuActivation<<<grid, block>>>(d_input, layer->d_output, layer->in_W, layer->in_H, /*isForward=*/true);
            break;
    }

    CUDA_CHECK(cudaGetLastError());
    return layer->d_output;
}

void layer_free(Layer *layer) {
    if (layer->d_output)       cudaFree(layer->d_output);
    if (layer->d_weights)      cudaFree(layer->d_weights);
    if (layer->d_bias)         cudaFree(layer->d_bias);
    if (layer->d_grad_weights) cudaFree(layer->d_grad_weights);
    if (layer->d_grad_bias)    cudaFree(layer->d_grad_bias);
}
