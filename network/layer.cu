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

    // gradient w.r.t. input — sized like the INPUT, produced by backward and handed back
    size_t in_size = layer->in_H * layer->in_W * layer->in_C * sizeof(float);
    cudaMalloc((void**)&layer->d_grad_input, in_size);

    if (layer->type == LayerType::CONV || layer->type == LayerType::FC || layer->type == LayerType::BATCHNORM) {
        size_t weight_size = layer->out_C * layer->in_C * layer->filter_H * layer->filter_W * sizeof(float);  // for CONV
        size_t bias_size = layer->out_C * sizeof(float);

        cudaMalloc((void**)&layer->d_weights, weight_size);
        cudaMalloc((void**)&layer->d_bias, bias_size);
        cudaMalloc((void**)&layer->d_grad_weights, weight_size);
        cudaMalloc((void**)&layer->d_grad_bias, bias_size);
    } else {
        layer->d_weights = nullptr;
        layer->d_bias = nullptr;
        layer->d_grad_weights = nullptr;
        layer->d_grad_bias = nullptr;
    }

    // POOL needs an argmax buffer (output-sized ints) to route gradients in backward
    if (layer->type == LayerType::POOL) {
        cudaMalloc((void**)&layer->argmax, layer->out_H * layer->out_W * sizeof(int));
    } else {
        layer->argmax = nullptr;
    }

    layer->d_input = nullptr;
}
float *layer_forward(Layer *layer, float *d_input) {
    layer->d_input = d_input;   // cache for the backward pass (Week 6)
    dim3 block(16, 16);
    dim3 grid((layer->out_W + block.x - 1) / block.x, (layer->out_H + block.y - 1) / block.y);

    switch (layer->type) {
        case LayerType::SIGMOID:
            sigmoidActivation<<<grid, block>>>(d_input, layer->d_output, /*doutMatrix=*/nullptr, layer->in_W, layer->in_H, /*isForward=*/true);
            break;
        case LayerType::BATCHNORM:
            // TODO(not runnable yet): needs per-channel d_mean/d_variance buffers
            // computed by a per-channel reduction. gamma=d_weights, beta=d_bias.
            // batchNormForwardPerChannel<<<grid1D, block1D>>>(
            //     d_input, layer->d_output,
            //     layer->d_mean, layer->d_variance,   // per-channel mean/variance
            //     layer->d_weights, layer->d_bias,    // gamma, beta
            //     layer->epsilon, layer->in_H, layer->in_W, layer->in_C);
            break;
        case LayerType::RELU:
            reLuActivation<<<grid, block>>>(d_input, layer->d_output, /*doutMatrix=*/nullptr, layer->in_W, layer->in_H, /*isForward=*/true);
            break;
        case LayerType::POOL:
            maxPool2D<<<grid, block>>>(d_input, layer->d_output, layer->argmax, layer->in_H, layer->in_W, layer->out_H, layer->out_W, layer->P, layer->S);
            break;
        case LayerType::SF:
            softMaxActivation<<<grid, block>>>(d_input, layer->d_output, /*doutMatrix=*/nullptr, layer->in_W, layer->in_H, /*isForward=*/true);
            break;
        case LayerType::FC:
            // TODO(not runnable yet): matmul_tiled is square-only; FC needs a
            // non-square matmul + out dims set in layer_setup.
            // matmul_tiled<<<grid, block>>>(d_input, layer->d_weights, layer->d_output, layer->in_W);
            break;
        case LayerType::CONV:
            // TODO(not runnable yet): conv2d_shared reads its filter from the
            // __constant__ c_filter (needs cudaMemcpyToSymbol first) and uses
            // extern __shared__ (needs a shared-mem size as the 3rd launch arg).
            // conv2d_shared<<<grid, block, shmem_bytes>>>(d_input, layer->d_output,
            //     layer->in_H, layer->in_W, layer->filter_H, layer->filter_W);
            break;
        case LayerType::LRELU:
            leakyreLuActivation<<<grid, block>>>(d_input, layer->d_output, /*doutMatrix=*/nullptr, layer->in_W, layer->in_H, /*isForward=*/true);
            break;
    }

    CUDA_CHECK(cudaGetLastError());
    return layer->d_output;
}

void layer_free(Layer *layer) {
    if (layer->d_output)       cudaFree(layer->d_output);
    if (layer->d_grad_input)   cudaFree(layer->d_grad_input);
    if (layer->d_weights)      cudaFree(layer->d_weights);
    if (layer->d_bias)         cudaFree(layer->d_bias);
    if (layer->d_grad_weights) cudaFree(layer->d_grad_weights);
    if (layer->d_grad_bias)    cudaFree(layer->d_grad_bias);
    if (layer->argmax)         cudaFree(layer->argmax);
}
