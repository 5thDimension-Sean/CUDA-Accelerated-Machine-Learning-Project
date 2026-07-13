#include "loss.cuh"
#include "common.cuh"
#include <cmath>
#include <cstdio>

__global__ void loss_forward_kernel(const float* predictions,
                                    const float* targets,
                                    float* loss,
                                    int numElements,
                                    LossType lossType) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = threadIdx.x;
    float partial = 0.0f;

    while (idx < numElements) {
        float diff = predictions[idx] - targets[idx];
        if (lossType == LossType::MSE) {
            partial += diff * diff;
        } else {
            // predictions are post-softmax
            float clipped = fmaxf(predictions[idx], 1e-7f);
            partial += -targets[idx] * logf(clipped);
        }
        idx += blockDim.x;
    }

    sdata[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *loss = sdata[0] / static_cast<float>(numElements);
    }
}

__global__ void loss_backward_kernel(const float* predictions,
                                     const float* targets,
                                     float* gradients,
                                     int numElements,
                                     LossType lossType) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
        float pred = predictions[idx];
        float target = targets[idx];

        if (lossType == LossType::MSE) {
            gradients[idx] = 2.0f * (pred - target) / static_cast<float>(numElements);
        } else {
            // Assumption: predictions are post-softmax probabilities.
            gradients[idx] = (pred - target) / static_cast<float>(numElements);
        }
    }
}

void loss_forward(const float* predictions,
                  const float* targets,
                  float* loss,
                  int numElements,
                  LossType lossType) {
    if (numElements <= 0) {
        *loss = 0.0f;
        return;
    }

    size_t bytes = numElements * sizeof(float);
    float *d_predictions = nullptr;
    float *d_targets = nullptr;
    float *d_loss = nullptr;

    CUDA_CHECK(cudaMalloc(&d_predictions, bytes));
    CUDA_CHECK(cudaMalloc(&d_targets, bytes));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_predictions, predictions, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

    dim3 block(256);
    dim3 grid(1);
    loss_forward_kernel<<<grid, block, block.x * sizeof(float)>>>(
        d_predictions, d_targets, d_loss, numElements, lossType);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_predictions));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_loss));
}

void loss_backward(const float* predictions,
                   const float* targets,
                   float* gradients,
                   int numElements,
                   LossType lossType) {
    if (numElements <= 0) {
        return;
    }

    size_t bytes = numElements * sizeof(float);
    float *d_predictions = nullptr;
    float *d_targets = nullptr;
    float *d_gradients = nullptr;

    CUDA_CHECK(cudaMalloc(&d_predictions, bytes));
    CUDA_CHECK(cudaMalloc(&d_targets, bytes));
    CUDA_CHECK(cudaMalloc(&d_gradients, bytes));

    CUDA_CHECK(cudaMemcpy(d_predictions, predictions, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets, bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((numElements + block.x - 1) / block.x);
    loss_backward_kernel<<<grid, block>>>(
        d_predictions, d_targets, d_gradients, numElements, lossType);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gradients, d_gradients, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_predictions));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_gradients));
}

#ifndef BUILD_AS_LIBRARY
int main() {
    const int numElements = 6;
    float predictions[numElements] = {0.8f, 0.2f, 0.7f, 0.3f, 0.9f, 0.1f};
    float targets[numElements] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    float loss = 0.0f;
    float gradients[numElements] = {0.0f};

    loss_forward(predictions, targets, &loss, numElements, LossType::MSE);
    printf("MSE loss: %.6f\n", loss);
    loss_backward(predictions, targets, gradients, numElements, LossType::MSE);
    printf("MSE gradients: ");
    for (int i = 0; i < numElements; ++i) {
        printf("%.4f ", gradients[i]);
    }
    printf("\n");

    loss_forward(predictions, targets, &loss, numElements, LossType::CrossEntropy);
    printf("Cross-entropy loss: %.6f\n", loss);
    loss_backward(predictions, targets, gradients, numElements, LossType::CrossEntropy);
    printf("Cross-entropy gradients: ");
    for (int i = 0; i < numElements; ++i) {
        printf("%.4f ", gradients[i]);
    }
    printf("\n");

    return 0;
}
#endif