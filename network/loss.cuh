#ifndef NETWORK_LOSS_CUH
#define NETWORK_LOSS_CUH

#include <cuda_runtime.h>

enum class LossType
{
    MSE,
    CrossEntropy
};

__global__ void loss_forward_kernel(const float* predictions,
                                    const float* targets,
                                    float* loss,
                                    int numElements,
                                    LossType lossType);

__global__ void loss_backward_kernel(const float* predictions,
                                     const float* targets,
                                     float* gradients,
                                     int numElements,
                                     LossType lossType);

void loss_forward(const float* predictions,
                  const float* targets,
                  float* loss,
                  int numElements,
                  LossType lossType);

void loss_backward(const float* predictions,
                   const float* targets,
                   float* gradients,
                   int numElements,
                   LossType lossType);

#endif 
