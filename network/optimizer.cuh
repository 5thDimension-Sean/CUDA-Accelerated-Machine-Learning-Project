#ifndef NETWORK_LOSS_CUH
#define NETWORK_LOSS_CUH

#include <cuda_runtime.h>

enum class LossType
{
    SGD, 
    MOMENTUM
};

__global__ void sgd_kernel(const float* predictions,
                                    const float* targets,
                                    float* loss,
                                    int numElements,
                                    LossType lossType);

__global__ void momentum_kernel(const float* predictions,
                                     const float* targets,
                                     float* gradients,
                                     int numElements,
                                     LossType lossType);

void sgd(const float* predictions,
                 const float* targets,
                 float* loss,
                 int numElements,
                 LossType lossType);

void momentum(const float* predictions,
                    const float* targets,
                    float* gradients,
                    int numElements,
                    LossType lossType);

#endif 
