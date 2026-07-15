#ifndef NETWORK_OPTIMZER_CUH
#define NETWORK_OPTIMZER_CUH

#include <cuda_runtime.h>

enum class LossType
{
    SGD, 
    Momentum
};

__global__ void loss_forward_kernel(const float* predictions,
                                    const float* targets,
                                    float* loss,
                                    int numElements,
                                    LossType lossType);



void loss_forward(const float* predictions,
                  const float* targets,
                  float* loss,
                  int numElements,
                  LossType lossType);


#endif 
