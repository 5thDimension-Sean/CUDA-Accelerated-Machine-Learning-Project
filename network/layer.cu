#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <iostream>
#include "layer.cuh"


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



void layer_setup(Layer *layer){

}

float* layer_forward(Layer *layer, float *d_input){
    switch (layer->type) {
        case CONV:

}

void layer_free(Layer *Layer){

}