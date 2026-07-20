// ============================================================================
// conv2d_mc.cuh — declarations for multi-channel 2D convolution.
// Definitions live in conv2d_mc.cu.
//   input  [C_in x H x W], filter [C_out x C_in x FH x FW], bias [C_out]
//   output [C_out x outH x outW]
// ============================================================================
#pragma once

// forward kernel (device)
__global__ void conv2d_mc_forward(const float *input, const float *filter, const float *bias,
                                  float *output, int C_in, int C_out, int H, int W, int FH, int FW);
            
__global__ void conv2d_mc_backward_bias(const float *dOut, float *dBias,
                                        int C_out, int outH, int outW);

__global__ void conv2d_mc_backward_weights(const float *dOut, const float *input, float *dFilter,
                                           int C_in, int C_out, int H, int W, int FH, int FW);                                       

__global__ void conv2d_mc_backward_input(const float *dOut, const float *filter, float *dInput,
                                         int C_in, int C_out, int H, int W, int FH, int FW);

void conv2d_mc(const float *input, const float *filter, const float *bias, float *output,
               int C_in, int C_out, int H, int W, int FH, int FW);

void conv2d_mc_backward(const float *dOut,               
                        const float *input, const float *filter,
                        float *dInput, float *dFilter, float *dBias, //3 output
                        int C_in, int C_out, int H, int W, int FH, int FW);
