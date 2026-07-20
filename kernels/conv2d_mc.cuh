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

// host wrapper (malloc + copy + launch + copy-back + free)
void conv2d_mc(const float *input, const float *filter, const float *bias, float *output,
               int C_in, int C_out, int H, int W, int FH, int FW);
