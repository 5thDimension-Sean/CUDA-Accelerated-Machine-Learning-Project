// ============================================================================
// pooling.cuh — declarations for the 2D max pooling kernel + wrapper.
// Definitions live in pooling.cu.
// ============================================================================
#pragma once

// --- kernel (device) ---
__global__ void maxPool2D(const float *input, float *output, int *argmax,
                          int H, int W, int out_H, int out_W, int P, int S);

// --- host wrapper (malloc + copy + launch + copy-back + free) ---
void maxPoolWrapKernel(float *h_input, float *h_output, int *argmax, int H, int W, int P, int S);
