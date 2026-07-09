// ============================================================================
// activations.cuh — declarations for the activation kernels + wrapper.
// Definitions live in activations.cu.
// ============================================================================
#pragma once

// which activation the wrapper should dispatch to
enum class ActivationType {
    ReLU,
    LRELU,
    Sigmoid,
    Softmax   // forward-only (no backward kernel)
};

// --- kernels (device) ---
__global__ void sigmoidActivation(float *z_matrix, float *activation_matrix,
                                  int width, int height, bool isForward);
__global__ void reLuActivation(float *z_matrix, float *activation_matrix,
                               int width, int height, bool isForward);
__global__ void leakyreLuActivation(float *z_matrix, float *activation_matrix,
                                    int width, int height, bool isForward);
__global__ void softMaxActivation(float *z_matrix, float *activation_matrix,
                                  int width, int height, bool isForward);

// --- host wrapper (dispatches by type, malloc + copy + launch + copy-back + free) ---
void wrapperKernel(float *host_z_matrix, float *host_activation_matrix,
                   int width, int height, bool isForward, ActivationType type);
