// ============================================================================
// conv2d.cu — Week 4
// Goal: Implement 2D convolution in CUDA.
//
// You will build this in stages:
//   Stage 1 — Naive:          one thread per output pixel, reads from global memory
//   Stage 2 — Constant memory: move the filter kernel to __constant__ memory
//   Stage 3 — Shared memory:  load input tiles + halo cells into shared memory
//
// Test cases to verify correctness:
//   - Identity filter [[0,0,0],[0,1,0],[0,0,0]] → output = input
//   - Blur filter (all 1/9)                     → gaussian blur
//   - Sobel filter                              → edge detection
//
// Resources:
//   - CUDA Programming Guide, Ch. 3 (constant memory)
//   - "GPU Gems" Ch. 39 (convolution optimization)
// ============================================================================

// TODO — Week 4: implement here
