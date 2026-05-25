// ============================================================================
// matmul.cu — Week 3
// Goal: Implement matrix multiplication C = A × B in CUDA.
//
// You will build this in 3 stages:
//   Stage 1 — Naive:     one thread per output element, reads from global memory
//   Stage 2 — Tiled:     load tiles into shared memory to reduce global reads
//   Stage 3 — Vectorized: use float4 for 128-bit loads
//
// Key concept: shared memory is ~100x faster than global memory.
// Tiling is THE fundamental GPU optimization technique.
//
// Resources:
//   - https://siboehm.com/articles/22/CUDA-MMM  (best practical guide)
//   - CUDA Programming Guide, Ch. 3
// ============================================================================

// TODO — Week 3: implement here
// Start with the naive kernel, verify correctness, then optimize.
