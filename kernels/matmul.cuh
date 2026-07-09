// ============================================================================
// matmul.cuh — declarations for the matrix-multiply kernels.
// Definitions live in matmul.cu.
// NOTE: all three assume SQUARE N x N matrices (single int N). matmul_tiled
// hardcodes a 32x32 tile, so it MUST be launched with block(32, 32).
// ============================================================================
#pragma once

// C = A * B, all N x N. One thread per output element.
__global__ void matmul_naive(const float *A, const float *B, float *C, int N);

// tiled shared-memory version. REQUIRES block(32, 32) — tile is 32x32.
__global__ void matmul_tiled(const float *A, const float *B, float *C, int N);

// float4-vectorized version (alignment / divisibility constraints apply).
__global__ void float4_vectorized(const float *A, const float *B, float *C, int N);
