cmake_minimum_required(VERSION 3.18)
project(cuda_ml CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Detect GPU architecture automatically — change if your GPU is not listed
# 86 = RTX 3000 series, 89 = RTX 4000 series, 75 = RTX 2000 series, 61 = GTX 1000 series
# Find yours at: https://developer.nvidia.com/cuda-gpus
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# Output binaries to /build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compiler flags — optimize for speed, enable all warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -Wall")

# ── Week 1: Vector Addition ──────────────────────────────────────────────────
add_executable(vector_add kernels/vector_add.cu)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Week 2: Reduction Benchmark
add_executable(benchmark benchmarks/benchmark.cu)
set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Week 2: Memory Coalescing Benchmark
add_executable(memory_benchmark benchmarks/memory_benchmark.cu)
set_target_properties(memory_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Week 2: Streams Benchmark
add_executable(streams_benchmark benchmarks/streams_benchmark.cu)
set_target_properties(streams_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Week 3: Matrix Multiplication Benchmark
add_executable(matmul_benchmark benchmarks/matmul_benchmark.cu)
set_target_properties(matmul_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(matmul_benchmark cublas)
# ── Week 4: 2D Convolution ───────────────────────────────────────────────────
add_executable(conv2d_benchmark benchmarks/conv2d_benchmark.cu) 
set_target_properties(conv2d_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_executable(imageTest kernels/imageTest.cu) 
set_target_properties(imageTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# ── Week 5: Activations ──────────────────────────────────────────────────────
add_executable(activations kernels/activations.cu)
set_target_properties(activations PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Uncomment each block as you reach that week.
