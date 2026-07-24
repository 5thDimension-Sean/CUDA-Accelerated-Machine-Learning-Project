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
add_executable(batchnorm kernels/batchnorm.cu)
set_target_properties(batchnorm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_executable(loss network/loss.cu)
set_target_properties(loss PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(loss PRIVATE network kernels)
add_executable(pooling kernels/pooling.cu)
set_target_properties(pooling PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Week 5: Layer abstraction — forward pass test (RELU -> POOL chain).
# Links the test main + layer code + the kernels it calls. BUILD_AS_LIBRARY
# skips the individual main()s in the kernel files so there's only one main.
add_executable(forward_pass
    network/test_forward_pass.cu
    network/layer.cu
    kernels/activations.cu
    kernels/pooling.cu)
set_target_properties(forward_pass PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(forward_pass PRIVATE BUILD_AS_LIBRARY)
target_include_directories(forward_pass PRIVATE network kernels)
add_executable(optimizer network/optimizer.cu)
set_target_properties(optimizer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_executable(fc kernels/fc.cu)
set_target_properties(fc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_executable(train_xor
    network/train_xor.cu
    kernels/fc.cu
    network/loss.cu
    network/optimizer.cu)
set_target_properties(train_xor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(train_xor PRIVATE BUILD_AS_LIBRARY)
target_include_directories(train_xor PRIVATE network kernels)
add_executable(mnist_train
    data/mnist_train.cu
    kernels/fc.cu
    network/optimizer.cu)
set_target_properties(mnist_train PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(mnist_train PRIVATE BUILD_AS_LIBRARY)
target_include_directories(mnist_train PRIVATE network kernels)

add_executable(conv2d_mc kernels/conv2d_mc.cu)
set_target_properties(conv2d_mc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(mnist_cnn
      data/mnist_cnn.cu
      kernels/conv2d_mc.cu
      kernels/pooling.cu
      kernels/activations.cu
      kernels/fc.cu
      network/loss.cu
      network/optimizer.cu)
  set_target_properties(mnist_cnn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(mnist_cnn PRIVATE BUILD_AS_LIBRARY)
  target_include_directories(mnist_cnn PRIVATE network kernels)

add_executable(cnn_optimization
      data/cnn_optimization.cu
      )
  set_target_properties(cnn_optimization PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(cnn_optimization PRIVATE BUILD_AS_LIBRARY)
  target_include_directories(cnn_optimization PRIVATE network kernels)

# Uncomment each block as you reach that week.
