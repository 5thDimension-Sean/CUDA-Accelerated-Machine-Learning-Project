**#CUDA Accelerated Machine Learning Project**

Uses NVIDIA’s parallel computing framework to perform data-intensive tasks using the GPU rather than the CPU. This ensures a huge reduction in training time and makes it possible to deal with huge data sets.

**#Technical Overview**

It is a 3 layer GPU computing stack.
- Top: Application Layer(Python benchmarking scripts and C++ inference demo to feed data to the GPU)
- Middle: CUDA Kernel library(cu files implmenting neural network operations(GEMM, convlution, activations, pooling, batch normalization) as parallel GPU kernels.)
- Bottom: Hardware(GPU's streaming multprocessor units executing warps of 32 threas in parallel against the kernel code)

**#System Prerequisites**

- NVIDIA GPU(Compute Capability 7.5+, i.e RTX 20xx or newer)
- CUDA Toolkit 12.x+
- MSVC 2022 as host compiler
- CMake 3.18+ with Ninja build system
- Windows 10/11 or Linux(Ubuntu 20.04+)
- 8GB+ system RAM, 4GB+ VRAM

**#Workflow**

Training through images -> CPU preprocessing -> cudaMemcpy for GPU memory transfer -> forward pass over CUDA kernels -> loss computation -> backwards pass over CUDA kernels -> weight update -> repeat
Inference: single image -> GPU -> detection output -> NMS -> bounding boxes on screen

**#Data Flow: **
Host CPU gives and initializes data -> cudaMemcpy transfers to GPU global memory -> kernels load tiles into shared memory -> threads compute in parallel -> results go back to global memory -> cudaMemcpy back to host for display

**#Sample Results**
```
Vector Addition: N = 16777216 elements (67.1 MB per vector)

Launching kernel: 65536 blocks * 256 threads = 16777216 total threads

Results (averaged over 10 runs):
  GPU time : 1.262 ms
  CPU time : 12.630 ms
  Speedup  : 10.01x
  Correct  : YES



Memory bandwidth (GPU): 142.5 GB/s
```

**#Getting started**
1. Clone the repo
   
  ```git clone https://github.com/5thDimension-Sean/CUDA-Accelerated-Machine-Learning-Project.git```
  
2. Verify CUDA installation
   
   ```nvcc --version```
   
4. Configure the build system

   Open the project folder in VS Code. When prompted, select the CUDA Release configure preset. This preset targets Ninja as the build system and points directly to the CUDA and MSVC 2022 compilers.
   If the preset prompt does not appear automatically:
   Ctrl+Shift+P -> CMake: Select Configure Preset → CUDA Release
   
4. Build

   F7  (or Ctrl+Shift+P -> CMake: Build)
   
5. Run Verification Kernel

   ```& ".\build\bin\vector_add.exe"```
   
   Sample Output:
   ```
     Vector Addition: N = 16777216 elements (67.1 MB per vector)

     Launching kernel: 65536 blocks * 256 threads = 16777216 total threads
    
     Results (averaged over 10 runs):
        GPU time : 1.262 ms
        CPU time : 12.630 ms
        Speedup  : 10.01x
        Correct  : YES


     Memory bandwidth (GPU): 142.5 GB/s

   
**#Acknowledgements**

  https://www.cse.iitd.ac.in/~rijurekha/col730_2022/cudabook.pdf

  https://developer.nvidia.com/cuda/toolkit
  
  https://developer.nvidia.com/nsight-systems/get-started
  
  https://developer.nvidia.com/tools-overview/nsight-compute/get-started
  
  Claude Sonnet 4.6 for all setup files and code for vector_add.cu


Exact figures may vary depending on GPU. A speedup of more than 1x and Correct: YES signify a successful compilation.
Workflow
Whenever changes occur to .cu or .cpp code, rebuild using the F7 shortcut in VS Code. CMake uses an incremental build process; only altered files are rebuilt. There is no need to reset configuration between compilations.
With the addition of new kernel targets weekly, uncomment the corresponding add_executable line in CMakeLists.txt and perform another rebuild.

For profiling purposes, NVIDIA Nsight is used in this project:

Nsight Systems – provides a timeline of the entire operation cycle. To run a test, launch the program from within Nsight Systems.
Nsight Compute – profiling of individual CUDA kernels. Use for occupancy calculation, memory bandwidth, and warp utilization analysis.

To ensure consistent results when benchmarking, shut down other running applications and ignore the initial run of each kernel, as it serves as GPU warmup.
