# CUDA-Accelerated Machine Learning

**Train and run neural networks on the GPU instead of the CPU.**

Hand-written CUDA kernels for data-intensive deep-learning workloads, built for speed on modern NVIDIA hardware. Massive parallelism delivers substantially shorter training times and the ability to work with large datasets.

![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)


---

## Overview

This project uses NVIDIA's CUDA parallel-computing framework to offload data-intensive machine-learning tasks from the CPU to the GPU. By running thousands of threads in parallel, it achieves large reductions in training time and scales to datasets that would be impractical on a CPU alone.

---

## Architecture

A three-layer GPU computing stack:

```
┌───────────────────────────────────────────────────────────┐
│  APPLICATION LAYER                                          │
│  Python benchmarking scripts and C++ inference demo         │
│  Feeds data to the GPU                                      │
├───────────────────────────────────────────────────────────┤
│  CUDA KERNEL LIBRARY  (.cu files)                           │
│  Neural-net operations as parallel GPU kernels:             │
│  GEMM, Convolution, Activations, Pooling, BatchNorm         │
├───────────────────────────────────────────────────────────┤
│  HARDWARE                                                   │
│  GPU Streaming Multiprocessors executing warps of           │
│  32 threads in parallel against the kernel code             │
└───────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement       | Minimum spec                                        |
| ----------------- | --------------------------------------------------- |
| **GPU**           | NVIDIA, Compute Capability 7.5+ (RTX 20xx or newer) |
| **CUDA Toolkit**  | 12.x+                                               |
| **Host compiler** | MSVC 2022                                            |
| **Build system**  | CMake 3.18+ with Ninja                              |
| **OS**            | Windows 10/11 or Linux (Ubuntu 20.04+)              |
| **Memory**        | 8 GB+ system RAM, 4 GB+ VRAM                         |

---

## Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/5thDimension-Sean/CUDA-Accelerated-Machine-Learning-Project.git
```

**2. Verify your CUDA install**
```bash
nvcc --version
```

**3. Configure the build**
> Open the project folder in VS Code. When prompted, select the **CUDA Release** configure preset. It targets Ninja and points directly at the CUDA and MSVC 2022 compilers.
>
> If the prompt does not appear: `Ctrl+Shift+P`, then **CMake: Select Configure Preset**, then **CUDA Release**.

**4. Build**
> Press `F7` (or `Ctrl+Shift+P`, then **CMake: Build**).

**5. Run the verification kernel**
```powershell
& ".\build\bin\vector_add.exe"
```

**6. Confirm success**
> A speedup greater than 1x and `Correct: YES` indicate that everything compiled and ran successfully.

---

## Sample Results

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

> Note: exact figures vary by GPU. The numbers above are a reference point, not a target.

---

## Executables

Run any of these from the project root after building. The notes below are inferred from each target's name; provide your summer plan and they will be replaced with exact descriptions.

| Command | Description |
| ------- | ----------- |
| `& ".\build\bin\vector_add.exe"` | Verification kernel. Element-wise vector addition; confirms the CUDA setup works and reports GPU-vs-CPU speedup. |
| `& ".\build\bin\benchmark.exe"` | General kernel performance benchmark suite. |
| `& ".\build\bin\memory_benchmark.exe"` | Measures memory bandwidth and host-to-device transfer throughput. |
| `&".\build\bin\streams_benchmark.exe"` | Tests CUDA streams, overlapping compute with memory transfers. |
| `& ".\build\bin\matmul_benchmark.exe"` | Times matrix multiplication (GEMM) on the GPU. |
| `python benchmarks\gemm_benchmark.py` | Python-driven GEMM benchmark. |
| `& ".\build\bin\conv2d_benchmark.exe"` | Benchmarks the 2D convolution kernel. |
| `& ".\build\bin\imageTest.exe"` | Image processing and inference demo test. |
| `& ".\build\bin\activations.exe"` | Exercises activation-function kernels (ReLU, sigmoid, and others). |

---

## Data Flow

**Training loop**
```
images -> CPU preprocessing -> cudaMemcpy to GPU -> forward pass (CUDA kernels)
       -> loss computation -> backward pass (CUDA kernels) -> weight update -> repeat
```

**Inference**
```
single image -> GPU -> detection output -> NMS -> bounding boxes on screen
```

**Memory path**
```
Host CPU initializes data
   -> cudaMemcpy transfers to GPU global memory
   -> kernels load tiles into shared memory
   -> threads compute in parallel
   -> results written back to global memory
   -> cudaMemcpy back to host for display
```

---

## Development Workflow

Whenever you change a `.cu` or `.cpp` file, rebuild with `F7`. CMake builds incrementally, so only changed files recompile, and there is no need to re-configure between builds.

To add a new kernel target, uncomment the corresponding `add_executable` line in `CMakeLists.txt`, then rebuild. New kernel targets are added weekly.

---

## Profiling with NVIDIA Nsight

| Tool | Purpose |
| ---- | ------- |
| **Nsight Systems** | Provides a timeline view of the entire operation cycle. Launch the program from within Nsight Systems to capture a trace. |
| **Nsight Compute** | Per-kernel profiling for occupancy, memory bandwidth, and warp-utilization analysis. |

For consistent benchmarks, close other running applications and ignore the first run of each kernel, as it serves as GPU warmup.

---

## Acknowledgements

- [CUDA Programming Book (IIT Delhi)](https://www.cse.iitd.ac.in/~rijurekha/col730_2022/cudabook.pdf)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)
- Claude Sonnet 4.6 for setup files and code for `vector_add.cu`
