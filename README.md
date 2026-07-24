# CUDA-Accelerated Machine Learning

**Train and run neural networks on the GPU instead of the CPU.**

CUDA kernels for data-intensive deep-learning workloads, built for speed on modern NVIDIA hardware. Parallelism delivers substantially shorter training times and the ability to work with large datasets.

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
│  GEMM, Convolution, Activations, Pooling, BatchNorm, FC     │
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

### 1 · Clone
```bash
git clone https://github.com/5thDimension-Sean/CUDA-Accelerated-Machine-Learning-Project.git
```

### 2 · Verify CUDA
```bash
nvcc --version      # should report CUDA 12.x
nvidia-smi          # should list your GPU
```

### 3 · Configure
Open the project folder in VS Code and select the **CUDA Release** configure preset when prompted — it targets Ninja and points at the CUDA + MSVC 2022 compilers.

> If the prompt doesn't appear: `Ctrl+Shift+P` → **CMake: Select Configure Preset** → **CUDA Release**.

### 4 · Build
Press `F7`  ·  or  `Ctrl+Shift+P` → **CMake: Build**

### 5 · Run the verification kernel
```powershell
& ".\build\bin\vector_add.exe"
```

### 6 · Confirm success
A **speedup > 1×** and **`Correct: YES`** mean everything compiled and ran correctly.
---
### 7 · Extra Datasets
```https://www.kaggle.com/datasets/oddrationale/mnist-in-csv```


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

## Results — MNIST Classification

Two networks trained from scratch on MNIST and evaluated on the full **10,000-image held-out test set**. Every layer — convolution, max-pooling, fully-connected, softmax — runs as a hand-written CUDA kernel with hand-derived backpropagation. No autograd, no ML frameworks for the core math. The convolution and pooling gradients were verified against finite-difference numerical gradients before training.

| Model   | Architecture                                    | Parameters | Train acc | **Test acc** |
| ------- | ----------------------------------------------- | ---------- | --------- | ------------ |
| MLP     | 784 → 128 → 10  (ReLU, softmax)                 | 101,770    | 94.83%    | 92.77%       |
| **CNN** | 1→8 conv → pool → 8→16 conv → pool → FC → 10     | **5,258**  | 99.09%    | **97.44%**   |

The CNN beats the dense baseline by **+4.67 points on held-out data while using ~19× fewer parameters.** Convolution's weight-sharing captures spatial structure the MLP can't, and it generalizes better — a 1.6-point train/test gap versus the MLP's 2.1.

**CNN training** — SGD, learning rate 0.001, He initialization, 20 epochs over 10,000 images:

```
epoch  0  loss = 0.9107
epoch  5  loss = 0.1068
epoch 10  loss = 0.0641
epoch 15  loss = 0.0431
epoch 19  loss = 0.0320
train accuracy = 99.09% (9909/10000)
TEST  accuracy = 97.44% (9744/10000)
```

---

## Executables

Everything below builds from hand-written CUDA — no ML frameworks for the core math. Run any target from the project root after building.

### Foundations & profiling
| Command | What it does |
| ------- | ------------ |
| `& ".\build\bin\vector_add.exe"` | Verification kernel: element-wise vector addition; confirms the CUDA setup and reports GPU-vs-CPU speedup. |
| `& ".\build\bin\benchmark.exe"` | Parallel reduction: naive → shared-memory → warp-level, timed against a CPU baseline. |
| `& ".\build\bin\memory_benchmark.exe"` | Coalesced vs. uncoalesced global-memory access, showing the bandwidth cost of stride. |
| `& ".\build\bin\streams_benchmark.exe"` | CUDA streams: overlapping compute with async host↔device transfers. |

### Core math
| Command | What it does |
| ------- | ------------ |
| `& ".\build\bin\matmul_benchmark.exe"` | GEMM: naive → tiled → `float4`-vectorized → cuBLAS, across matrix sizes. |
| `python benchmarks\gemm_benchmark.py` | Plots the GEMM benchmark results. |
| `& ".\build\bin\conv2d_benchmark.exe"` | 2D convolution: naive → constant-memory → shared-memory tiled → depthwise. |
| `& ".\build\bin\imageTest.exe"` | Applies the convolution kernels to a real image (Gaussian blur, Sobel edges). |

### Neural-network layers
| Command | What it does |
| ------- | ------------ |
| `& ".\build\bin\activations.exe"` | ReLU, Leaky ReLU, Sigmoid, Softmax — forward **and** backward. |
| `& ".\build\bin\batchnorm.exe"` | Batch Normalization, forward and backward (parallel mean/variance reductions). |
| `& ".\build\bin\pooling.exe"` | 2D max pooling, forward and backward (argmax gradient routing). |
| `& ".\build\bin\fc.exe"` | Fully-connected layer, forward and backward. |
| `& ".\build\bin\forward_pass.exe"` | Chains layers through the `Layer` abstraction (device-to-device forward pass). |

### Training
| Command | What it does |
| ------- | ------------ |
| `& ".\build\bin\loss.exe"` | MSE and cross-entropy loss, forward and backward. |
| `& ".\build\bin\optimizer.exe"` | SGD and momentum weight-update kernels. |
| `& ".\build\bin\train_xor.exe"` | End-to-end training loop: a from-scratch network that learns XOR, loss driven to ~0. |
| `python benchmarks\xor_loss_plot.py` | Plots the XOR training-loss curve. |
| `python data\prepare_mnist.py` | Preprocesses the MNIST CSV files into normalized `float32` binaries (train + test sets). |
| `python data\prepare_mnist_test.py` | second MNIST testing dataset |
| `& ".\build\bin\mnist_train.exe"` | Trains an MLP (784 → 128 → 10, ReLU + softmax, cross-entropy) on MNIST digits, then reports **train vs. test accuracy** on held-out data. |
| `& ".\build\bin\conv2d_mc.exe"` | Multi-channel 2D convolution, forward **and** backward — the CNN building block. Gradients verified against finite differences. |
| `& ".\build\bin\mnist_cnn.exe"` | End-to-end CNN on MNIST (conv → pool → conv → pool → FC), fully hand-written forward + backprop. Reports train vs. test accuracy — **97.44% test**, beating the MLP baseline. |

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
