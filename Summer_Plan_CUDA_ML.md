# CUDA-Accelerated Machine Learning Project
## Summer Plan: June 18 – August 21

> **Goal:** Build a CUDA-accelerated object detection system from scratch — implementing matrix multiplication, convolution, and neural network layers directly in CUDA C++, then applying them to real-time AI inference. No high-level frameworks for the core math. Benchmark everything against CPU. Ship a portfolio-ready GitHub repo.

---

## Prerequisites Checklist (Complete Before June 18)

- [ ] Verify NVIDIA GPU model and CUDA Compute Capability at [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus)
- [ ] Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (12.x recommended)
- [ ] Install [cuDNN](https://developer.nvidia.com/cudnn) (requires free NVIDIA account)
- [ ] Install VS Code + C/C++ Extension + CMake Tools extension
- [ ] Install Python 3.11+ (for benchmarking visualization with matplotlib)
- [ ] Create a GitHub repo: `cuda-ml-from-scratch`
- [ ] Install NVIDIA Nsight Systems (free profiler, bundled with CUDA Toolkit)

---

## Project Architecture (What You're Building)

```
cuda-ml-from-scratch/
├── kernels/
│   ├── matmul.cu          ← Custom CUDA matrix multiplication
│   ├── conv2d.cu          ← Custom CUDA 2D convolution
│   ├── activations.cu     ← ReLU, Sigmoid, Softmax kernels
│   ├── pooling.cu         ← Max pooling kernel
│   └── batchnorm.cu       ← Batch normalization kernel
├── network/
│   ├── layer.cu           ← Layer abstraction (forward + backward)
│   ├── model.cu           ← CNN model builder
│   └── optimizer.cu       ← SGD/Adam update kernel
├── detection/
│   ├── yolo_head.cu       ← Bounding box prediction head
│   ├── nms.cu             ← Non-max suppression in CUDA
│   └── inference.cu       ← Real-time inference pipeline
├── benchmarks/
│   ├── cpu_baseline.cpp   ← NumPy/OpenCV CPU reference
│   └── benchmark.py       ← Timing + chart generation
├── data/
│   └── prepare_dataset.py ← Download and preprocess COCO subset
└── demo/
    └── webcam_demo.py     ← Real-time webcam inference
```

---

## Week-by-Week Schedule

---

### WEEK 1 — June 18–24: Foundations & Environment
**Theme: Learn how GPUs think**

**Daily Breakdown:**

| Day | Focus | Tasks |
|-----|-------|-------|
| Wed Jun 18 | Setup Day | Install CUDA Toolkit, cuDNN, VS Code. Verify with `nvcc --version` and `nvidia-smi`. Set up GitHub repo. |
| Thu Jun 19 | GPU Architecture | Read NVIDIA's "CUDA Programming Guide" Ch. 1–2 (free online). Understand: SMs, warps, threads, blocks, grids. Watch: "CUDA Programming" by NVIDIA Developer YouTube. |
| Fri Jun 20 | First Kernels | Complete NVIDIA's [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/). Write: vector addition kernel. Compile with `nvcc`. Run it. |
| Sat Jun 21 | Memory Model | Learn: global vs shared vs local vs constant memory. Implement: vector addition with timing (`cudaEvent_t`). Compare with CPU loop. |
| Sun Jun 22 | Thread Organization | Experiment with different block/grid sizes. Learn why 32 threads/warp matters. Read: "Professional CUDA C Programming" Ch. 2. |
| Mon Jun 23 | Error Handling | Implement proper CUDA error checking macro. Write a kernel that intentionally errors — catch it. |
| Tue Jun 24 | Week 1 Deliverable | Clean up code, push to GitHub. Deliverable: Vector addition kernel with proper error handling, timing output, and CPU comparison. Write a README section explaining what you built. |

**Resources:**
- [CUDA by Example (free PDF)](https://developer.download.nvidia.com/books/cuda-by-example/cuda-by-example-sample.pdf)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- YouTube: "CUDA Programming" — NVIDIA Developer channel

**Deliverable:** `kernels/vector_add.cu` — benchmarked, documented, pushed to GitHub.

---

### WEEK 2 — June 25 – July 1: Parallel Patterns & Profiling
**Theme: Make the GPU work smart, not just fast**

| Day | Focus | Tasks |
|-----|-------|-------|
| Wed Jun 25 | Parallel Reduction | Implement parallel sum reduction. Start naive (atomic adds), then optimize with shared memory. |
| Thu Jun 26 | Shared Memory Deep Dive | Learn bank conflicts. Implement shared memory optimized reduction. Measure speedup with `cudaEvent_t`. |
| Fri Jun 27 | NVIDIA Nsight | Profile your vector add and reduction with Nsight Systems. Learn to read the timeline. Identify memory vs compute bottlenecks. |
| Sat Jun 28 | Memory Coalescing | Learn what coalesced vs uncoalesced access means. Write both — measure the difference. This is critical for all future kernels. |
| Sun Jun 29 | Streams & Async | Introduction to CUDA streams. Implement async memory transfers. |
| Mon Jun 30 | Warp-Level Primitives | Learn `__syncwarp()`, shuffle instructions. Implement warp-level reduction. |
| Tue Jul 1 | Week 2 Deliverable | Deliverable: Fully optimized reduction kernel with Nsight profiling screenshots, committed to GitHub with a benchmark table in the README. |

**Deliverable:** `kernels/reduction.cu` + Nsight screenshots in `/docs/profiling/`

---

### WEEK 3 — July 2–8: Matrix Multiplication (GEMM)
**Theme: The heartbeat of every neural network**

> This is the most important week. Every layer in a neural network is fundamentally matrix multiplication. Mastering this is the skill that separates real GPU engineers from framework users.

| Day | Focus | Tasks |
|-----|-------|-------|
| Wed Jul 2 | Naive GEMM | Implement matrix multiplication: C = A × B. Each thread computes one output element. Test correctness against CPU result. |
| Thu Jul 3 | Tiled GEMM | Implement shared memory tiling. Load tiles of A and B into shared memory. This is the classic GPU optimization — understand every line. |
| Fri Jul 4 | Benchmark GEMM | Profile naive vs tiled. Try different tile sizes (16×16, 32×32). Plot speedup vs matrix size. |
| Sat Jul 5 | Vectorized Loads | Use `float4` for 128-bit loads. Measure improvement. |
| Sun Jul 6 | Double Buffering | Implement software pipelining — prefetch next tile while computing current. |
| Mon Jul 7 | cuBLAS Comparison | Compare your GEMM against NVIDIA's `cuBLAS` (the professional library). You won't beat it, but you'll understand WHY it's faster. |
| Tue Jul 8 | Week 3 Deliverable | Deliverable: `kernels/matmul.cu` — naive + tiled + vectorized. Benchmark chart comparing CPU / naive CUDA / tiled CUDA / cuBLAS across sizes 256×256 to 4096×4096. |

**Key Concept:** Your tiled GEMM should be ~5-10x faster than naive CUDA, and cuBLAS will be another 2-5x faster. Understanding that gap is the actual lesson.

**Deliverable:** `kernels/matmul.cu` + `benchmarks/gemm_benchmark.py` with matplotlib charts.

---

### WEEK 4 — July 9–15: 2D Convolution in CUDA
**Theme: The core of computer vision**

| Day | Focus | Tasks |
|-----|-------|-------|
| Wed Jul 9 | Convolution Theory | Review: what a convolution does mathematically. Implement CPU convolution in Python first to verify understanding. |
| Thu Jul 10 | Naive Conv2D Kernel | Implement direct convolution in CUDA. Each output pixel = one thread. Test on a grayscale image with edge detection filter. |
| Fri Jul 11 | Constant Memory | Move the convolution filter to constant memory (`__constant__`). Measure speedup. |
| Sat Jul 12 | Shared Memory Conv | Implement conv with shared memory input tiles + halo cells (border handling). |
| Sun Jul 13 | Depthwise Conv | Implement depthwise separable convolution (used in MobileNet, efficient for mobile AI). |
| Mon Jul 14 | Visual Test | Apply your CUDA convolution to an actual image: Gaussian blur, Sobel edge detection, sharpening. Save output images. |
| Tue Jul 15 | Week 4 Deliverable | Deliverable: `kernels/conv2d.cu` — tested on real images, benchmarked, with visual outputs in `/docs/samples/`. |

**Deliverable:** `kernels/conv2d.cu` + example images showing your filters applied.

---

### WEEK 5 — July 16–22: Neural Network Layers
**Theme: Turn primitives into a learning machine**

| Day | Focus | Tasks |
|-----|-------|-------|
| Wed Jul 16 | Activation Functions | Implement ReLU, Leaky ReLU, Sigmoid, Softmax as CUDA kernels. Test numerically against PyTorch. |
| Thu Jul 17 | Batch Normalization | Implement forward pass of BatchNorm in CUDA (requires parallel mean and variance reduction). |
| Fri Jul 18 | Max Pooling | Implement 2D max pooling kernel. Test correctness. |
| Sat Jul 19 | Layer Abstraction | Design a `Layer` struct/class in C++ that holds forward/backward function pointers, weight tensors, and gradient tensors. |
| Sun Jul 20 | Loss Function | Implement cross-entropy loss and MSE loss kernels. |
| Mon Jul 21 | Forward Pass Pipeline | Chain your layers: Conv → BatchNorm → ReLU → Pool → FC. Run an image through. Verify shapes match at each stage. |
| Tue Jul 22 | Week 5 Deliverable | Deliverable: All layer kernels in `kernels/`, layer abstraction in `network/layer.cu`, forward pass working end-to-end on random data. |

**Deliverable:** Full set of NN layer kernels. A `test_forward_pass.cu` that builds a small net and runs data through it.

---

### WEEK 6 — July 23–29: Backpropagation in CUDA
**Theme: Teaching the GPU to learn**

> This is the hardest week. Backprop through a GPU-resident network requires computing gradients for every kernel you've written. Take your time — understanding this deeply is what sets you apart from everyone who just uses PyTorch.

| Day | Focus | Tasks |
|-----|-------|-------|
| Wed Jul 23 | Backprop Theory | Review: chain rule, gradient flow through each layer type. Derive gradients for Conv, BatchNorm, ReLU on paper first. |
| Thu Jul 24 | Gradient of Activations | Implement backward kernels for ReLU, Sigmoid. These are element-wise and straightforward — build confidence. |
| Fri Jul 25 | Gradient of Conv | Implement backward pass for Conv2D (gradient w.r.t. inputs and w.r.t. filter weights). This is the hard one. |
| Sat Jul 26 | Gradient of BatchNorm | Implement backward pass for BatchNorm. Requires parallel reductions again. |
| Sun Jul 27 | Gradient of Pooling | Implement backward pass for Max Pool (gradient routes only through the max element). |
| Mon Jul 28 | SGD Optimizer Kernel | Implement stochastic gradient descent update: `w = w - lr * dw`. Implement momentum SGD. |
| Tue Jul 29 | Training Loop | Write the full training loop: forward → compute loss → backward → update weights. Train a tiny network on MNIST digits (or XOR) to verify learning works. Watch loss go down. |

**Deliverable:** `network/optimizer.cu`, full backward pass for all layers, a working training loop that demonstrably reduces loss on a simple dataset.

---

### WEEK 7 — July 30 – August 5: Object Detection Application
**Theme: Apply everything to a real AI problem**

| Day | Focus | Tasks |
|-----|-------|-------|
| Wed Jul 30 | Dataset Setup | Download a subset of COCO dataset (use `data/prepare_dataset.py`). Choose 5-10 object classes. Preprocess to 416×416. |
| Thu Jul 31 | Model Architecture | Design a YOLO-inspired architecture using your custom layers. Keep it lightweight: ~5 conv layers + detection head. Draw the architecture diagram. |
| Fri Aug 1 | Detection Head | Implement the bounding box prediction head in CUDA: predict (x, y, w, h, confidence, class) per grid cell. |
| Sat Aug 2 | Loss Function | Implement YOLO-style detection loss: localization loss + confidence loss + classification loss. |
| Sun Aug 3 | Non-Max Suppression | Implement NMS in CUDA — suppress overlapping bounding boxes in parallel. |
| Mon Aug 4 | Training Run | Train on your COCO subset. Expect 20-40+ epochs. Use your profiler to ensure GPU utilization stays high. |
| Tue Aug 5 | Week 7 Deliverable | Deliverable: Working end-to-end object detector using your custom CUDA kernels. Can detect objects in test images. |

**Deliverable:** `detection/` directory — full object detection pipeline.

---

### WEEK 8 — August 6–12: Optimization & Benchmarking
**Theme: Make it fast, then prove it**

| Day | Focus | Tasks |
|-----|-------|-------|
| Wed Aug 6 | Full Profiling Session | Profile the complete inference pipeline with Nsight. Find the top 3 bottlenecks. |
| Thu Aug 7 | Optimize Bottleneck #1 | Apply targeted optimizations: shared memory, vectorized loads, kernel fusion where possible. |
| Fri Aug 8 | Optimize Bottleneck #2 | Continue optimization. Measure improvement after each change. |
| Sat Aug 9 | CPU Baseline | Implement the same network in pure Python/NumPy as a CPU baseline. |
| Sun Aug 10 | Benchmark Suite | Write `benchmarks/benchmark.py`: measures latency at batch sizes 1, 4, 16, 32. Plot GPU vs CPU speedup. |
| Mon Aug 11 | Real-Time Webcam Demo | Connect inference pipeline to webcam with OpenCV. Target: >10 FPS real-time detection. Record a demo video. |
| Tue Aug 12 | Benchmark Report | Generate final benchmark charts: throughput (images/sec), latency (ms), GPU utilization %, memory usage. |

**Deliverable:** `benchmarks/` with final performance charts. A demo video. Documented speedup numbers.

---

### WEEK 9 — August 13–21: Portfolio & Launch
**Theme: Make it impossible to ignore**

| Day | Focus | Tasks |
|-----|-------|-------|
| Wed Aug 13 | Code Cleanup | Refactor all code: consistent naming, remove dead code, add inline comments explaining the "why" not just the "what". |
| Thu Aug 14 | Documentation | Write thorough documentation for every CUDA kernel: what it does, the optimization technique used, the performance characteristics. |
| Fri Aug 15 | README | Write a world-class GitHub README: architecture diagram, benchmark results table, setup instructions, demo GIF. |
| Sat Aug 16 | Technical Write-Up | Write a 1,500-word technical blog post on Medium or dev.to: "I Built a CUDA Object Detector from Scratch." Explain the key insights. |
| Sun Aug 17 | Demo Video | Record a clean 3-5 minute demo: explain the project, show the webcam working, walk through a key piece of CUDA code. |
| Mon Aug 18 | GitHub Polish | Add GitHub topics, license, contribution guide. Make it searchable. |
| Tue Aug 19 | Share & Submit | Share to r/CUDA, r/MachineLearning, Hacker News (Show HN). DM robotics/AI professors at UMD with the project link. |
| Wed-Thu Aug 20-21 | Buffer + Reflection | Catch-up days. Extend any under-developed week. Write a personal reflection on what you learned. |

**Final Deliverable:** A complete, public GitHub repository that any NVIDIA recruiter or university professor can open and understand in 5 minutes.

---

## Daily Schedule Template
*(3-4 hours/day, 5-6 days/week — leaves room for life)*

```
Morning (1.5 hrs):  Read / learn the concept for the day
Afternoon (2 hrs):  Implement the CUDA kernel or feature
Evening (30 min):   Push to GitHub, update README, journal what was hard
```

---

## Learning Resources (All Free Unless Noted)

| Resource | What It's For | Cost |
|----------|--------------|------|
| [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | Official reference | Free |
| [CUDA by Example](https://developer.download.nvidia.com/books/cuda-by-example/cuda-by-example-sample.pdf) | Beginner walkthroughs | Free |
| "Programming Massively Parallel Processors" (Kirk & Hwu) | Deep understanding, used in CMU/MIT courses | ~$75 |
| [NVIDIA Developer Blog](https://developer.nvidia.com/blog/) | Real-world CUDA optimization articles | Free |
| [Nsight Systems](https://developer.nvidia.com/nsight-systems) | GPU profiling | Free |
| [Simon Boehm's CUDA MatMul Blog](https://siboehm.com/articles/22/CUDA-MMM) | Best practical GEMM guide on the internet | Free |
| [nand2tetris](https://www.nand2tetris.org/) | Optional: understand hardware from first principles | Free |
| Google Colab Pro | Cloud backup, sharing demos | $10/month |

---

## Cost Diagram

> Since you already have an NVIDIA GPU, hardware costs are minimal. Here's every recommended purchase:

### Tier 1 — Essential ($0–$85)

| Item | Purpose | Est. Cost |
|------|---------|-----------|
| CUDA Toolkit + cuDNN | Core development tools | **Free** |
| Nsight Systems / Nsight Compute | GPU profiling | **Free** |
| VS Code + extensions | IDE | **Free** |
| Python + PyTorch (for comparison) | Benchmarking & visualization | **Free** |
| "Programming Massively Parallel Processors" (4th ed.) | The definitive textbook — used in MIT/CMU GPU courses | ~$75 |
| **Tier 1 Total** | | **~$75** |

### Tier 2 — Strongly Recommended ($130–$180)

| Item | Purpose | Est. Cost |
|------|---------|-----------|
| External SSD (1TB, USB 3.2) | Store COCO dataset (~25GB) + model checkpoints | ~$65–90 |
| USB Webcam (1080p) | Real-time inference demo | ~$35–60 |
| Google Colab Pro (2 months) | Cloud backup, share demo notebooks, run experiments if your GPU is busy | ~$20 |
| **Tier 2 Total** | | **~$120–$170** |

### Tier 3 — Optional Upgrades ($0–$200)

| Item | Purpose | Est. Cost |
|------|---------|-----------|
| NVIDIA Jetson Orin Nano | Deploy your model on embedded hardware — links to robotics background | ~$150 |
| USB-C Hub / extra cables | Connectivity, monitors | ~$30–50 |
| Upgraded cooling (if desktop) | Sustained GPU compute runs hot | ~$30–60 |
| **Tier 3 Total** | | **~$0–$260** |

### Total Estimated Cost

| Scenario | Total |
|---------|-------|
| Bare minimum (book only) | **~$75** |
| Full recommended setup | **~$195–$255** |
| With Jetson for robotics tie-in | **~$345–$515** |

---

## What This Project Proves (For College Apps & Career)

When you're done, you can say — with a working GitHub repo to prove it:

- "I implemented matrix multiplication, 2D convolution, and backpropagation directly in CUDA C++ without frameworks"
- "I achieved X× speedup over CPU baseline on a custom object detection pipeline"
- "I profiled and optimized GPU memory access patterns using NVIDIA Nsight"

That puts you in a category of applicants — and eventually engineers — that is extremely rare at any level, let alone as a high school student.

---

## Quick Reference: Key CUDA Concepts by Week

| Week | Concept | Why It Matters |
|------|---------|---------------|
| 1 | Thread/block/grid hierarchy | Every kernel you write depends on this |
| 2 | Shared memory + coalescing | The difference between slow and fast kernels |
| 3 | Tiled GEMM | Foundation of all neural network math |
| 4 | Convolution | Foundation of all computer vision |
| 5 | Layer composition | How frameworks like PyTorch are actually built |
| 6 | Backpropagation in CUDA | What NO tutorial covers — your differentiator |
| 7 | Object detection pipeline | The application that ties everything together |
| 8 | Profiling + optimization | What separates junior from senior GPU engineers |
| 9 | Documentation + portfolio | What gets you the internship/research opportunity |
