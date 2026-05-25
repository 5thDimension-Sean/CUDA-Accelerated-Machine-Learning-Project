# benchmark.py — Week 8
# Measures GPU vs CPU performance and generates comparison charts.
#
# Run: python benchmarks/benchmark.py
# Requires: pip install matplotlib numpy torch

import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
SIZES      = [256, 512, 1024, 2048, 4096]   # matrix sizes to benchmark
BATCH_SIZES = [1, 4, 16, 32]                # inference batch sizes
REPEATS    = 10                              # runs per measurement (take average)

# ── Timing helper ─────────────────────────────────────────────────────────────
def time_cpu_matmul(n, repeats=REPEATS):
    """Time numpy matrix multiplication as CPU baseline."""
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)

    # Warmup
    _ = A @ B

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        C = A @ B
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)   # ms

    return np.mean(times)

def time_gpu_matmul(n, repeats=REPEATS):
    """Time PyTorch GPU matrix multiplication."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("  No CUDA GPU found — skipping GPU benchmark")
            return None

        device = torch.device("cuda")
        A = torch.randn(n, n, device=device)
        B = torch.randn(n, n, device=device)

        # Warmup
        _ = A @ B
        torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            C = A @ B
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        return np.mean(times)

    except ImportError:
        print("PyTorch not installed — run: pip install torch")
        return None

# ── Run benchmarks ────────────────────────────────────────────────────────────
def run_matmul_benchmark():
    print("Matrix Multiplication Benchmark")
    print("=" * 50)
    print(f"{'Size':>8}  {'CPU (ms)':>10}  {'GPU (ms)':>10}  {'Speedup':>8}")
    print("-" * 50)

    cpu_times = []
    gpu_times = []
    speedups  = []

    for n in SIZES:
        cpu_ms = time_cpu_matmul(n)
        gpu_ms = time_gpu_matmul(n)

        cpu_times.append(cpu_ms)

        if gpu_ms is not None:
            gpu_times.append(gpu_ms)
            speedup = cpu_ms / gpu_ms
            speedups.append(speedup)
            print(f"{n:>8}  {cpu_ms:>10.2f}  {gpu_ms:>10.2f}  {speedup:>7.1f}x")
        else:
            gpu_times.append(None)
            speedups.append(None)
            print(f"{n:>8}  {cpu_ms:>10.2f}  {'N/A':>10}  {'N/A':>8}")

    return cpu_times, gpu_times, speedups

# ── Plot results ─────────────────────────────────────────────────────────────
def plot_results(sizes, cpu_times, gpu_times, speedups):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GPU vs CPU Performance — CUDA ML Project", fontsize=14, fontweight="bold")

    # Plot 1: Latency comparison
    ax1.plot(sizes, cpu_times, "o-", label="CPU (NumPy)", color="#888780", linewidth=2)
    if any(t is not None for t in gpu_times):
        valid_gpu = [(s, t) for s, t in zip(sizes, gpu_times) if t is not None]
        ax1.plot(*zip(*[(s, t) for s, t in valid_gpu]), "o-",
                 label="GPU (CUDA)", color="#534AB7", linewidth=2)
    ax1.set_xlabel("Matrix Size (N×N)")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Latency (lower is better)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Speedup
    valid = [(s, sp) for s, sp in zip(sizes, speedups) if sp is not None]
    if valid:
        s_vals, sp_vals = zip(*valid)
        bars = ax2.bar(range(len(s_vals)), sp_vals, color="#1D9E75", alpha=0.85)
        ax2.set_xticks(range(len(s_vals)))
        ax2.set_xticklabels([str(s) for s in s_vals])
        ax2.set_xlabel("Matrix Size (N×N)")
        ax2.set_ylabel("Speedup (×)")
        ax2.set_title("GPU Speedup over CPU")
        ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax2.grid(True, alpha=0.3, axis="y")

        # Label each bar
        for bar, val in zip(bars, sp_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}×", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig("benchmarks/matmul_benchmark.png", dpi=150, bbox_inches="tight")
    print("\nChart saved: benchmarks/matmul_benchmark.png")
    plt.show()

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cpu_times, gpu_times, speedups = run_matmul_benchmark()
    plot_results(SIZES, cpu_times, gpu_times, speedups)

    if any(s is not None for s in speedups):
        valid = [s for s in speedups if s is not None]
        print(f"\nPeak speedup: {max(valid):.1f}× at N={SIZES[speedups.index(max(valid))]}")
        print("This is what goes in your GitHub README.")
