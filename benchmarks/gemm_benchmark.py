import matplotlib.pyplot as plt

# Hard-coded values from matmul_benchmark.exe — update these after each run
sizes = [256, 512, 1024, 2048, 4096]

naive_ms  = [0.106, 0.561, 9.483, 80.942, 636.052]
tiled_ms  = [0.104, 1.302, 9.913, 78.885, 632.828]
float4_ms = [0.034, 0.173, 2.061, 19.787, 153.685]
cublas_ms = [1.465, 0.146, 1.098, 14.081, 117.864]
cpu_ms    = [13.897, 173.424, 1413.980, 46888.213, None]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('CUDA Matrix Multiplication Benchmark', fontsize=16, fontweight='bold')

# Chart 1: Execution time (log scale)
ax1.plot(sizes, naive_ms,  marker='o', label='Naive CUDA')
ax1.plot(sizes, tiled_ms,  marker='s', label='Tiled CUDA')
ax1.plot(sizes, float4_ms, marker='^', label='Float4 CUDA')
ax1.plot(sizes, cublas_ms, marker='D', label='cuBLAS')
ax1.plot(sizes[:4], cpu_ms[:4], marker='x', linestyle='--', label='CPU')
ax1.set_xlabel('Matrix Size (N)')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Execution Time')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# Chart 2: Speedup vs CPU (N=4096 excluded — no CPU baseline)
sizes_cpu      = sizes[:4]
naive_speedup  = [cpu_ms[i] / naive_ms[i]  for i in range(4)]
tiled_speedup  = [cpu_ms[i] / tiled_ms[i]  for i in range(4)]
float4_speedup = [cpu_ms[i] / float4_ms[i] for i in range(4)]
cublas_speedup = [cpu_ms[i] / cublas_ms[i] for i in range(4)]

ax2.plot(sizes_cpu, naive_speedup,  marker='o', label='Naive CUDA')
ax2.plot(sizes_cpu, tiled_speedup,  marker='s', label='Tiled CUDA')
ax2.plot(sizes_cpu, float4_speedup, marker='^', label='Float4 CUDA')
ax2.plot(sizes_cpu, cublas_speedup, marker='D', label='cuBLAS')
ax2.set_xlabel('Matrix Size (N)')
ax2.set_ylabel('Speedup vs CPU')
ax2.set_title('Speedup over CPU Baseline')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('gemm_benchmark.png', dpi=150)
plt.show()
