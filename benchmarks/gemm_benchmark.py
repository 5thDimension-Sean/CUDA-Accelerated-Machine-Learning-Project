import matplotlib.pyplot as plt
import numpy as np
#This file for time sake will have hard coded values. edit the arrays to create plots for your own benchmarks, which can be ran by checking the commands on the README
sizes = [256, 512, 1024, 2048, 4096]

naive_ms  = [0.106, 0.561, 9.483, 80.942, 636.052]
tiled_ms  = [0.104, 1.302, 9.913, 78.885, 632.828]
float4_ms = [0.034, 0.173, 2.061, 19.787, 153.685]
cublas_ms = [1.465, 0.146, 1.098, 14.081, 117.864]
cpu_ms    = [13.897, 173.424, 1413.980, 46888.213, None]
#Naive time
fig, ax = plt.subplots()
ax.plot(sizes, naive_ms, label='Naive')
plt.show()
#Tiled Time
ax[1].plot(sizes, tiled_ms, label='Tiled')

#CPU baseline
ax[2].plot(sizes, cpu_ms, label='CPU')
#cuBlas Matmul time
ax[3].plot(sizes, cublas_ms, label='cuBlas')
#Vectorized matmul time
ax[4].plot(sizes, float4_ms, label='Vectorized')
#comparison
ax[5].plot(sizes, [cpu_ms[i]/tiled_ms[i] for i in range(len(sizes))], label='Tiled vs CPU')
ax[6].plot(sizes, [cpu_ms[i]/cublas_ms[i] for i in range(len(sizes))], label='cuBlas vs CPU')
ax[7].plot(sizes, [cpu_ms[i]/float4_ms[i] for i in range(len(sizes))], label='Vectorized vs CPU')
