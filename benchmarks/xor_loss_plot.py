#!/usr/bin/env python3
"""
Plot the XOR training loss curve produced by network/train_xor.cu.

The C++ trainer writes `loss_curve.csv` (columns: epoch,loss). This script
styles it into a portfolio-ready figure that highlights the characteristic
"plateau then breakthrough" shape of learning XOR with a hidden layer.

Usage:
    python benchmarks/xor_loss_plot.py [path/to/loss_curve.csv]
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ---- locate the CSV (arg, cwd, or build dir) --------------------------------
CANDIDATES = [
    sys.argv[1] if len(sys.argv) > 1 else None,
    "loss_curve.csv",
    "build/bin/loss_curve.csv",
    "build/loss_curve.csv",
]
csv_path = next((p for p in CANDIDATES if p and os.path.exists(p)), None)
if csv_path is None:
    sys.exit("Could not find loss_curve.csv — pass its path as an argument.")

data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
epochs, loss = data[:, 0], data[:, 1]

# ---- find the "breakthrough": epoch of steepest descent ---------------------
d_loss = np.gradient(loss)
break_idx = int(np.argmin(d_loss))          # most negative slope
break_epoch, break_loss = epochs[break_idx], loss[break_idx]

# ---- palette ----------------------------------------------------------------
INK      = "#1b1f24"   # near-black text
ACCENT   = "#4c6ef5"   # indigo line
ACCENT2  = "#f03e3e"   # red marker
PLATEAU  = "#adb5bd"   # grey for the plateau band
GRID     = "#e9ecef"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.edgecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# plateau band (start of training up to the breakthrough)
ax.axvspan(epochs[0], break_epoch, color=PLATEAU, alpha=0.12, zorder=0)

# the loss curve
ax.plot(epochs, loss, color=ACCENT, linewidth=2.4, zorder=3,
        solid_capstyle="round", label="MSE loss")

# breakthrough marker + annotation
ax.scatter([break_epoch], [break_loss], color=ACCENT2, s=70, zorder=4,
           edgecolor="white", linewidth=1.5)
ax.annotate(
    f"breakthrough\n~epoch {int(break_epoch)}",
    xy=(break_epoch, break_loss),
    xytext=(break_epoch + 0.18 * (epochs[-1] - epochs[0]), break_loss + 0.06),
    fontsize=11, color=ACCENT2, weight="bold", ha="left", va="center",
    arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.6,
                    connectionstyle="arc3,rad=-0.2"),
)

# annotate the plateau
ax.annotate(
    "plateau\n(≈0.25, saddle)",
    xy=(epochs[max(1, break_idx // 3)], 0.25),
    fontsize=10.5, color="#495057", ha="center", va="bottom",
)

# start / end value callouts
ax.annotate(f"start  {loss[0]:.4f}", xy=(epochs[0], loss[0]),
            xytext=(6, 6), textcoords="offset points", fontsize=10, color="#495057")
ax.annotate(f"final  {loss[-1]:.4f}", xy=(epochs[-1], loss[-1]),
            xytext=(-6, 10), textcoords="offset points", fontsize=10,
            color=ACCENT, weight="bold", ha="right")

# ---- titles -----------------------------------------------------------------
ax.set_title("XOR Training Loss — a network learning from scratch in CUDA",
             fontsize=15, weight="bold", pad=14, loc="left")
ax.text(0.0, 1.008,
        "Single hidden layer (2→4→1), sigmoid, MSE. Flat near 0.25, then it cracks.",
        transform=ax.transAxes, fontsize=10.5, color="#868e96", va="bottom")

ax.set_xlabel("Epoch", fontsize=12.5, labelpad=8)
ax.set_ylabel("Mean squared error", fontsize=12.5, labelpad=8)

# ---- clean up the frame -----------------------------------------------------
ax.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_linewidth(1.1)
ax.margins(x=0.01)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(ScalarFormatter())

fig.tight_layout()
out = "xor_loss.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved {out}  ({len(epochs)} points, final loss {loss[-1]:.4f})")
