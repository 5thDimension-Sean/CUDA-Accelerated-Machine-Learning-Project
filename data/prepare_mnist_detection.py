
import numpy as np

CANVAS = 64          # output image side (px)
DIGIT  = 28          # MNIST digit side (px)
S      = 4           # grid cells per side (CANVAS must be divisible by S)
C      = 10          # number of classes
K      = 1           # digits per image (v1 = 1; raise to 2-3 for multi-object + NMS)


assert CANVAS % S == 0, "CANVAS must be divisible by S"
CELL = CANVAS // S     
VALS = 5 + C         


def generate(src_x, src_y, out_prefix, num, seed):
    """Composite `num` detection images from the digits in src_x/src_y."""
    rng = np.random.default_rng(seed)
    digits = np.fromfile(src_x, dtype=np.float32).reshape(-1, DIGIT, DIGIT)
    labels = np.fromfile(src_y, dtype=np.float32).reshape(-1, C).argmax(1)
    Nsrc = digits.shape[0]

    X    = np.zeros((num, CANVAS, CANVAS), dtype=np.float32)
    Y    = np.zeros((num, S, S, VALS),     dtype=np.float32)
    META = np.full ((num, K, 5), -1.0,     dtype=np.float32)

    for i in range(num):
        used_cells = set()
        for k in range(K):
            j   = int(rng.integers(Nsrc))
            d   = digits[j]
            lab = int(labels[j])
            tx  = int(rng.integers(0, CANVAS - DIGIT + 1))
            ty  = int(rng.integers(0, CANVAS - DIGIT + 1))
            cx, cy = tx + DIGIT/2.0, ty + DIGIT/2.0
            gx, gy = int(cx // CELL), int(cy // CELL)
            if (gx, gy) in used_cells:        # one box per cell (YOLOv1 limit)
                continue
            used_cells.add((gx, gy))
            X[i, ty:ty+DIGIT, tx:tx+DIGIT] = np.maximum(X[i, ty:ty+DIGIT, tx:tx+DIGIT], d)
            Y[i, gy, gx, 0]       = 1.0
            Y[i, gy, gx, 1]       = (cx - gx*CELL) / CELL
            Y[i, gy, gx, 2]       = (cy - gy*CELL) / CELL
            Y[i, gy, gx, 3]       = DIGIT / float(CANVAS)
            Y[i, gy, gx, 4]       = DIGIT / float(CANVAS)
            Y[i, gy, gx, 5 + lab] = 1.0
            META[i, k] = [cx, cy, DIGIT, DIGIT, lab]

    X.tofile(f"{out_prefix}_X.bin")
    Y.reshape(num, -1).tofile(f"{out_prefix}_Y.bin")
    META.tofile(f"{out_prefix}_meta.bin")
    print(f"[{out_prefix}] {num} imgs  X:{X.shape}->{num}x{CANVAS*CANVAS}  "
          f"Y:{num}x{S*S*VALS}  (from {Nsrc} digits, seed {seed})")


generate("mnist_X.bin",      "mnist_Y.bin",      "det",      num=10000, seed=42)
generate("mnist_test_X.bin", "mnist_test_Y.bin", "det_test", num=10000, seed=123)
