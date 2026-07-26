import numpy as np


SRC_X  = "mnist_X.bin"
SRC_Y  = "mnist_Y.bin"
CANVAS = 64          # output image side (px)
DIGIT  = 28          # MNIST digit side (px)
S      = 4           # grid cells per side (CANVAS must be divisible by S)
C      = 10          # number of classes
NUM    = 10000       # detection images to generate
K      = 1           # digits per image (v1 = 1; raise to 2-3 once the pipeline is green)
SEED   = 42


assert CANVAS % S == 0, "CANVAS must be divisible by S"
CELL = CANVAS // S       # 16
VALS = 5 + C             # per-cell vector length = 15
np.random.seed(SEED)

digits = np.fromfile(SRC_X, dtype=np.float32).reshape(-1, DIGIT, DIGIT)
onehot = np.fromfile(SRC_Y, dtype=np.float32).reshape(-1, C)
labels = onehot.argmax(1)
Nsrc = digits.shape[0]
print(f"source: {Nsrc} digits")

X    = np.zeros((NUM, CANVAS, CANVAS), dtype=np.float32)
Y    = np.zeros((NUM, S, S, VALS),     dtype=np.float32)
META = np.full ((NUM, K, 5), -1.0,     dtype=np.float32)

for i in range(NUM):
    used_cells = set()
    for k in range(K):
        j   = np.random.randint(Nsrc)
        d   = digits[j]
        lab = int(labels[j])

        tx = np.random.randint(0, CANVAS - DIGIT + 1)
        ty = np.random.randint(0, CANVAS - DIGIT + 1)
        cx = tx + DIGIT / 2.0
        cy = ty + DIGIT / 2.0
        gx = int(cx // CELL)
        gy = int(cy // CELL)

        if (gx, gy) in used_cells:
            continue
        used_cells.add((gx, gy))

        X[i, ty:ty+DIGIT, tx:tx+DIGIT] = np.maximum(X[i, ty:ty+DIGIT, tx:tx+DIGIT], d)

        Y[i, gy, gx, 0]       = 1.0                       # confidence
        Y[i, gy, gx, 1]       = (cx - gx*CELL) / CELL     # x offset in cell
        Y[i, gy, gx, 2]       = (cy - gy*CELL) / CELL     # y offset in cell
        Y[i, gy, gx, 3]       = DIGIT / float(CANVAS)     # w (canvas-normalized)
        Y[i, gy, gx, 4]       = DIGIT / float(CANVAS)     # h
        Y[i, gy, gx, 5 + lab] = 1.0                       # class one-hot
        META[i, k] = [cx, cy, DIGIT, DIGIT, lab]

X.tofile("det_X.bin")
Y.reshape(NUM, -1).tofile("det_Y.bin")
META.tofile("det_meta.bin")

print(f"wrote det_X.bin   {X.shape}  -> {NUM} x {CANVAS*CANVAS}")
print(f"wrote det_Y.bin   {(NUM, S, S, VALS)}  -> {NUM} x {S*S*VALS}")
print(f"wrote det_meta.bin {(NUM, K, 5)}")

# quick sanity print for image 0
gy, gx = np.argwhere(Y[0, :, :, 0] > 0)[0]
print(f"sample0: box center cell ({gx},{gy})  label={int(Y[0,gy,gx,5:].argmax())}  "
      f"x={Y[0,gy,gx,1]:.3f} y={Y[0,gy,gx,2]:.3f} w={Y[0,gy,gx,3]:.3f} h={Y[0,gy,gx,4]:.3f}")
