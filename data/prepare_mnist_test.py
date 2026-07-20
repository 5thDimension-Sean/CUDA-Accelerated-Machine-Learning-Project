import numpy as np

data = np.loadtxt("mnist_test.csv/mnist_test.csv", delimiter=",", skiprows=1)
labels = data[:, 0].astype(int)
pixels = (data[:, 1:] / 255.0).astype(np.float32)  # Normalize pixel values to [0, 1]
onehot = np.eye(10, dtype=np.float32)[labels]

pixels.tofile("mnist_X.bin")
onehot.tofile("mnist_Y.bin")
print("wrote", pixels.shape[0], "samples")

