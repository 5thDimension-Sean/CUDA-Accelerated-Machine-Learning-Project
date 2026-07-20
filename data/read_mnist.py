import numpy as np

# Read the data back
data = np.fromfile("mnist_X.bin", dtype=np.float32)
data = data.reshape(-1, 784) 

print(data.shape)