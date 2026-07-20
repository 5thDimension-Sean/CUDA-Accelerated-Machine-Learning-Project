import numpy as np

# Read the data back
data = np.fromfile("mnist_test_X.bin", dtype=np.float32)
data = data.reshape(-1, 784) 

print(data.shape)


data1 = np.fromfile("mnist_test_Y.bin", dtype=np.float32)
data1 = data1.reshape(-1, 784) 

print(data1.shape)