// ============================================================================
// activations.cu — Week 5
// Goal: Implement neural network activation functions as CUDA kernels.
//
// Functions to implement:
//   - ReLU:       f(x) = max(0, x)         backward: 1 if x > 0 else 0
//   - Leaky ReLU: f(x) = max(0.01x, x)     backward: 1 if x > 0 else 0.01
//   - Sigmoid:    f(x) = 1 / (1 + e^-x)   backward: f(x) * (1 - f(x))
//   - Softmax:    f(x_i) = e^x_i / sum(e^x) (requires parallel reduction)
//
// Each activation needs TWO kernels: forward pass + backward pass (gradient).
// The backward pass is used in Week 6 (backpropagation).
//
// Verify against PyTorch:
//   import torch
//   x = torch.tensor([...])
//   torch.nn.functional.relu(x)
// ============================================================================

// TODO — Week 5: implement here
