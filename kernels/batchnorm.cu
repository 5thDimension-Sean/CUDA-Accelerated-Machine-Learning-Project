// ============================================================================
// batchnorm.cu — Week 5
// Goal: Implement Batch Normalization forward + backward in CUDA.
//
// Forward pass:
//   1. Compute mean across the batch (parallel reduction)
//   2. Compute variance across the batch (parallel reduction)
//   3. Normalize: x_hat = (x - mean) / sqrt(variance + epsilon)
//   4. Scale and shift: y = gamma * x_hat + beta
//
// Backward pass requires gradients w.r.t. gamma, beta, and input x.
// This is the hardest backward pass you'll write — derive it on paper first.
//
// Resources:
//   - Original BatchNorm paper: arxiv.org/abs/1502.03167
//   - The backward pass derivation: kevinzakka.github.io/2016/09/14/batch_normalization
// ============================================================================

// TODO — Week 5: implement here
