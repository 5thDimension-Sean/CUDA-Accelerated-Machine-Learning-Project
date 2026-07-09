// ============================================================================
// test_forward_pass.cu — Week 5
// Drives the Layer abstraction: builds a RELU -> POOL chain and runs one input
// through it device-to-device (single copy in, single copy out).
//
// Build with -DBUILD_AS_LIBRARY so the kernel files' own main()s are skipped.
// ============================================================================
#include "layer.cuh"

int main() {
    // 4x4 single-channel input. Negatives so ReLU visibly clamps them to 0.
    const int H = 4, W = 4, C = 1;
    float h_input[H * W] = {
        -1,  3, -2,  9,
         8, -4,  7,  5,
        -1, -3,  6,  0,
        -5,  2, -4,  1
    };

    // --- copy input onto the device once ---
    float *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, H * W * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, H * W * sizeof(float), cudaMemcpyHostToDevice));

    // --- build the layers ---
    Layer relu = {};
    relu.type = LayerType::RELU;
    relu.in_H = H; relu.in_W = W; relu.in_C = C;
    layer_setup(&relu);

    Layer pool = {};
    pool.type = LayerType::POOL;
    pool.in_H = H; pool.in_W = W; pool.in_C = C;
    pool.P = 2; pool.S = 2;
    layer_setup(&pool);

    // --- forward: input -> ReLU -> Pool, staying on the device the whole time ---
    float *a = layer_forward(&relu, d_input);
    float *out = layer_forward(&pool, a);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- copy the final result back to the host once ---
    int out_n = pool.out_H * pool.out_W * pool.out_C;
    float *h_out = (float*)malloc(out_n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out, out, out_n * sizeof(float), cudaMemcpyDeviceToHost));

    // After ReLU the negatives become 0; then 2x2 max-pool over the 4x4:
    //   ReLU'd:            max-pooled 2x2:
    //   0 3 0 9            8 9
    //   8 0 7 5            2 6
    //   0 0 6 0
    //   0 2 0 1
    printf("RELU -> POOL output (%dx%d):\n", pool.out_H, pool.out_W);
    for (int i = 0; i < pool.out_H; i++) {
        for (int j = 0; j < pool.out_W; j++) printf("%.4f ", h_out[i * pool.out_W + j]);
        printf("\n");
    }

    free(h_out);
    cudaFree(d_input);
    layer_free(&relu);
    layer_free(&pool);
    return 0;
}
