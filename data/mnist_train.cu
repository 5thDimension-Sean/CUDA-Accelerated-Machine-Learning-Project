#include "..\kernels\common.cuh"       // CUDA_CHECK
#include "fc.cuh"           // ff/fb
#include "activations.cuh"  // srs
#include "loss.cuh"         // ce
#include "optimizer.cuh"    // sgd
#include <cmath>
#include <cstdio>

void load_bin(const char *path, float *dst, size_t count){
    FILE *f = fopen(path, "rb");
    if (!f) { printf("could not open %s\n", path); exit(1); }
    size_t got = fread(dst, sizeof(float), count, f);
    if(got != count){
        printf("Error reading file %s: expected %zu floats, got %zu\n", path, count, got);
        exit(1);
    }else{
        fclose(f);
    }
}
int main(){
    const int N = 1000, IN = 784, HIDDEN = 128, OUT = 10; //1000 can be changed / n size can be changed
    float *X, *Y;
    float *W1, *b1, *W2, *b2;
    CUDA_CHECK(cudaMallocHost((void**)&W1, HIDDEN * IN  * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&b1, HIDDEN       * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&W2, OUT * HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&b2, OUT          * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&X,  N * IN * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&Y,  N * OUT * sizeof(float)));
    load_bin("mnist_X.bin", X, N*IN);
    load_bin("mnist_Y.bin", Y, N*OUT);
    srand(42);  // fixed seed

    float scale1 = sqrtf(2.0f / IN); 
    for (int i = 0; i < HIDDEN * IN; i++)
        W1[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale1;
    for (int i = 0; i < HIDDEN; i++) b1[i] = 0.0f;

    float scale2 = sqrtf(2.0f / HIDDEN); 
    for (int i = 0; i < OUT * HIDDEN; i++)
        W2[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale2;
    for (int i = 0; i < OUT; i++) b2[i] = 0.0f;
    //gradient buffers
    static float z1[N * HIDDEN] = {0.0f};
    static float a1[N * HIDDEN] = {0.0f};
    static float z2[N * OUT] = {0.0f};
    static float a2[N * OUT] = {0.0f};
    static float dZ2[N * OUT] = {0.0f};
    static float dW2[OUT * HIDDEN] = {0.0f};
    static float db2[OUT] = {0.0f};
    static float dA1[N * HIDDEN] = {0.0f};
    static float dZ1[N * HIDDEN] = {0.0f};
    static float dW1[HIDDEN * IN] = {0.0f};
    static float db1[HIDDEN] = {0.0f};
    static float dX[N * IN] = {0.0f};
    float lr = 0.5f;
     for (int epoch = 0; epoch < 1000; ++epoch) {
        fc_forward(X, W1, b1, z1, N, IN, HIDDEN);
        for (int i = 0; i < N * HIDDEN; ++i) {
            a1[i] = z1[i] > 0 ? z1[i] : 0; // ReLU activation
        }

        fc_forward(a1, W2, b2, z2, N, HIDDEN, OUT);
       for (int n = 0; n < N; ++n) {
            float m = z2[n*OUT];                            
            for (int c = 1; c < OUT; ++c) if (z2[n*OUT+c] > m) m = z2[n*OUT+c];

            float sum = 0.0f;                   
            for (int c = 0; c < OUT; ++c) {
                a2[n*OUT+c] = expf(z2[n*OUT+c] - m);
                sum += a2[n*OUT+c];
            }
            for (int c = 0; c < OUT; ++c) a2[n*OUT+c] /= sum; 
        }

        float loss = 0.0f;
        for (int i = 0; i < N*OUT; ++i) {
            loss +=Y[i] * logf(a2[i] + 1e-8f); // Cross-entropy loss
        }
        loss = -loss / N;
       for (int i = 0; i < N * OUT; ++i)
        dZ2[i] = (a2[i] - Y[i]) / N;

        fc_backward(dZ2, a1, W2, dW2, db2, dA1, N, HIDDEN, OUT);

        for (int i = 0; i < N * HIDDEN; ++i)
            dZ1[i] = dA1[i] * (z1[i] > 0 ? 1.0f : 0.0f);

        fc_backward(dZ1, X, W1, dW1, db1, dX, N, IN, HIDDEN);

        sgd(W1, dW1, lr, HIDDEN * IN);    // 128 * 784
        sgd(b1, db1, lr, HIDDEN);         // 128
        sgd(W2, dW2, lr, OUT * HIDDEN);   // 10 * 128
        sgd(b2, db2, lr, OUT);            // 10

    //100,352 weights
    }
    int correct = 0;
    for (int n = 0; n < N; ++n) {
      int pred = 0, truth = 0;
      for (int c = 1; c < OUT; ++c) {
          if (a2[n*OUT + c] > a2[n*OUT + pred]) pred = c;   // predicted digit
          if (Y [n*OUT + c] > Y [n*OUT + truth]) truth = c; // true digit
      }
      if (pred == truth) correct++;
    }
     std::printf("final accuracy = %.2f%% (%d/%d)\n", 100.0f * correct / N, correct, N);

    for (int n = 0; n < 10; ++n) {
      int pred = 0, truth = 0;
      for (int c = 1; c < OUT; ++c) {
          if (a2[n*OUT + c] > a2[n*OUT + pred]) pred = c;
          if (Y [n*OUT + c] > Y [n*OUT + truth]) truth = c;
      }
      std::printf("sample %d: predicted %d, actual %d\n", n, pred, truth);
    }
    load_bin("mnist_test_X.bin", X, N*IN);
    load_bin("mnist_test_Y.bin", Y, N*OUT);
    fc_forward(X, W1, b1, z1, N, IN, HIDDEN);
    for (int i=0;i<N*HIDDEN;++i) a1[i] = z1[i]>0?z1[i]:0;
    fc_forward(a1, W2, b2, z2, N, HIDDEN, OUT);
    for (int n=0;n<N;++n){ /* softmax */ }
    int correct = 0;
    for (int n = 0; n < N; ++n) {
      int pred = 0, truth = 0;
      for (int c = 1; c < OUT; ++c) {
          if (a2[n*OUT + c] > a2[n*OUT + pred]) pred = c;   // predicted digit
          if (Y [n*OUT + c] > Y [n*OUT + truth]) truth = c; // true digit
      }
      if (pred == truth) correct++;
    }
     std::printf("final accuracy(test) = %.2f%% (%d/%d)\n", 100.0f * correct / N, correct, N);

    for (int n = 0; n < 10; ++n) {
      int pred = 0, truth = 0;
      for (int c = 1; c < OUT; ++c) {
          if (a2[n*OUT + c] > a2[n*OUT + pred]) pred = c;
          if (Y [n*OUT + c] > Y [n*OUT + truth]) truth = c;
      }
      std::printf("sample %d: predicted %d, actual %d\n", n, pred, truth);
    }
    return 0;
}