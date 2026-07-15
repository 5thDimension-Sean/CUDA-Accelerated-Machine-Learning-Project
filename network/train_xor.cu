#include "fc.cuh"
#include "optimizer.cuh"
#include "common.cuh"
#include <cmath>
#include <cstdio>

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

int main() {
    const int batch = 4;
    const int in = 2;
    const int hidden = 4;
    const int out = 1;

    float X[batch * in] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    float targets[batch * out] = {0.0f, 1.0f, 1.0f, 0.0f};

    float W1[hidden * in] = {
        0.5f, -0.3f,
       -0.4f,  0.8f,
        0.2f,  0.6f,
       -0.7f,  0.1f
    };
    float b1[hidden] = {0.0f, 0.0f, 0.0f, 0.0f};

    float W2[out * hidden] = {0.3f, -0.6f, 0.4f, -0.2f};
    float b2[out] = {0.0f};

    float z1[batch * hidden] = {0.0f};
    float a1[batch * hidden] = {0.0f};
    float z2[batch * out] = {0.0f};
    float a2[batch * out] = {0.0f};

    float dA2[batch * out] = {0.0f};
    float dZ2[batch * out] = {0.0f};
    float dW2[out * hidden] = {0.0f};
    float db2[out] = {0.0f};
    float dA1[batch * hidden] = {0.0f};
    float dZ1[batch * hidden] = {0.0f};
    float dW1[hidden * in] = {0.0f};
    float db1[hidden] = {0.0f};
    float dX[batch * in] = {0.0f};
    float lr = 0.5f;

    for (int epoch = 0; epoch < 1000; ++epoch) {
        fc_forward(X, W1, b1, z1, batch, in, hidden);
        for (int i = 0; i < batch * hidden; ++i) {
            a1[i] = sigmoid(z1[i]);
        }

        fc_forward(a1, W2, b2, z2, batch, hidden, out);
        for (int i = 0; i < batch * out; ++i) {
            a2[i] = sigmoid(z2[i]);
        }

        float loss = 0.0f;
        for (int i = 0; i < batch * out; ++i) {
            float diff = a2[i] - targets[i];
            loss += diff * diff;
        }
        loss /= static_cast<float>(batch * out);

        for (int i = 0; i < batch * out; ++i) {
            dA2[i] = (2.0f / static_cast<float>(batch * out)) * (a2[i] - targets[i]);
            dZ2[i] = dA2[i] * a2[i] * (1.0f - a2[i]);
        }

        fc_backward(dZ2, a1, W2, dW2, db2, dA1, batch, hidden, out);

        for (int i = 0; i < batch * hidden; ++i) {
            dZ1[i] = dA1[i] * a1[i] * (1.0f - a1[i]);
        }
        fc_backward(dZ1, X, W1, dW1, db1, dX, batch, in, hidden);

        sgd(W1, dW1, lr, hidden * in);
        sgd(b1, db1, lr, hidden);
        sgd(W2, dW2, lr, out * hidden);
        sgd(b2, db2, lr, out);

        if ((epoch % 100) == 0) {
            std::printf("epoch %d loss = %.4f\n", epoch, loss);
        }
    }

    std::printf("final predictions:\n");
    for (int i = 0; i < batch; ++i) {
        std::printf("sample %d :  %.4f\n", i, a2[i]);
    }

    return 0;
}