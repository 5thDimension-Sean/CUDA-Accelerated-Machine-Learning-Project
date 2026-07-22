#include "common.cuh"
#include "conv2d_mc.cuh"
#include "pooling.cuh"
#include "activations.cuh"
#include "fc.cuh"
#include "loss.cuh"
#include "optimizer.cuh"
#include <cmath>
#include <cstdlib>

void load_bin(const char *path, float *dst, size_t count){
    FILE *f = fopen(path, "rb");
    if (!f) { printf("could not open %s\n", path); exit(1); }
    size_t got = fread(dst, sizeof(float), count, f);
    if (got != count) { printf("short read on %s\n", path); exit(1); }
    fclose(f);
}

struct Net {
    float *conv1_f, *conv1_b;  
    float *conv2_f, *conv2_b;
    float *fc_W,    *fc_b;    
};


struct Grads {
    float *conv1_f, *conv1_b;
    float *conv2_f, *conv2_b;
    float *fc_W,    *fc_b;
};


struct Acts {
    float *conv1_out, *relu1_out, *pool1_out; int *argmax1;  
    float *conv2_out, *relu2_out, *pool2_out; int *argmax2;  
    float *logits, *probs;                  
};


void forward(const float *image, const Net *net, Acts *a){
    int C_in=1,  C_out=8,  H=28, W=28, FH=3, FW=3;
    conv2d_mc(image, net->conv1_f, net->conv1_b, a->conv1_out, C_in,  C_out,  H, W, FH, FW);
    for (int i = 0; i < 8*26*26; ++i)
        a->relu1_out[i] = a->conv1_out[i] > 0.0f ? a->conv1_out[i] : 0.0f;
    H=26, W=26;
    int P=2, S=2,C=8;
    maxPoolWrapKernel(a->relu1_out, a->pool1_out, a->argmax1, H, W, P, S, C);
    conv2d_mc(a->pool1_out, net->conv2_f, net->conv2_b, a->conv2_out, C_in=8, C_out=16, H=13, W=13, FH=3, FW=3);
    for (int i = 0; i < 16*11*11; ++i)
        a->relu2_out[i] = a->conv2_out[i] > 0.0f ? a->conv2_out[i] : 0.0f;
    maxPoolWrapKernel(a->relu2_out, a->pool2_out, a->argmax2,H=11, W=11, P=2, S=2, C=16);
    int batch=1, in=400, out=10;
    fc_forward(a->pool2_out, net->fc_W, net->fc_b, a->logits, batch, in, out);
    float m = a->logits[0];
    for (int c = 1; c < 10; ++c) if (a->logits[c] > m) m = a->logits[c];
    float sum = 0.0f;
    for (int c = 0; c < 10; ++c) { a->probs[c] = expf(a->logits[c] - m); sum += a->probs[c]; }
    for (int c = 0; c < 10; ++c) a->probs[c] /= sum;
}


void backward(const float *image, int label, const Net *net, const Acts *a, Grads *g){
    float *dY          = (float*)malloc(10   * sizeof(float));
    float *d_pool2     = (float*)malloc(400  * sizeof(float));
    float *d_relu2     = (float*)malloc(1936 * sizeof(float));
    float *d_conv2_out = (float*)malloc(1936 * sizeof(float));
    float *d_pool1     = (float*)malloc(1352 * sizeof(float));
    float *d_relu1     = (float*)malloc(5408 * sizeof(float));
    float *d_conv1_out = (float*)malloc(5408 * sizeof(float));
    float *d_image     = (float*)malloc(784  * sizeof(float));
    for (int c = 0; c < 10; ++c) {
        dY[c] = a->probs[c] - (c==label ? 1 : 0);
    }
    int batch=1, in=400, out=10;
    fc_backward(dY, a->pool2_out, net->fc_W,  g->fc_W, g->fc_b, d_pool2,   batch, in, out);
    int H=11, W=11, P=2, S=2, C=16;
    backMaxPoolWrapKernel(d_pool2, d_relu2, a->argmax2,   H, W, P, S, C);
    for (int i = 0; i < 16*11*11; ++i)
        d_conv2_out[i] = d_relu2[i] * (a->conv2_out[i] > 0.0f ? 1.0f : 0.0f);
    int C_in=8, C_out=16, FH=3, FW=3;
    H = 13, W = 13;
    conv2d_mc_backward(d_conv2_out, a->pool1_out, net->conv2_f,  d_pool1, g->conv2_f, g->conv2_b,  C_in, C_out, H, W, FH, FW);
    backMaxPoolWrapKernel(d_pool1, d_relu1, a->argmax1,   H=26, W=26, P=2, S=2, C=8);
    for (int i = 0; i < 5408; ++i) {
        d_conv1_out[i] = d_relu1[i] * (a->conv1_out[i] > 0 ? 1 : 0);
    }
    conv2d_mc_backward(d_conv1_out, image, net->conv1_f,  d_image, g->conv1_f, g->conv1_b,  C_in=1, C_out=8, H=28, W=28, FH=3, FW=3);
}


void update(Net *net, const Grads *g, float lr){
    sgd(net->conv1_f, g->conv1_f, lr, 72);
    sgd(net->conv1_b, g->conv1_b, lr, 8);
    sgd(net->conv2_f, g->conv2_f, lr, 1152);
    sgd(net->conv2_b, g->conv2_b, lr, 16);
    sgd(net->fc_W,    g->fc_W,    lr, 4000);
    sgd(net->fc_b,    g->fc_b,    lr, 10);
}

int main(){
    Net net;
    Grads g;
    Acts a;
        // weights
    net.conv1_f = (float*)malloc(72   * sizeof(float));
    net.conv1_b = (float*)malloc(8    * sizeof(float));
    net.conv2_f = (float*)malloc(1152 * sizeof(float));
    net.conv2_b = (float*)malloc(16   * sizeof(float));
    net.fc_W    = (float*)malloc(4000 * sizeof(float));
    net.fc_b    = (float*)malloc(10   * sizeof(float));


    g.conv1_f = (float*)malloc(72   * sizeof(float));
    g.conv1_b = (float*)malloc(8    * sizeof(float));
    g.conv2_f = (float*)malloc(1152 * sizeof(float));
    g.conv2_b = (float*)malloc(16   * sizeof(float));
    g.fc_W    = (float*)malloc(4000 * sizeof(float));
    g.fc_b    = (float*)malloc(10   * sizeof(float));


    a.conv1_out = (float*)malloc(5408 * sizeof(float));
    a.relu1_out = (float*)malloc(5408 * sizeof(float));
    a.pool1_out = (float*)malloc(1352 * sizeof(float));
    a.argmax1   = (int*)  malloc(1352 * sizeof(int));      // ints!
    a.conv2_out = (float*)malloc(1936 * sizeof(float));
    a.relu2_out = (float*)malloc(1936 * sizeof(float));
    a.pool2_out = (float*)malloc(400  * sizeof(float));
    a.argmax2   = (int*)  malloc(400  * sizeof(int));       // ints!
    a.logits    = (float*)malloc(10   * sizeof(float));
    a.probs     = (float*)malloc(10   * sizeof(float));  
    const int N = 1000;     
    const int EPOCHS = 50;
    float lr = 0.001f;
    float *X     = (float*)malloc((size_t)N*784 * sizeof(float));
    float *Y     = (float*)malloc((size_t)N*10  * sizeof(float));   
    int   *label = (int*)  malloc(N * sizeof(int));
    srand(42); 
    // conv1: fan_in = 1*3*3 = 9
    float s1 = sqrtf(2.0f / 9.0f);
    for (int i = 0; i < 72; ++i) net.conv1_f[i] = ((float)rand()/RAND_MAX*2.0f - 1.0f) * s1;
    for (int i = 0; i < 8;  ++i) net.conv1_b[i] = 0.0f;
    // conv2: fan_in = 8*3*3 = 72
    float s2 = sqrtf(2.0f / 72.0f);
    for (int i = 0; i < 1152; ++i) net.conv2_f[i] = ((float)rand()/RAND_MAX*2.0f - 1.0f) * s2;
    for (int i = 0; i < 16;   ++i) net.conv2_b[i] = 0.0f;
    // fc: fan_in = 400
    float s3 = sqrtf(2.0f / 400.0f);
    for (int i = 0; i < 4000; ++i) net.fc_W[i] = ((float)rand()/RAND_MAX*2.0f - 1.0f) * s3;
    for (int i = 0; i < 10;   ++i) net.fc_b[i] = 0.0f;

    load_bin("mnist_X.bin", X, (size_t)N*784);
    load_bin("mnist_Y.bin", Y, (size_t)N*10);
    for (int s = 0; s < N; ++s) {            
        int t = 0;
        for (int c = 1; c < 10; ++c) if (Y[s*10+c] > Y[s*10+t]) t = c;
        label[s] = t;
    }
    for (int epoch = 0; epoch < EPOCHS; ++epoch){
        float loss = 0.0f;
      for (int s = 0; s < N; ++s) {
          const float *img = &X[s*784];
          forward(img, &net, &a);          // pass structs by pointer
          if (epoch == 0 && s == 0) {
                printf("logits: "); for (int c=0;c<10;++c) printf("%.3f ", a.logits[c]); printf("\n");
                printf("probs:  "); for (int c=0;c<10;++c) printf("%.3f ", a.probs[c]); printf("\n");
                printf("sample0 loss = %.4f\n", -logf(a.probs[label[0]] + 1e-8f));
            }

          loss += -logf(a.probs[label[s]] + 1e-8f);   
          backward(img, label[s], &net, &a, &g);
          update(&net, &g, lr);
      }
      printf("epoch %d  loss = %.4f\n", epoch, loss / N);
    }
    int correct = 0;
    for (int s = 0; s < N; ++s) {
        forward(&X[s*784], &net, &a);
        int pred = 0;
        for (int c = 1; c < 10; ++c) if (a.probs[c] > a.probs[pred]) pred = c;
        if (pred == label[s]) correct++;
    }
    printf("train accuracy = %.2f%% (%d/%d)\n", 100.0f*correct/N, correct, N);
    free(net.conv1_f);
    free(net.conv1_b);
    free(net.conv2_f);
    free(net.conv2_b);
    free(net.fc_W);
    free(net.fc_b);
    free(g.conv1_f);
    free(g.conv1_b);
    free(g.conv2_f);
    free(g.conv2_b);
    free(g.fc_W);
    free(g.fc_b);
    free(a.conv1_out);
    free(a.relu1_out);
    free(a.pool1_out);
    free(a.argmax1);
    free(a.conv2_out);
    free(a.relu2_out);
    free(a.pool2_out);
    free(a.argmax2);
    free(a.logits);
    free(a.probs);
    free(X);
    free(Y);
    free(label);
}