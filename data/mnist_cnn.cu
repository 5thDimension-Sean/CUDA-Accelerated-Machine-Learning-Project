#include "common.cuh"
#include "conv2d_mc.cu"
#include "pooling.cu"
#include "activations.cu"
#include "fc.cu"
#include "loss.cu"
#include "optimizer.cu"

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
    ReLu(a->conv1_out, a->relu1_out);

}


void backward(const float *image, int label, const Net *net, const Acts *a, Grads *g){

}


void update(Net *net, const Grads *g, float lr){

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
    const int EPOCHS = 3;
    float lr = 0.01f;
    float *X     = (float*)malloc((size_t)N*784 * sizeof(float));
    float *Y     = (float*)malloc((size_t)N*10  * sizeof(float));   
    int   *label = (int*)  malloc(N * sizeof(int));
    load_bin("mnist_X.bin", X, (size_t)N*784);
    load_bin("mnist_Y.bin", Y, (size_t)N*10);
    for (int s = 0; s < N; ++s) {                 // one-hot → int label (argmax)
        int t = 0;
        for (int c = 1; c < 10; ++c) if (Y[s*10+c] > Y[s*10+t]) t = c;
        label[s] = t;
    }

    for (int epoch = 0; epoch < EPOCHS; ++epoch)
      for (int s = 0; s < N; ++s) {
          const float *img = &X[s*784];
          forward(img, &net, &a);          // pass structs by pointer
          backward(img, label[s], &net, &a, &g);
          update(&net, &g, lr);
      }
    
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
}