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


void forward(const float *image, const Net *net, Acts *a);


void backward(const float *image, int label, const Net *net, const Acts *a, Grads *g);


void update(Net *net, const Grads *g, float lr);

int main(){
    
}