
#include "common.cuh"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>

__global__ void conv2d_mc_forward(const float*, const float*, const float*, float*, int,int,int,int,int,int);
__global__ void maxPool2D(const float*, float*, int*, int,int,int,int,int,int,int);
__global__ void backMaxPool2D(const float*, const int*, float*, int,int,int);
__global__ void conv2d_mc_backward_bias   (const float*, float*, int,int,int);
__global__ void conv2d_mc_backward_weights (const float*, const float*, float*, int,int,int,int,int,int);
__global__ void conv2d_mc_backward_input   (const float*, const float*, float*, int,int,int,int,int,int);
__global__ void fc_forward_kernel         (const float*, const float*, const float*, float*, int,int,int);
__global__ void fc_backward_weights_kernel(const float*, const float*, float*, int,int,int);
__global__ void fc_backward_bias_kernel   (const float*, float*, int,int);
__global__ void fc_backward_input_kernel  (const float*, const float*, float*, int,int,int);
__global__ void sgd_kernel(float*, const float*, float, int);

__global__ void relu_forward(const float* in, float* out, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] > 0.f ? in[i] : 0.f;
}

__global__ void relu_backward(const float* dOut, const float* preact, float* dIn, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) dIn[i] = preact[i] > 0.f ? dOut[i] : 0.f;   // mask by pre-activation
}


__global__ void softmax_ce_grad(const float* logits, float* probs, float* dY, int label, float* loss){
    float m = logits[0];
    for (int c=1;c<10;++c) if (logits[c] > m) m = logits[c];
    float sum = 0.f;
    for (int c=0;c<10;++c){ probs[c] = expf(logits[c]-m); sum += probs[c]; }
    for (int c=0;c<10;++c) probs[c] /= sum;
    for (int c=0;c<10;++c) dY[c] = probs[c] - (c==label ? 1.f : 0.f);
    atomicAdd(loss, -logf(probs[label] + 1e-8f));
}


void load_bin(const char *path, float *dst, size_t count){
    FILE *f = fopen(path, "rb");
    if (!f) { printf("could not open %s\n", path); exit(1); }
    size_t got = fread(dst, sizeof(float), count, f);
    if (got != count) { printf("short read on %s\n", path); exit(1); }
    fclose(f);
}

// All pointers below are DEVICE pointers.
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

struct Back {           
    float *dY;                     
    float *d_pool2;                   
    float *d_relu2, *d_conv2_out;  
    float *d_pool1;               
    float *d_relu1, *d_conv1_out;      
    float *d_image_grad;            
};

void forward(const float *d_image, const Net *net, Acts *a){
    dim3 b2(16,16), b3(16,16,1);       // reusable block shapes

    dim3 g_conv1((26+15)/16, (26+15)/16, 8);
    conv2d_mc_forward<<<g_conv1, b3>>>(d_image, net->conv1_f, net->conv1_b, a->conv1_out, 1,8,28,28,3,3);
    relu_forward<<<(5408+255)/256, 256>>>(a->conv1_out, a->relu1_out, 5408);

    dim3 g_pool1((13+15)/16, (13+15)/16, 8);
    maxPool2D<<<g_pool1, b2>>>(a->relu1_out, a->pool1_out, a->argmax1, 26,26, 13,13, 2,2, 8);

    dim3 g_conv2((11+15)/16, (11+15)/16, 16);
    conv2d_mc_forward<<<g_conv2, b3>>>(a->pool1_out, net->conv2_f, net->conv2_b, a->conv2_out, 8,16,13,13,3,3);
    relu_forward<<<(1936+255)/256, 256>>>(a->conv2_out, a->relu2_out, 1936);

    dim3 g_pool2((5+15)/16, (5+15)/16, 16);
    maxPool2D<<<g_pool2, b2>>>(a->relu2_out, a->pool2_out, a->argmax2, 11,11, 5,5, 2,2, 16);

    dim3 g_fc((1+15)/16, (10+15)/16);
    fc_forward_kernel<<<g_fc, b2>>>(a->pool2_out, net->fc_W, net->fc_b, a->logits, 1,400,10);
}

void backward(const float *d_image, int label, const Net *net,
              const Acts *a, Grads *g, const Back *bp, float *d_loss){
    dim3 b2(16,16), b3(16,16,1);

    softmax_ce_grad<<<1,1>>>(a->logits, a->probs, bp->dY, label, d_loss);

    dim3 g_fcw((400+15)/16, (10+15)/16);   // grid.x=in(400), grid.y=out(10)
    dim3 g_fcx((1+15)/16,   (400+15)/16);  // grid.x=batch(1), grid.y=in(400)
    fc_backward_weights_kernel<<<g_fcw, b2>>>(bp->dY, a->pool2_out, g->fc_W, 1,400,10);
    fc_backward_bias_kernel   <<<(10+15)/16, 16>>>(bp->dY, g->fc_b, 1,10);
    fc_backward_input_kernel  <<<g_fcx, b2>>>(bp->dY, net->fc_W, bp->d_pool2, 1,400,10);

    cudaMemset(bp->d_relu2, 0, 1936*sizeof(float));
    dim3 g_bp2((5+15)/16, (5+15)/16, 16);
    backMaxPool2D<<<g_bp2, b2>>>(bp->d_pool2, a->argmax2, bp->d_relu2, 5,5, 16);
    relu_backward<<<(1936+255)/256, 256>>>(bp->d_relu2, a->conv2_out, bp->d_conv2_out, 1936);
    dim3 g_cbi2((13+15)/16, (13+15)/16, 8);   // dInput dims: C_in=8, 13x13
    conv2d_mc_backward_bias   <<<(16+255)/256, 256>>>(bp->d_conv2_out, g->conv2_b, 16,11,11);
    conv2d_mc_backward_weights<<<(1152+255)/256, 256>>>(bp->d_conv2_out, a->pool1_out, g->conv2_f, 8,16,13,13,3,3);
    conv2d_mc_backward_input  <<<g_cbi2, b3>>>(bp->d_conv2_out, net->conv2_f, bp->d_pool1, 8,16,13,13,3,3);
    cudaMemset(bp->d_relu1, 0, 5408*sizeof(float));
    dim3 g_bp1((13+15)/16, (13+15)/16, 8);
    backMaxPool2D<<<g_bp1, b2>>>(bp->d_pool1, a->argmax1, bp->d_relu1, 13,13, 8);
    relu_backward<<<(5408+255)/256, 256>>>(bp->d_relu1, a->conv1_out, bp->d_conv1_out, 5408);
    dim3 g_cbi1((28+15)/16, (28+15)/16, 1);   // dInput dims: C_in=1, 28x28
    conv2d_mc_backward_bias   <<<(8+255)/256, 256>>>(bp->d_conv1_out, g->conv1_b, 8,26,26);
    conv2d_mc_backward_weights<<<(72+255)/256, 256>>>(bp->d_conv1_out, d_image, g->conv1_f, 1,8,28,28,3,3);
    conv2d_mc_backward_input  <<<g_cbi1, b3>>>(bp->d_conv1_out, net->conv1_f, bp->d_image_grad, 1,8,28,28,3,3);
}

void update(Net *net, const Grads *g, float lr){
    sgd_kernel<<<(72+255)/256,   256>>>(net->conv1_f, g->conv1_f, lr, 72);
    sgd_kernel<<<(8+255)/256,    256>>>(net->conv1_b, g->conv1_b, lr, 8);
    sgd_kernel<<<(1152+255)/256, 256>>>(net->conv2_f, g->conv2_f, lr, 1152);
    sgd_kernel<<<(16+255)/256,   256>>>(net->conv2_b, g->conv2_b, lr, 16);
    sgd_kernel<<<(4000+255)/256, 256>>>(net->fc_W,    g->fc_W,    lr, 4000);
    sgd_kernel<<<(10+255)/256,   256>>>(net->fc_b,    g->fc_b,    lr, 10);
}

int main(){
    Net net; Grads g; Acts a; Back bp;


    CUDA_CHECK(cudaMalloc(&bp.dY,          10   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_pool2,     400  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_relu2,     1936 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_conv2_out, 1936 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_pool1,     1352 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_relu1,     5408 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_conv1_out, 5408 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_image_grad, 784 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&net.conv1_f, 72   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv1_b, 8    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv2_f, 1152 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv2_b, 16   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.fc_W,    4000 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.fc_b,    10   * sizeof(float)));


    CUDA_CHECK(cudaMalloc(&g.conv1_f, 72   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv1_b, 8    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv2_f, 1152 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv2_b, 16   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.fc_W,    4000 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.fc_b,    10   * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&a.conv1_out, 5408 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.relu1_out, 5408 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.pool1_out, 1352 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.argmax1,   1352 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&a.conv2_out, 1936 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.relu2_out, 1936 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.pool2_out, 400  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.argmax2,   400  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&a.logits,    10   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.probs,     10   * sizeof(float)));

    float *d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));

    const int N = 10000;
    const int EPOCHS = 10;
    float lr = 0.001f;

    float *X     = (float*)malloc((size_t)N*784 * sizeof(float));
    float *Y     = (float*)malloc((size_t)N*10  * sizeof(float));
    int   *label = (int*)  malloc(N * sizeof(int));

    float *h_c1f=(float*)malloc(72*sizeof(float)),   *h_c1b=(float*)malloc(8*sizeof(float));
    float *h_c2f=(float*)malloc(1152*sizeof(float)), *h_c2b=(float*)malloc(16*sizeof(float));
    float *h_fW =(float*)malloc(4000*sizeof(float)), *h_fb =(float*)malloc(10*sizeof(float));

    srand(42);
    float s1 = sqrtf(2.0f/9.0f);          // conv1: fan_in = 1*3*3 = 9
    for (int i=0;i<72;++i) h_c1f[i]=((float)rand()/RAND_MAX*2.0f-1.0f)*s1;
    for (int i=0;i<8; ++i) h_c1b[i]=0.0f;
    float s2 = sqrtf(2.0f/72.0f);         // conv2: fan_in = 8*3*3 = 72
    for (int i=0;i<1152;++i) h_c2f[i]=((float)rand()/RAND_MAX*2.0f-1.0f)*s2;
    for (int i=0;i<16;  ++i) h_c2b[i]=0.0f;
    float s3 = sqrtf(2.0f/400.0f);        // fc: fan_in = 400
    for (int i=0;i<4000;++i) h_fW[i]=((float)rand()/RAND_MAX*2.0f-1.0f)*s3;
    for (int i=0;i<10;  ++i) h_fb[i]=0.0f;

    CUDA_CHECK(cudaMemcpy(net.conv1_f, h_c1f, 72*sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net.conv1_b, h_c1b, 8*sizeof(float),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net.conv2_f, h_c2f, 1152*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net.conv2_b, h_c2b, 16*sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net.fc_W,    h_fW,  4000*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net.fc_b,    h_fb,  10*sizeof(float),   cudaMemcpyHostToDevice));


    load_bin("mnist_X.bin", X, (size_t)N*784);
    load_bin("mnist_Y.bin", Y, (size_t)N*10);
    for (int s=0;s<N;++s){ int t=0; for(int c=1;c<10;++c) if(Y[s*10+c]>Y[s*10+t]) t=c; label[s]=t; }

    float *d_X;
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)N*784*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X, X, (size_t)N*784*sizeof(float), cudaMemcpyHostToDevice));

    float h_logits[10], h_probs[10];   // small host scratch for prints/eval


    auto t0 = std::chrono::high_resolution_clock::now();
    for (int epoch=0; epoch<EPOCHS; ++epoch){
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));    // reset accumulator
        for (int s=0; s<N; ++s){
            const float *d_img = d_X + (size_t)s*784;
            forward(d_img, &net, &a);
            backward(d_img, label[s], &net, &a, &g, &bp, d_loss);   // softmax+loss inside
            update(&net, &g, lr);

            if (epoch==0 && s==0){                            // debug: copy DOWN first
                CUDA_CHECK(cudaMemcpy(h_logits, a.logits, 10*sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_probs,  a.probs,  10*sizeof(float), cudaMemcpyDeviceToHost));
                printf("logits: "); for(int c=0;c<10;++c) printf("%.3f ", h_logits[c]); printf("\n");
                printf("probs:  "); for(int c=0;c<10;++c) printf("%.3f ", h_probs[c]);  printf("\n");
                printf("sample0 loss = %.4f\n", -logf(h_probs[label[0]] + 1e-8f));
            }
        }
        float h_loss = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost)); // syncs
        printf("epoch %d  loss = %.4f\n", epoch, h_loss / N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("training time: %.2f s  (%.2f ms/sample)\n",
           std::chrono::duration<double>(t1-t0).count(),
           std::chrono::duration<double,std::milli>(t1-t0).count()/(EPOCHS*N));

    int correct = 0;
    for (int s=0;s<N;++s){
        forward(d_X + (size_t)s*784, &net, &a);
        CUDA_CHECK(cudaMemcpy(h_logits, a.logits, 10*sizeof(float), cudaMemcpyDeviceToHost));
        int pred=0; for(int c=1;c<10;++c) if(h_logits[c]>h_logits[pred]) pred=c;
        if (pred==label[s]) correct++;
    }
    printf("train accuracy = %.2f%% (%d/%d)\n", 100.0f*correct/N, correct, N);

    const int NT = 10000;
    float *Xt=(float*)malloc((size_t)NT*784*sizeof(float));
    float *Yt=(float*)malloc((size_t)NT*10*sizeof(float));
    int   *labelt=(int*)malloc(NT*sizeof(int));
    load_bin("mnist_test_X.bin", Xt, (size_t)NT*784);
    load_bin("mnist_test_Y.bin", Yt, (size_t)NT*10);
    for (int s=0;s<NT;++s){ int t=0; for(int c=1;c<10;++c) if(Yt[s*10+c]>Yt[s*10+t]) t=c; labelt[s]=t; }

    float *d_Xt;
    CUDA_CHECK(cudaMalloc(&d_Xt, (size_t)NT*784*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Xt, Xt, (size_t)NT*784*sizeof(float), cudaMemcpyHostToDevice));

    int tc = 0;
    for (int s=0;s<NT;++s){
        forward(d_Xt + (size_t)s*784, &net, &a);
        CUDA_CHECK(cudaMemcpy(h_logits, a.logits, 10*sizeof(float), cudaMemcpyDeviceToHost));
        int pred=0; for(int c=1;c<10;++c) if(h_logits[c]>h_logits[pred]) pred=c;
        if (pred==labelt[s]) tc++;
    }
    printf("TEST accuracy = %.2f%% (%d/%d)\n", 100.0f*tc/NT, tc, NT);

    cudaFree(d_X); cudaFree(d_Xt); cudaFree(d_loss);
    cudaFree(net.conv1_f); cudaFree(net.conv1_b); cudaFree(net.conv2_f);
    cudaFree(net.conv2_b); cudaFree(net.fc_W);    cudaFree(net.fc_b);
    cudaFree(g.conv1_f);   cudaFree(g.conv1_b);   cudaFree(g.conv2_f);
    cudaFree(g.conv2_b);   cudaFree(g.fc_W);      cudaFree(g.fc_b);
    cudaFree(a.conv1_out); cudaFree(a.relu1_out); cudaFree(a.pool1_out); cudaFree(a.argmax1);
    cudaFree(a.conv2_out); cudaFree(a.relu2_out); cudaFree(a.pool2_out); cudaFree(a.argmax2);
    cudaFree(a.logits);    cudaFree(a.probs);
    cudaFree(bp.dY); cudaFree(bp.d_pool2); cudaFree(bp.d_relu2); cudaFree(bp.d_conv2_out);
    cudaFree(bp.d_pool1); cudaFree(bp.d_relu1); cudaFree(bp.d_conv1_out); cudaFree(bp.d_image_grad);

    free(X); free(Y); free(label); free(Xt); free(Yt); free(labelt);
    free(h_c1f); free(h_c1b); free(h_c2f); free(h_c2b); free(h_fW); free(h_fb);
    return 0;
}
