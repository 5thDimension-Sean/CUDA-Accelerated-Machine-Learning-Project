#include "common.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <algorithm>

void load_bin(const char *path, float *dst, size_t count){
    FILE *f = fopen(path, "rb");
    if (!f) { printf("could not open %s\n", path); exit(1); }
    size_t got = fread(dst, sizeof(float), count, f);
    if (got != count) { printf("short read on %s\n", path); exit(1); }
    fclose(f);
}
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


__global__ void detection_loss_grad(const float* pred, const float* target, float* dY, float* loss, int S, int C, float lam_coord, float lam_noobj){
    int vals = 5+ C; //should be 15
    int cells = S*S; //16
    float total = 0.0f;
    for(int cell = 0; cell <cells; ++cell){
        int base = cell * vals;
        bool has = target[base+0] > 0.5f;
        for(int k = 0; k < vals; ++k){
            float w; 
            if(k==0) w = has ? 1.0f : lam_noobj; //confidence
            else if(k>=1 && k<=4) w = has ? lam_coord : 0.0f; //bbox
            else w = has ? 1.0f : 0.0f; //class
            float diff = pred[base+k] - target[base+k];
            total += w * diff * diff;
            dY[base+k] = w * diff;
        }
    }
    atomicAdd(loss, total);
}

struct Net {
    float *conv1_f, *conv1_b;
    float *conv2_f, *conv2_b;
    float *conv3_f, *conv3_b;
    float *fc_W,    *fc_b;
};

struct Grads {
    float *conv1_f, *conv1_b;
    float *conv2_f, *conv2_b;
    float *conv3_f, *conv3_b;
    float *fc_W,    *fc_b;
};

struct Acts {
    float *conv1_out, *relu1_out, *pool1_out; int *argmax1;
    float *conv2_out, *relu2_out, *pool2_out; int *argmax2;
    float *conv3_out, *relu3_out, *pool3_out; int *argmax3;
    float *logits, *probs;
    float *preds;
};

struct Back {           
    float *dY;                     
    float *d_pool2, *d_pool3;                   
    float *d_relu2, *d_conv2_out;  
    float *d_pool1;               
    float *d_relu1, *d_conv1_out;      
    float *d_image_grad;      
    float *d_relu3, *d_conv3_out;          
};

static void put_px(unsigned char* img, int W, int H, int x, int y,
                   unsigned char r, unsigned char gg, unsigned char b){
    if (x < 0 || x >= W || y < 0 || y >= H) return;
    int o = (y*W + x)*3;
    img[o] = r; img[o+1] = gg; img[o+2] = b;
}

// draw a hollow rectangle from a center+size box
static void draw_rect(unsigned char* img, int W, int H,
                      float cx, float cy, float w, float h,
                      unsigned char r, unsigned char gg, unsigned char b){
    int l = (int)(cx - w/2), rt = (int)(cx + w/2);
    int t = (int)(cy - h/2), bo = (int)(cy + h/2);
    for (int x = l; x <= rt; ++x){ put_px(img,W,H,x,t,r,gg,b); put_px(img,W,H,x,bo,r,gg,b); }
    for (int y = t; y <= bo; ++y){ put_px(img,W,H,l,y,r,gg,b); put_px(img,W,H,rt,y,r,gg,b); }
}

void draw_prediction(const float* canvas, const float* h_preds,
                     const float* meta, const char* path){
    const int W = 64, H = 64;
    unsigned char* img = (unsigned char*)malloc(W*H*3);

    // grayscale canvas [0,1] -> RGB [0,255]
    for (int i = 0; i < W*H; ++i){
        int v = (int)(canvas[i]*255.0f); v = v<0?0:(v>255?255:v);
        img[i*3] = img[i*3+1] = img[i*3+2] = (unsigned char)v;
    }

    // decode predicted box = max-confidence cell (same math as eval_one)
    int best = 0; float mc = h_preds[0];
    for (int cell = 1; cell < 16; ++cell)
        if (h_preds[cell*15] > mc){ mc = h_preds[cell*15]; best = cell; }
    int gx = best%4, gy = best/4, base = best*15;
    float cx = (gx + h_preds[base+1])*16.0f, cy = (gy + h_preds[base+2])*16.0f;
    float pw = h_preds[base+3]*64.0f,        ph = h_preds[base+4]*64.0f;

    draw_rect(img, W, H, meta[0], meta[1], meta[2], meta[3], 0,255,0);   // GT   = green
    draw_rect(img, W, H, cx, cy, pw, ph,                    255,0,0);   // pred = red

    stbi_write_png(path, W, H, 3, img, W*3);
    free(img);
}

float iou_xywh(float ax, float ay, float aw, float ah, float bx, float by, float bw, float bh) {
    float leftA = ax - aw / 2.0f;
    float rightA = ax + aw / 2.0f;
    float topA = ay - ah / 2.0f;
    float botA = ay + ah / 2.0f;

    float leftB = bx - bw / 2.0f;
    float rightB = bx + bw / 2.0f;
    float topB = by - bh / 2.0f;
    float botB = by + bh / 2.0f;

    // Overlap width and height (fixed the typo max(leftA, leftA) to max(leftA, leftB))
    float w_overlap = std::max(0.0f, std::min(rightA, rightB) - std::max(leftA, leftB));
    float h_overlap = std::max(0.0f, std::min(botA, botB) - std::max(topA, topB));

    float intersection = w_overlap * h_overlap;
    float areaA = aw * ah;
    float areaB = bw * bh;
    float union_area = areaA + areaB - intersection;

    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return intersection / union_area;
}

int eval_one(const float *h_preds, const float *meta, float *iou_out, int *pred_class_out) {
    int best_cell = 0;
    float max_conf = h_preds[0 * 15 + 0];

    for (int cell = 1; cell < 16; ++cell) {
        float conf = h_preds[cell * 15 + 0];
        if (conf > max_conf) {
            max_conf = conf;
            best_cell = cell;
        }
    }

    int gx = best_cell % 4;
    int gy = best_cell / 4;

    int base = best_cell * 15;
    float x = h_preds[base + 1];
    float y = h_preds[base + 2];
    float w = h_preds[base + 3];
    float h = h_preds[base + 4];

    float cx = (gx + x) * 16.0f;
    float cy = (gy + y) * 16.0f;
    float pw = w * 64.0f;
    float ph = h * 64.0f;

    int pred_class = 0;
    float max_class_val = h_preds[base + 5];

    for (int c = 1; c < 10; c++) {
        float class_val = h_preds[base + 5 + c];
        if (class_val > max_class_val) {
            max_class_val = class_val;
            pred_class = c;
        }
    }

    float gcx = meta[0];
    float gcy = meta[1];
    float gw  = meta[2];
    float gh  = meta[3];
    int glabel = (int)meta[4];

    float iou = iou_xywh(cx, cy, pw, ph, gcx, gcy, gw, gh);

    if (iou_out != NULL) {
        *iou_out = iou;
    }
    if (pred_class_out != NULL) {
        *pred_class_out = pred_class;
    }

    return (pred_class == glabel && iou > 0.5f) ? 1 : 0;
}

void forward(const float* d_image, const Net * net, const Acts * a){
    //conv, relu, max...3 iterations forward fc
    dim3 conv1grid(4, 4, 8);
    dim3 conv1block(16, 16, 1);
    conv2d_mc_forward<<<conv1grid, conv1block>>>(d_image, net->conv1_f, net->conv1_b, a->conv1_out, 1,8,64,64,3,3);
    dim3 relu1Grid(121);
    dim3 relu1Block(256);
    relu_forward<<<relu1Grid, relu1Block>>>(a->conv1_out, a->relu1_out, 30752);
    dim3 pool1Grid(2, 2, 8);
    dim3 pool1Block(16, 16);
    maxPool2D<<<pool1Grid, pool1Block>>>(a->relu1_out, a->pool1_out, a->argmax1, 62,62, 31,31, 2,2, 8);
    //2
    dim3 conv2grid(2, 2, 16);
    dim3 conv2block(16, 16, 1);
    conv2d_mc_forward<<<conv2grid, conv2block>>>(a->pool1_out, net->conv2_f, net->conv2_b, a->conv2_out, 8,16,31,31,3,3);
    dim3 relu2Grid(53);
    dim3 relu2Block(256);
    relu_forward<<<relu2Grid, relu2Block>>>(a->conv2_out, a->relu2_out, 13456);
    dim3 pool2Grid(1, 1, 16);
    dim3 pool2Block(16, 16);
    maxPool2D<<<pool2Grid, pool2Block>>>(a->relu2_out, a->pool2_out, a->argmax2, 29,29, 14,14, 2,2, 16);
    //3
    dim3 conv3Grid(1, 1, 32);
    dim3 conv3Block(16, 16, 1);
    conv2d_mc_forward<<<conv3Grid, conv3Block>>>(a->pool2_out, net->conv3_f, net->conv3_b, a->conv3_out, 16,32,14,14,3,3);
    dim3 relu3Grid(18);
    dim3 relu3Block(256);
    relu_forward<<<relu3Grid, relu3Block>>>(a->conv3_out, a->relu3_out, 4608);
    dim3 pool3Grid(1, 1, 32);
    dim3 pool3Block(16, 16);
    maxPool2D<<<pool3Grid, pool3Block>>>(a->relu3_out, a->pool3_out, a->argmax3, 12,12, 6,6, 2,2, 32);
    dim3 fcForwardGrid(1, 15);
    dim3 fcForwardBlock(16, 16);
    fc_forward_kernel<<<fcForwardGrid, fcForwardBlock>>>(a->pool3_out, net->fc_W, net->fc_b, a->preds, 1,1152,240);
}

void backward(const float* d_image, const float* d_target, const Net * net, const Acts * a, const Grads * g, const Back * bp, float* d_loss){
 dim3 b2(16,16), b3(16,16,1);
    detection_loss_grad<<<1,1>>>(a->preds, d_target, bp->dY, d_loss, 4, 10, 5.0f, 0.5f);

    dim3 g_fcw((1152+15)/16,(240+15)/16);   // grid.x=in(1152), grid.y=out(240)
    dim3 g_fcx((1+15)/16,   (1152+15)/16);  // grid.x=batch(1), grid.y=in(1152)
    fc_backward_weights_kernel<<<g_fcw,b2>>>(bp->dY, a->pool3_out, g->fc_W, 1,1152,240);
    fc_backward_bias_kernel   <<<(240+15)/16,16>>>(bp->dY, g->fc_b, 1,240);
    fc_backward_input_kernel  <<<g_fcx,b2>>>(bp->dY, net->fc_W, bp->d_pool3, 1,1152,240);

    cudaMemset(bp->d_relu3, 0, 4608*sizeof(float));
    dim3 g_bp3((6+15)/16,(6+15)/16,32);
    backMaxPool2D<<<g_bp3,b2>>>(bp->d_pool3, a->argmax3, bp->d_relu3, 6,6, 32);
    relu_backward<<<(4608+255)/256,256>>>(bp->d_relu3, a->conv3_out, bp->d_conv3_out, 4608);
    dim3 g_cbi3((14+15)/16,(14+15)/16,16);   // dInput dims: C_in=16, 14x14
    conv2d_mc_backward_bias   <<<(32+255)/256,256>>>(bp->d_conv3_out, g->conv3_b, 32,12,12);
    conv2d_mc_backward_weights<<<(4608+255)/256,256>>>(bp->d_conv3_out, a->pool2_out, g->conv3_f, 16,32,14,14,3,3);
    conv2d_mc_backward_input  <<<g_cbi3,b3>>>(bp->d_conv3_out, net->conv3_f, bp->d_pool2, 16,32,14,14,3,3);

    cudaMemset(bp->d_relu2, 0, 13456*sizeof(float));
    dim3 g_bp2((14+15)/16,(14+15)/16,16);
    backMaxPool2D<<<g_bp2,b2>>>(bp->d_pool2, a->argmax2, bp->d_relu2, 14,14, 16);
    relu_backward<<<(13456+255)/256,256>>>(bp->d_relu2, a->conv2_out, bp->d_conv2_out, 13456);
    dim3 g_cbi2((31+15)/16,(31+15)/16,8);    // dInput dims: C_in=8, 31x31
    conv2d_mc_backward_bias   <<<(16+255)/256,256>>>(bp->d_conv2_out, g->conv2_b, 16,29,29);
    conv2d_mc_backward_weights<<<(1152+255)/256,256>>>(bp->d_conv2_out, a->pool1_out, g->conv2_f, 8,16,31,31,3,3);
    conv2d_mc_backward_input  <<<g_cbi2,b3>>>(bp->d_conv2_out, net->conv2_f, bp->d_pool1, 8,16,31,31,3,3);

    cudaMemset(bp->d_relu1, 0, 30752*sizeof(float));
    dim3 g_bp1((31+15)/16,(31+15)/16,8);
    backMaxPool2D<<<g_bp1,b2>>>(bp->d_pool1, a->argmax1, bp->d_relu1, 31,31, 8);
    relu_backward<<<(30752+255)/256,256>>>(bp->d_relu1, a->conv1_out, bp->d_conv1_out, 30752);
    dim3 g_cbi1((64+15)/16,(64+15)/16,1);    // dInput dims: C_in=1, 64x64
    conv2d_mc_backward_bias   <<<(8+255)/256,256>>>(bp->d_conv1_out, g->conv1_b, 8,62,62);
    conv2d_mc_backward_weights<<<(72+255)/256,256>>>(bp->d_conv1_out, d_image, g->conv1_f, 1,8,64,64,3,3);
    conv2d_mc_backward_input  <<<g_cbi1,b3>>>(bp->d_conv1_out, net->conv1_f, bp->d_image_grad, 1,8,64,64,3,3);
}

void update(Net *net, const Grads *g, float lr){
    sgd_kernel<<<(72+255)/256,    256>>>(net->conv1_f, g->conv1_f, lr, 72);
    sgd_kernel<<<(8+255)/256,     256>>>(net->conv1_b, g->conv1_b, lr, 8);
    sgd_kernel<<<(1152+255)/256,  256>>>(net->conv2_f, g->conv2_f, lr, 1152);
    sgd_kernel<<<(16+255)/256,    256>>>(net->conv2_b, g->conv2_b, lr, 16);
    sgd_kernel<<<(4608+255)/256,  256>>>(net->conv3_f, g->conv3_f, lr, 4608);
    sgd_kernel<<<(32+255)/256,    256>>>(net->conv3_b, g->conv3_b, lr, 32);
    sgd_kernel<<<(276480+255)/256,256>>>(net->fc_W,    g->fc_W,    lr, 276480);
    sgd_kernel<<<(240+255)/256,   256>>>(net->fc_b,    g->fc_b,    lr, 240);
}

int main(){
    Net net; Grads g; Acts a; Back bp;
    CUDA_CHECK(cudaMalloc(&bp.dY,          240   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_pool3,     1152  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_relu3,     4608  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_conv3_out, 4608  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_pool2,     3136  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_relu2,     13456 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_conv2_out, 13456 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_pool1,     7688  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_relu1,     30752 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_conv1_out, 30752 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bp.d_image_grad, 4096 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv1_f, 72     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv1_b, 8      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv2_f, 1152   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv2_b, 16     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv3_f, 4608   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.conv3_b, 32     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.fc_W,    276480 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net.fc_b,    240    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv1_f, 72     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv1_b, 8      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv2_f, 1152   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv2_b, 16     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv3_f, 4608   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.conv3_b, 32     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.fc_W,    276480 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.fc_b,    240    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.conv1_out, 30752 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.relu1_out, 30752 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.pool1_out, 7688  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.argmax1,   7688  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&a.conv2_out, 13456 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.relu2_out, 13456 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.pool2_out, 3136  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.argmax2,   3136  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&a.conv3_out, 4608  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.relu3_out, 4608  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.pool3_out, 1152  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a.argmax3,   1152  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&a.preds,     240   * sizeof(float)));

    int N = 10000;
    int EPOCHS = 50;
    float lr = 0.003f;
    float *d_loss; CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    float *d_X;    CUDA_CHECK(cudaMalloc(&d_X, (size_t)N*4096 * sizeof(float)));
    float *d_T;    CUDA_CHECK(cudaMalloc(&d_T, (size_t)N*240  * sizeof(float)));
    float *h_c1f=(float*)malloc(72*sizeof(float)),     *h_c1b=(float*)malloc(8*sizeof(float));
    float *h_c2f=(float*)malloc(1152*sizeof(float)),   *h_c2b=(float*)malloc(16*sizeof(float));
    float *h_c3f=(float*)malloc(4608*sizeof(float)),   *h_c3b=(float*)malloc(32*sizeof(float));
    float *h_fW =(float*)malloc(276480*sizeof(float)), *h_fb =(float*)malloc(240*sizeof(float));
    //init
    srand(42);
    float s1 = sqrtf(2.0f/9.0f);      // conv1: fan_in = 1*3*3 = 9
    for (int i=0;i<72;  ++i) h_c1f[i]=((float)rand()/RAND_MAX*2.0f-1.0f)*s1;
    for (int i=0;i<8;   ++i) h_c1b[i]=0.0f;
    float s2 = sqrtf(2.0f/72.0f);     // conv2: fan_in = 8*3*3 = 72
    for (int i=0;i<1152;++i) h_c2f[i]=((float)rand()/RAND_MAX*2.0f-1.0f)*s2;
    for (int i=0;i<16;  ++i) h_c2b[i]=0.0f;
    float s3 = sqrtf(2.0f/144.0f);    // conv3: fan_in = 16*3*3 = 144
    for (int i=0;i<4608;++i) h_c3f[i]=((float)rand()/RAND_MAX*2.0f-1.0f)*s3;
    for (int i=0;i<32;  ++i) h_c3b[i]=0.0f;
    float s4 = sqrtf(2.0f/1152.0f);   // fc: fan_in = 1152
    for (int i=0;i<276480;++i) h_fW[i]=((float)rand()/RAND_MAX*2.0f-1.0f)*s4;
    for (int i=0;i<240; ++i) h_fb[i]=0.0f; 
    //upload
    CUDA_CHECK(cudaMemcpy(net.conv1_f, h_c1f, 72*sizeof(float),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net.conv1_b, h_c1b, 8*sizeof(float),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net.conv2_f, h_c2f, 1152*sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net.conv2_b, h_c2b, 16*sizeof(float),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net.conv3_f, h_c3f, 4608*sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net.conv3_b, h_c3b, 32*sizeof(float),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net.fc_W,    h_fW,  276480*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net.fc_b,    h_fb,  240*sizeof(float),    cudaMemcpyHostToDevice));

    float *X = (float*)malloc((size_t)N*4096 * sizeof(float)); 
    float *T = (float*)malloc((size_t)N*240  * sizeof(float));  

    load_bin("det_X.bin", X, (size_t)N*4096);
    load_bin("det_Y.bin", T, (size_t)N*240);

    CUDA_CHECK(cudaMemcpy(d_X, X, (size_t)N*4096 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T, T, (size_t)N*240  * sizeof(float), cudaMemcpyHostToDevice));
    float *META = (float*)malloc((size_t)N*15 * sizeof(float));   
    load_bin("det_meta.bin", META, (size_t)N*15);
    for (int epoch = 0; epoch < EPOCHS; ++epoch){
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));    
        for (int s = 0; s < N; ++s){
            const float *d_img    = d_X + (size_t)s*4096;
            const float *d_target = d_T + (size_t)s*240;
            forward(d_img, &net, &a);
            backward(d_img, d_target, &net, &a, &g, &bp, d_loss);
            update(&net, &g, lr);
        }
        float h_loss = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        printf("epoch %d  loss = %.4f\n", epoch, h_loss / N);
    }
        load_bin("det_test_X.bin",    X,    (size_t)N*4096);
    load_bin("det_test_meta.bin", META, (size_t)N*5);
    CUDA_CHECK(cudaMemcpy(d_X, X, (size_t)N*4096*sizeof(float), cudaMemcpyHostToDevice));
    float h_preds[240];
    int correct = 0, class_ok = 0; float iou_sum = 0.0f;
    for (int s = 0; s < N; ++s){
        forward(d_X + (size_t)s*4096, &net, &a);
        cudaMemcpy(h_preds, a.preds, 240*sizeof(float), cudaMemcpyDeviceToHost);
        float iou; int pc;
        correct  += eval_one(h_preds, &META[s*5], &iou, &pc);
        iou_sum  += iou;
        class_ok += (pc == (int)META[s*5+4]);
    }
    printf("detection acc = %.2f%%   mean IoU = %.3f   class acc = %.2f%%\n", 100.0f*correct/N, iou_sum/N, 100.0f*class_ok/N);

    for (int s = 0; s < 12; ++s){
        forward(d_X + (size_t)s*4096, &net, &a);
        cudaMemcpy(h_preds, a.preds, 240*sizeof(float), cudaMemcpyDeviceToHost);
        char path[64];
        sprintf(path, "det_pred_%02d.png", s);
        draw_prediction(X + (size_t)s*4096, h_preds, &META[s*5], path);
    }
    printf("wrote det_pred_00..11.png\n");

     cudaFree(d_X); cudaFree(d_T); cudaFree(d_loss);

    cudaFree(net.conv1_f); cudaFree(net.conv1_b); cudaFree(net.conv2_f); cudaFree(net.conv2_b);
    cudaFree(net.conv3_f); cudaFree(net.conv3_b); cudaFree(net.fc_W);    cudaFree(net.fc_b);

    cudaFree(g.conv1_f);   cudaFree(g.conv1_b);   cudaFree(g.conv2_f);   cudaFree(g.conv2_b);
    cudaFree(g.conv3_f);   cudaFree(g.conv3_b);   cudaFree(g.fc_W);      cudaFree(g.fc_b);

    cudaFree(a.conv1_out); cudaFree(a.relu1_out); cudaFree(a.pool1_out); cudaFree(a.argmax1);
    cudaFree(a.conv2_out); cudaFree(a.relu2_out); cudaFree(a.pool2_out); cudaFree(a.argmax2);
    cudaFree(a.conv3_out); cudaFree(a.relu3_out); cudaFree(a.pool3_out); cudaFree(a.argmax3);
    cudaFree(a.preds);

    cudaFree(bp.dY);
    cudaFree(bp.d_pool3); cudaFree(bp.d_relu3); cudaFree(bp.d_conv3_out);
    cudaFree(bp.d_pool2); cudaFree(bp.d_relu2); cudaFree(bp.d_conv2_out);
    cudaFree(bp.d_pool1); cudaFree(bp.d_relu1); cudaFree(bp.d_conv1_out);
    cudaFree(bp.d_image_grad);

    free(X); free(T); free(META);
    free(h_c1f); free(h_c1b); free(h_c2f); free(h_c2b);
    free(h_c3f); free(h_c3b); free(h_fW);  free(h_fb);
    return 0;
}