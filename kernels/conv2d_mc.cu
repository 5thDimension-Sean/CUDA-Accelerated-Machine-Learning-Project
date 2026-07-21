// ============================================================================
// conv2d_mc.cu — Week 7
// Multi-channel 2D convolution (the building block of a real CNN):
//   input  [C_in  x H x W]
//   filter [C_out x C_in x FH x FW]   (one filter per output channel,
//                                       each spanning ALL input channels)
//   bias   [C_out]
//   output [C_out x outH x outW],  outH = H-FH+1, outW = W-FW+1
// ============================================================================
#include "common.cuh"
#include "conv2d_mc.cuh"
#include <cstdio>

__global__ void conv2d_mc_forward(const float *input, const float *filter, const float *bias,
                                  float *output, int C_in, int C_out, int H, int W, int FH, int FW){
    int outH = H - FH + 1;
    int outW = W - FW + 1;

    int ox = blockIdx.x * blockDim.x + threadIdx.x;   // output column
    int oy = blockIdx.y * blockDim.y + threadIdx.y;   // output row
    int oc = blockIdx.z;                              // output channel (one filter)

    if (ox >= outW || oy >= outH || oc >= C_out) return;

    float sum = bias[oc];
    for (int ic = 0; ic < C_in; ++ic) {              // sum over input channels
        for (int fy = 0; fy < FH; ++fy) {
            for (int fx = 0; fx < FW; ++fx) {
                float in_val = input [ic*(H*W)         + (oy+fy)*W + (ox+fx)];
                float w_val  = filter[oc*(C_in*FH*FW)  + ic*(FH*FW) + fy*FW + fx];
                sum += in_val * w_val;
            }
        }
    }
    output[oc*(outH*outW) + oy*outW + ox] = sum;
}

void conv2d_mc(const float *input, const float *filter, const float *bias, float *output,
               int C_in, int C_out, int H, int W, int FH, int FW){
    int outH = H - FH + 1, outW = W - FW + 1;
    size_t bytes_in     = (size_t)C_in  * H * W          * sizeof(float);
    size_t bytes_filter = (size_t)C_out * C_in * FH * FW * sizeof(float);
    size_t bytes_bias   = (size_t)C_out                  * sizeof(float);
    size_t bytes_out    = (size_t)C_out * outH * outW    * sizeof(float);

    float *d_in, *d_filter, *d_bias, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,     bytes_in));
    CUDA_CHECK(cudaMalloc(&d_filter, bytes_filter));
    CUDA_CHECK(cudaMalloc(&d_bias,   bytes_bias));
    CUDA_CHECK(cudaMalloc(&d_out,    bytes_out));

    CUDA_CHECK(cudaMemcpy(d_in,     input,  bytes_in,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, filter, bytes_filter, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias,   bias,   bytes_bias,   cudaMemcpyHostToDevice));

    dim3 block(16, 16, 1);
    dim3 grid((outW + 15) / 16, (outH + 15) / 16, C_out);   // z-axis = output channel
    conv2d_mc_forward<<<grid, block>>>(d_in, d_filter, d_bias, d_out,
                                       C_in, C_out, H, W, FH, FW);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, d_out, bytes_out, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_out));
}

__global__ void conv2d_mc_backward_bias(const float *dOut, float *dBias,
                                        int C_out, int outH, int outW){
                                            int oc = blockIdx.x * blockDim.x + threadIdx.x;   // output channel
                                            float g = 0.0f;
                                            if (oc >= C_out) return;
                                            for (int i = 0; i < outH*outW; ++i) g += dOut[oc*(outH*outW) + i];
                                            dBias[oc] = g;
}

__global__ void conv2d_mc_backward_weights(const float *dOut, const float *input, float *dFilter,
                                           int C_in, int C_out, int H, int W, int FH, int FW){
                                            int idx = blockIdx.x * blockDim.x + threadIdx.x;   // index in dFilter
                                            int fx = idx % FW;
                                            int fy = idx / FW % FH;
                                            int ic = idx / (FW * FH) % C_in;
                                            int oc = idx/(FW*FH*C_in);
                                            if(idx >= C_out * C_in * FH * FW) return;
                                            float g = 0.0f;
                                            int outH = H-FH + 1, outW = W-FW + 1;
                                            for (int j = 0; j < outW; ++j)
                                                g += dOut[oc*(outH*outW) + j*outW + j] * input[ic*(H*W) + (j+fy)*W + (j+fx)];
                                            dFilter[idx] = g;
                                            
                                        

}                             

__global__ void conv2d_mc_backward_input(const float *dOut, const float *filter, float *dInput,
                                         int C_in, int C_out, int H, int W, int FH, int FW){
                                            int ix = blockIdx.x * blockDim.x + threadIdx.x;   // input column
                                            int iy = blockIdx.y * blockDim.y + threadIdx.y; // input row
                                            int iz = blockIdx.z;                              // input channel    
                                            if(ix >= W || iy >= H || iz >= C_in) return;
                                            int outH = H-FH + 1, outW = W-FW + 1;
                                            float sum = 0.0f;
                                            for(int oc = 0; oc < C_out; ++oc){
                                                for(int fy = 0; fy < FH; ++fy){
                                                    for(int fx = 0; fx < FW; ++fx){
                                                        int oy = iy - fy;
                                                        int ox = ix - fx;
                                                        if(oy >= 0 && oy < outH && ox >= 0 && ox < outW){
                                                            sum += dOut[oc*(outH*outW) + oy*outW + ox] *
                                                                   filter[oc*(C_in*FH*FW) + iz*(FH*FW) + fy*FW + fx];
                                                        }
                                                    }
                                                }
                                            }
                                            dInput[iz*(H*W) + iy*W + ix] = sum;

}


void conv2d_mc_backward(const float *dOut,               
                        const float *input, const float *filter,
                        float *dInput, float *dFilter, float *dBias, //3 output
                        int C_in, int C_out, int H, int W, int FH, int FW){

    int outH = H - FH + 1, outW = W - FW + 1;
    size_t bytes_in     = (size_t)C_in  * H * W          * sizeof(float);
    size_t bytes_filter = (size_t)C_out * C_in * FH * FW * sizeof(float);
    size_t bytes_bias   = (size_t)C_out                  * sizeof(float);
    size_t bytes_out    = (size_t)C_out * outH * outW    * sizeof(float);

    float *d_dOut, *d_input, *d_filter, *d_dInput, *d_dFilter, *d_dBias;
    CUDA_CHECK(cudaMalloc(&d_dOut,    bytes_out));
    CUDA_CHECK(cudaMalloc(&d_input,   bytes_in));
    CUDA_CHECK(cudaMalloc(&d_filter,  bytes_filter));
    CUDA_CHECK(cudaMalloc(&d_dInput,  bytes_in));
    CUDA_CHECK(cudaMalloc(&d_dFilter, bytes_filter));
    CUDA_CHECK(cudaMalloc(&d_dBias,   bytes_bias));

    CUDA_CHECK(cudaMemcpy(d_dOut,   dOut,   bytes_out,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input,  input,  bytes_in,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, filter, bytes_filter, cudaMemcpyHostToDevice));


    conv2d_mc_backward_bias<<<(C_out + 255) / 256, 256>>>(
        d_dOut, d_dBias, C_out, outH, outW);

    int nW = C_out * C_in * FH * FW;
    conv2d_mc_backward_weights<<<(nW + 255) / 256, 256>>>(
        d_dOut, d_input, d_dFilter, C_in, C_out, H, W, FH, FW);

    dim3 inBlock(16, 16, 1);
    dim3 inGrid((W + 15) / 16, (H + 15) / 16, C_in);
    conv2d_mc_backward_input<<<inGrid, inBlock>>>(
        d_dOut, d_filter, d_dInput, C_in, C_out, H, W, FH, FW);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dInput,  d_dInput,  bytes_in,     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dFilter, d_dFilter, bytes_filter, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dBias,   d_dBias,   bytes_bias,   cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dOut));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_dInput));
    CUDA_CHECK(cudaFree(d_dFilter));
    CUDA_CHECK(cudaFree(d_dBias));
}

static float forward_loss(const float *in, const float *filt, const float *b,
                          int C_in, int C_out, int H, int W, int FH, int FW){
    int outH = H-FH+1, outW = W-FW+1, nOut = C_out*outH*outW;
    float *out = (float*)malloc(sizeof(float)*nOut);
    conv2d_mc(in, filt, b, out, C_in, C_out, H, W, FH, FW);
    float L = 0; for (int i = 0; i < nOut; ++i) L += out[i];
    free(out);
    return L;
}

#ifndef BUILD_AS_LIBRARY
int main(){
    const int C_in=2, C_out=3, H=5, W=5, FH=3, FW=3;
    const int outH=H-FH+1, outW=W-FW+1;
    int nIn=C_in*H*W, nF=C_out*C_in*FH*FW, nOut=C_out*outH*outW;

    float *in=(float*)malloc(sizeof(float)*nIn);
    float *filt=(float*)malloc(sizeof(float)*nF);
    float *b=(float*)malloc(sizeof(float)*C_out);
    for (int i=0;i<nIn;++i) in[i]   = (float)((i*7)%13)/13.0f - 0.5f;  // arbitrary spread
    for (int i=0;i<nF; ++i) filt[i] = (float)((i*5)%11)/11.0f - 0.5f;
    for (int i=0;i<C_out;++i) b[i] = 0.1f*i;

    // analytic gradients, with dOut = ones 
    float *dOut=(float*)malloc(sizeof(float)*nOut); for(int i=0;i<nOut;++i)dOut[i]=1.0f;
    float *dIn=(float*)malloc(sizeof(float)*nIn);
    float *dF =(float*)malloc(sizeof(float)*nF);
    float *dB =(float*)malloc(sizeof(float)*C_out);
    conv2d_mc_backward(dOut, in, filt, dIn, dF, dB, C_in, C_out, H, W, FH, FW);

    const float eps=1e-3f;
    float maxF=0, maxI=0;
    for (int k=0;k<nF;++k){                              // check dFilter
        float s=filt[k];
        filt[k]=s+eps; float Lp=forward_loss(in,filt,b,C_in,C_out,H,W,FH,FW);
        filt[k]=s-eps; float Lm=forward_loss(in,filt,b,C_in,C_out,H,W,FH,FW);
        filt[k]=s;
        maxF=fmaxf(maxF, fabsf((Lp-Lm)/(2*eps) - dF[k]));
    }
    for (int k=0;k<nIn;++k){                             // check dInput
        float s=in[k];
        in[k]=s+eps; float Lp=forward_loss(in,filt,b,C_in,C_out,H,W,FH,FW);
        in[k]=s-eps; float Lm=forward_loss(in,filt,b,C_in,C_out,H,W,FH,FW);
        in[k]=s;
        maxI=fmaxf(maxI, fabsf((Lp-Lm)/(2*eps) - dIn[k]));
    }
    printf("dFilter max |analytic - numeric| = %.6f\n", maxF);
    printf("dInput  max |analytic - numeric| = %.6f\n", maxI);
    printf("dBias = "); for(int i=0;i<C_out;++i) printf("%.1f ", dB[i]);
    printf(" (expect %d each)\n", outH*outW);
    return 0;
}
#endif
