#include "common.cuh"

struct Det { float cx, cy, w, h, score; int cls; };

__device__ float iou_dev(float ax, float ay, float aw, float ah,
                         float bx, float by, float bw, float bh){


}

__global__ void nms_kernel(const float* boxes, const float* scores, int n, float iou_thresh, int* keep){

}

int nms(const Det* cand, int n, float iou_thresh, Det* out){
    float *d_boxes;
    float *d_scores;
    int *d_keep;
    CUDA_CHECK(cudaMalloc(&d_boxes, (size_t)n*4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, (size_t)n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_keep, (size_t)n*sizeof(int)));
    nms_kernel<<<(n+255)/256, 256>>>(d_boxes, d_scores, n, iou_thresh, d_keep);
    float h_boxes[16*4], h_scores[16]; int h_keep[16];
    for (int i = 0; i < n; ++i){
        h_boxes[i*4+0] = cand[i].cx;
        h_boxes[i*4+1] = cand[i].cy;
        h_boxes[i*4+2] = cand[i].w;
        h_boxes[i*4+3] = cand[i].h;
        h_scores[i]    = cand[i].score;
    }
    CUDA_CHECK(cudaMemcpy(d_boxes, h_boxes, (size_t)n*4*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores,  h_scores, (size_t)n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_keep, h_keep, (size_t)n*sizeof(int), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaFree(d_boxes));
    CUDA_CHECK(cudaFree(d_boxes));
    CUDA_CHECK(cudaFree(d_boxes));

    free(h_boxes);
    free(h_scores);
    free(h_keep);
}

