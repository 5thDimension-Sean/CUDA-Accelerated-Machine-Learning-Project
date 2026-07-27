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
}

