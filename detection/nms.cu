#include "common.cuh"

struct Det { float cx, cy, w, h, score; int cls; };

__device__ float iou_dev(float ax, float ay, float aw, float ah,
                         float bx, float by, float bw, float bh){
    float leftA = ax - aw / 2.0f, rightA = ax + aw / 2.0f;
    float topA  = ay - ah / 2.0f, botA   = ay + ah / 2.0f;
    float leftB = bx - bw / 2.0f, rightB = bx + bw / 2.0f;
    float topB  = by - bh / 2.0f, botB   = by + bh / 2.0f;

    float w_overlap = std::max(0.0f, std::min(rightA, rightB) - std::max(leftA, leftB));
    float h_overlap = std::max(0.0f, std::min(botA, botB) - std::max(topA, topB));

    float intersection = w_overlap * h_overlap;
    float union_area   = aw*ah + bw*bh - intersection;
    if (union_area <= 0.0f) return 0.0f;
    return intersection / union_area;

}

__global__ void nms_kernel(const float* boxes, const float* scores, int n, float iou_thresh, int* keep){
    
}

int nms(const Det* cand, int n, float iou_thresh, Det* out){
    float *d_boxes;
    float *d_scores;
    int *d_keep;
    float h_boxes[16*4], h_scores[16]; int h_keep[16];
    for (int i = 0; i < n; ++i){
        h_boxes[i*4+0] = cand[i].cx;
        h_boxes[i*4+1] = cand[i].cy;
        h_boxes[i*4+2] = cand[i].w;
        h_boxes[i*4+3] = cand[i].h;
        h_scores[i]    = cand[i].score;
    }
    CUDA_CHECK(cudaMalloc(&d_boxes, (size_t)n*4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, (size_t)n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_keep, (size_t)n*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_boxes, h_boxes, (size_t)n*4*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores,  h_scores, (size_t)n*sizeof(float), cudaMemcpyHostToDevice));
    nms_kernel<<<(n+255)/256, 256>>>(d_boxes, d_scores, n, iou_thresh, d_keep);

    CUDA_CHECK(cudaMemcpy(h_keep, d_keep, (size_t)n*sizeof(int), cudaMemcpyHostToDevice));
    
    int m = 0;                                    // compact survivors into out[]
    for (int i = 0; i < n; ++i)
        if (h_keep[i]) out[m++] = cand[i];        // carries cls along, host-side

    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_keep);
    return m;
}

