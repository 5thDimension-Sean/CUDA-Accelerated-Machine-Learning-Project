#include "common.cuh"

struct Det { float cx, cy, w, h, score; int cls; };

__device__ float iou_dev(float ax, float ay, float aw, float ah,
                         float bx, float by, float bw, float bh){


}

__global__ void nms_kernel(const float* boxes, const float* scores, int n, float iou_thresh, int* keep){

}

int nms(const Det* cand, int n, float iou_thresh, Det* out){

}

#ifndef BUILD_AS_LIBRARY
int main(){

    return 0;
}
#endif