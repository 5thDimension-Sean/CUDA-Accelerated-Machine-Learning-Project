#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image.h"
#include "../stb_image_write.h"
#include "conv2d.cu"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main(){
    const char* file_name = "images.jpg";
    const char* saveout_file_name = "images_edit.png";
    int w, h, c = 0;

    unsigned char* image_buffer = stbi_load(file_name, &w, &h, &c, 0);

    printf("%d %d %d\n", w, h, c);
    unsigned char r, g, b;
    for(int i = 0; i < w * h * c; i += c){
        r = image_buffer[i];
        g = image_buffer[i + 1];
        b = image_buffer[i + 2];
        //converting to grayscale
        unsigned char y = 0.299 * r + 0.587 * g + 0.114 * b;
        image_buffer[i] = y;
        image_buffer[i + 1] = y;
        image_buffer[i + 2] = y;
    }
    //cuda events

    const int FH = 3;
    const int FW = 3;
    size_t size = w*h * sizeof(float);
    size_t filter_size = FH*FW * sizeof(float);
    float* d_input = nullptr;
    cudaMalloc((void**)&d_input, size);
    float* d_filter = nullptr;
    cudaMalloc((void**)&d_filter, filter_size);
    float* d_output = nullptr;
    cudaMalloc((void**)&d_output, size);
    const int outH = h - FH + 1;
    const int outW = w - FW + 1;
    float* h_output = (float*)malloc(outH * outW * sizeof(float));
    float* h_input = (float*)malloc(size);
    float* h_filter = (float*)malloc(filter_size);
    h_filter[0] = 1/16.f; h_filter[1] = 2/16.f; h_filter[2] = 1/16.f;
    h_filter[3] = 2/16.f; h_filter[4] = 4/16.f; h_filter[5] = 2/16.f;
    h_filter[6] = 1/16.f; h_filter[7] = 2/16.f; h_filter[8] = 1/16.f;
    //fillin with gaussian blur filter values
    // Configure execution configuration block and grid sizes
    for(int i = 0; i < w * h; ++i) {
        h_input[i] = image_buffer[i * c]/255.0f; // Assuming grayscale image, take the first channel
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice));
    dim3 gridDim((outW + 31) / 32, (outH + 31) / 32);
    dim3 blockDim(32, 32);
    CUDA_CHECK(cudaMemset(d_output, 0, outW * outH * sizeof(float)));
    conv2d_naive<<<gridDim, blockDim>>>(d_input, d_filter, d_output, h, w, FH, FW);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, outW * outH * sizeof(float), cudaMemcpyDeviceToHost));
    unsigned char* out_img = (unsigned char*)malloc(outH * outW);
    for (int i = 0; i < outH * outW; i++)
        out_img[i] = (unsigned char)(h_output[i] * 255.0f);
    stbi_write_png(saveout_file_name, outW, outH, 1, out_img, outW);
    free(out_img);
    stbi_image_free(image_buffer);
    free(h_input);
    free(h_filter);
    free(h_output);
    free(out_img);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    stbi_image_free(image_buffer);

    return 0;
}