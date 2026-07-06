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

        image_buffer[i] = 255 - r;
        image_buffer[i + 1] = 255 - g;
        image_buffer[i + 2] = 255 - b;
        //converting to grayscale
        unsigned char y = 0.299 * r + 0.587 * g + 0.114 * b;
        image_buffer[i] = y;
        image_buffer[i + 1] = y;
        image_buffer[i + 2] = y;
    }
    stbi_write_png(saveout_file_name, w, h, c, image_buffer, w * c);
    stbi_image_free(image_buffer);

}