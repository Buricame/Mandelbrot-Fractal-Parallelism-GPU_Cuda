
#include <iostream>
#include <fstream>
#include <complex>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// '<<<' operatörü visual studioda çalışmadığı için makro atama yapıldı
#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block)                 <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem)         <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

// Yükseklik genişlik ve iterasyon sayısını ayarlama
#define WIDTH 100000
#define HEIGHT 100000
#define MAX_ITER 1000

// CUDA için kernelda global değişkeni kullanılarak mandelbrot hesaplanmıştır
__global__ void mandelbrot(int* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < WIDTH && y < HEIGHT) {
        double dx = 3.0 / WIDTH;
        double dy = 2.0 / HEIGHT;

        double x0 = x * dx - 2.0;
        double y0 = y * dy - 1.0;

        double zx = 0.0;
        double zy = 0.0;
        double zx2 = 0.0;
        double zy2 = 0.0;

        int iter = 0;
        while (zx2 + zy2 < 4 && iter < MAX_ITER) {
            zy = 2 * zx * zy + y0;
            zx = zx2 - zy2 + x0;
            zx2 = zx * zx;
            zy2 = zy * zy;
            iter++;
        }

        output[y * WIDTH + x] = iter;
    }
}

// Çıktıyı bmp resim dosyası olarak kaydeder
void saveBMP(int* data, const char* filename) {
    std::ofstream outFile(filename, std::ios::out | std::ios::binary);

    // BMP header
    unsigned char header[54] = {
        0x42, 0x4D, 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0, 40, 0,
        0, 0, (unsigned char)(WIDTH & 0xff), (unsigned char)((WIDTH >> 8) & 0xff), (unsigned char)((WIDTH >> 16) & 0xff), (unsigned char)((WIDTH >> 24) & 0xff),
        (unsigned char)(HEIGHT & 0xff), (unsigned char)((HEIGHT >> 8) & 0xff), (unsigned char)((HEIGHT >> 16) & 0xff), (unsigned char)((HEIGHT >> 24) & 0xff),
        1, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    // Write header
    outFile.write(reinterpret_cast<const char*>(header), sizeof(header));

    // Write image data
    for (int y = HEIGHT - 1; y >= 0; --y) {
        for (int x = 0; x < WIDTH; ++x) {
            unsigned char color = (data[y * WIDTH + x] == MAX_ITER) ? 0 : 255;
            outFile.write(reinterpret_cast<const char*>(&color), sizeof(color));
            outFile.write(reinterpret_cast<const char*>(&color), sizeof(color));
            outFile.write(reinterpret_cast<const char*>(&color), sizeof(color));
        }
        // Padding to make sure each row's size is a multiple of 4
        const unsigned char pad = 0;
        for (int p = 0; p < (4 - (WIDTH * 3) % 4) % 4; ++p) {
            outFile.write(reinterpret_cast<const char*>(&pad), sizeof(pad));
        }
    }

    outFile.close();
}

int main() {
    int* output;
    cudaMallocManaged(&output, WIDTH * HEIGHT * sizeof(int));

    //Cuda thread yapılandırması belirlenir.(Yükseklik,Genişlik)
    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    //Hesaplama süresini hesaplamak için başlangıç ve bitiş süreleri tanımlanır ve tutulur
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    mandelbrot KERNEL_ARGS2(grid, block) (output); // Cuda kerneldeki kodu çalıştırır.
    cudaDeviceSynchronize(); //Hesaplama yaparken Cuda çekirdekleri arasında senkronizasyonu sağlar

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Hesaplama süresini miliseconds float olarak hesaplayan kod
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

    saveBMP(output, "mandelbrot.bmp"); // Resmi bmp formatı ile diske bu isimle kaydeder

    cudaFree(output); // Cudaları serbest bırakır ve işlemi sonlandırır

    return 0;
}
