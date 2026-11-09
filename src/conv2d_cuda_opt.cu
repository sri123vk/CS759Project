#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "conv2d.h"
#define BLOCK_x 32
#define BLOCK_y 8 //incur launch and scheduling/resource overhead , if you dont configure it runtime/ based on matrix size
namespace {

// Report CUDA errors instead of continuing with invalid results.
void checkCuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

} // namespace


// One GPU thread computes one output pixel.
__global__ void conv2d_kernel(
    const float *image,
    float *output,
    const float *mask,
    int H,
    int W,
    int K
) {
    // y maps to image rows; x maps to image columns.
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::int64_t col = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Some threads can lie outside the output image.
    if (row >= H || col >= W)
        return;

    const std::int64_t radius = K / 2;
    float sum = 0.0f;

    // Does the whole mask fit inside the image?
    const bool mask_inside = row >= radius && row + radius < H && col >= radius && col + radius < W;

    if (mask_inside) {
        // Interior pixel: no boundary checks are needed.
        const float *image_start = image + (row - radius) * W + (col - radius);

        // i: mask row
        for (std::int64_t i = 0; i < K; ++i) {
            const float *image_row = image_start + i * W;
            const float *mask_row = mask + i * K;

            // j: mask column
            for (std::int64_t j = 0; j < K; ++j) {
                sum += image_row[j] * mask_row[j];
            }
        }
    }
    else {
        // Boundary pixel: skip positions outside the image.
        // Skipping them is equivalent to zero padding.
        for (std::int64_t i = 0; i < K; ++i) {
            const std::int64_t y = row + i - radius;
            const bool row_in_bounds = (y >= 0 && y < H);
            const float *mask_row = mask + i * K;

            for (std::int64_t j = 0; j < K; ++j) {
                const std::int64_t x = col + j - radius;
                const bool col_in_bounds = (x >= 0 && x < W);
                if (row_in_bounds && col_in_bounds) {
                    sum += image[y * W + x] * mask_row[j];
                }
            }
        }
    }

    output[row * W + col] = sum;
}


// Host function. image, output, and mask point to CPU memory.
void conv2d_cuda(
    const float *image,
    float *output,
    const float *mask,
    int H,
    int W,
    int K
) {
    if (image == nullptr || output == nullptr || mask == nullptr ||
        H <= 0 || W <= 0 || K <= 0 || K % 2 == 0) {
        throw std::invalid_argument("Invalid image or mask parameters");
    }

    // Calculate allocation sizes using size_t.
    const std::size_t h = static_cast<std::size_t>(H);
    const std::size_t w = static_cast<std::size_t>(W);
    const std::size_t k = static_cast<std::size_t>(K);

    const std::size_t max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);

    if (h > max_elements / w || k > max_elements / k) {
        throw std::overflow_error("Image or mask allocation size is too large");
    }

    const std::size_t image_bytes = h * w * sizeof(float);
    const std::size_t mask_bytes = k * k * sizeof(float);

    float *d_image = nullptr;
    float *d_output = nullptr;
    float *d_mask = nullptr;

    // Release any buffers that were successfully allocated.
    auto cleanup = [&]() {
        if (d_image)  cudaFree(d_image);
        if (d_output) cudaFree(d_output);
        if (d_mask)   cudaFree(d_mask);
    };

    try {
        //Allocate GPU memory.
        checkCuda(
            cudaMalloc(reinterpret_cast<void **>(&d_image), image_bytes),
            "cudaMalloc image"
        );

        checkCuda(
            cudaMalloc(reinterpret_cast<void **>(&d_output), image_bytes),
            "cudaMalloc output"
        );

        checkCuda(
            cudaMalloc(reinterpret_cast<void **>(&d_mask), mask_bytes),
            "cudaMalloc mask"
        );

        // Copy input image and mask from CPU to GPU.
        checkCuda(cudaMemcpy(d_image, image, image_bytes, cudaMemcpyHostToDevice),"copy image to GPU");

        checkCuda(cudaMemcpy(d_mask, mask, mask_bytes, cudaMemcpyHostToDevice),"copy mask to GPU");

        // Configure and launch the GPU kernel.
        const dim3 block(BLOCK_x, BLOCK_y);

        const dim3 grid((w + block.x - 1) / block.x,(h + block.y - 1) / block.y);

        conv2d_kernel<<<grid, block>>>(d_image, d_output, d_mask, H, W, K);

        // Check for launch errors and execution errors.
        checkCuda(cudaGetLastError(), "launch kernel");
        checkCuda(cudaDeviceSynchronize(), "execute kernel");

        // Copy the result from GPU back to CPU.
        checkCuda(cudaMemcpy(output, d_output, image_bytes, cudaMemcpyDeviceToHost),"copy output to CPU");
    }
    catch (...) {
        cleanup();
        throw;
    }

    // Release GPU memory.
    cleanup();
}