#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "conv2d.h"
#define BLOCK_x 32
#define BLOCK_y 8 // The existing one-call correctness-test wrapper uses 32x8.
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
__global__ void conv2d_kernel_opt(
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


// Check launch dimensions once before timing. The kernel launch itself still
// has a CUDA error check, which catches additional resource limitations.
bool cuda_block_supported(int H, int W, int bx, int by) {
    if (H <= 0 || W <= 0 || bx <= 0 || by <= 0) return false;

    int device = 0;
    cudaDeviceProp prop{};
    checkCuda(cudaGetDevice(&device), "get CUDA device");
    checkCuda(cudaGetDeviceProperties(&prop, device), "get CUDA device properties");

    const auto threads = static_cast<std::uint64_t>(bx) * static_cast<std::uint64_t>(by);
    const auto grid_x = (static_cast<std::uint64_t>(W) + bx - 1) / bx;
    const auto grid_y = (static_cast<std::uint64_t>(H) + by - 1) / by;
    return bx <= prop.maxThreadsDim[0] && by <= prop.maxThreadsDim[1] &&
           threads <= static_cast<std::uint64_t>(prop.maxThreadsPerBlock) &&
           grid_x <= static_cast<std::uint64_t>(prop.maxGridSize[0]) &&
           grid_y <= static_cast<std::uint64_t>(prop.maxGridSize[1]);
}

// Use cuda_conv_alloc's buffers/events; caller validates dimensions once.
CudaTimes cuda_conv_run_opt(const CudaConv& c, const float* in, float* out,
                            const float* k, int bx, int by) {
    if (c.H <= 0 || c.W <= 0 || c.K <= 0 || bx <= 0 || by <= 0) {
        throw std::invalid_argument("Invalid image, filter, or block size");
    }
    const std::size_t image_bytes = static_cast<std::size_t>(c.H) * c.W * sizeof(float);
    const std::size_t mask_bytes = static_cast<std::size_t>(c.K) * c.K * sizeof(float);
    cudaEvent_t e[4];
    for (int i = 0; i < 4; ++i) e[i] = (cudaEvent_t)c.events[i];

    const dim3 block(bx, by);
    const dim3 grid((static_cast<std::size_t>(c.W) + block.x - 1) / block.x,
                    (static_cast<std::size_t>(c.H) + block.y - 1) / block.y);

    checkCuda(cudaEventRecord(e[0]), "record H2D start");
    checkCuda(cudaMemcpy(c.d_in, in, image_bytes, cudaMemcpyHostToDevice), "copy image to GPU");
    checkCuda(cudaMemcpy(c.d_k, k, mask_bytes, cudaMemcpyHostToDevice), "copy filter to GPU");
    checkCuda(cudaEventRecord(e[1]), "record H2D end");
    conv2d_kernel_opt<<<grid, block>>>(c.d_in, c.d_out, c.d_k, c.H, c.W, c.K);
    checkCuda(cudaGetLastError(), "launch optimized kernel");
    checkCuda(cudaEventRecord(e[2]), "record kernel end");
    checkCuda(cudaMemcpy(out, c.d_out, image_bytes, cudaMemcpyDeviceToHost), "copy output to CPU");
    checkCuda(cudaEventRecord(e[3]), "record D2H end");
    checkCuda(cudaEventSynchronize(e[3]), "synchronize");

    CudaTimes times;
    checkCuda(cudaEventElapsedTime(&times.h2d_ms, e[0], e[1]), "measure H2D");
    checkCuda(cudaEventElapsedTime(&times.kernel_ms, e[1], e[2]), "measure kernel");
    checkCuda(cudaEventElapsedTime(&times.d2h_ms, e[2], e[3]), "measure D2H");
    return times;
}

// Host function. image, output, and mask point to CPU memory.
void conv2d_cuda_opt(
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

        conv2d_kernel_opt<<<grid, block>>>(d_image, d_output, d_mask, H, W, K);

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
