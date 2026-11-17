#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
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
    const int row = int(blockIdx.y) * blockDim.y + threadIdx.y;
    const int col = int(blockIdx.x) * blockDim.x + threadIdx.x;

    // Some threads can lie outside the output image.
    if (row >= H || col >= W)
        return;

    const int radius = K / 2;
    float sum = 0.0f;

    // Does the whole mask fit inside the image?
    const bool mask_inside = row >= radius && row + radius < H && col >= radius && col + radius < W;

    if (mask_inside) {
        // Interior pixel: no boundary checks are needed.
        const float *image_start = image + (row - radius) * W + (col - radius);

        // i: mask row
        for (int i = 0; i < K; ++i) {
            const float *image_row = image_start + i * W;
            const float *mask_row = mask + i * K;

            // j: mask column
            for (int j = 0; j < K; ++j) {
                sum += image_row[j] * mask_row[j];
            }
        }
    }
    else {
        // Boundary pixel: skip positions outside the image.
        // Skipping them is equivalent to zero padding.
        for (int i = 0; i < K; ++i) {
            const int y = row + i - radius;
            const bool row_in_bounds = (y >= 0 && y < H);
            const float *mask_row = mask + i * K;

            for (int j = 0; j < K; ++j) {
                const int x = col + j - radius;
                const bool col_in_bounds = (x >= 0 && x < W);
                if (row_in_bounds && col_in_bounds) {
                    sum += image[y * W + x] * mask_row[j];
                }
            }
        }
    }

    output[row * W + col] = sum;
}



namespace {

// All threads in a block cooperate to load the image patch needed by that
// block's output pixels. The patch includes a K/2-pixel halo on each side.
__global__ void conv2d_kernel_opt_tiled(
    const float *image, float *output, const float *mask, int H, int W, int K
) {
    extern __shared__ float tile[];

    const int radius = K / 2;
    const int tile_width = static_cast<int>(blockDim.x) + K - 1;
    const int tile_height = static_cast<int>(blockDim.y) + K - 1;
    const int base_row = static_cast<int>(blockIdx.y) * blockDim.y;
    const int base_col = static_cast<int>(blockIdx.x) * blockDim.x;

    // Usually the tile has more elements than there are threads, so each
    // thread loads one or more elements. Threads whose output lies beyond the
    // image still participate: they may be needed to load a halo element.
    for (int ty = static_cast<int>(threadIdx.y); ty < tile_height;
         ty += static_cast<int>(blockDim.y)) {
        const int y = base_row + ty - radius;
        for (int tx = static_cast<int>(threadIdx.x); tx < tile_width;
             tx += static_cast<int>(blockDim.x)) {
            const int x = base_col + tx - radius;
            const bool in_bounds = y >= 0 && y < H && x >= 0 && x < W;
            tile[ty * tile_width + tx] = in_bounds ? image[y * W + x] : 0.0f;
        }
    }

    // Every thread must finish loading before any thread uses the tile.
    __syncthreads();

    const int row = base_row + threadIdx.y;
    const int col = base_col + threadIdx.x;
    if (row >= H || col >= W) return;

    float sum = 0.0f;
    const bool mask_inside = row >= radius && row + radius < H &&
                             col >= radius && col + radius < W;
    if (mask_inside) {
        // Interior: every tap is valid.
        for (int i = 0; i < K; ++i) {
            const float *tile_row = tile + (static_cast<int>(threadIdx.y) + i) * tile_width
                                        + threadIdx.x;
            const float *mask_row = mask + static_cast<int>(i) * K;
            for (int j = 0; j < K; ++j) {
                sum += tile_row[j] * mask_row[j];
            }
        }
    } else {
        // Boundary: skip invalid taps, matching the original kernel exactly.
        for (int i = 0; i < K; ++i) {
            const int y = row + i - radius;
            const float *tile_row = tile + (static_cast<int>(threadIdx.y) + i) * tile_width
                                        + threadIdx.x;
            const float *mask_row = mask + static_cast<int>(i) * K;
            for (int j = 0; j < K; ++j) {
                const int x = col + j - radius;
                if (y >= 0 && y < H && x >= 0 && x < W) {
                    sum += tile_row[j] * mask_row[j];
                }
            }
        }
    }
    output[row * W + col] = sum;
}

// Return dynamic shared-memory bytes for the tiled kernel. Zero selects the
// original kernel when the tile or block is too large for this GPU/kernel.
std::size_t shared_tile_bytes(int K, int bx, int by) {
    if (K <= 0 || bx <= 0 || by <= 0) return 0;

    int device = 0;
    checkCuda(cudaGetDevice(&device), "get CUDA device");

    // A benchmark repeats the same configuration. Avoid re-querying its
    // device/kernel limits on every iteration, while respecting device changes.
    struct Cache {
        bool valid = false;
        int device = -1, K = 0, bx = 0, by = 0;
        std::size_t bytes = 0;
    };
    static thread_local Cache cache;
    if (cache.valid && cache.device == device && cache.K == K &&
        cache.bx == bx && cache.by == by) return cache.bytes;

    cudaDeviceProp prop{};
    cudaFuncAttributes attr{};
    checkCuda(cudaGetDeviceProperties(&prop, device), "get CUDA device properties");
    checkCuda(cudaFuncGetAttributes(&attr, conv2d_kernel_opt_tiled),
              "get shared-memory kernel properties");

    // Only use the default per-block limit; no opt-in memory is required.
    const std::size_t available = prop.sharedMemPerBlock > attr.sharedSizeBytes
        ? prop.sharedMemPerBlock - attr.sharedSizeBytes : 0;
    const std::size_t kernel_limit = attr.maxDynamicSharedSizeBytes > 0
        ? static_cast<std::size_t>(attr.maxDynamicSharedSizeBytes) : 0;
    const std::size_t limit = available < kernel_limit ? available : kernel_limit;
    const auto width = static_cast<int>(bx) + static_cast<int>(K) - 1;
    const auto height = static_cast<int>(by) + static_cast<int>(K) - 1;
    const auto threads = static_cast<int>(bx) * static_cast<int>(by);
    const auto max_elements = static_cast<int>(limit / sizeof(float));
    std::size_t bytes = 0;
    if (threads <= static_cast<int>(attr.maxThreadsPerBlock) &&
        width <= static_cast<int>(std::numeric_limits<int>::max()) &&
        height <= static_cast<int>(std::numeric_limits<int>::max()) &&
        width <= max_elements && height <= max_elements / width) {
        bytes = static_cast<std::size_t>(width * height) * sizeof(float);
    }

    cache = {true, device, K, bx, by, bytes};
    // Diagnostic only; do not enable while collecting benchmark timings.
    if (std::getenv("CONV_CUDA_TRACE") != nullptr) {
        std::fprintf(stderr, "cuda_opt: %s, block=%dx%d, K=%d, shared=%zu bytes\n",
                     bytes ? "shared tile" : "original fallback", bx, by, K, bytes);
    }
    return bytes;
}

void launch_opt(dim3 grid, dim3 block, std::size_t shared_bytes,
                const float *image, float *output, const float *mask,
                int H, int W, int K) {
    if (shared_bytes != 0) {
        conv2d_kernel_opt_tiled<<<grid, block, shared_bytes>>>(image, output, mask, H, W, K);
    } else {
        conv2d_kernel_opt<<<grid, block>>>(image, output, mask, H, W, K);
    }
}

} // namespace

// Check launch dimensions once before timing. The kernel launch itself still
// has a CUDA error check, which catches additional resource limitations.
bool cuda_block_supported(int H, int W, int bx, int by) {
    if (H <= 0 || W <= 0 || bx <= 0 || by <= 0) return false;

    int device = 0;
    cudaDeviceProp prop{};
    checkCuda(cudaGetDevice(&device), "get CUDA device");
    checkCuda(cudaGetDeviceProperties(&prop, device), "get CUDA device properties");

    const auto threads = static_cast<int>(bx) * static_cast<int>(by);
    const auto grid_x = (static_cast<int>(W) + bx - 1) / bx;
    const auto grid_y = (static_cast<int>(H) + by - 1) / by;
    return bx <= prop.maxThreadsDim[0] && by <= prop.maxThreadsDim[1] &&
           threads <= static_cast<int>(prop.maxThreadsPerBlock) &&
           grid_x <= static_cast<int>(prop.maxGridSize[0]) &&
           grid_y <= static_cast<int>(prop.maxGridSize[1]);
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

    const std::size_t shared_bytes = shared_tile_bytes(c.K, bx, by);

    checkCuda(cudaEventRecord(e[0]), "record H2D start");
    checkCuda(cudaMemcpy(c.d_in, in, image_bytes, cudaMemcpyHostToDevice), "copy image to GPU");
    checkCuda(cudaMemcpy(c.d_k, k, mask_bytes, cudaMemcpyHostToDevice), "copy filter to GPU");
    checkCuda(cudaEventRecord(e[1]), "record H2D end");
    launch_opt(grid, block, shared_bytes, c.d_in, c.d_out, c.d_k, c.H, c.W, c.K);
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

        const std::size_t shared_bytes = shared_tile_bytes(K, BLOCK_x, BLOCK_y);
        launch_opt(grid, block, shared_bytes, d_image, d_output, d_mask, H, W, K);

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
