// 2-D cross-correlation experiments beyond the direct baseline in
// conv2d_cuda.cu (impl name `cuda`).
//
// All variants take C input channels (CHW layout, see conv2d.h) and sum them
// into one output plane, adding channel by channel and, within a channel, in
// the same row/column order as before. Shared-tile variants load one
// channel's tile at a time, so their shared memory does not grow with C.
// Constant-memory variants hold all C filter slices in the constant symbol,
// so they need C*K*K <= 4096.
//
// cuda_opt:         shared input tile, filter in global memory.
// cuda_const:       one output per thread, filter in constant memory.
// cuda_tiled_const: shared input tile plus constant-memory filter.
// cuda_multiout:             two outputs per thread, global image and filter.
// cuda_multiout_const:       two outputs per thread, constant-memory filter.
// cuda_multiout_tiled:       two outputs per thread, shared-memory image tile.
// cuda_multiout_tiled_const: two outputs per thread, shared tile and constant filter.
//
// Constant-memory symbols are shared by launches in a CUDA context. A mutex
// protects uploads and launches through this file from concurrent callers.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#include "conv2d.h"

namespace {

// 4096 floats = 16 KiB, holding all C filter slices: C*K*K <= 4096 (for one
// channel, odd square filters through 63 x 63 fit). Larger filters are
// rejected explicitly rather than silently changing implementations in a
// performance experiment.
constexpr std::size_t kConstantFloats = 4096;
__constant__ float g_variant_filter[kConstantFloats];
std::mutex g_constant_filter_mutex;

void check_variant(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Same one-thread-per-output structure as the direct baseline, but reading
// filter coefficients from a constant symbol. Bounds are clipped outside the
// inner loop, just as in the project's direct baseline.
__global__ void conv2d_variant_constant(
    const float* image, float* output, int C, int H, int W, int K
) {
    // The host validates that all 32-bit indices and intermediate values fit.
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= H || col >= W) return;

    const int radius = K / 2;
    const int i_lo = max(0, radius - row);
    const int i_hi = min(K, H - row + radius);
    const int j_lo = max(0, radius - col);
    const int j_hi = min(K, W - col + radius);
    const std::size_t plane = static_cast<std::size_t>(H) * W;

    float sum = 0.0f;
    for (int ch = 0; ch < C; ++ch) {
        const float* channel = image + ch * plane;
        // Index the symbol directly (not through a pointer) so the reads
        // stay in the constant address space. C*K*K <= 4096 fits an int.
        const int f0 = ch * K * K;
        for (int i = i_lo; i < i_hi; ++i) {
            const int y = row + i - radius;
            for (int j = j_lo; j < j_hi; ++j) {
                const int x = col + j - radius;
                sum += channel[y * W + x] * g_variant_filter[f0 + i * K + j];
            }
        }
    }
    output[row * W + col] = sum;
}

// Read a filter coefficient from constant memory or from the global buffer.
// Templating lets each pair of kernels differ only in where the filter lives.
template <bool UseConstant>
__device__ __forceinline__ float filter_weight(const float* filter, int index) {
    if constexpr (UseConstant) {
        return g_variant_filter[index];
    } else {
        return filter[index];
    }
}

// Each block cooperatively loads its output-sized image tile plus the halo,
// one input channel at a time. The filter is independent of the shared input
// tile: cuda_opt reads it from global memory, cuda_tiled_const from constant
// memory. Every thread, including those whose output lies outside the image
// in a partial edge block, participates in every load and every barrier.
template <bool UseConstant>
__global__ void conv2d_variant_tiled(
    const float* image, float* output, const float* filter,
    int C, int H, int W, int K
) {
    extern __shared__ float tile[];
    const int radius = K / 2;
    const int tile_width = static_cast<int>(blockDim.x) + K - 1;
    const int tile_height = static_cast<int>(blockDim.y) + K - 1;
    const int base_y = static_cast<int>(blockIdx.y * blockDim.y);
    const int base_x = static_cast<int>(blockIdx.x * blockDim.x);
    const std::size_t plane = static_cast<std::size_t>(H) * W;

    const int row = base_y + static_cast<int>(threadIdx.y);
    const int col = base_x + static_cast<int>(threadIdx.x);
    const bool active = row < H && col < W;
    const bool interior = row >= radius && row + radius < H &&
                          col >= radius && col + radius < W;
    float sum = 0.0f;

    for (int ch = 0; ch < C; ++ch) {
        const float* channel = image + ch * plane;
        const int f0 = ch * K * K;

        // Wait until every thread has finished reading the previous
        // channel's tile before overwriting it.
        if (ch > 0) __syncthreads();
        for (int ty = static_cast<int>(threadIdx.y); ty < tile_height;
             ty += static_cast<int>(blockDim.y)) {
            const int y = base_y + ty - radius;
            for (int tx = static_cast<int>(threadIdx.x); tx < tile_width;
                 tx += static_cast<int>(blockDim.x)) {
                const int x = base_x + tx - radius;
                tile[ty * tile_width + tx] = y >= 0 && y < H && x >= 0 && x < W
                    ? channel[y * W + x] : 0.0f;
            }
        }
        __syncthreads();

        if (interior) {
            for (int i = 0; i < K; ++i) {
                const float* tile_row = tile +
                    (static_cast<int>(threadIdx.y) + i) * tile_width + threadIdx.x;
                for (int j = 0; j < K; ++j) {
                    sum += tile_row[j] * filter_weight<UseConstant>(filter, f0 + i * K + j);
                }
            }
        } else if (active) {
            // Skip invalid taps, including 0 * infinity or 0 * NaN at borders.
            for (int i = 0; i < K; ++i) {
                const int y = row + i - radius;
                const float* tile_row = tile +
                    (static_cast<int>(threadIdx.y) + i) * tile_width + threadIdx.x;
                for (int j = 0; j < K; ++j) {
                    const int x = col + j - radius;
                    if (y >= 0 && y < H && x >= 0 && x < W) {
                        sum += tile_row[j] * filter_weight<UseConstant>(filter, f0 + i * K + j);
                    }
                }
            }
        }
    }
    if (active) output[row * W + col] = sum;
}

// Each thread computes two horizontally separated output pixels. For each
// q=0,1, adjacent lanes address adjacent columns (for warp-wide block rows).
// Each coefficient loaded by a thread can be used by both accumulators.
// This variant does not use shared memory or constant memory.
__global__ void conv2d_variant_multiout(
    const float* image, float* output, const float* filter,
    int C, int H, int W, int K
) {
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int col0 = static_cast<int>(blockIdx.x * (2U * blockDim.x) + threadIdx.x);
    const int col1 = col0 + static_cast<int>(blockDim.x);
    if (row >= H || col0 >= W) return;

    const int radius = K / 2;
    const bool have_second = col1 < W;
    const bool interior = have_second && row >= radius && row + radius < H &&
                          col0 >= radius && col1 + radius < W;
    const std::size_t plane = static_cast<std::size_t>(H) * W;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (int ch = 0; ch < C; ++ch) {
        const float* channel = image + ch * plane;
        const float* channel_filter = filter + ch * K * K;
        if (interior) {
            const float* first = channel + (row - radius) * W + col0 - radius;
            const float* second = channel + (row - radius) * W + col1 - radius;
            for (int i = 0; i < K; ++i) {
                const float* in0 = first + i * W;
                const float* in1 = second + i * W;
                const float* weights = channel_filter + i * K;
                for (int j = 0; j < K; ++j) {
                    const float coefficient = weights[j];
                    sum0 += in0[j] * coefficient;
                    sum1 += in1[j] * coefficient;
                }
            }
        } else {
            for (int i = 0; i < K; ++i) {
                const int y = row + i - radius;
                if (y < 0 || y >= H) continue;
                for (int j = 0; j < K; ++j) {
                    const int x0 = col0 + j - radius;
                    const int x1 = col1 + j - radius;
                    const bool valid0 = x0 >= 0 && x0 < W;
                    const bool valid1 = have_second && x1 >= 0 && x1 < W;
                    if (valid0 || valid1) {
                        const float coefficient = channel_filter[i * K + j];
                        if (valid0) sum0 += channel[y * W + x0] * coefficient;
                        if (valid1) sum1 += channel[y * W + x1] * coefficient;
                    }
                }
            }
        }
    }

    output[row * W + col0] = sum0;
    if (have_second) output[row * W + col1] = sum1;
}

// Compile separate kernels for the requested combinations. Each thread uses
// one coefficient for two accumulators; this is an independent optimization
// from shared image tiles and constant-memory filter coefficients.
template <bool UseShared, bool UseConstant>
__global__ void conv2d_variant_multiout_combined(
    const float* image, float* output, const float* filter,
    int C, int H, int W, int K
) {
    extern __shared__ float tile[];
    const int bx = static_cast<int>(blockDim.x);
    const int by = static_cast<int>(blockDim.y);
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int radius = K / 2;
    const int base_y = static_cast<int>(blockIdx.y * blockDim.y);
    const int base_x = static_cast<int>(blockIdx.x * (2U * blockDim.x));
    const int tile_width = 2 * bx + K - 1;
    const std::size_t plane = static_cast<std::size_t>(H) * W;

    const int row = base_y + ty;
    const int col0 = base_x + tx;
    const int col1 = col0 + bx;
    // No early return: in the shared variants every thread must reach every
    // barrier in the channel loop, so out-of-image threads just skip the sums.
    const bool active = row < H && col0 < W;
    const bool have_second = col1 < W;
    const bool interior = have_second && row >= radius && row + radius < H &&
                          col0 >= radius && col1 + radius < W;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (int ch = 0; ch < C; ++ch) {
        const float* channel = image + ch * plane;
        const int f0 = ch * K * K;

        if constexpr (UseShared) {
            // A block produces 2*bx by by outputs. Load this channel's input
            // tile, including both halos. Every thread helps, including
            // threads whose output positions are outside the image in a
            // partial edge block. The first barrier keeps the previous
            // channel's tile intact until every thread has finished with it.
            if (ch > 0) __syncthreads();
            const int tile_height = by + K - 1;
            for (int tile_y = ty; tile_y < tile_height; tile_y += by) {
                const int y = base_y + tile_y - radius;
                for (int tile_x = tx; tile_x < tile_width; tile_x += bx) {
                    const int x = base_x + tile_x - radius;
                    tile[tile_y * tile_width + tile_x] =
                        y >= 0 && y < H && x >= 0 && x < W
                        ? channel[y * W + x] : 0.0f;
                }
            }
            __syncthreads();
        }

        if (interior) {
            for (int i = 0; i < K; ++i) {
                const float* in0;
                const float* in1;
                if constexpr (UseShared) {
                    in0 = tile + (ty + i) * tile_width + tx;
                    in1 = in0 + bx;
                } else {
                    const int y = row + i - radius;
                    in0 = channel + y * W + col0 - radius;
                    in1 = channel + y * W + col1 - radius;
                }
                for (int j = 0; j < K; ++j) {
                    const float coefficient =
                        filter_weight<UseConstant>(filter, f0 + i * K + j);
                    sum0 += in0[j] * coefficient;
                    sum1 += in1[j] * coefficient;
                }
            }
        } else if (active) {
            // Match the direct implementation: invalid image taps are skipped,
            // not multiplied by padding zeros (important for NaN/Inf weights).
            for (int i = 0; i < K; ++i) {
                const int y = row + i - radius;
                if (y < 0 || y >= H) continue;
                for (int j = 0; j < K; ++j) {
                    const int x0 = col0 + j - radius;
                    const int x1 = col1 + j - radius;
                    const bool valid0 = x0 >= 0 && x0 < W;
                    const bool valid1 = have_second && x1 >= 0 && x1 < W;
                    if (valid0 || valid1) {
                        const float coefficient =
                            filter_weight<UseConstant>(filter, f0 + i * K + j);
                        if constexpr (UseShared) {
                            const int tile_index = (ty + i) * tile_width + tx + j;
                            if (valid0) sum0 += tile[tile_index] * coefficient;
                            if (valid1) sum1 += tile[tile_index + bx] * coefficient;
                        } else {
                            if (valid0) sum0 += channel[y * W + x0] * coefficient;
                            if (valid1) sum1 += channel[y * W + x1] * coefficient;
                        }
                    }
                }
            }
        }
    }

    if (!active) return;
    output[row * W + col0] = sum0;
    if (have_second) output[row * W + col1] = sum1;
}

bool is_multiout(CudaVariant variant) {
    return variant == CudaVariant::MultiOutput ||
           variant == CudaVariant::MultiOutputConstant ||
           variant == CudaVariant::MultiOutputTiled ||
           variant == CudaVariant::MultiOutputTiledConstant;
}

bool uses_shared_tile(CudaVariant variant) {
    return variant == CudaVariant::Tiled ||
           variant == CudaVariant::TiledConstant ||
           variant == CudaVariant::MultiOutputTiled ||
           variant == CudaVariant::MultiOutputTiledConstant;
}

bool uses_constant_filter(CudaVariant variant) {
    return variant == CudaVariant::Constant ||
           variant == CudaVariant::TiledConstant ||
           variant == CudaVariant::MultiOutputConstant ||
           variant == CudaVariant::MultiOutputTiledConstant;
}

struct LaunchConfig {
    dim3 block;
    dim3 grid;
    std::size_t shared_bytes = 0;
};

// Check the actual kernel's limits. Cache the last configuration so repeated
// timing iterations do not repeatedly query properties or function attributes.
LaunchConfig get_launch_config(int H, int W, int K, int bx, int by,
                               CudaVariant variant) {
    int device = 0;
    check_variant(cudaGetDevice(&device), "get CUDA device");

    struct Cache {
        bool valid = false;
        int device = -1, H = 0, W = 0, K = 0, bx = 0, by = 0;
        CudaVariant variant = CudaVariant::Constant;
        LaunchConfig config{};
    };
    static thread_local Cache cache;
    if (cache.valid && cache.device == device && cache.H == H && cache.W == W &&
        cache.K == K && cache.bx == bx && cache.by == by &&
        cache.variant == variant) return cache.config;

    cudaDeviceProp prop{};
    cudaFuncAttributes attr{};
    check_variant(cudaGetDeviceProperties(&prop, device), "get CUDA properties");
    switch (variant) {
    case CudaVariant::Constant:
        check_variant(cudaFuncGetAttributes(&attr, conv2d_variant_constant),
                      "get constant kernel properties");
        break;
    case CudaVariant::Tiled:
        check_variant(cudaFuncGetAttributes(&attr, conv2d_variant_tiled<false>),
                      "get tiled kernel properties");
        break;
    case CudaVariant::TiledConstant:
        check_variant(cudaFuncGetAttributes(&attr, conv2d_variant_tiled<true>),
                      "get tiled constant kernel properties");
        break;
    case CudaVariant::MultiOutput:
        check_variant(cudaFuncGetAttributes(&attr, conv2d_variant_multiout),
                      "get multiple-output kernel properties");
        break;
    case CudaVariant::MultiOutputConstant:
        check_variant(cudaFuncGetAttributes(&attr, conv2d_variant_multiout_combined<false, true>),
                      "get multiple-output constant kernel properties");
        break;
    case CudaVariant::MultiOutputTiled:
        check_variant(cudaFuncGetAttributes(&attr, conv2d_variant_multiout_combined<true, false>),
                      "get multiple-output tiled kernel properties");
        break;
    case CudaVariant::MultiOutputTiledConstant:
        check_variant(cudaFuncGetAttributes(&attr, conv2d_variant_multiout_combined<true, true>),
                      "get multiple-output tiled constant kernel properties");
        break;
    default:
        throw std::invalid_argument("unknown CUDA variant");
    }

    const auto threads = static_cast<std::uint64_t>(bx) * by;
    if (bx > prop.maxThreadsDim[0] || by > prop.maxThreadsDim[1] ||
        threads > static_cast<std::uint64_t>(prop.maxThreadsPerBlock) ||
        threads > static_cast<std::uint64_t>(attr.maxThreadsPerBlock)) {
        throw std::invalid_argument("block exceeds this GPU or kernel's limits");
    }

    const auto columns_per_block = static_cast<std::uint64_t>(bx) *
        (is_multiout(variant) ? 2 : 1);
    const auto grid_x = (static_cast<std::uint64_t>(W) + columns_per_block - 1)
                        / columns_per_block;
    const auto grid_y = (static_cast<std::uint64_t>(H) + by - 1) / by;
    if (grid_x > static_cast<std::uint64_t>(prop.maxGridSize[0]) ||
        grid_y > static_cast<std::uint64_t>(prop.maxGridSize[1])) {
        throw std::invalid_argument("grid exceeds the GPU's limits");
    }

    // Match the 32-bit indexing used by the project's existing direct and
    // shared kernels. Reject inputs whose products or padded coordinates
    // would overflow instead of silently mis-indexing.
    const auto int_max = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    if (static_cast<std::uint64_t>(H) * W > int_max ||
        static_cast<std::uint64_t>(K) * K > int_max ||
        grid_x * columns_per_block + K > int_max ||
        grid_y * static_cast<std::uint64_t>(by) + K > int_max) {
        throw std::invalid_argument("image/filter or padded grid exceeds 32-bit indexing limits");
    }

    LaunchConfig result;
    result.block = dim3(static_cast<unsigned>(bx), static_cast<unsigned>(by));
    result.grid = dim3(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));

    if (uses_shared_tile(variant)) {
        const auto tile_w = columns_per_block + K - 1;
        const auto tile_h = static_cast<std::uint64_t>(by) + K - 1;
        const std::size_t available = prop.sharedMemPerBlock > attr.sharedSizeBytes
            ? prop.sharedMemPerBlock - attr.sharedSizeBytes : 0;
        const std::size_t kernel_limit = attr.maxDynamicSharedSizeBytes > 0
            ? static_cast<std::size_t>(attr.maxDynamicSharedSizeBytes) : 0;
        const std::size_t limit = available < kernel_limit ? available : kernel_limit;
        const auto capacity = static_cast<std::uint64_t>(limit / sizeof(float));
        if (tile_w > capacity || tile_h > capacity / tile_w) {
            throw std::invalid_argument(
                "shared input tile exceeds this kernel's per-block limit");
        }
        result.shared_bytes = static_cast<std::size_t>(tile_w * tile_h) * sizeof(float);
    }

    cache = {true, device, H, W, K, bx, by, variant, result};
    return result;
}

} // namespace

CudaTimes cuda_conv_run_variant(const CudaConv& c, const float* in, float* out,
                                const float* k, int bx, int by,
                                CudaVariant variant) {
    if (!in || !out || !k || !c.d_in || !c.d_out || !c.d_k ||
        c.C <= 0 || c.H <= 0 || c.W <= 0 || c.K <= 0 || c.K % 2 == 0 ||
        bx <= 0 || by <= 0) {
        throw std::invalid_argument("invalid CUDA variant arguments");
    }
    for (void* event : c.events) {
        if (!event) throw std::invalid_argument("CUDA events are not initialized");
    }
    const bool uses_constant = uses_constant_filter(variant);
    // One output plane of H*W; C input planes; C filter slices of K*K.
    const auto image_elements =
        static_cast<std::uint64_t>(c.H) * static_cast<std::uint64_t>(c.W);
    const auto input_elements = static_cast<std::uint64_t>(c.C) * image_elements;
    const auto filter_elements = static_cast<std::uint64_t>(c.C) *
        static_cast<std::uint64_t>(c.K) * static_cast<std::uint64_t>(c.K);
    const auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (input_elements / c.C != image_elements || input_elements > max_elements ||
        filter_elements > max_elements) {
        throw std::overflow_error("image or filter byte count is too large");
    }
    // Kernels index the filter with 32-bit ints (ch*K*K + i*K + j).
    if (filter_elements > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("C*K*K exceeds 32-bit filter indexing");
    }
    if (uses_constant && filter_elements > kConstantFloats) {
        throw std::invalid_argument(
            "constant-memory variant requires C*K*K <= 4096 (odd K <= 63 for C = 1)");
    }
    const std::size_t image_bytes = static_cast<std::size_t>(image_elements) * sizeof(float);
    const std::size_t input_bytes = static_cast<std::size_t>(input_elements) * sizeof(float);
    const std::size_t filter_bytes = static_cast<std::size_t>(filter_elements) * sizeof(float);
    const LaunchConfig launch = get_launch_config(c.H, c.W, c.K, bx, by, variant);

    cudaEvent_t e[4];
    for (int i = 0; i < 4; ++i) e[i] = static_cast<cudaEvent_t>(c.events[i]);

    // Do not change the constant symbol while an earlier call in this process
    // may still be reading it. The lock covers upload through completion.
    std::unique_lock<std::mutex> guard(g_constant_filter_mutex, std::defer_lock);
    if (uses_constant) guard.lock();

    try {
        check_variant(cudaEventRecord(e[0]), "record H2D start");
        check_variant(cudaMemcpy(c.d_in, in, input_bytes, cudaMemcpyHostToDevice),
                      "copy image to GPU");
        if (uses_constant) {
            check_variant(cudaMemcpyToSymbol(g_variant_filter, k, filter_bytes, 0,
                                           cudaMemcpyHostToDevice),
                          "copy filter to constant memory");
        } else {
            check_variant(cudaMemcpy(c.d_k, k, filter_bytes, cudaMemcpyHostToDevice),
                          "copy filter to GPU");
        }
        check_variant(cudaEventRecord(e[1]), "record H2D end");

        switch (variant) {
        case CudaVariant::Constant:
            conv2d_variant_constant<<<launch.grid, launch.block>>>(
                c.d_in, c.d_out, c.C, c.H, c.W, c.K);
            break;
        case CudaVariant::Tiled:
            conv2d_variant_tiled<false><<<launch.grid, launch.block, launch.shared_bytes>>>(
                c.d_in, c.d_out, c.d_k, c.C, c.H, c.W, c.K);
            break;
        case CudaVariant::TiledConstant:
            conv2d_variant_tiled<true><<<launch.grid, launch.block, launch.shared_bytes>>>(
                c.d_in, c.d_out, c.d_k, c.C, c.H, c.W, c.K);
            break;
        case CudaVariant::MultiOutput:
            conv2d_variant_multiout<<<launch.grid, launch.block>>>(
                c.d_in, c.d_out, c.d_k, c.C, c.H, c.W, c.K);
            break;
        case CudaVariant::MultiOutputConstant:
            conv2d_variant_multiout_combined<false, true><<<launch.grid, launch.block>>>(
                c.d_in, c.d_out, c.d_k, c.C, c.H, c.W, c.K);
            break;
        case CudaVariant::MultiOutputTiled:
            conv2d_variant_multiout_combined<true, false><<<launch.grid, launch.block, launch.shared_bytes>>>(
                c.d_in, c.d_out, c.d_k, c.C, c.H, c.W, c.K);
            break;
        case CudaVariant::MultiOutputTiledConstant:
            conv2d_variant_multiout_combined<true, true><<<launch.grid, launch.block, launch.shared_bytes>>>(
                c.d_in, c.d_out, c.d_k, c.C, c.H, c.W, c.K);
            break;
        default:
            throw std::invalid_argument("unknown CUDA variant");
        }
        check_variant(cudaGetLastError(), "launch CUDA variant kernel");
        check_variant(cudaEventRecord(e[2]), "record kernel end");
        check_variant(cudaMemcpy(out, c.d_out, image_bytes, cudaMemcpyDeviceToHost),
                      "copy output to CPU");
        check_variant(cudaEventRecord(e[3]), "record D2H end");
        check_variant(cudaEventSynchronize(e[3]), "synchronize CUDA variant");

        CudaTimes times;
        check_variant(cudaEventElapsedTime(&times.h2d_ms, e[0], e[1]), "measure H2D");
        check_variant(cudaEventElapsedTime(&times.kernel_ms, e[1], e[2]), "measure kernel");
        check_variant(cudaEventElapsedTime(&times.d2h_ms, e[2], e[3]), "measure D2H");
        return times;
    } catch (...) {
        // Keep the symbol protected until outstanding work has stopped, even
        // if a later API call reports an error before the event synchronizes.
        if (uses_constant) (void)cudaDeviceSynchronize();
        throw;
    }
}
