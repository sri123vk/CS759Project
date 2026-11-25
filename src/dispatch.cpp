// Dispatcher (see include/dispatch.h).

#include "dispatch.h"
#include "dispatch_model.h"
#include "conv2d.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

using ConvFn = void (*)(const float*, float*, const float*, int, int, int, int);

// The CPU implementations compiled into this build.
ConvFn cpu_fn(const std::string& impl) {
    if (impl == "seq") return conv2d_seq;
#ifdef CONV_HAVE_OMP
    if (impl == "omp") return conv2d_omp;
#endif
#ifdef CONV_HAVE_OMP_SIMD
    if (impl == "omp_simd") return conv2d_omp_simd;
#endif
#ifdef CONV_HAVE_OMP_BLOCKED
    if (impl == "omp_blocked") return conv2d_omp_blocked;
#endif
#ifdef CONV_HAVE_OMP_TILED
    if (impl == "omp_tiled") return conv2d_omp_tiled;
#endif
    return nullptr;
}

#if defined(CONV_HAVE_CUDA) && defined(CONV_HAVE_CUDA_VARIANTS)
bool cuda_variant(const std::string& impl, CudaVariant& v) {
    if (impl == "cuda_opt") v = CudaVariant::Tiled;
    else if (impl == "cuda_const") v = CudaVariant::Constant;
    else if (impl == "cuda_tiled_const") v = CudaVariant::TiledConstant;
    else if (impl == "cuda_multiout") v = CudaVariant::MultiOutput;
    else if (impl == "cuda_multiout_const") v = CudaVariant::MultiOutputConstant;
    else if (impl == "cuda_multiout_tiled") v = CudaVariant::MultiOutputTiled;
    else if (impl == "cuda_multiout_tiled_const") v = CudaVariant::MultiOutputTiledConstant;
    else return false;
    return true;
}
#endif

// Whether this build can run impl on the problem.
bool available(const DispatchModel& m, int C, int K) {
    if (!m.gpu) return cpu_fn(m.impl) != nullptr;
#if defined(CONV_HAVE_CUDA) && defined(CONV_HAVE_CUDA_VARIANTS)
    if (std::strstr(m.impl, "const") && (long long)C * K * K > 4096) return false;
    if (std::strcmp(m.impl, "cuda") == 0) return true;
    CudaVariant v;
    return cuda_variant(m.impl, v);
#else
    (void)C;
    (void)K;
    return false;
#endif
}

}  // namespace

DispatchChoice conv2d_choose(int C, int H, int W, int K) {
    const char* dev = std::getenv("CONV_AUTO_DEVICE");
    const bool cpu_only = dev && std::strcmp(dev, "cpu") == 0;
    const bool gpu_only = dev && std::strcmp(dev, "gpu") == 0;

    const double n = (double)H * W;
    DispatchChoice best{"", false, std::numeric_limits<double>::infinity()};
    for (const DispatchModel& m : kDispatchModels) {
        if ((cpu_only && m.gpu) || (gpu_only && !m.gpu) || !available(m, C, K)) continue;
        const double t = m.c0 + m.c1 * n * C + m.c2 * n + m.c3 * n * C * K * K;
        if (t < best.predicted_ms) best = {m.impl, m.gpu, t};
    }
    // Always runnable: seq is in every build (and in the model; this only
    // matters if the model were regenerated without it).
    if (best.impl.empty()) best = {"seq", false, std::numeric_limits<double>::quiet_NaN()};
    return best;
}

void conv2d_auto(const float* in, float* out, const float* k,
                 int C, int H, int W, int K) {
    const DispatchChoice c = conv2d_choose(C, H, W, K);
    if (!c.gpu) {
        cpu_fn(c.impl)(in, out, k, C, H, W, K);
        return;
    }
#if defined(CONV_HAVE_CUDA) && defined(CONV_HAVE_CUDA_VARIANTS)
    if (c.impl == "cuda") {
        conv2d_cuda(in, out, k, C, H, W, K);
        return;
    }
    CudaVariant v;
    if (!cuda_variant(c.impl, v)) throw std::logic_error("dispatch: unknown CUDA implementation " + c.impl);
    CudaConv buf = cuda_conv_alloc(C, H, W, K);
    try {
        cuda_conv_run_variant(buf, in, out, k, 32, 8, v);
    } catch (...) {
        cuda_conv_free(buf);
        throw;
    }
    cuda_conv_free(buf);
#endif
}
