// Small helpers shared by the drivers: input generation, binary I/O,
// result comparison, and timing.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

// Fill with uniform values in [-1, 1] from a fixed seed.
inline void fill_random(std::vector<float>& v, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(gen);
}

// Raw float32 dump, no header. Dimensions are passed separately
// (the Python side uses np.fromfile(...).reshape(H, W)).
inline bool write_bin(const std::string& path, const std::vector<float>& v) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t n = std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    return n == v.size();
}

inline bool read_bin(const std::string& path, std::vector<float>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    size_t n = std::fread(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    return n == v.size();
}

// Per-pixel sum of |k[i][j] * in[y][x]| over the filter window (zero padding),
// computed in double. This is the scale of the rounding error any FP32
// implementation can make at that pixel, whatever its summation order.
inline std::vector<double> sum_abs_terms(const std::vector<float>& in,
                                         const std::vector<float>& k,
                                         int H, int W, int K) {
    const int R = K / 2;
    std::vector<double> s((size_t)H * W, 0.0);
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            double acc = 0.0;
            for (int i = 0; i < K; ++i) {
                const int y = r + i - R;
                if (y < 0 || y >= H) continue;
                for (int j = 0; j < K; ++j) {
                    const int x = c + j - R;
                    if (x < 0 || x >= W) continue;
                    acc += std::fabs((double)k[(size_t)i * K + j] *
                                     (double)in[(size_t)y * W + x]);
                }
            }
            s[(size_t)r * W + c] = acc;
        }
    }
    return s;
}

struct ErrStats {
    double max_abs = 0.0;    // max |got - ref|
    double max_rel = 0.0;    // max |got - ref| / |ref| where |ref| > 1e-3 (informational)
    double max_ratio = 0.0;  // max |got - ref| / bound; must be <= 1 to pass
    size_t nonfinite = 0;    // entries where got or ref is NaN or Inf
};

// Compares got against ref pixel by pixel.
//
// Tolerance: a K*K-term FP32 dot product has error at most
// gamma_n * sum|terms| with n = K*K and gamma_n = n*u / (1 - n*u),
// u = 2^-24 (Higham, Accuracy and Stability of Numerical Algorithms, sec. 3.1).
// This holds for any summation order and with or without FMA. Both ref and
// got carry such an error, so we allow twice that. A fixed relative tolerance
// does not work here: pixels whose terms cancel to a small result can
// legitimately differ by much more than 1e-5 relative once FMA (the CUDA
// default) changes the rounding. Real indexing bugs produce errors on the
// order of the terms themselves, far above this bound.
//
// NaN and Inf are counted explicitly. A plain std::max over the differences
// would skip them, because every comparison with NaN is false.
inline ErrStats compare(const std::vector<float>& ref,
                        const std::vector<float>& got,
                        const std::vector<double>& scale, int K) {
    ErrStats e;
    if (ref.size() != got.size() || ref.size() != scale.size()) {
        e.max_ratio = INFINITY;
        return e;
    }
    const double u = std::ldexp(1.0, -24);
    const double nu = (double)K * K * u;
    const double gamma = nu / (1.0 - nu);

    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) {
            ++e.nonfinite;
            continue;
        }
        const double d = std::fabs((double)ref[i] - (double)got[i]);
        e.max_abs = std::max(e.max_abs, d);

        const double mag = std::fabs((double)ref[i]);
        if (mag > 1e-3) e.max_rel = std::max(e.max_rel, d / mag);

        const double bound = 2.0 * gamma * scale[i];
        const double ratio = bound > 0.0 ? d / bound : (d > 0.0 ? INFINITY : 0.0);
        e.max_ratio = std::max(e.max_ratio, ratio);
    }
    return e;
}

inline bool within_tol(const ErrStats& e) {
    return e.nonfinite == 0 && e.max_ratio <= 1.0;
}

// Milliseconds elapsed while running f().
template <typename F>
double time_ms(F&& f) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
