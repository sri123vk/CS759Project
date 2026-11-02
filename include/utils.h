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

struct ErrStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
};

// Relative error is only measured where |ref| > 1e-3, otherwise values
// near zero blow it up for no real reason.
inline ErrStats compare(const std::vector<float>& ref,
                        const std::vector<float>& got) {
    ErrStats e;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = std::fabs((double)ref[i] - (double)got[i]);
        e.max_abs = std::max(e.max_abs, d);
        double mag = std::fabs((double)ref[i]);
        if (mag > 1e-3) e.max_rel = std::max(e.max_rel, d / mag);
    }
    return e;
}

inline bool within_tol(const ErrStats& e) {
    return e.max_abs <= 1e-4 && e.max_rel <= 1e-5;
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
