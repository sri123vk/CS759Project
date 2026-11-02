# Performance Comparison of Sequential, OpenMP, and CUDA-Based 2D Convolution

ECE 759 · Fall 2026 · Final Project

**Team:** Srimathi Ravisankar, Kshitija Sunil Rangrej, Vanessa Bennett, Frey Wysong

## Overview

One 2D convolution, implemented three ways (sequential C++, OpenMP, CUDA)
and benchmarked across image sizes 256²–2048² and filter sizes 3×3, 5×5, 7×7.
We also study OpenMP thread count and scheduling, and CUDA thread-block shape.

## Specification

Every implementation computes, with `R = K / 2`:

```
out[r][c] = Σᵢ Σⱼ k[i][j] · in[r+i−R][c+j−R]
```

| | |
|---|---|
| Operation | Cross-correlation (filter not flipped), as in image-processing and CNN libraries |
| Precision | FP32 input, filter, output, and accumulator |
| Layout | Row-major; pixel `(r, c)` at `in[r*W + c]` |
| Filter | Square, odd `K` ∈ {3, 5, 7}, centered |
| Boundary | Zero padding; output is `H × W` |
| Sum order | Filter row `i`, then column `j` |

Shared interface (`include/conv2d.h`):

```cpp
void conv2d_<variant>(const float* in, float* out, const float* k,
                      int H, int W, int K);
```

## Methodology

**Inputs.** Uniform in [−1, 1] from fixed seeds. Exportable as raw float32 for
Python cross-checks.

**Correctness.** Parallel outputs are checked against the sequential version,
which is checked against `scipy.signal.correlate2d(in, k, mode="same", boundary="fill")`.
Pass: max absolute error ≤ 1e-4 and max relative error ≤ 1e-5 (on entries
with |ref| > 1e-3). GPU results need not match bit-for-bit (FMA, summation order).

**Timing.** Warm-up runs, then the median of N timed runs.

- CPU: `std::chrono::steady_clock`
- CUDA: `cudaEvent` for kernel-only time; end-to-end time adds host↔device copies

All reported numbers come from one cluster node. Laptop timings are for
development only.

## Layout

```
include/      interface, timing, and I/O helpers
src/          implementations and benchmark driver
validation/   Python reference checks
scripts/      benchmark sweeps and Slurm jobs
results/      raw CSV output and hardware info
plots/        figures
report/       final report
```

## Build and run

```
make                                         # builds bin/conv
./bin/conv --impl seq --size 1024 --K 5 --header
```

Options: `--size N` (or `--H`, `--W`), `--K`, `--iters`, `--warmup`,
`--check`, `--dump DIR`, `--header`. Output is one CSV row per run.

## Status

- [x] Sequential baseline
- [ ] OpenMP
- [ ] CUDA
- [ ] Benchmarks and plots
- [ ] Report
