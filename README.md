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
Pass: no NaN/Inf in the output, and at every pixel
`|got − ref| ≤ 2·γ(K²)·Σ|k·in|` where `γ(n) = n·u / (1 − n·u)`, `u = 2⁻²⁴`.
This is the standard FP32 dot-product error bound (Higham §3.1). It holds
for any summation order, with or without FMA, so GPU results need not match
bit-for-bit. The CSV reports `err_ratio` (worst error ÷ bound, pass ≤ 1);
max absolute and relative error are reported for information only.

**Timing.** Warm-up runs, then the median of N timed runs.

- CPU: `std::chrono::steady_clock`
- CUDA: the context is created and device memory allocated before timing
  (allocation reported separately as `alloc_ms`). End-to-end time
  (`median_ms`) is host wall-clock for host→device copy + kernel +
  device→host copy. `h2d_ms`, `kernel_ms`, `d2h_ms` are per-stage
  `cudaEvent` times (medians).

All reported numbers come from one cluster node. Laptop timings are for
development only.

## Layout

```
include/      interface, timing, and I/O helpers
src/          implementations and benchmark driver
tests/        C++ correctness tests (make test)
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
./bin/conv --impl cuda --size 2048 --K 7 --block 16 16 --check
```

CUDA is built in automatically when `nvcc` is on the PATH (`make CUDA=0`
to skip it). To target a specific GPU, e.g. `make NVCCFLAGS="-O3 -std=c++17 -arch=sm_75"`.

Options: `--size N` (or `--H`, `--W`), `--K`, `--iters`, `--warmup`,
`--block BX BY` (CUDA thread block, default 32 8), `--check`, `--dump DIR`,
`--header`. Output is one CSV row per run with columns
`impl,H,W,K,bx,by,iters,median_ms,min_ms,h2d_ms,kernel_ms,d2h_ms,alloc_ms,max_abs_err,max_rel_err,err_ratio,nonfinite,pass`;
CUDA-only columns are empty for CPU implementations.

## Tests

```
make test                          # build bin/test_conv and run every test
./bin/test_conv --list             # implementations being tested
./bin/test_conv --impl omp -v      # only names containing "omp", show passes
```

Every test runs on every implementation that is built in: `seq` always,
`omp/<static|dynamic>/t<threads>` once `src/conv2d_omp.cpp` exists, and
`cuda/<bx>x<by>` for several block shapes when CUDA is built in. Tests cover exact-answer cases
(identity, shifted single tap, impulse, all-ones), awkward shapes, random
data against a double-precision reference using the same error bound as
`--check`, out-of-bounds writes, unwritten pixels, and run-to-run
repeatability. OpenMP results must be bit-identical to `seq`.
Exit status is 0 only if everything passes. Run it before every benchmark sweep.

## Benchmarks

The scripts in `scripts/` run the experiments for the report by calling
`bin/conv` across a set of configurations. They don't change any
implementation code. For more detail, see [BENCHMARK.md](BENCHMARK.md).

```
make && make test                        # benchmark only code that passes its tests
scripts/bench.sh --quick                 # ~10 s smoke run of every experiment
scripts/bench.sh -o results/final all    # full sweep -> results/final/*.csv
python3 scripts/plot.py results/final    # -> plots/final/*.png + summary.md
sbatch scripts/euler.slurm               # build, test, sweep and plot on one cluster node
```

What is benchmarked:

| Experiment | Configurations | Measures |
|---|---|---|
| `baseline` | every implementation × 256², 512², 1024², 2048² × K = 3, 5, 7 | time per call, speedup over `seq`, CUDA kernel-only vs end-to-end and the time spent in each stage |
| `omp` | OpenMP × 1, 2, 4, … cores threads × `static` / `dynamic` / `guided` | thread scaling and scheduling policy |
| `cuda_blocks` | CUDA × 13 thread-block shapes (e.g. 32×8, 16×16, 1×128) | effect of block shape and memory coalescing |
| `torch` | PyTorch `conv2d` (cuDNN on the GPU, oneDNN on the CPU), same problems as `baseline` | an external reference point |

Every run except `seq` is checked against `seq` with `--check`, and failures
are logged to `errors.log`. Each results directory gets a `sysinfo.txt` with
the hardware, compiler flags and git commit. Sweep parameters
(implementations, sizes, filter sizes, threads, block shapes, iteration
counts) can be overridden with environment variables. See
`scripts/bench.sh --help`.

The plots need Python 3 with pandas and matplotlib. The `torch` experiment
needs PyTorch and is skipped if it isn't installed.

## Status

- [x] Sequential baseline
- [ ] OpenMP
- [ ] CUDA 
- [ ] Benchmarks and plots
- [ ] Report
