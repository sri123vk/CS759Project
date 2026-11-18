# Performance Comparison of Sequential, OpenMP, and CUDA-Based 2D Convolution

ECE 759 · Fall 2026 · Final Project

**Team:** Srimathi Ravisankar, Kshitija Sunil Rangrej, Vanessa Bennett, and Frey Wysong

## Overview

This project implements and compares a single-channel 2D convolution on the CPU and GPU. It includes a sequential baseline, three OpenMP implementations, and five CUDA implementations. The experiments examine image and filter size, CPU thread count and scheduling, GPU block shape, memory optimizations, and the cost of transfers between host and device. An optional PyTorch benchmark provides a library comparison.

The benchmark driver calls the operation convolution, but the mathematical operation is **cross-correlation**, as in `torch.nn.functional.conv2d`: the filter is not flipped.

## Specification

For an input image of height `H` and width `W`, a square filter of odd size `K`, and radius `R = K / 2`, each output pixel is

```text
out[r, c] = sum over i, j of k[i, j] * in[r + i - R, c + j - R]
```

Input positions outside the image are treated as zero. The output has the same `H × W` dimensions as the input. Arrays are row-major and use FP32 values. The driver accepts a positive odd `K`; the usual benchmark sizes are `K = 3, 5, 7`. Individual GPU variants can impose additional limits.

## Implementations

Each name below is accepted by `bin/conv --impl` when the corresponding source is included in the build.

| Implementation | Description |
| --- | --- |
| `seq` | Sequential CPU reference. |
| `omp` | Parallelizes output rows with OpenMP and runtime scheduling. |
| `omp_simd` | Combines OpenMP row parallelism with SIMD across output columns. |
| `omp_blocked` | Computes blocks of output columns using local accumulators and SIMD, with separate edge handling. |
| `cuda` | Direct CUDA baseline; one thread computes one output pixel. |
| `cuda_opt` | Threads cooperate to load a shared-memory input tile, including its halo. It falls back to the original optimized kernel if the tile cannot be used with the GPU's shared-memory limits. |
| `cuda_const` | One output per thread, with filter coefficients in constant memory. |
| `cuda_tiled_const` | Shared-memory input tile and constant-memory filter. |
| `cuda_multiout` | Computes two output pixels per thread using a global-memory filter. |

The constant-memory variants require odd `K <= 63`. The tiled variants also depend on the GPU's block and shared-memory limits. A tiled kernel is an optimization to evaluate, not a guarantee of better performance for every input or block size.

The driver defaults to a `32 × 8` CUDA block. Use `--block BX BY` to select another shape, or `--block auto` to try a small set of supported shapes and select the one with the lowest trial median kernel time. Auto-tuning is performed for the current workload; its cost is reported separately and excluded from the benchmark times.

## Repository layout

```text
include/       Shared interfaces and utilities
src/           CPU/CUDA implementations and benchmark driver
tests/         C++ correctness tests
validation/    Independent SciPy/PyTorch validation
scripts/       Slurm jobs, benchmark sweeps, and plotting scripts
results/       Benchmark results and logs, when generated
plots/         Generated plots
report/        Report materials
```

The additional sequential experiment in `src/conv2d_seq_opt_for_largesize.cpp` is present in the repository but is not registered in the current Makefile or benchmark driver.

## Requirements

- A C++17 compiler, `make`, and OpenMP support for the CPU implementations.
- NVIDIA CUDA Toolkit (`nvcc`) and an NVIDIA GPU for the CUDA implementations.
- Python 3 with NumPy and SciPy **or** NumPy and PyTorch for independent validation.
- Python 3 with pandas and Matplotlib for `scripts/plot.py`; Matplotlib for `scripts/plot_cuda_blocks.py`.
- A CUDA-enabled PyTorch installation for the optional PyTorch GPU benchmark.

On Euler, load an appropriate CUDA module before building or submitting a job and make sure it is available in the job environment. Run GPU tests and benchmarks on a GPU compute node rather than a login node.

## Build

From the repository root:

```bash
# CPU-only build
make -B CUDA=0 bin/conv bin/test_conv

# CUDA + OpenMP build, after nvcc is available
make -B CUDA=1 bin/conv bin/test_conv
```

`make` automatically enables CUDA when `nvcc` is on `PATH`, unless `CUDA=0` is specified. The Makefile includes each OpenMP variant whose source file is present. `make -B` forces a rebuild, which is useful after changing CUDA build settings or adding source files: `make` does not necessarily notice changes in build flags.

If a CUDA library cannot be found, check that the CUDA module and `CUDA_LIBDIR` point to the toolkit installed on the system. Run `./bin/test_conv --list` to see the test names compiled into the executable. On a GPU system, check the list from the allocated node.

## Run the driver

The driver generates reproducible input and filter values, performs warm-up iterations, and prints a CSV row. Examples from the project root:

```bash
# CPU
./bin/conv --impl seq --size 1024 --K 3 --iters 10 --header
OMP_NUM_THREADS=8 OMP_SCHEDULE=static \
  ./bin/conv --impl omp_simd --size 1024 --K 7 --check --header

# GPU: run these on an allocated GPU compute node
./bin/conv --impl cuda --size 1024 --K 7 \
  --block 32 8 --warmup 3 --iters 10 --check --header
./bin/conv --impl cuda_opt --H 1024 --W 1024 --K 7 \
  --block auto --warmup 3 --iters 10 --check --header
./bin/conv --impl cuda_tiled_const --size 1024 --K 7 \
  --block 32 8 --check --header
```

`--size N` selects an `N × N` image; use `--H` and `--W` for a rectangular image. `--check` compares the output against `seq`. Without `--check`, a printed `pass=1` does **not** mean that a correctness comparison was performed.

For a diagnostic of which `cuda_opt` path is selected, set `CONV_CUDA_TRACE=1` for a test run. Leave it unset during timed experiments.

## Correctness tests

The C++ test program checks identity, shifted single-tap, impulse, all-ones, random, and repeatability cases. It covers small and non-square images and partial CUDA blocks. Random results are compared with an independently written double-precision reference using an FP32 error bound; the suite also checks for nonfinite and unwritten outputs.

```bash
# List available test configurations.
./bin/test_conv --list

# Examples for a CUDA-enabled build, on a GPU compute node.
./bin/test_conv --impl cuda/7x3
./bin/test_conv --impl cuda_opt/7x3
./bin/test_conv --impl cuda_const/7x3
./bin/test_conv --impl cuda_tiled_const/7x3
./bin/test_conv --impl cuda_multiout/7x3

# All registered C++ tests; external validation is skipped in this command.
make CUDA=1 test VALIDATE=0
```

`--impl` filters by substring: `--impl cuda` can select several CUDA implementations. Use `--list --impl cuda` to see which configurations match. For a CPU-only build, use `make CUDA=0 test VALIDATE=0`.

### Independent library validation

`validation/validate.py` compares dumped output against a float64 SciPy `correlate2d` reference, or PyTorch `conv2d` if SciPy is unavailable. The reference is itself checked against known answers. A result passes when it has no NaN/Inf and its maximum error is within the implemented FP32 error bound.

By default, `make test` runs the external check when its selected Python can import the required packages; otherwise it prints `SKIPPED`. Set `VALIDATE=1` to require that check. The default `make validate` implementations are `seq` and, for a CUDA build, `cuda`; explicitly list other variants to validate them too:

```bash
# On a GPU compute node with the validation Python environment active:
make CUDA=1 validate PYTHON=python3 \
  VALIDATE_DIR=data/validate_variants \
  VALIDATE_SHAPES="33x65x3 33x65x7" \
  VALIDATE_IMPLS="seq cuda cuda_opt cuda_const cuda_tiled_const cuda_multiout"
```

The validation target deletes existing `*.bin` files in `VALIDATE_DIR` before writing new dumps, so use a dedicated directory. To validate existing dumps separately, run `python3 validation/validate.py DUMP_DIR -v`. Dump filenames identify the image size, filter size, and implementation, but not the block shape.

## Euler: test and benchmark the variants

Submit the existing job from the repository root:

```bash
sbatch scripts/cuda_variants.slurm
squeue -u "$USER"
```

Before submission, check that the script's partition, GPU request, CPU count, and time limit are valid for your Euler allocation. The script builds both executables, checks the expected test registrations, tests the OpenMP variants and five CUDA variants, and runs the `baseline`, `omp`, and `cuda_blocks` sweeps. It can also locate the repository when submitted from a subdirectory; set `PROJECT_DIR` if the intended checkout is elsewhere.

For job ID `JOBID`, the Slurm output is `conv_variants_JOBID.out` in the submission directory. Benchmark results are written under `results/cuda_variants_JOBID/`, including:

| File | Contents |
| --- | --- |
| `implementations.txt` | Implementations accepted by this build and selected for the sweep. |
| `baseline.csv` | Selected implementations across image and filter sizes. |
| `omp.csv` | OpenMP thread-count and scheduling sweep. |
| `cuda_blocks.csv` | CUDA block-shape sweep. |
| `errors.log` | Errors and failed checks, when produced. |
| `sysinfo.txt` | Captured system information, if the capture script ran successfully. |

The script currently selects image sizes `1024` and `2048`, filter sizes `3` and `7`, five block shapes, 10 timed iterations, and 3 warm-up iterations. These are settings in the Slurm script; edit them there to change this job's sweep.

## Custom benchmark sweeps and library comparison

`scripts/bench.sh` accepts `baseline`, `omp`, `cuda_blocks`, `torch`, or `all`. It probes the requested implementations and skips names that the driver cannot run. `baseline` compares implementations over the selected sizes; it is an experiment name, not a separate convolution implementation. See [BENCHMARK.md](BENCHMARK.md) and `bash scripts/bench.sh --help` for the complete settings.

For example, inside an allocated GPU job:

```bash
export IMPLS="seq cuda cuda_opt cuda_const cuda_tiled_const cuda_multiout"
export SIZES="256 512"
export KS="3 5"
export BLOCK_SIZES="256 512"
export BLOCK_KS="3 5"
export BLOCKS="32x8 16x16"
export WARMUP=3 ITERS=10 CHECK=1
export PYTHON="$(command -v python3)"

srun bash scripts/bench.sh -o "results/compare_${SLURM_JOB_ID}" \
  baseline cuda_blocks torch
```

The `torch` experiment uses the Python interpreter specified by `PYTHON` and writes `torch.csv`. If that Python cannot import PyTorch, the experiment is skipped; if CUDA is unavailable to PyTorch, it runs the CPU comparison only. Check the job output and the CSV to confirm which comparisons ran. The existing `cuda_variants.slurm` does not request the `torch` experiment.

## Timing and interpreting results

The driver reports `median_ms` and `min_ms` over timed iterations after warm-up. For CUDA, device buffers are allocated before those iterations and `alloc_ms` is reported separately. `median_ms` is host wall-clock time per call, including host-to-device copies, kernel execution, and the device-to-host copy. `h2d_ms`, `kernel_ms`, and `d2h_ms` are CUDA-event stage medians. Independently calculated stage medians need not add exactly to the overall median.

Compare implementations using the same image size, filter size, block shape, warm-up, and iteration count. Inspect `kernel_ms` to study the kernel and `median_ms` for the measured copy-and-compute call. GPU transfer and launch overhead can hide a kernel improvement. Repeat runs on the same GPU and record the software and hardware configuration before drawing conclusions; a variant's name does not imply it will always be faster.

The repository contains previous benchmark data. In the recorded `cuda_variants_122741` run, the log reports 284 successful benchmark configurations, zero check failures, and zero errors. Its `sysinfo.txt` contains a permission error instead of system information, so that run alone does not establish the hardware used. Verify `sysinfo.txt` for new runs; if the capture script is not executable in a checkout, fix its permissions before rerunning.

## Plot results

The general plotter reads the benchmark **CSV directory** and creates PNG figures and a Markdown summary:

```bash
python3 scripts/plot.py results/cuda_variants_JOBID
# Output: plots/cuda_variants_JOBID/
```

For a focused `cuda` versus `cuda_opt` comparison, the comparison script reads the **Slurm output log**, matches configurations in the `baseline` and `cuda_blocks` sections, and plots their printed median times:

```bash
python3 scripts/plot_cuda_blocks.py conv_variants_JOBID.out \
  -o plots/cuda_comparison_JOBID
```

This produces `cuda_median_comparison.pdf` and `matched_cuda_medians.csv`. The log comparison uses the printed end-to-end median, not `kernel_ms`; consult `cuda_blocks.csv` and the general plots for kernel-stage comparisons. Run commands from the repository root or adjust the paths to the files' actual locations.

## Scope and limitations

- The implemented operation is single-channel, same-size, zero-padded FP32 cross-correlation, not general multi-channel convolution.
- `--check` uses the project's sequential implementation as its reference; use the external validator for an independent library comparison.
- Tests exercise a finite set of inputs and block shapes. The external validation target checks only the implementations and shapes requested.
- GPU performance and supported block/shared-memory configurations depend on the device, toolkit, and workload. PyTorch/cuDNN algorithm selection and overhead can also affect comparisons.
