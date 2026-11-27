# CUDA convolution experiments

> **Historical note.** This guide describes how the first CUDA variants were
> merged into the project. It is kept for reference; the main
> [README](../README.md#implementations) describes the current code. Since
> this was written:
>
> - There are eight CUDA implementations, not five: `cuda_multiout_const`,
>   `cuda_multiout_tiled` and `cuda_multiout_tiled_const` were added (see
>   [MULTIOUT_README.md](../MULTIOUT_README.md)).
> - Every implementation takes multi-channel input (`--C`). The
>   constant-memory variants need `C·K² ≤ 4096` (odd `K ≤ 63` for one channel).
> - `cuda_opt` no longer falls back to a direct kernel and there is no
>   `CONV_CUDA_TRACE`. If the tile doesn't fit, it throws and `bin/conv`
>   exits with status 3.
> - The CUDA sources live in `src/cuda/` (`src/cuda/conv2d_cuda_variants.cu`).
>   The `files/` directory mentioned below isn't in the repository; only
>   `cuda_variants.patch` is, and it no longer applies to the current code.
> - All variants have since been built and tested on an Euler GPU:
>   `scripts/euler.slurm` runs `make test CUDA=1` and stops on a failure
>   before the benchmark run `results/euler-127842/`, which includes all eight.
>   The "Validation status" section below describes only what had been
>   checked when this was written.

This package adds three named CUDA implementations to the existing project.
Together with the project's existing `cuda` and the supplied `cuda_opt`, they
let you compare five separate variants on the same inputs.

| CLI name | Kernel | What changes |
| --- | --- | --- |
| `cuda` | Existing `conv2d_kernel` | Direct, one output per thread; image and filter in global memory. |
| `cuda_opt` | Existing `conv2d_kernel_opt_tiled` | Image tile and halo in shared memory; filter in global memory. |
| `cuda_const` | `conv2d_variant_constant` | Direct image access; filter in constant memory. |
| `cuda_tiled_const` | `conv2d_variant_tiled_constant` | Shared image tile and halo; filter in constant memory. |
| `cuda_multiout` | `conv2d_variant_multiout` | Two horizontal outputs per thread; filter remains in global memory. |

The two existing `.cu` files are not replaced by this package. These variants
are for one FP32 image channel, an odd square filter, and same-sized output
with zero padding. They perform cross-correlation, matching the current
project reference. Multi-channel convolution is not included.

The supplied `conv2d_cuda_opt(10).cu` was used as the basis for `cuda_opt`.
That version falls back to a direct kernel if the tile is unsupported. Set
`CONV_CUDA_TRACE=1` when running it and look for `shared tile`; a result from
`original fallback` should not be presented as a tiled-kernel result.

## Integrate into your existing checkout

The `files/` directory contains complete versions of each changed project
file, arranged by path. These were prepared against the project's
`Cudaexperiments` branch and the CUDA file you supplied. Your checkout may
have moved on: use the files as a comparison and merge the changes if so;
do not overwrite teammates' work.

1. Add `files/src/conv2d_cuda_variants.cu` as
   `src/conv2d_cuda_variants.cu`. If that name already exists in your
   checkout, inspect it before copying.
2. In `include/conv2d.h`, add the `CudaVariant` enum and
   `cuda_conv_run_variant` declaration shown in the supplied header.
3. In `Makefile`, build and link `bin/conv2d_cuda_variants.o`, define
   `CONV_HAVE_CUDA_VARIANTS`, and link host code with `-pthread` as shown.
4. In `src/main.cpp`, register the three new CLI names and dispatch them to
   `cuda_conv_run_variant`. The supplied version also recognizes them for
   `--block auto`.
5. In `tests/test_conv.cpp`, register all three variants at `32x8` and `7x3`;
   add the `cuda_opt/7x3` case. The existing tests remain in place.
6. In `scripts/bench.sh`, add the three names to `IMPLS` and recognize them
   as CUDA implementations in the baseline sweep.
7. Add `files/scripts/cuda_variants.slurm` as
   `scripts/cuda_variants.slurm`; adapt the `cd` path and Slurm resources if
   your checkout or partition is different.

For a checkout close to the referenced branch, `cuda_variants.patch` contains
the same changes. From the repository root, run `git status --short` and then
`git apply --check /path/to/cuda_variants.patch` before choosing to apply it.
If the check fails, merge the relevant changes from `files/` manually. The
patch does not touch either existing CUDA implementation.

## Build and test on Euler

Load the CUDA module you normally use, and from the repository root run:

```bash
make CUDA=1 bin/conv bin/test_conv
sbatch scripts/cuda_variants.slurm
```

The Slurm script checks the shared-memory path, runs the existing C++ tests
for all five implementations using an awkward `7x3` block, then benchmarks
`seq` and the five CUDA versions across image sizes, filter sizes, and block
shapes. It writes the log in the Slurm submission directory and CSV files to
`results/cuda_variants_JOBID/baseline.csv` and `cuda_blocks.csv`.

You can try one configuration inside a GPU allocation with:

```bash
srun ./bin/conv --impl cuda_tiled_const --H 33 --W 65 --K 7 \
    --block 32 8 --warmup 3 --iters 20 --check --header
```

Replace the implementation name to try another version. `--block auto` is
also available, but for an experiment comparing kernels use the same explicit
block shape first. If your Slurm script exports `IMPLS`, include the new names
there: an exported list overrides the defaults in `bench.sh`.

## Read the results

- Compare `kernel_ms` to study kernel execution. Compare `median_ms` to study
  the complete measured call, which includes the image and filter transfers,
  kernel, output transfer, and host overhead; buffer allocation is measured
  separately. Each column is summarized independently, so their medians need
  not sum exactly.
- The constant filter is uploaded on every call and that copy is counted in
  `h2d_ms`. Calls using the constant symbol are serialized within the process
  so a different filter cannot overwrite it while a kernel is running.
- Constant-memory variants require odd `K <= 63` (the fixed symbol holds
  4096 floats) and reject larger filters. The tiled-constant kernel reports
  unsupported block/shared-memory configurations rather than falling back.
- All new kernels use 32-bit indexing and reject shapes whose indexing would
  overflow. Block and grid dimensions are derived from the runtime arguments.
- Re-run experiments, alternate the execution order when practical, and report
  the GPU and compilation settings. A new variant is not guaranteed to be
  faster; correctness and timing must be checked on the actual device.

## Validation status

The CPU-only build and existing sequential tests passed. The CUDA kernel logic
was also exercised in a CPU simulation against a double-precision reference,
including partial blocks and boundary cases. This environment had no CUDA
toolkit or GPU, so actual `nvcc` compilation, GPU correctness and performance
must still be verified on Euler.
