# Additional CUDA multi-output variants

This package adds three CUDA implementations while retaining `cuda_multiout`:

| Implementation | Image | Filter | Outputs per thread |
| --- | --- | --- | ---: |
| `cuda_multiout` | Global memory | Global memory | 2 |
| `cuda_multiout_const` | Global memory | Constant memory | 2 |
| `cuda_multiout_tiled` | Shared-memory tile | Global memory | 2 |
| `cuda_multiout_tiled_const` | Shared-memory tile | Constant memory | 2 |

The package contains six replacement project files, plus this guide:

```text
include/conv2d.h
src/conv2d_cuda_variants.cu
src/main.cpp
tests/test_conv.cpp
scripts/bench.sh
scripts/cuda_variants.slurm
```

The CUDA source was based on the supplied `conv2d_cuda_variants(4).cu` and
matched the GitHub `main` source with blob SHA
`1246c76dfd382c395c3980a59593767657c6b19b` at the time of this work. The
other files were taken from the same repository's `main` branch. If your local
checkout has additional changes, review and merge the changes into those files
instead of replacing them blindly.

## Important implementation details

- Each thread computes columns `col0` and `col0 + blockDim.x`, as in the
  existing `cuda_multiout` implementation.
- The shared variants load a tile of width `2 * blockDim.x + K - 1` and
  height `blockDim.y + K - 1`, including halo pixels. Every thread participates
  in tile loading and synchronization, including threads on partial blocks.
- The constant-filter variants use the existing `g_variant_filter` symbol,
  mutex, and 4,096-float capacity. The project accepts positive odd `K` up to
  63 for these variants. The global-filter tiled variant has no 63 limit but
  the shared tile must fit on the GPU.
- Image boundary taps are skipped, preserving the existing behavior for
  filters containing NaN or infinity as well as ordinary finite values.
- The existing CUDA, OpenMP, and sequential implementations are retained.

## Build on Euler

From the project root, after loading your usual CUDA environment:

```bash
make CUDA=1 bin/conv bin/test_conv
```

The header and source both change, so install all package files before
building. Replacing only the CUDA source will not register the new names and
can fail to compile against an older `conv2d.h`.

## Correctness tests on a GPU node

In a Slurm GPU allocation or GPU batch script, run:

```bash
./bin/test_conv --list | grep -F cuda_multiout

for impl in cuda_multiout cuda_multiout_const cuda_multiout_tiled cuda_multiout_tiled_const; do
    for block in 7x3 32x8; do
        echo "Testing $impl/$block"
        srun ./bin/test_conv --impl "$impl/$block"
    done
done
```

The project's existing tests include tiny and rectangular images, filter
sizes 1, 3, 5, and 7, partial blocks, an independent reference comparison,
and repeatability checks. The altered `scripts/cuda_variants.slurm` also
builds and tests the new names and includes them in its benchmark sweep.

## Small benchmark

Run the following inside a Slurm GPU allocation, from the project root:

```bash
for impl in cuda_multiout cuda_multiout_const cuda_multiout_tiled cuda_multiout_tiled_const; do
    srun ./bin/conv --impl "$impl" --H 33 --W 65 --K 3 \
        --block 32 8 --iters 10 --warmup 3 --check --header
done
```

Compare `kernel_ms` for device-kernel performance and `median_ms` for the
end-to-end call including transfers. A tiny 33x65 case is useful for testing
but is not representative of performance on larger images.

## Verification performed here

The CUDA toolchain and a GPU were not available in the editing environment, so
no NVCC build or GPU execution is claimed. The CPU-only project builds and
the existing sequential and OpenMP tests passed; host-side CUDA dispatch was
syntax-checked and exercised with CPU stubs. The actual new kernel body was
also executed using a CPU thread/barrier emulation against an independent CPU
reference for partial blocks, rectangular images, filters up to 63x63, and
nonfinite filter values. This is a logic check, not a substitute for the
Euler GPU tests above.
