#!/bin/bash
#SBATCH --job-name=conv_variants
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=conv_variants_%j.out

set -euo pipefail

# An explicit PROJECT_DIR selects one checkout when several copies exist.
# Otherwise, find the project above the sbatch submission directory.
start_dir=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z $start_dir || ! -d $start_dir ]]; then
    echo "ERROR: Set PROJECT_DIR to your project root or submit from inside it." >&2
    exit 1
fi
project_dir=$(cd "$start_dir" && pwd -P)
while [[ ! -f "$project_dir/Makefile" || ! -f "$project_dir/scripts/bench.sh" || ! -d "$project_dir/src" ]]; do
    if [[ -n ${PROJECT_DIR:-} ]]; then
        echo "ERROR: PROJECT_DIR is not a project root: $project_dir" >&2
        exit 1
    fi
    parent=$(dirname "$project_dir")
    if [[ $parent == "$project_dir" ]]; then
        echo "ERROR: Could not find the project above $start_dir. Set PROJECT_DIR to its root." >&2
        exit 1
    fi
    project_dir=$parent
done
cd "$project_dir"
echo "Project directory: $PWD"

# Build missing or out-of-date programs in this checkout.
echo "Building bin/conv and bin/test_conv"
if ! make CUDA=1 bin/conv bin/test_conv; then
    echo "ERROR: Build failed. Check the compiler output above and load the CUDA module if needed." >&2
    exit 1
fi

# Make does not automatically detect every change in build flags. If a test
# registration is absent, force one rebuild and check again.
expected_tests=(
    omp/static/t1 omp_simd/static/t1 omp_blocked/static/t1
    cuda/7x3 cuda_opt/7x3 cuda_const/7x3 cuda_tiled_const/7x3 cuda_multiout/7x3
    cuda_multiout_const/7x3 cuda_multiout_tiled/7x3 cuda_multiout_tiled_const/7x3
)
check_test_names() {
    local test_list name missing=0
    if ! test_list=$(OMP_NUM_THREADS=1 ./bin/test_conv --list); then
        echo "Could not list tests in $PWD/bin/test_conv" >&2
        return 1
    fi
    for name in "${expected_tests[@]}"; do
        if ! grep -Fqx -- "$name" <<< "$test_list"; then
            echo "Missing test registration: $name" >&2
            missing=1
        fi
    done
    return "$missing"
}

if ! check_test_names; then
    echo "Rebuilding all targets once to refresh the compiled variants"
    if ! make -B CUDA=1 bin/conv bin/test_conv; then
        echo "ERROR: Forced rebuild failed. See the compiler output above." >&2
        exit 1
    fi
    if ! check_test_names; then
        echo "ERROR: Tests are still missing. Check tests/test_conv.cpp, the OpenMP sources, and Makefile CUDA/OpenMP flags." >&2
        exit 1
    fi
fi
echo "Test executable: $(readlink -f ./bin/test_conv)"

for impl in omp omp_simd omp_blocked; do
    echo "Testing $impl"
    OMP_NUM_THREADS=1 srun ./bin/test_conv --impl "$impl/static/t1"
done

echo "Checking cuda_opt on a small image with partial blocks"
srun ./bin/conv --impl cuda_opt \
    --H 33 --W 65 --K 7 --block 32 8 \
    --warmup 0 --iters 1 --check --header

for impl in cuda cuda_opt cuda_const cuda_tiled_const cuda_multiout \
            cuda_multiout_const cuda_multiout_tiled cuda_multiout_tiled_const; do
    echo "Testing $impl with partial blocks"
    srun ./bin/test_conv --impl "$impl/7x3"
done

# Test automatic block selection for one workload.
echo "Testing automatic block selection for cuda_multiout_const"

srun ./bin/conv \
    --impl cuda_multiout_const \
    --size 2048 \
    --K 3 \
    --block auto \
    --warmup 3 \
    --iters 10 \
    --check \
    --header

export IMPLS="seq omp omp_simd omp_blocked cuda cuda_opt cuda_const cuda_tiled_const cuda_multiout cuda_multiout_const cuda_multiout_tiled cuda_multiout_tiled_const"
export SIZES="256 512 1024 2048"
export KS="3 5 7 11"
# OpenMP scaling: use the same workloads.
export OMP_SIZES="$SIZES"
export OMP_KS="$KS"
# CUDA block sweep: use the same workloads.
export BLOCK_SIZES="$SIZES"
export BLOCK_KS="$KS"
export BLOCKS="32x8 32x16 16x16 8x8 1x128"
export BLOCK="auto"
export ITERS=10
export WARMUP=3
export CHECK=1

out="results/cuda_variants_${SLURM_JOB_ID}"
srun bash scripts/bench.sh -o "$out" baseline omp cuda_blocks
echo "Results: $out"
cat "$out/implementations.txt"

