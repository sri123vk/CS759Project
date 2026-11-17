#!/bin/bash
#SBATCH --job-name=conv_variants
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=conv_variants_%j.out

set -euo pipefail
cd /srv/home/rangrej/CS759Project

if [[ ! -x ./bin/conv || ! -x ./bin/test_conv ]]; then
    echo "Build both programs first: make CUDA=1 bin/conv bin/test_conv" >&2
    exit 1
fi

echo "Confirming the existing shared-memory path"
CONV_CUDA_TRACE=1 srun ./bin/conv --impl cuda_opt \
    --H 33 --W 65 --K 7 --block 32 8 \
    --warmup 0 --iters 1 --check --header

for impl in cuda cuda_opt cuda_const cuda_tiled_const cuda_multiout; do
    echo "Testing $impl with partial blocks"
    srun ./bin/test_conv --impl "$impl/7x3"
done

# These override bench.sh defaults for this job only.
export IMPLS="seq cuda cuda_opt cuda_const cuda_tiled_const cuda_multiout"
export SIZES="1024 2048"
export KS="3 7"
export BLOCK_SIZES="1024 2048"
export BLOCK_KS="3 7"
export BLOCKS="32x8 32x16 16x16 8x8 1x128"
export BLOCK="32 8"
export ITERS=10
export WARMUP=3
export CHECK=1

out="results/cuda_variants_${SLURM_JOB_ID}"
srun bash scripts/bench.sh -o "$out" baseline cuda_blocks
echo "Results: $out"
cat "$out/implementations.txt"
