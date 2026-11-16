#!/bin/bash
#SBATCH --job-name=cuda_compare
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=compare_cuda_%j.out

set -euo pipefail

cd /srv/home/rangrej/CS759Project

for run in 1 2 3 4; do
    echo "Run $run"

    # Alternate which implementation runs first.
    if (( run % 2 == 1 )); then
        impls=(cuda cuda_opt)
    else
        impls=(cuda_opt cuda)
    fi

    for impl in "${impls[@]}"; do
        srun ./bin/conv \
            --impl "$impl" \
            --size 1024 --K 3 \
            --block 32 8 \
            --warmup 5 --iters 50 \
            --check --header
    done
done
