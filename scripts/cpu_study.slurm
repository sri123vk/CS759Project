#!/usr/bin/env bash
# CPU-side experiments on one cluster node:
#   1. compiler-vectorization study: sse2 / avx2 / native builds (scripts/vec_study.sh)
#   2. OpenMP thread scaling and schedules for every CPU implementation,
#      including omp_tiled (scripts/bench.sh omp)
#
#   sbatch scripts/cpu_study.slurm
#
# Runs on the same node type as the CUDA runs (Genoa nodes of the
# instruction partition); sysinfo.txt in each output directory records the
# node. Everything is built with the system g++, so seq and the OpenMP
# versions in one output directory always come from the same compiler.
#SBATCH -J conv-cpu
#SBATCH -p instruction
#SBATCH --constraint=genoa
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 0-00:40:00   # instruction partition allows at most 45 min
#SBATCH -o results/slurm-cpu-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
g++ --version | head -1

OUT=results/cpu-$SLURM_JOB_ID
mkdir -p "$OUT"

# 1. Builds bin/vec/{sse2,avx2,native}, runs make test on each, benchmarks each.
scripts/vec_study.sh -o "$OUT/vec"

# 2. Default build (the one every other experiment uses).
make clean
make -j "$SLURM_CPUS_PER_TASK" CUDA=0
make test CUDA=0 VALIDATE=0
IMPLS="seq omp omp_simd omp_blocked omp_tiled" TORCH=0 \
    scripts/bench.sh -o "$OUT/bench" omp \
    || echo "bench.sh reported failures; see $OUT/bench/errors.log"

# Plots need pandas + matplotlib; skip if they're not available here.
python3 scripts/plot_vec.py "$OUT/vec" -o "$OUT/vec/plots" \
    || echo "plotting skipped; run scripts/plot_vec.py $OUT/vec elsewhere"
python3 scripts/plot.py "$OUT/bench" -o "$OUT/bench/plots" \
    || echo "plotting skipped; run scripts/plot.py $OUT/bench elsewhere"
echo "done -> $OUT"
