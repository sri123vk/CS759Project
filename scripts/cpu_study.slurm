#!/usr/bin/env bash
# CPU-side experiments on one cluster node:
#   1. compiler-vectorization study (default vs -march=native, scripts/vec_study.sh)
#   2. OpenMP thread scaling and baseline for the CPU implementations,
#      including omp_tiled (scripts/bench.sh omp baseline)
#
#   sbatch scripts/cpu_study.slurm
#
# Runs on the same node type as the CUDA runs (Genoa nodes of the
# instruction partition), so CPU and GPU numbers come from one machine
# type; sysinfo.txt in each output directory records the node.
#SBATCH -J conv-cpu
#SBATCH -p instruction
#SBATCH --constraint=genoa
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 0-01:00:00
#SBATCH -o results/slurm-cpu-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc 2>/dev/null || echo "warning: module load gcc failed"

OUT=results/cpu-$SLURM_JOB_ID
mkdir -p "$OUT"

# 1. Builds bin/vec/{default,native}, runs make test on both, benchmarks both.
scripts/vec_study.sh -o "$OUT/vec"

# 2. Default build (the one every other experiment uses).
make clean
make -j "$SLURM_CPUS_PER_TASK" CUDA=0
make test CUDA=0 VALIDATE=0
IMPLS="seq omp omp_simd omp_blocked omp_tiled" \
    scripts/bench.sh -o "$OUT/bench" omp baseline \
    || echo "bench.sh reported failures; see $OUT/bench/errors.log"

# Plots need pandas + matplotlib; skip if they're not available here.
python3 scripts/plot_vec.py "$OUT/vec" -o "$OUT/vec/plots" \
    || echo "plotting skipped; run scripts/plot_vec.py $OUT/vec elsewhere"
python3 scripts/plot.py "$OUT/bench" -o "$OUT/bench/plots" \
    || echo "plotting skipped; run scripts/plot.py $OUT/bench elsewhere"
echo "done -> $OUT"
