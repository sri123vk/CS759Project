#!/usr/bin/env bash
# Full benchmark on one cluster node: build, test, sweep, plot.
#   sbatch scripts/euler.slurm               # everything
#   sbatch scripts/euler.slurm baseline omp  # selected experiments
#
# ADJUST BEFORE THE FIRST RUN: partition, module names (`module avail`), and
# -c (use every core on the node, so the OpenMP scaling sweep reaches it).
# All reported numbers must come from one node type; sysinfo.txt records it.
#SBATCH -J conv-bench
#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 0-03:00:00
#SBATCH -o results/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load nvidia/cuda gcc 2>/dev/null || echo "warning: module load failed; check module names"

# Build on the compute node, so the binaries match the hardware that is
# benchmarked (not the login node).
make clean
make -j "$SLURM_CPUS_PER_TASK"
make test   # never benchmark code that fails its correctness tests

OUT=results/euler-$SLURM_JOB_ID
scripts/bench.sh -o "$OUT" "${@:-all}" || echo "bench.sh reported failures; see $OUT/errors.log"

# Plots need pandas + matplotlib; skip quietly if they're not available.
python3 scripts/plot.py "$OUT" || echo "plotting skipped; run scripts/plot.py $OUT elsewhere"
