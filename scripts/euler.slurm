#!/usr/bin/env bash
# Full benchmark on one cluster node: build, test, sweep, plot.
#   sbatch scripts/euler.slurm               # everything
#   sbatch scripts/euler.slurm baseline omp  # selected experiments
#
# ADJUST BEFORE THE FIRST RUN: partition, module names (`module avail`), and
# -c (use every core on the node, so the OpenMP scaling sweep reaches it).
# Module names can also be set per job:
#   sbatch --export=ALL,MODULES="nvidia/cuda/13.0.0 gcc/13.2.0" scripts/euler.slurm
# All reported numbers must come from one node type; sysinfo.txt records it.
#SBATCH -J conv-bench
#SBATCH -p research
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 0-03:00:00
#SBATCH -o results/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Load each module separately: Lmod rejects the whole line if one name is
# unknown. Errors are shown, not hidden.
MODULES=${MODULES:-"nvidia/cuda gcc"}
set +u   # Lmod's shell function reads unset variables
for m in $MODULES; do
    module load "$m" || echo "warning: 'module load $m' failed; see 'module avail'" >&2
done
set -u
command -v nvcc >/dev/null || {
    echo "error: nvcc not found after loading: $MODULES" >&2
    echo "       find the CUDA module with 'module avail cuda' and set MODULES" >&2
    exit 1
}

# Build on the compute node, so the binaries match the hardware that is
# benchmarked (not the login node). CUDA=1 makes a missing nvcc a build
# error instead of a silent CPU-only build.
make clean
make -j "$SLURM_CPUS_PER_TASK" CUDA=1
make test CUDA=1   # never benchmark code that fails its correctness tests

OUT=results/euler-$SLURM_JOB_ID
scripts/bench.sh -o "$OUT" "${@:-all}" || echo "bench.sh reported failures; see $OUT/errors.log"

# Plots need pandas + matplotlib; skip quietly if they're not available.
python3 scripts/plot.py "$OUT" || echo "plotting skipped; run scripts/plot.py $OUT elsewhere"
