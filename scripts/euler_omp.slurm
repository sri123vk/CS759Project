#!/usr/bin/env bash
# OpenMP-only rerun, spliced into an earlier full run (default: euler-127842).
# Rebuilds without CUDA, tests and benchmarks only omp / omp_simd / omp_blocked
# (no seq, CUDA or PyTorch timings), then replaces the omp* rows of the base
# run with the new ones and plots the combined results.
#
#   sbatch scripts/euler_omp.slurm                       # baseline omp channels
#   sbatch scripts/euler_omp.slurm channels              # selected experiments
#   sbatch --export=ALL,BASE=results/euler-NNNNNN scripts/euler_omp.slurm
#
# The new numbers are only comparable with the base run on the same node and
# CPU count: euler-127842 ran on euler01 with cpus-per-task=16, which is what
# -w and -c below request. Change both if you use another base run (its
# sysinfo.txt has them). No GPU is requested; the job only needs the CPUs.
#
# Output:
#   results/euler-<job>-omp/              the rerun on its own
#   results/<base>+omp-<job>/             base run with its omp* rows replaced
#   plots/<base>+omp-<job>/               figures and summary.md for the combined run
#SBATCH -J conv-omp
#SBATCH -p research
#SBATCH -w euler01
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 0-02:00:00
#SBATCH -o results/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

BASE=${BASE:-results/euler-127842}
[[ -d $BASE ]] || { echo "error: base run $BASE not found" >&2; exit 1; }

# Same compiler as the base run (g++ 13.2.0 for euler-127842; see its sysinfo.txt).
MODULES=${MODULES:-"gcc"}
set +u   # Lmod's shell function reads unset variables
for m in $MODULES; do
    module load "$m" || echo "warning: 'module load $m' failed; see 'module avail'" >&2
done
set -u

make clean
make -j "$SLURM_CPUS_PER_TASK" CUDA=0 bin/conv bin/test_conv

# Unit tests for the OpenMP versions only (each compared with seq), then the
# external check of their output against SciPy/PyTorch if Python has it.
./bin/test_conv --impl omp
if python3 -c "import numpy, scipy" 2>/dev/null || python3 -c "import numpy, torch" 2>/dev/null; then
    make validate CUDA=0 VALIDATE_IMPLS="omp omp_simd omp_blocked" VALIDATE_DIR=data/validate-omp-$SLURM_JOB_ID
else
    echo "*** external check skipped: python3 has neither numpy+scipy nor numpy+torch" >&2
fi

# Benchmarks: the experiments whose CSVs contain OpenMP rows. --check still
# compares each run with seq for correctness, but no seq timings are recorded.
(($#)) || set -- baseline omp channels
RERUN=results/euler-$SLURM_JOB_ID-omp
IMPLS="omp omp_simd omp_blocked" TORCH=0 \
    scripts/bench.sh -o "$RERUN" "$@" \
    || echo "bench.sh reported failures; see $RERUN/errors.log"

OUT=results/$(basename "$BASE")+omp-$SLURM_JOB_ID
python3 scripts/merge_results.py "$BASE" "$RERUN" "$OUT"
python3 scripts/plot.py "$OUT" || echo "plotting skipped; run scripts/plot.py $OUT elsewhere"
