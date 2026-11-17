
#!/usr/bin/env bash
# Benchmark sweeps for the final report. Drives the existing benchmark driver
# (bin/conv) and collects its CSV rows; scripts/plot.py turns a results
# directory into the report's figures and tables.
#
# Usage: scripts/bench.sh [-o OUTDIR] [--quick] EXPERIMENT...
#
# Experiments (BENCHMARK.md maps each to the proposal and reviews):
#   baseline     every implementation x SIZES x KS               execution time, speedup,
#                (CUDA rows include h2d/kernel/d2h stages)       kernel-only vs end-to-end (H1, H2)
#   omp          OpenMP: OMP_NUM_THREADS x OMP_SCHEDULE          thread scaling, static vs dynamic
#   cuda_blocks  CUDA: thread-block shapes (--block BX BY)       block configuration (H3)
#   torch        PyTorch conv2d (cuDNN / CPU), same problems     external reference (review)
#   all          all of the above
#
# Implementations: every name in IMPLS that bin/conv accepts. When teammates
# add a variant to the driver (e.g. --impl cuda_tiled), add its name to IMPLS
# (or to the default list below); names starting with "omp" also go through
# the omp sweep, names starting with "cuda" through the cuda_blocks sweep.
#
# Settings (environment variables, defaults in brackets):
#   IMPLS        ["seq omp cuda cuda_opt cuda_const cuda_tiled_const cuda_multiout
#                  omp_simd omp_blocked cuda_tiled cuda_tiled_mo"]
#                names the driver doesn't know are skipped
#   SIZES        ["256 512 1024 2048"]        KS ["3 5 7"]   (proposal)
#   OMP_SIZES    ["512 2048"]   OMP_KS ["3 7"]   THREADS [1 2 4 ... nproc]
#   SCHEDULES    ["static dynamic guided"]
#   BLOCK_SIZES  ["1024 2048"]  BLOCK_KS ["3 7"]
#   BLOCKS       ["32x8 32x4 32x16 32x32 64x4 64x8 128x1 256x1 8x8 16x8 16x16 1x128 4x64"]
#   BLOCK        ["32 8"]       CUDA block for baseline; auto for cuda, cuda_opt,
#                               cuda_const, cuda_tiled_const, and cuda_multiout
#   ITERS        timed runs per point [auto: 50 up to 512^2, 20 at 1024^2, 10 above]
#   WARMUP       [2]
#   CHECK        [1]  --check every run against seq (0 to skip)
#   CONV [bin/conv]   PYTHON [python3]
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONV=${CONV:-$ROOT/bin/conv}
PYTHON=${PYTHON:-python3}
OUT=""
QUICK=0
EXPS=()

while (($#)); do
    case $1 in
        -o) OUT=$2; shift 2 ;;
        --quick) QUICK=1; shift ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) EXPS+=("$1"); shift ;;
    esac
done
((${#EXPS[@]})) || EXPS=(all)
[[ " ${EXPS[*]} " == *" all "* ]] && EXPS=(baseline omp cuda_blocks torch)

NPROC=$(nproc 2>/dev/null || echo 4)
pow2_threads() {
    local t=1 out=""
    while ((t < NPROC)); do out+="$t "; t=$((t * 2)); done
    echo "$out$NPROC"
}

if ((QUICK)); then
    : "${SIZES:=256 1024}" "${KS:=3 7}" "${OMP_SIZES:=1024}" "${OMP_KS:=3}"
    : "${BLOCK_SIZES:=1024}" "${BLOCK_KS:=3}" "${BLOCKS:=32x8 16x16 128x1 1x128}"
    : "${SCHEDULES:=static dynamic}" "${ITERS:=5}"
fi
IMPLS=${IMPLS:-"seq omp cuda cuda_opt cuda_const cuda_tiled_const cuda_multiout omp_simd omp_blocked cuda_tiled cuda_tiled_mo"}
SIZES=${SIZES:-"256 512 1024 2048"}
KS=${KS:-"3 5 7"}
OMP_SIZES=${OMP_SIZES:-"512 2048"}
OMP_KS=${OMP_KS:-"3 7"}
THREADS=${THREADS:-$(pow2_threads)}
SCHEDULES=${SCHEDULES:-"static dynamic guided"}
BLOCK_SIZES=${BLOCK_SIZES:-"1024 2048"}
BLOCK_KS=${BLOCK_KS:-"3 7"}
# Warp-wide rows (coalesced), squares, thin rows, and columns (1xN: uncoalesced).
BLOCKS=${BLOCKS:-"32x8 32x4 32x16 32x32 64x4 64x8 128x1 256x1 8x8 16x8 16x16 1x128 4x64"}
BLOCK=${BLOCK:-"32 8"}
WARMUP=${WARMUP:-2}
ITERS=${ITERS:-}
CHECK=${CHECK:-1}

# Stable thread placement for the scaling runs (override by setting them).
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

[[ -x $CONV ]] || { echo "error: $CONV not found; run make first" >&2; exit 1; }
OUT=${OUT:-$ROOT/results/$(hostname -s)-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
ERRLOG=$OUT/errors.log
"$ROOT/scripts/sysinfo.sh" > "$OUT/sysinfo.txt" 2>&1

# Which implementations does this build of the driver accept?
AVAILABLE=()
for i in $IMPLS; do