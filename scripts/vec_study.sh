#!/usr/bin/env bash
# Compiler-vectorization study (review: "compiler vectorizations").
#
# Builds the CPU implementations once per target instruction set and
# benchmarks every build on the same problems. Each build names its -march
# explicitly, because the compiler's own default differs between systems
# (Ubuntu's g++ targets plain x86-64 = SSE2; the g++ on Euler's AlmaLinux 10
# targets x86-64-v3 = AVX2 + FMA). Default builds:
#
#   sse2    -march=x86-64      4 floats per vector, no FMA
#   avx2    -march=x86-64-v3   8 floats per vector, FMA
#   native  -march=native      whatever this CPU has (AVX-512 = 16 floats on
#                              Euler's Zen 4 nodes) and tuning for it
#
# Comparing them shows how much each implementation gains from wider
# vectors, and the compiler's own report shows which loops it vectorized
# at what width.
#
# Usage: scripts/vec_study.sh [-o OUTDIR] [--quick]
#
# Output (OUTDIR, default results/vec-<host>-<date>):
#   builds.txt                    build names in order (the first is the reference)
#   <build>/omp.csv               bench.sh "omp" experiment for each build
#   <build>/sysinfo.txt           hardware and the flags actually used
#   vec_<build>.txt               GCC -fopt-info-vec report (implementation sources only)
#   vec_<build>_inorder.txt       loops GCC vectorized only as an in-order reduction
#   isa.txt                       the instruction-set options each build turns on
#   test_<build>.log              unit-test output for each build
#
# Then: python3 scripts/plot_vec.py OUTDIR
#
# Nothing here changes the normal build: the builds go to bin/vec/<build>,
# and bin/conv is not touched. CUDA is left out (CUDA=0) because this study
# is CPU-only.
#
# Settings (environment variables, defaults in brackets):
#   VEC_BUILDS   ["sse2=-march=x86-64 avx2=-march=x86-64-v3 native=-march=native"]
#                name=extra flags, space-separated; the first is the reference
#   VEC_IMPLS    ["seq omp omp_simd omp_blocked omp_tiled"]
#   OMP_SIZES    ["512 1024 2048"]   OMP_KS ["3 5 7"]
#   THREADS      ["1 <nproc>"]        single core isolates the vector effect;
#                                     all cores shows whether it survives memory limits
#   BASE_FLAGS   ["-O3 -std=c++17 -Wall -Wextra"]   must match the Makefile's CXXFLAGS
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

OUT=""
QUICK=0
while (($#)); do
    case $1 in
        -o) OUT=$2; shift 2 ;;
        --quick) QUICK=1; shift ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

NPROC=$(nproc)
if [[ ${SLURM_CPUS_PER_TASK:-} =~ ^[1-9][0-9]*$ ]] && ((NPROC > SLURM_CPUS_PER_TASK)); then
    NPROC=$SLURM_CPUS_PER_TASK
fi
VEC_BUILDS=${VEC_BUILDS:-"sse2=-march=x86-64 avx2=-march=x86-64-v3 native=-march=native"}
VEC_IMPLS=${VEC_IMPLS:-"seq omp omp_simd omp_blocked omp_tiled"}
BASE_FLAGS=${BASE_FLAGS:-"-O3 -std=c++17 -Wall -Wextra"}
if ((QUICK)); then
    : "${OMP_SIZES:=512}" "${OMP_KS:=3 7}" "${ITERS:=5}"
fi
export OMP_SIZES=${OMP_SIZES:-"512 1024 2048"}
export OMP_KS=${OMP_KS:-"3 5 7"}
export THREADS=${THREADS:-"1 $NPROC"}
export SCHEDULES=static
export ITERS=${ITERS:-}

OUT=${OUT:-results/vec-$(hostname -s)-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)   # absolute, so -o works with relative or absolute paths

NAMES=()
declare -A EXTRA
for b in $VEC_BUILDS; do
    NAMES+=("${b%%=*}")
    EXTRA[${b%%=*}]=${b#*=}
done
printf '%s\n' "${NAMES[@]}" > "$OUT/builds.txt"

# Which instruction-set extensions each build enables on this machine.
: > "$OUT/isa.txt"
for name in "${NAMES[@]}"; do
    {
        echo "# $name: g++ ${EXTRA[$name]} on $(hostname)"
        g++ ${EXTRA[$name]} -Q --help=target 2>/dev/null \
            | grep -E '^\s+-march=|^\s+-m(sse4\.2|avx|avx2|fma|avx512f|avx512vl)\s' \
            | sed 's/^\s*//'
        echo
    } >> "$OUT/isa.txt"
done

# build NAME
# Fresh build into bin/vec/NAME, unit tests on that build, then the driver
# again with GCC's vectorization report. The report flag only writes the
# report; it doesn't change the generated code.
build() {
    local name=$1 bin=bin/vec/$1
    local flags="$BASE_FLAGS ${EXTRA[$1]}"
    echo "== build $name: $flags" >&2
    rm -rf "$bin"
    make --no-print-directory CUDA=0 BIN="$bin" CXXFLAGS="$flags" test VALIDATE=0 \
        > "$OUT/test_$name.log" 2>&1 \
        || { echo "unit tests failed for $name build; see $OUT/test_$name.log" >&2; exit 1; }
    grep -E 'ALL PASSED|FAIL' "$OUT/test_$name.log" | tail -1 | sed 's/^/   unit tests: /' >&2

    local report=$OUT/vec_${name}_all.txt
    make --no-print-directory CUDA=0 BIN="$bin" \
        CXXFLAGS="$flags -fopt-info-vec-optimized=$report" "$bin/conv" > /dev/null
    # Keep only lines about the implementations, not the driver or headers.
    grep -E '^src/([a-z]+/)?conv2d_' "$report" | sort -u > "$OUT/vec_$name.txt" || true
    rm -f "$report"

    # "loop vectorized" can be misleading for a sum into one float: without
    # -ffast-math GCC may only vectorize it as an in-order (fold-left)
    # reduction, where the multiplies run in vector lanes but the adds still
    # happen one after another to keep the rounding identical. List those
    # loops separately so the summary can flag them.
    local f tmp
    tmp=$(mktemp -d)
    : > "$OUT/vec_${name}_inorder.txt"
    for f in $(cut -d: -f1 "$OUT/vec_$name.txt" | sort -u); do
        g++ $flags -fopenmp -Iinclude -c "$f" -o "$tmp/x.o" \
            -fdump-tree-vect-details="$tmp/vect.txt"
        grep -E 'in-order \(fold-left\) reduction' "$tmp/vect.txt" \
            | cut -d: -f1-2 | sort -u >> "$OUT/vec_${name}_inorder.txt" || true
    done
    rm -rf "$tmp"
}

for name in "${NAMES[@]}"; do build "$name"; done

for name in "${NAMES[@]}"; do
    echo "== benchmark $name" >&2
    # Exported so bench.sh's sysinfo.txt records the flags of this build.
    CXXFLAGS="$BASE_FLAGS ${EXTRA[$name]}" \
    CONV="$ROOT/bin/vec/$name/conv" IMPLS="$VEC_IMPLS" \
        scripts/bench.sh -o "$OUT/$name" omp
done

echo "done -> $OUT   (next: python3 scripts/plot_vec.py $OUT)" >&2
