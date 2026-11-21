#!/usr/bin/env bash
# Compiler-vectorization study (review: "compiler vectorizations").
#
# Builds the CPU implementations twice, with the Makefile's default flags and
# with -march=native added, and benchmarks both on the same problems. The
# default build may only use the baseline x86-64 vector instructions (SSE2,
# 4 floats wide, no FMA); -march=native lets the compiler use whatever this
# CPU has (AVX2 = 8 floats, AVX-512 = 16 floats, FMA). Comparing the two
# shows how much each implementation gains from wider vectors, and the
# compiler's own report shows which loops it actually vectorized.
#
# Usage: scripts/vec_study.sh [-o OUTDIR] [--quick]
#
# Output (OUTDIR, default results/vec-<host>-<date>):
#   default/omp.csv, native/omp.csv    bench.sh "omp" experiment for each build
#   default/sysinfo.txt, native/...    hardware and the flags actually used
#   vec_default.txt, vec_native.txt    GCC -fopt-info-vec reports (implementation sources only)
#   vec_<build>_inorder.txt            loops GCC vectorized only as an in-order reduction
#   isa.txt                            what -march=native turns on for this CPU
#
# Then: python3 scripts/plot_vec.py OUTDIR
#
# Nothing here changes the normal build: the two builds go to bin/vec/default
# and bin/vec/native, and bin/conv is not touched. CUDA is left out (CUDA=0)
# because this study is CPU-only.
#
# Settings (environment variables, defaults in brackets):
#   VEC_IMPLS    ["seq omp omp_simd omp_blocked"]
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
VEC_IMPLS=${VEC_IMPLS:-"seq omp omp_simd omp_blocked"}
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

# Which instruction-set extensions -march=native enables on this machine.
{
    echo "# g++ -march=native on $(hostname)"
    g++ -march=native -Q --help=target 2>/dev/null \
        | grep -E '^\s+-march=|^\s+-m(sse4\.2|avx|avx2|fma|avx512f|avx512vl)\s' \
        | sed 's/^\s*//'
} > "$OUT/isa.txt"

# build NAME EXTRA_FLAGS
# Fresh build into bin/vec/NAME, unit tests on that build, then the driver
# again with GCC's vectorization report. The report flag only writes the
# report; it doesn't change the generated code.
build() {
    local name=$1 extra=$2 bin=bin/vec/$1
    local flags="$BASE_FLAGS${extra:+ $extra}"
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

build default ""
build native "-march=native"

for name in default native; do
    echo "== benchmark $name" >&2
    # Exported so bench.sh's sysinfo.txt records the flags of this build.
    CXXFLAGS="$BASE_FLAGS$([[ $name == native ]] && echo ' -march=native')" \
    CONV="$ROOT/bin/vec/$name/conv" IMPLS="$VEC_IMPLS" \
        scripts/bench.sh -o "$OUT/$name" omp
done

echo "done -> $OUT   (next: python3 scripts/plot_vec.py $OUT)" >&2
