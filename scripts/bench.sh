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
#   channels     seq + CUDA x CHANNELS x CH_SIZES x CH_KS        multi-channel input (review)
#                (and PyTorch on the same problems, if present)
#   channel_blocks  CUDA block shapes at CH_BLOCK_CS channels    block configuration with C > 1
#   all          all of the above
#
# Implementations: every name in IMPLS that bin/conv accepts. When teammates
# add a variant to the driver (e.g. --impl cuda_tiled), add its name to IMPLS
# (or to the default list below); names starting with "omp" also go through
# the omp sweep, names starting with "cuda" through the cuda_blocks sweep.
#
# Settings (environment variables, defaults in brackets):
#   IMPLS        ["seq omp omp_simd omp_blocked cuda cuda_opt cuda_const cuda_tiled_const cuda_multiout ..."]
#                also includes cuda_multiout_const, cuda_multiout_tiled,
#                and cuda_multiout_tiled_const
#                names the driver doesn't know are skipped
#   SIZES        ["16 32 64 128 256 512 1024 2048"]   KS ["3 5 7"]
#                (the sizes below 256 locate where parallel versions stop beating seq: H2)
#   OMP_SIZES    ["64 512 2048"]   OMP_KS ["3 7"]   THREADS [1 2 4 ... nproc]
#   SCHEDULES    ["static dynamic guided"]
#   BLOCK_SIZES  ["1024 2048"]  BLOCK_KS ["3 7"]
#   BLOCKS       [26 shapes: warp-wide rows, squares, rows wider than 32 up to 1024x1,
#                 and columns taller than 32 up to 1x512; see the default below]
#   BLOCK        ["32 8"]       CUDA block for baseline and channels; auto tunes supported CUDA variants
#   CHANNELS     ["1 3 8 16 32 64"]   CH_SIZES ["256 512 1024"]   CH_KS ["3 5 7"]
#   CH_BLOCK_CS  ["16"]   CH_BLOCK_SIZES ["1024"]   CH_BLOCK_KS ["3 7"]   (channel_blocks; shapes from BLOCKS)
#   ITERS        timed runs per point [auto: 200 up to 128^2, 50 up to 512^2, 20 at 1024^2, 10 above]
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
[[ " ${EXPS[*]} " == *" all "* ]] && EXPS=(baseline omp cuda_blocks torch channels channel_blocks)

NPROC=$(nproc 2>/dev/null || echo 4)
# Respect both the CPUs visible to this process and the Slurm allocation.
if [[ ${SLURM_CPUS_PER_TASK:-} =~ ^[1-9][0-9]*$ ]] && ((NPROC > SLURM_CPUS_PER_TASK)); then
    NPROC=$SLURM_CPUS_PER_TASK
fi
pow2_threads() {
    local t=1 out=""
    while ((t < NPROC)); do out+="$t "; t=$((t * 2)); done
    echo "$out$NPROC"
}

if ((QUICK)); then
    : "${SIZES:=256 1024}" "${KS:=3 7}" "${OMP_SIZES:=1024}" "${OMP_KS:=3}"
    : "${BLOCK_SIZES:=1024}" "${BLOCK_KS:=3}" "${BLOCKS:=32x8 16x16 128x1 1x128 256x4 1024x1}"
    : "${SCHEDULES:=static dynamic}" "${ITERS:=5}"
    : "${CHANNELS:=1 16}" "${CH_SIZES:=256}" "${CH_KS:=3}"
    : "${CH_BLOCK_CS:=4}" "${CH_BLOCK_SIZES:=256}" "${CH_BLOCK_KS:=3}"
fi
IMPLS=${IMPLS:-"seq omp omp_simd omp_blocked cuda cuda_opt cuda_const cuda_tiled_const cuda_multiout cuda_multiout_const cuda_multiout_tiled cuda_multiout_tiled_const cuda_tiled cuda_tiled_mo"}
SIZES=${SIZES:-"16 32 64 128 256 512 1024 2048"}
KS=${KS:-"3 5 7"}
OMP_SIZES=${OMP_SIZES:-"64 512 2048"}
OMP_KS=${OMP_KS:-"3 7"}
THREADS=${THREADS:-$(pow2_threads)}
SCHEDULES=${SCHEDULES:-"static dynamic guided"}
BLOCK_SIZES=${BLOCK_SIZES:-"1024 2048"}
BLOCK_KS=${BLOCK_KS:-"3 7"}
# Warp-wide rows (coalesced), squares, rows wider than a warp (64 to 1024
# threads across), and columns taller than 32 (1xN, 4x64 ...: uncoalesced).
# Shapes over 1024 threads are skipped; shapes a kernel can't run (e.g. a
# shared tile that doesn't fit) are recorded as skipped, not as errors.
BLOCKS=${BLOCKS:-"32x8 32x4 32x16 32x32 8x8 16x8 16x16 64x4 64x8 64x16 128x1 128x2 128x4 128x8 256x1 256x2 256x4 512x1 512x2 1024x1 1x128 1x256 1x512 4x64 8x64 16x64"}
BLOCK=${BLOCK:-"32 8"}
CHANNELS=${CHANNELS:-"1 3 8 16 32 64"}
CH_SIZES=${CH_SIZES:-"256 512 1024"}
CH_KS=${CH_KS:-"3 5 7"}
CH_BLOCK_CS=${CH_BLOCK_CS:-"16"}
CH_BLOCK_SIZES=${CH_BLOCK_SIZES:-"1024"}
CH_BLOCK_KS=${CH_BLOCK_KS:-"3 7"}
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
SKIPPED=()
for i in $IMPLS; do
    if "$CONV" --impl "$i" --size 16 --K 3 --iters 1 --warmup 0 >/dev/null 2>&1; then
        AVAILABLE+=("$i")
    else
        SKIPPED+=("$i")
    fi
done
((${#AVAILABLE[@]})) || { echo "error: $CONV accepts none of: $IMPLS" >&2; exit 1; }
echo "implementations: ${AVAILABLE[*]}" | tee "$OUT/implementations.txt" >&2
if ((${#SKIPPED[@]})); then
    echo "unavailable/skipped: ${SKIPPED[*]}" >&2
fi
impls_matching() { local i; for i in "${AVAILABLE[@]}"; do [[ $i == $1 ]] && echo "$i"; done; }

iters_for() {  # more samples for short runs
    if [[ -n $ITERS ]]; then echo "$ITERS"
    elif (($1 <= 128)); then echo 200
    elif (($1 <= 512)); then echo 50
    elif (($1 <= 1024)); then echo 20
    else echo 10; fi
}

N_OK=0 N_FAIL=0 N_ERR=0 N_SKIP=0
SKIPLOG=$OUT/skipped.log
# run CSV THREADS SCHEDULE ARGS...
# Runs bin/conv once and appends "threads,schedule,<driver row>" to CSV. The
# driver's own header (--header) becomes the file header, so the files stay
# right if teammates add columns to the driver.
run() {
    local csv=$1 threads=$2 sched=$3; shift 3
    local args=("$@" --warmup "$WARMUP" --header)
    ((CHECK)) && [[ " $* " != *" --impl seq "* ]] && args+=(--check)
    local out rc errfile
    errfile=$(mktemp)
    if [[ -n $sched ]]; then   # OpenMP run: thread count and schedule via the environment
        out=$(OMP_NUM_THREADS=$threads OMP_SCHEDULE=$sched "$CONV" "${args[@]}" 2>"$errfile"); rc=$?
    else
        out=$("$CONV" "${args[@]}" 2>"$errfile"); rc=$?
    fi
    # Status 3: a configuration this GPU or kernel can't run (e.g. a shared
    # tile too large for a wide block). Logged, not counted as an error.
    if ((rc == 3)); then
        N_SKIP=$((N_SKIP + 1))
        echo "${args[*]}: $(tr '\n' ' ' <"$errfile")" >>"$SKIPLOG"
        rm -f "$errfile"
        return
    fi
    cat "$errfile" >>"$ERRLOG"; rm -f "$errfile"
    case $rc in
        0) N_OK=$((N_OK + 1)) ;;
        2) N_FAIL=$((N_FAIL + 1)); echo "CHECK FAILED: ${args[*]}" | tee -a "$ERRLOG" >&2 ;;
        *) N_ERR=$((N_ERR + 1)); echo "ERROR rc=$rc: ${args[*]}" | tee -a "$ERRLOG" >&2; return ;;
    esac
    [[ -s $csv ]] || echo "threads,schedule,$(head -1 <<<"$out")" > "$csv"
    echo "$threads,$sched,$(tail -1 <<<"$out")" >> "$csv"
    # Find median_ms by name, so new driver columns don't shift it.
    local col
    col=$(head -1 <<<"$out" | tr , '\n' | grep -nx median_ms | cut -d: -f1)
    printf '  %-58s %10s ms\n' "$* ${threads:+t=$threads $sched}" \
        "$(tail -1 <<<"$out" | cut -d, -f"${col:-9}")" >&2
}

# In experiments where threads/schedule aren't the variable: seq runs on 1
# thread, OpenMP on all cores with a static schedule, CUDA has neither.
exp_baseline() {
    local csv=$OUT/baseline.csv i n k
    for i in "${AVAILABLE[@]}"; do
        for n in $SIZES; do for k in $KS; do
            local common=(--impl "$i" --size "$n" --K "$k" --iters "$(iters_for "$n")")
            case $i in
                omp*)  run "$csv" "$NPROC" static "${common[@]}" ;;
                cuda|cuda_opt|cuda_const|cuda_tiled_const|cuda_multiout|cuda_multiout_const|cuda_multiout_tiled|cuda_multiout_tiled_const)
                    if [[ $BLOCK == auto ]]; then
                        run "$csv" "" "" "${common[@]}" --block auto
                    else
                        run "$csv" "" "" "${common[@]}" --block $BLOCK
                    fi ;;
                cuda*)
                    if [[ $BLOCK == auto ]]; then
                        # Other CUDA implementations may not support auto tuning.
                        run "$csv" "" "" "${common[@]}" --block 32 8
                    else
                        run "$csv" "" "" "${common[@]}" --block $BLOCK
                    fi ;;
                *)     run "$csv" 1 "" "${common[@]}" ;;
            esac
        done; done
    done
}

exp_omp() {
    local csv=$OUT/omp.csv i n k t s
    local omps; omps=$(impls_matching 'omp*')
    [[ -n $omps ]] || { echo "  driver has no OpenMP implementation yet; skipping" >&2; return; }
    for n in $OMP_SIZES; do for k in $OMP_KS; do
        run "$csv" 1 "" --impl seq --size "$n" --K "$k" --iters "$(iters_for "$n")"  # speedup baseline
        for i in $omps; do for s in $SCHEDULES; do for t in $THREADS; do
            run "$csv" "$t" "$s" --impl "$i" --size "$n" --K "$k" --iters "$(iters_for "$n")"
        done; done; done
    done; done
}

exp_cuda_blocks() {
    local csv=$OUT/cuda_blocks.csv i n k b
    local cudas; cudas=$(impls_matching 'cuda*')
    [[ -n $cudas ]] || { echo "  driver has no CUDA implementation; skipping" >&2; return; }
    for i in $cudas; do
        for n in $BLOCK_SIZES; do for k in $BLOCK_KS; do for b in $BLOCKS; do
            (( ${b%x*} * ${b#*x} <= 1024 )) || continue
            run "$csv" "" "" --impl "$i" --size "$n" --K "$k" --block "${b%x*}" "${b#*x}" \
                --iters "$(iters_for "$n")"
        done; done; done
    done
}

exp_torch() {
    if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
        echo "  $PYTHON has no torch; skipping (see BENCHMARK.md)" >&2
        return
    fi
    local args=(--sizes $SIZES --ks $KS --threads "$NPROC" --warmup "$WARMUP"
                --out "$OUT/torch.csv")
    [[ -n $ITERS ]] && args+=(--iters "$ITERS")
    ((CHECK)) || args+=(--no-check)
    "$PYTHON" "$ROOT/scripts/bench_torch.py" "${args[@]}" 2>>"$ERRLOG" || N_ERR=$((N_ERR + 1))
}

# Multi-channel input: seq and every CUDA implementation over CHANNELS x
# CH_SIZES x CH_KS at the baseline block (the OpenMP versions are single-
# channel). C = 1 is included so each curve starts from the single-channel
# case measured the same way. PyTorch runs the same problems if available;
# cuDNN is built for multi-channel input, so this is its fairer comparison.
exp_channels() {
    local csv=$OUT/channels.csv i n k c
    local impls=()
    for i in "${AVAILABLE[@]}"; do [[ $i == omp* ]] || impls+=("$i"); done
    for i in "${impls[@]}"; do
        for n in $CH_SIZES; do for k in $CH_KS; do for c in $CHANNELS; do
            local common=(--impl "$i" --size "$n" --K "$k" --C "$c" --iters "$(iters_for "$n")")
            case $i in
                cuda*) run "$csv" "" "" "${common[@]}" --block $BLOCK ;;
                *)     run "$csv" 1 "" "${common[@]}" ;;
            esac
        done; done; done
    done
    if "$PYTHON" -c 'import torch' 2>/dev/null; then
        local args=(--sizes $CH_SIZES --ks $CH_KS --channels $CHANNELS --threads "$NPROC"
                    --warmup "$WARMUP" --out "$OUT/torch_channels.csv")
        [[ -n $ITERS ]] && args+=(--iters "$ITERS")
        ((CHECK)) || args+=(--no-check)
        "$PYTHON" "$ROOT/scripts/bench_torch.py" "${args[@]}" 2>>"$ERRLOG" || N_ERR=$((N_ERR + 1))
    else
        echo "  $PYTHON has no torch; multi-channel PyTorch reference skipped" >&2
    fi
}

# CUDA block shapes with multi-channel input: the shared-tile kernels reload
# their tile once per channel, so the best shape can differ from C = 1.
exp_channel_blocks() {
    local csv=$OUT/channel_blocks.csv i n k b c
    local cudas; cudas=$(impls_matching 'cuda*')
    [[ -n $cudas ]] || { echo "  driver has no CUDA implementation; skipping" >&2; return; }
    for c in $CH_BLOCK_CS; do for i in $cudas; do
        for n in $CH_BLOCK_SIZES; do for k in $CH_BLOCK_KS; do for b in $BLOCKS; do
            (( ${b%x*} * ${b#*x} <= 1024 )) || continue
            run "$csv" "" "" --impl "$i" --size "$n" --K "$k" --C "$c" \
                --block "${b%x*}" "${b#*x}" --iters "$(iters_for "$n")"
        done; done; done
    done; done
}

echo "results -> $OUT" >&2
for e in "${EXPS[@]}"; do
    declare -F "exp_$e" >/dev/null || { echo "unknown experiment '$e'" >&2; exit 1; }
    echo "== $e" >&2
    start=$SECONDS
    "exp_$e"
    echo "   ($((SECONDS - start)) s)" >&2
done
echo "done: $N_OK ok, $N_FAIL check failures, $N_ERR errors, $N_SKIP unsupported configurations skipped -> $OUT" >&2
((N_SKIP)) && echo "  skipped configurations are listed in $SKIPLOG" >&2
((N_FAIL + N_ERR == 0))
