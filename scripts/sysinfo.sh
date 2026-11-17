#!/usr/bin/env bash
# Hardware, software and build settings, saved next to every benchmark run
# (scripts/bench.sh writes it to <results>/sysinfo.txt) for the report's
# reproducibility section.
ROOT=$(cd "$(dirname "$0")/.." && pwd)

section() { printf '\n## %s\n' "$1"; }

echo "# conv2d benchmark system info"
echo "date:     $(date -Is)"
echo "host:     $(hostname)"
echo "git:      $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)$(git -C "$ROOT" diff --quiet 2>/dev/null || echo ' (dirty)')"
echo "branch:   $(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)"
[[ -n ${SLURM_JOB_ID:-} ]] && echo "slurm:    job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, cpus-per-task=${SLURM_CPUS_PER_TASK:-?}"

section "build"
grep -E "^(CXX|CXXFLAGS|NVCC|NVCCFLAGS) " "$ROOT/Makefile"
echo "CXXFLAGS/NVCCFLAGS overrides from the environment: ${CXXFLAGS:-none} / ${NVCCFLAGS:-none}"
g++ --version 2>/dev/null | head -1
command -v nvcc >/dev/null && nvcc --version | tail -1

section "cpu"
if command -v lscpu >/dev/null; then
    lscpu | grep -E '^(Model name|Socket|Core|Thread|CPU\(s\)|NUMA node\(s\)|L1d|L2|L3|CPU max MHz)'
fi
echo "simd: $(grep -o -w -E 'avx|avx2|fma|avx512f' /proc/cpuinfo 2>/dev/null | sort -u | tr '\n' ' ')"
echo "nproc: $(nproc 2>/dev/null)"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-} OMP_PROC_BIND=${OMP_PROC_BIND:-} OMP_PLACES=${OMP_PLACES:-}"
free -g 2>/dev/null | head -2

section "gpu"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.mem,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv
else
    echo "no nvidia-smi"
fi

section "python"
"${PYTHON:-python3}" - <<'EOF' 2>/dev/null || echo "python3 not available"
import sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda,
          "cudnn", torch.backends.cudnn.version(), "gpu", torch.cuda.is_available())
except ImportError:
    print("torch not installed")
EOF
