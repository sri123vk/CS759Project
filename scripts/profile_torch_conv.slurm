#!/bin/bash
#SBATCH --job-name=torch_profile
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=torch_profile_%j.out

set -eo pipefail

cd /srv/home/rangrej/CS759Project

module load conda/miniforge
bootstrap-conda
conda activate conv_torch312
export PYTHON="$(command -v python)"

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys is not on PATH. Check the available Euler modules."
    exit 1
fi

mkdir -p results
profile_base="results/torch_conv_trace_${SLURM_JOB_ID}"

srun --cpus-per-task="$SLURM_CPUS_PER_TASK" nsys profile \
    --trace=cuda,cudnn \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --sample=none \
    --cpuctxsw=none \
    --output="$profile_base" \
    "$PYTHON" -c '
import torch
import torch.cuda.profiler
import torch.nn.functional as F

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access a CUDA GPU")

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("cuDNN available:", torch.backends.cudnn.is_available())

# Match the relevant settings in your benchmark.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

H, W, K = 1024, 1024, 3

# Batch=1, input channels=1, output channels=1.
image = torch.rand((1, 1, H, W), device="cuda", dtype=torch.float32)
mask = torch.rand((1, 1, K, K), device="cuda", dtype=torch.float32)

# Warm-up occurs before the captured region.
for _ in range(3):
    F.conv2d(image, mask, padding=K // 2)

torch.cuda.synchronize()

# Capture one convolution after warm-up.
torch.cuda.profiler.start()
output = F.conv2d(image, mask, padding=K // 2)
torch.cuda.synchronize()
torch.cuda.profiler.stop()

print("Output shape:", tuple(output.shape))
'

# Print the GPU launch dimensions.
nsys stats --report cuda_gpu_trace \
    "$profile_base.nsys-rep" > "${profile_base}_launches.txt"

cat "${profile_base}_launches.txt"
