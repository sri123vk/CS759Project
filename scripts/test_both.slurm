#!/bin/bash
#SBATCH --job-name=conv_test
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=conv_test_%j.out

set -euo pipefail

cd /srv/home/rangrej/CS759Project

# Run the C++ correctness tests for each CUDA function.
echo "Testing conv2d_cuda"
srun ./bin/test_conv --impl cuda/32x8

echo "Testing conv2d_cuda_opt"
srun ./bin/test_conv --impl cuda_opt/32x8

# Create a separate directory for this job's validation files.
dump_dir="data/validate_opt_${SLURM_JOB_ID}"
mkdir -p "$dump_dir"

# Generate output from the original CUDA implementation.
echo "Generating cuda output"
srun ./bin/conv --impl cuda \
    --H 33 --W 65 --K 3 \
    --iters 1 --warmup 0 --dump "$dump_dir"

# Generate output from the optimized CUDA implementation.
echo "Generating cuda_opt output"
srun ./bin/conv --impl cuda_opt \
    --H 33 --W 65 --K 3 \
    --iters 1 --warmup 0 --dump "$dump_dir"

# Set up Conda after the CUDA programs finish.
set +u
module load conda/miniforge
bootstrap-conda

# Create the environment only if it does not exist.
conda_env_list="$(conda env list)"

if printf '%s\n' "$conda_env_list" |
    awk '$1 == "conv_validation" { found = 1 } END { exit !found }'
then
    echo "Using existing conv_validation environment"
else
    echo "Creating conv_validation environment"
    conda create -n conv_validation python numpy scipy -y
fi

conda activate conv_validation
set -u

# Check the Python environment.
python -c "import sys, numpy, scipy.signal; print('Using Python:', sys.executable)"

# Compare both CUDA outputs against the external reference.
echo "Running external validation"
python validation/validate.py "$dump_dir" -v
