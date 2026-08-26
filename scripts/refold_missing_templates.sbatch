#!/bin/bash
#SBATCH --job-name=refold-missing
#SBATCH --reservation=mkoziarski_gpu
#SBATCH --gres=gpu:NVIDIA_L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-31%4
#SBATCH --output=/hpf/projects/mkoziarski/zdeng/BindCraftCP/output/refold_missing/slurm/%x-%A_%a.out
#SBATCH --error=/hpf/projects/mkoziarski/zdeng/BindCraftCP/output/refold_missing/slurm/%x-%A_%a.err

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/hpf/projects/mkoziarski/zdeng/BindCraftCP}"
cd "$REPO_ROOT"

source ~/.bashrc >/dev/null 2>&1 || true
conda activate BindCraft

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

echo "Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true

python scripts/refold_missing_templates.py \
  --design-path output/HA \
  --design-path output/ClpP \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --shard-count "${SLURM_ARRAY_TASK_COUNT:-32}"
