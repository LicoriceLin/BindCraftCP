#!/bin/bash
#SBATCH --job-name=cholera-hallu-resume
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --partition=gen_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00

set -eo pipefail

echo "[$(date)] starting ${SLURM_JOB_ID:-no_job_id} on $(hostname)"
echo "CONFIG=${CONFIG:-}"

if [[ -z "${CONFIG:-}" ]]; then
  echo "CONFIG is required" >&2
  exit 2
fi

cd /hpf/projects/mkoziarski/zdeng/BindCraftCP
echo "[$(date)] loading conda"
source /hpf/projects/mkoziarski/zdeng/miniconda3/etc/profile.d/conda.sh
conda activate BindCraft
echo "[$(date)] conda env: ${CONDA_DEFAULT_ENV:-unknown}; python=$(which python)"
nvidia-smi
echo "[$(date)] starting epitopecraft"
python -u -m epitopecraft.main design-all-in-one "$CONFIG"
