#!/bin/bash
#SBATCH --job-name=cholera-s1506-mpnn100
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --partition=gen_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

set -eo pipefail

echo "[$(date)] starting ${SLURM_JOB_ID:-no_job_id} on $(hostname)"
cd /hpf/projects/mkoziarski/zdeng/BindCraftCP
export PYTHONPATH=/hpf/projects/mkoziarski/zdeng/BindCraftCP:${PYTHONPATH:-}

echo "[$(date)] loading conda"
source /hpf/projects/mkoziarski/zdeng/miniconda3/etc/profile.d/conda.sh
conda activate BindCraft

echo "[$(date)] conda env: ${CONDA_DEFAULT_ENV:-unknown}; python=$(which python)"
nvidia-smi

echo "[$(date)] starting s150-6 top4 MPNN100"
python -u input/cholera/hallu/run_s1506_top4_mpnn100.py
