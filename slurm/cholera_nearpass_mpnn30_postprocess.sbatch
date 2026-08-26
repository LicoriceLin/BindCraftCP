#!/bin/bash
#SBATCH --job-name=cholera-nearpass-post
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=general
#SBATCH --time=2-00:00:00

set -eo pipefail

echo "[$(date)] starting ${SLURM_JOB_ID:-no_job_id} on $(hostname)"
cd /hpf/projects/mkoziarski/zdeng/BindCraftCP
export PYTHONPATH=/hpf/projects/mkoziarski/zdeng/BindCraftCP:${PYTHONPATH:-}

echo "[$(date)] loading conda"
source /hpf/projects/mkoziarski/zdeng/miniconda3/etc/profile.d/conda.sh
conda activate BindCraft

echo "[$(date)] conda env: ${CONDA_DEFAULT_ENV:-unknown}; python=$(which python)"
echo "[$(date)] starting near-pass CPU post-processing"
python -u input/cholera/hallu/run_nearpass_mpnn30_postprocess.py --threads "${SLURM_CPUS_PER_TASK:-16}"
