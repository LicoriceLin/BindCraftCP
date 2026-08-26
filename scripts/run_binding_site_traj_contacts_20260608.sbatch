#!/bin/bash
#SBATCH --job-name=bstraj-contact
#SBATCH --reservation=mkoziarski_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --tmp=120G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/hpf/projects/mkoziarski/zdeng/BindCraftCP/output/gradient/binding_site_traj/slurm/%x-%A_%a.out
#SBATCH --error=/hpf/projects/mkoziarski/zdeng/BindCraftCP/output/gradient/binding_site_traj/slurm/%x-%A_%a.err

REPO_ROOT="${REPO_ROOT:-/hpf/projects/mkoziarski/zdeng/BindCraftCP}"
cd "$REPO_ROOT"

source ~/.bashrc >/dev/null 2>&1 || true
if ! command -v conda >/dev/null 2>&1; then
  source /hpf/projects/mkoziarski/zdeng/miniconda3/etc/profile.d/conda.sh
fi

set -eo pipefail

conda activate BindCraft

set -u

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

mkdir -p output/gradient/binding_site_traj/slurm

CASES=(
  "ALS3 full cycpep 20"
  "ALS3 8 cycpep 20"
  "ClpP full cycpep 20"
  "ClpP 8 cycpep 20"
  "HA full miniprotein 50"
  "HA 8 miniprotein 50"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
read -r TARGET SEGMENT BINDER_TYPE BINDER_LEN <<< "${CASES[$TASK_ID]}"

echo "Job ${SLURM_JOB_ID:-local}_${TASK_ID} on $(hostname)"
echo "Case: target=${TARGET} segment=${SEGMENT} binder_type=${BINDER_TYPE} binder_len=${BINDER_LEN}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true

python output/gradient/ana/binding_site_pipeline/analyze_als3_hotspot_trajectory_contacts.py \
  --target "$TARGET" \
  --segment "$SEGMENT" \
  --binder-type "$BINDER_TYPE" \
  --binder-len "$BINDER_LEN" \
  --run-gradient \
  --overwrite-run \
  --iters "${ITERS:-100}" \
  --num-models "${NUM_MODELS:-1}" \
  --log-every "${LOG_EVERY:-10}"
