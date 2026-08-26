#!/bin/bash
set -eo pipefail

cd /hpf/projects/mkoziarski/zdeng/BindCraftCP
source /hpf/projects/mkoziarski/zdeng/miniconda3/etc/profile.d/conda.sh
conda activate BindCraft
export PYTHONPATH=/hpf/projects/mkoziarski/zdeng/BindCraftCP:${PYTHONPATH:-}

echo "[$(date)] starting cholera nearpass MPNN2 on $(hostname)"
echo "run_label=${CHOLERA_MPNN2_RUN_LABEL:-nearpass_mpnn2_cn502} global_seed=${CHOLERA_MPNN2_GLOBAL_SEED:-42} n_mpnn=${CHOLERA_MPNN2_N_MPNN:-2}"
echo "conda=${CONDA_DEFAULT_ENV:-unknown} python=$(which python)"
nvidia-smi
python -u input/cholera/hallu/run_nearpass_mpnn2_cn502.py
status=$?
echo "[$(date)] finished with status ${status}"
exit "${status}"
