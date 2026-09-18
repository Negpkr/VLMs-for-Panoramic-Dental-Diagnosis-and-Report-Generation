#!/bin/bash
#SBATCH -p Virtual
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=prep-cvlms
#SBATCH --output=/home/s222393187/Dental/runs/prep_cvlms_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/prep_cvlms_%j.err
set -uo pipefail
cd /home/s222393187/Dental
source /opt/python/3.11/anaconda/etc/profile.d/conda.sh 2>/dev/null || true
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "PREP-CVLMS host=$(hostname) start=$(date)"
conda env remove -n custom-vlms -y >/dev/null 2>&1 || true
bash new_models/env_setup/setup_custom_vlms.sh; echo "SETUP rc=$?"
conda run -n custom-vlms python -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)" 2>&1 | tail -2
conda run -n custom-vlms python -c "import open_flamingo;print('open_flamingo OK')" 2>&1 | tail -1
echo "PREP_CVLMS_DONE $(date)"
