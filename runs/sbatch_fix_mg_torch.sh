#!/bin/bash
#SBATCH -p Virtual
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --job-name=fix-mg-torch
#SBATCH --output=/home/s222393187/Dental/runs/fix_mg_torch_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/fix_mg_torch_%j.err
set -uo pipefail
source /opt/python/3.11/anaconda/etc/profile.d/conda.sh 2>/dev/null || true
echo "FIXMG host=$(hostname) start=$(date)"
conda run -n medgemma-env pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0; echo "PIP rc=$?"
conda run -n medgemma-env python -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)"
echo "FIXMG_DONE $(date)"
