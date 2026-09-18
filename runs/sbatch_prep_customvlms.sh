#!/bin/bash
#SBATCH -p Virtual
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=prep-customvlms
#SBATCH --output=/home/s222393187/Dental/runs/prep_customvlms_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/prep_customvlms_%j.err
set -uo pipefail
cd /home/s222393187/Dental
source /opt/python/3.11/anaconda/etc/profile.d/conda.sh 2>/dev/null || true
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "PREP host=$(hostname) start=$(date)"
bash new_models/env_setup/setup_custom_vlms.sh; echo "SETUP rc=$?"
conda run -n custom-vlms python download_new_models.py --download --group custom; echo "DOWNLOAD rc=$?"
echo "clones present:"; ls -1 new_models/repos 2>&1
echo "PREP_CUSTOMVLMS_DONE $(date)"
