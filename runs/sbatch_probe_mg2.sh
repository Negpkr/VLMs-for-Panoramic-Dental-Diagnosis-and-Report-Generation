#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=probe-mg2
#SBATCH --output=/home/s222393187/Dental/runs/probe_mg2_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/probe_mg2_%j.err
set -uo pipefail
cd /home/s222393187/Dental
source /opt/python/3.11/anaconda/etc/profile.d/conda.sh 2>/dev/null || true
conda run --no-capture-output -n medgemma-env python new_models/diag/probe_mg.py
echo "PROBE2_DONE rc=$?"
