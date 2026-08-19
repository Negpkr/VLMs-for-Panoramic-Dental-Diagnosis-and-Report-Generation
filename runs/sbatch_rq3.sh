#!/bin/bash
# RQ3-only job. Does not run Tufts or DENTEX.
# Submit when you have a free GPU slot (you already have 2 jobs: dentex 27255, tufts 27256).
# Default: next free L40S. To run sooner on an idle RTX 4500, change the gres line to:
#   #SBATCH --gres=gpu:rtx4500:1
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=g46-1gpu-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=rq3-pr-all
#SBATCH --output=/home/s222393187/Dental/runs/rq3_l40s_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/rq3_l40s_%j.err

set -euo pipefail
cd /home/s222393187/Dental
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true
bash runs/run_rq3.sh
