#!/bin/bash
# Tufts-only job on a free L40S. Does not run DENTEX or RQ3.
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -w g46-1gpu-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=tufts-1000
#SBATCH --output=/home/s222393187/Dental/runs/tufts_l40s_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/tufts_l40s_%j.err

set -euo pipefail
cd /home/s222393187/Dental
export CASES=0
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true
# Tufts only — not runs/run_all.sh
bash runs/run_tufts_strategies.sh
