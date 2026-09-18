#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=tufts-new5
#SBATCH --output=/home/s222393187/Dental/runs/tufts_new_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/tufts_new_%j.err
set -euo pipefail
cd /home/s222393187/Dental
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true
export MODELS="medgemma qwen25_vl llava_onevision"
bash runs/run_tufts_new.sh
