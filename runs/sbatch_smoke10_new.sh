#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --job-name=smoke10-hf5
#SBATCH --output=/home/s222393187/Dental/runs/smoke10_hf5_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/smoke10_hf5_%j.err
set -euo pipefail
cd /home/s222393187/Dental
export CASES=10 STRATEGIES=zero_shot OUT=Results/model_comparison_smoke
export MODELS="medgemma qwen25_vl llava_onevision"
echo "SMOKE10 HF5 host=$(hostname) models=$MODELS"; nvidia-smi --query-gpu=name,memory.free --format=csv || true
bash runs/run_tufts_new.sh
echo "SMOKE10_HF5_DONE"
