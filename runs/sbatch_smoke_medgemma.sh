#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=smoke-mg
#SBATCH --output=/home/s222393187/Dental/runs/smoke_mg_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/smoke_mg_%j.err
set -euo pipefail
cd /home/s222393187/Dental
export CASES=3 STRATEGIES=zero_shot OUT=Results/model_comparison_smoke MODELS="medgemma"
echo "SMOKE-MG host=$(hostname)"; nvidia-smi --query-gpu=name,memory.free --format=csv || true
bash runs/run_tufts_new.sh
echo "SMOKE_MG_DONE"
