#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=smoke-llavarad
#SBATCH --output=/home/s222393187/Dental/runs/smoke_llavarad_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/smoke_llavarad_%j.err
set -uo pipefail
cd /home/s222393187/Dental
echo "SMOKE-LLAVARAD host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true
export CASES=5 STRATEGIES=zero_shot OUT=Results/model_comparison_smoke MODELS="llava_rad"
bash runs/run_tufts_new.sh
echo "SMOKE_LLAVARAD_DONE"
