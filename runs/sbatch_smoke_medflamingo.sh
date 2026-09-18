#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=smoke-medflam
#SBATCH --output=/home/s222393187/Dental/runs/smoke_medflam_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/smoke_medflam_%j.err
set -uo pipefail
cd /home/s222393187/Dental
echo "SMOKE-MEDFLAM host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true
export CASES=5 STRATEGIES=zero_shot OUT=Results/model_comparison_smoke MODELS="med_flamingo"
bash runs/run_tufts_new.sh
echo "SMOKE_MEDFLAM_DONE"
