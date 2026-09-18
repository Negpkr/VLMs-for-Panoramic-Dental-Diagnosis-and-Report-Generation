#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=mg-persist
#SBATCH --output=/home/s222393187/Dental/runs/mg_persist_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/mg_persist_%j.err
set -euo pipefail
cd /home/s222393187/Dental
export CASES=6 STRATEGIES=zero_shot OUT=Results/model_comparison_smoke MODELS="medgemma"
echo "MG-PERSIST host=$(hostname)"
bash runs/run_tufts_new.sh
echo "MG_PERSIST_DONE"
