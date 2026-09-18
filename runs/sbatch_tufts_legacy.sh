#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=tufts-legacy
#SBATCH --output=/home/s222393187/Dental/runs/tufts_legacy_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/tufts_legacy_%j.err
set -uo pipefail
cd /home/s222393187/Dental
echo "TUFTS-LEGACY host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true
export MODELS="llava_rad med_flamingo"
bash runs/run_tufts_new.sh
echo "TUFTS_LEGACY_DONE"
