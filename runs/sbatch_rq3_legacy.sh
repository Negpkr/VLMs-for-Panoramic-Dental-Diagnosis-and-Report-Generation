#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=rq3-legacy
#SBATCH --output=/home/s222393187/Dental/runs/rq3_legacy_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/rq3_legacy_%j.err
set -uo pipefail
cd /home/s222393187/Dental
echo "RQ3-LEGACY host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true
export MODELS="llava_rad med_flamingo"
bash runs/run_rq3_new.sh
echo "RQ3_LEGACY_DONE"
