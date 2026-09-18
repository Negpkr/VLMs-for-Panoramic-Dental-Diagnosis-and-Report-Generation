#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=dentex-legacy
#SBATCH --output=/home/s222393187/Dental/runs/dentex_legacy_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/dentex_legacy_%j.err
set -uo pipefail
cd /home/s222393187/Dental
echo "DENTEX-LEGACY host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true
export MODELS="llava_rad med_flamingo"
bash runs/run_dentex_new.sh
echo "DENTEX_LEGACY_DONE"
