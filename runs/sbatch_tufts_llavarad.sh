#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=g20-2gpu-1,g20-2gpu-2,g20-2gpu-3,g20-2gpu-4,g20-2gpu-5,g20-2gpu-6,g24-2gpu-1,g24-2gpu-2,g24-2gpu-3,g46-1gpu-1,g46-1gpu-2,g46-1gpu-3,g46-1gpu-4,g46-2gpu-1,g46-2gpu-2,g48-1gpu-1,g48-1gpu-2,g48-2gpu-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=tufts-llavarad
#SBATCH --output=/home/s222393187/Dental/runs/tufts_llavarad_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/tufts_llavarad_%j.err
set -uo pipefail
cd /home/s222393187/Dental
echo "TUFTS-LLAVARAD host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv||true
export MODELS="llava_rad"
bash runs/run_tufts_new.sh
echo "TUFTS_LLAVARAD_DONE"
