#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=g20-2gpu-1,g20-2gpu-2,g20-2gpu-3,g20-2gpu-4,g20-2gpu-5,g20-2gpu-6,g24-2gpu-1,g24-2gpu-2,g24-2gpu-3,g46-1gpu-1,g46-1gpu-2,g46-1gpu-3,g46-1gpu-4,g46-2gpu-1,g46-2gpu-2,g48-1gpu-1,g48-1gpu-2,g48-2gpu-1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --job-name=smk-lr3
#SBATCH --output=/home/s222393187/Dental/runs/smk_lr3_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/smk_lr3_%j.err
set -uo pipefail
cd /home/s222393187/Dental
export CASES=3 STRATEGIES=zero_shot OUT=Results/model_comparison_smoke MODELS="llava_rad"
bash runs/run_tufts_new.sh
echo "SMK_LR3_DONE"
