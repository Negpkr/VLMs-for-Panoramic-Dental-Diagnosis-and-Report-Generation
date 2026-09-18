#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=g48-1gpu-1,g48-1gpu-2,g48-2gpu-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --job-name=chk-lmed
#SBATCH --output=/home/s222393187/Dental/runs/chk_lmed_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/chk_lmed_%j.err
set -uo pipefail
cd /home/s222393187/Dental
source activate_env.sh
echo "CHK-LMED host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true
python -u dentex_vlm_comparison.py --split qed --cases 10 --strategy zero_shot \
  --models llava_med --out-dir Results/dentex_comparison_smoke --tag zs_llavamed_check
echo "CHK_LMED_DONE"
