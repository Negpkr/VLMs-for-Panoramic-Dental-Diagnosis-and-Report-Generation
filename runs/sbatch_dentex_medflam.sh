#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=dentex-medflam
#SBATCH --output=/home/s222393187/Dental/runs/dentex_medflam_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/dentex_medflam_%j.err
set -uo pipefail
cd /home/s222393187/Dental
echo "DENTEX-MEDFLAM host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv||true
export MODELS="med_flamingo"
bash runs/run_dentex_new.sh
echo "DENTEX_MEDFLAM_DONE"
