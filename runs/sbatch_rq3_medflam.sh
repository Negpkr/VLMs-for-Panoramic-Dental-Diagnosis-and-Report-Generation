#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=rq3-medflam
#SBATCH --output=/home/s222393187/Dental/runs/rq3_medflam_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/rq3_medflam_%j.err
set -uo pipefail
cd /home/s222393187/Dental
echo "RQ3-MEDFLAM host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv||true
export MODELS="med_flamingo"
bash runs/run_rq3_new.sh
echo "RQ3_MEDFLAM_DONE"
