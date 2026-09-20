#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=g16-8gpu-1,g16-8gpu-2,g48-1gpu-1,g48-1gpu-2,g48-2gpu-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=rq3-coreplain
#SBATCH --output=/home/s222393187/Dental/runs/rq3_coreplain_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/rq3_coreplain_%j.err
set -uo pipefail
cd /home/s222393187/Dental
export VLM_SCHEMA_REINFORCE=0    # plain prompt = same protocol as the original 4
echo "RQ3-COREPLAIN host=$(hostname) reinforce=$VLM_SCHEMA_REINFORCE"; nvidia-smi --query-gpu=name,memory.total --format=csv||true
export MODELS="medgemma qwen25_vl llava_onevision"
bash runs/run_rq3_new.sh
echo "RQ3_COREPLAIN_DONE"
