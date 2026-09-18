#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=probe-mg
#SBATCH --output=/home/s222393187/Dental/runs/probe_mg_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/probe_mg_%j.err
set -uo pipefail
cd /home/s222393187/Dental
source /opt/python/3.11/anaconda/etc/profile.d/conda.sh 2>/dev/null || true
conda run --no-capture-output -n medgemma-env python \
  /tmp/claude-8063/-home-s222393187-Dental--claude-worktrees-dental-folder-access-1f4b32/410b55cb-df29-4b8b-966f-8f9691ffbedc/scratchpad/probe_mg.py 2>/tmp/probe_mg_err.txt
echo "PROBE_DONE rc=$?"
echo "--- stderr tail ---"; tail -5 /tmp/probe_mg_err.txt
