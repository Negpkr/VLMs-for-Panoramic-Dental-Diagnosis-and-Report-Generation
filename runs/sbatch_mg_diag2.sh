#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=mg-diag2
#SBATCH --output=/home/s222393187/Dental/runs/mg_diag2_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/mg_diag2_%j.err
set -uo pipefail
cd /home/s222393187/Dental
source /opt/python/3.11/anaconda/etc/profile.d/conda.sh 2>/dev/null || true
P=/tmp/claude-8063/-home-s222393187-Dental--claude-worktrees-dental-folder-access-1f4b32/410b55cb-df29-4b8b-966f-8f9691ffbedc/scratchpad/p_describe.txt
R=Tufts_Dental_Database/Radiographs
for IMG in 1.JPG 100.JPG 1000.JPG; do
  echo "===== DESCRIBE $IMG ====="
  conda run --no-capture-output -n medgemma-env python -m new_models.workers.worker_medgemma \
     --image $R/$IMG --prompt-file $P --max-new-tokens 80 2>/dev/null
  echo
done
echo "MG_DIAG2_DONE"
