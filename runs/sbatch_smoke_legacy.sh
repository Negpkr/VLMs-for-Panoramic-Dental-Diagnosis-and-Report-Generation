#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=smoke-legacy
#SBATCH --output=/home/s222393187/Dental/runs/smoke_legacy_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/smoke_legacy_%j.err
set -uo pipefail
cd /home/s222393187/Dental
source /opt/python/3.11/anaconda/etc/profile.d/conda.sh 2>/dev/null || true
echo "SMOKE-LEGACY host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "=== custom-vlms CUDA sanity ==="
conda run --no-capture-output -n custom-vlms python -c "import torch;print('torch',torch.__version__,'avail',torch.cuda.is_available(),'dev',(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'));x=torch.randn(8,8).cuda(); print('matmul ok', (x@x).sum().item()!=None)" 2>&1 | tail -6
echo "=== run legacy smoke (5 cases, zero_shot) ==="
export CASES=5 STRATEGIES=zero_shot OUT=Results/model_comparison_smoke MODELS="llava_rad med_flamingo"
bash runs/run_tufts_new.sh
echo "SMOKE_LEGACY_DONE"
