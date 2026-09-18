#!/bin/bash
#SBATCH -p Virtual
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --job-name=dl-two
#SBATCH --output=/home/s222393187/Dental/runs/dl_two_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/dl_two_%j.err
set -uo pipefail
cd /home/s222393187/Dental
source /opt/python/3.11/anaconda/etc/profile.d/conda.sh 2>/dev/null || true
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1          # avoid Xet shared-session cascade failures
export HF_HUB_ENABLE_HF_TRANSFER=0
echo "DLTWO host=$(hostname) start=$(date)"

# Remove any partial/stub dirs from the poisoned run so we fetch clean.
rm -rf ~/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B \
       ~/.cache/huggingface/hub/models--llava-hf--llava-onevision-qwen2-7b-ov-hf

echo "=== disk before ==="; du -sh ~/.cache/huggingface/hub

conda run -n dental-llava python - <<'PY'
import os, time
from huggingface_hub import snapshot_download
for repo in ["OpenGVLab/InternVL2_5-8B", "llava-hf/llava-onevision-qwen2-7b-ov-hf"]:
    t0=time.time()
    try:
        p=snapshot_download(repo_id=repo)
        print(f"OK   {repo}  {(time.time()-t0)/60:.1f} min -> {p}", flush=True)
    except Exception as e:
        print(f"FAIL {repo}: {type(e).__name__}: {str(e)[:200]}", flush=True)
PY

echo "=== disk after ==="; du -sh ~/.cache/huggingface/hub
echo "=== sizes ==="
du -sh ~/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B 2>/dev/null
du -sh ~/.cache/huggingface/hub/models--llava-hf--llava-onevision-qwen2-7b-ov-hf 2>/dev/null
echo "DLTWO_DONE $(date)"
