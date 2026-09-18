#!/bin/bash
#SBATCH -p Virtual
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=prep-mg-dl
#SBATCH --output=/home/s222393187/Dental/runs/prep_mg_dl_%j.log
#SBATCH --error=/home/s222393187/Dental/runs/prep_mg_dl_%j.err
set -uo pipefail
cd /home/s222393187/Dental
source /opt/python/3.11/anaconda/etc/profile.d/conda.sh 2>/dev/null || true
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "PREP host=$(hostname) start=$(date)"

echo "=== disk before ==="; du -sh ~/.cache/huggingface/hub 2>/dev/null

echo "=== HF access check (hf_it2t group) ==="
conda run -n dental-llava python - <<'PY'
from huggingface_hub import HfApi
api=HfApi()
for r in ["OralGPT/OralGPT-Omni-7B-Instruct","OpenGVLab/InternVL2_5-8B",
          "llava-hf/llava-onevision-qwen2-7b-ov-hf","google/medgemma-4b-it",
          "Qwen/Qwen2.5-VL-7B-Instruct"]:
    try:
        info=api.model_info(r); print(f"ACCESS_OK    {r}  gated={getattr(info,'gated',None)}")
    except Exception as e:
        print(f"ACCESS_BLOCK {r}: {type(e).__name__}: {str(e)[:120]}")
PY

echo "=== build medgemma-env ==="
bash new_models/env_setup/setup_medgemma.sh; echo "MEDGEMMA_ENV_SETUP rc=$?"

echo "=== download hf_it2t weights (skips cached) ==="
conda run -n dental-llava python download_new_models.py --download --group hf_it2t; echo "DOWNLOAD rc=$?"

echo "=== disk after ==="; du -sh ~/.cache/huggingface/hub 2>/dev/null
echo "=== cached models ==="; ls -1 ~/.cache/huggingface/hub | grep models--
echo "PREP_MG_DL_DONE $(date)"
