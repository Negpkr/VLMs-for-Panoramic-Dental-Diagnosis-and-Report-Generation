#!/usr/bin/env bash
# Activate the dental-llava environment for this cluster
module purge 2>/dev/null || true
module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true
unset PYTHONPATH
source /opt/python/3.11/anaconda/etc/profile.d/conda.sh
conda activate /home/s222393187/.conda/envs/dental-llava
# Prefer freer GPU on luthin (GPU 0 is often busy). Keep Slurm's assignment if set.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES=1
fi
echo "Env: dental-llava | Python: $(which python) | CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Four-way VLM comparison helpers:
#   python download_models.py
#   CUDA_VISIBLE_DEVICES=1 python vlm_comparison.py --cases 125
#   python comparison_report.py
