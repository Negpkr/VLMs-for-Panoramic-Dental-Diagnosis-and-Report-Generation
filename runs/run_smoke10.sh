#!/usr/bin/env bash
# 10-case smoke test for per-class metrics. Does not start the full DENTEX matrix.
set -euo pipefail
cd /home/s222393187/Dental
source activate_env.sh
export CUDA_VISIBLE_DEVICES=1
echo "===== Tufts 10-case zero-shot ====="
python -u vlm_comparison.py --cases 10 --strategy zero_shot --tag smoke10_zs
echo "===== DENTEX val 10-case zero-shot ====="
python -u dentex_vlm_comparison.py --split val --cases 10 --strategy zero_shot --tag smoke10_zs_val
echo "SMOKE10_DONE"
