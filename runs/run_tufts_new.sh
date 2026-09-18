#!/usr/bin/env bash
# Tufts full-dataset prompting matrix for the FIVE NEW VLMs. Does NOT start itself.
# Writes to Results/model_comparison_new (paper results in model_comparison are untouched).
set -euo pipefail
cd /home/s222393187/Dental
source activate_env.sh
OUT="${OUT:-Results/model_comparison_new}"
MODELS="${MODELS:-medgemma qwen25_vl llava_onevision llava_rad med_flamingo}"
CASES="${CASES:-0}"
STRATEGIES="${STRATEGIES:-zero_shot few_shot cot}"
mkdir -p "$OUT" runs
echo "TUFTS(new) models=$MODELS strategies=$STRATEGIES out=$OUT GPU=${CUDA_VISIBLE_DEVICES:-unset}"
python -u vlm_comparison.py --cases "$CASES" --strategy zero_shot --parse-mode strict --dry-run
for S in $STRATEGIES; do
  echo "===== START TUFTS(new) strategy=$S ====="
  python -u vlm_comparison.py --cases "$CASES" --strategy "$S" --parse-mode strict \
    --models $MODELS --out-dir "$OUT" --tag "${S}_new"
done
echo "TUFTS_NEW_ALL_DONE"
