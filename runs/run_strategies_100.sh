#!/usr/bin/env bash
set -euo pipefail
cd /home/s222393187/Dental
source activate_env.sh
export CUDA_VISIBLE_DEVICES=1
OUT=Results/model_comparison
MODELS="llava llava_med huatuogpt_vision dentvlm"
CASES=100

for STRATEGY in zero_shot few_shot cot; do
  echo "===== START strategy=$STRATEGY cases=$CASES ====="
  python -u vlm_comparison.py \
    --cases "$CASES" \
    --strategy "$STRATEGY" \
    --parse-mode strict \
    --models $MODELS \
    --out-dir "$OUT" \
    --tag "${STRATEGY}_100"
  # generate report for newest matching json
  LATEST=$(ls -t "$OUT"/comparison_${STRATEGY}_100_*.json | head -1)
  python comparison_report.py --results "$LATEST" --out-dir "$OUT"
  echo "===== DONE strategy=$STRATEGY -> $LATEST ====="
done
echo "ALL_STRATEGIES_DONE"
