#!/usr/bin/env bash
# Tufts full-dataset prompting matrix (1000 usable cases). Does not start itself.
#
# Frozen protocol: pooled 32-slot Missing vs Present, parse Missing teeth: only
# (Total missing is not scored), Macro F1 headline.
#
# When you are ready, use runs/run_all.sh or:
#   cd /home/s222393187/Dental
#   nohup bash runs/run_tufts_strategies.sh > runs/tufts_strategies_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail
cd /home/s222393187/Dental
source activate_env.sh
# Do not override Slurm's GPU. On luthin, activate_env.sh defaults to GPU 1.

OUT=Results/model_comparison
MODELS="llava llava_med huatuogpt_vision dentvlm"
CASES="${CASES:-0}"
STRATEGIES="${STRATEGIES:-zero_shot few_shot cot}"
mkdir -p "$OUT" runs

is_done() {
  local json="$1"
  local log="${json%.json}.log"
  [[ -f "$json" ]] && [[ -f "$log" ]] && grep -q "COMPARISON_DONE" "$log"
}

latest_json() {
  local strategy="$1"
  ls -t "$OUT"/comparison_${strategy}_full_*.json 2>/dev/null | head -1 || true
}

echo "Tufts launcher cases=${CASES:-all} strategies=$STRATEGIES models=$MODELS GPU=${CUDA_VISIBLE_DEVICES}"
echo "DRY RUN first — aborting if cases do not load."
python -u vlm_comparison.py --cases "$CASES" --strategy zero_shot --parse-mode strict --dry-run

for STRATEGY in $STRATEGIES; do
  echo "===== START TUFTS strategy=$STRATEGY cases=$CASES ====="
  LATEST="$(latest_json "$STRATEGY")"
  RESUME_ARGS=()
  if [[ -n "${LATEST}" ]]; then
    if is_done "$LATEST"; then
      echo "===== SKIP strategy=$STRATEGY (already complete: $LATEST) ====="
      continue
    fi
    echo "===== RESUME strategy=$STRATEGY from $LATEST ====="
    RESUME_ARGS=(--resume-from "$LATEST")
  fi
  python -u vlm_comparison.py \
    --cases "$CASES" \
    --strategy "$STRATEGY" \
    --parse-mode strict \
    --models $MODELS \
    --out-dir "$OUT" \
    --tag "${STRATEGY}_full" \
    "${RESUME_ARGS[@]}"
  echo "===== DONE TUFTS strategy=$STRATEGY ====="
done
echo "TUFTS_ALL_STRATEGIES_DONE"
