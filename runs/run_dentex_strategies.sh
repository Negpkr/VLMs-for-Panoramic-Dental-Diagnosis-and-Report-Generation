#!/usr/bin/env bash
# DENTEX full-dataset prompting matrix. Does not start itself.
#
# Official QED labels only: SPLIT=qed (752 scored = 755 on disk minus 3 held-out train exemplars).
# Noisy LabelMe test included: SPLIT=all (same 3 train IDs excluded).
#
# When you are ready:
#   SPLIT=qed nohup bash runs/run_dentex_strategies.sh > runs/dentex_strategies_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail
cd /home/s222393187/Dental
source activate_env.sh
# Do not override Slurm's GPU. On luthin, activate_env.sh defaults to GPU 1.

OUT=Results/dentex_comparison
MODELS="llava llava_med huatuogpt_vision dentvlm"
SPLIT="${SPLIT:-qed}"
STRATEGIES="${STRATEGIES:-zero_shot few_shot cot}"
mkdir -p "$OUT" runs

is_done() {
  local json="$1"
  local log="${json%.json}.log"
  [[ -f "$json" ]] && [[ -f "$log" ]] && grep -q "DENTEX_COMPARISON_DONE" "$log"
}

latest_json() {
  local strategy="$1"
  ls -t "$OUT"/dentex_${strategy}_${SPLIT}_*.json 2>/dev/null | head -1 || true
}

echo "DENTEX launcher split=$SPLIT strategies=$STRATEGIES models=$MODELS GPU=${CUDA_VISIBLE_DEVICES}"
echo "DRY RUN first — aborting if the split does not load."
python -u dentex_vlm_comparison.py --split "$SPLIT" --strategy zero_shot --dry-run

for STRATEGY in $STRATEGIES; do
  echo "===== START DENTEX strategy=$STRATEGY split=$SPLIT ====="
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
  python -u dentex_vlm_comparison.py \
    --split "$SPLIT" \
    --cases 0 \
    --strategy "$STRATEGY" \
    --models $MODELS \
    --out-dir "$OUT" \
    --tag "${STRATEGY}_${SPLIT}" \
    "${RESUME_ARGS[@]}"
  echo "===== DONE strategy=$STRATEGY ====="
done
echo "===== PAPER FIGURES split=$SPLIT ====="
python -u paper_figures.py --dentex-split "$SPLIT" --out-dir Paper/figures_main \
  || echo "PAPER_FIGURES_SKIPPED (DENTEX results are still saved)"
echo "DENTEX_ALL_STRATEGIES_DONE"
