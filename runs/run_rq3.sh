#!/usr/bin/env bash
# RQ3: all PR-Reports cases, four VLMs, BLEU / ROUGE-L / BERTScore.
# Does not start itself. Does not run Tufts or DENTEX.
#
#   sbatch runs/sbatch_rq3.sh
#   # or, if already on a GPU node:
#   bash runs/run_rq3.sh
set -euo pipefail
cd /home/s222393187/Dental
source activate_env.sh

OUT=Results/rq3_reports
MODELS="${MODELS:-llava llava_med huatuogpt_vision dentvlm}"
mkdir -p "$OUT" runs

python -c "import bert_score" 2>/dev/null || pip install -q bert-score

is_done() {
  local json="$1"
  local log="${json%.json}.log"
  [[ -f "$json" ]] && [[ -f "$log" ]] && grep -q "RQ3_COMPARISON_DONE" "$log"
}

latest_json() {
  ls -t "$OUT"/rq3_pr_all_*.json 2>/dev/null | head -1 || true
}

echo "RQ3 launcher models=$MODELS GPU=${CUDA_VISIBLE_DEVICES:-unset}"
python -u rq3_report_comparison.py --prepare --dry-run

LATEST="$(latest_json)"
RESUME_ARGS=()
if [[ -n "${LATEST}" ]]; then
  if is_done "$LATEST"; then
    echo "===== SKIP RQ3 (already complete: $LATEST) ====="
    echo "RQ3_ALL_DONE"
    exit 0
  fi
  echo "===== RESUME RQ3 from $LATEST ====="
  RESUME_ARGS=(--resume-from "$LATEST")
fi

python -u rq3_report_comparison.py \
  --models $MODELS \
  --out-dir "$OUT" \
  "${RESUME_ARGS[@]}"
echo "RQ3_ALL_DONE"
