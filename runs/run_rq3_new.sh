#!/usr/bin/env bash
# RQ3 PR-Reports for the FIVE NEW VLMs. Does NOT start itself.
set -euo pipefail
cd /home/s222393187/Dental
source activate_env.sh
OUT="${OUT:-Results/rq3_reports_new}"
MODELS="${MODELS:-medgemma qwen25_vl llava_onevision llava_rad}"
mkdir -p "$OUT" runs
python -c "import bert_score" 2>/dev/null || pip install -q bert-score
echo "RQ3(new) models=$MODELS out=$OUT GPU=${CUDA_VISIBLE_DEVICES:-unset}"
python -u rq3_report_comparison.py --prepare --dry-run
python -u rq3_report_comparison.py --models $MODELS --out-dir "$OUT"
echo "RQ3_NEW_ALL_DONE"
