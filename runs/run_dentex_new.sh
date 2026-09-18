#!/usr/bin/env bash
# DENTEX QED prompting matrix for the FIVE NEW VLMs. Does NOT start itself.
set -euo pipefail
cd /home/s222393187/Dental
source activate_env.sh
OUT="${OUT:-Results/dentex_comparison_new}"
MODELS="${MODELS:-medgemma qwen25_vl llava_onevision llava_rad med_flamingo}"
SPLIT="${SPLIT:-qed}"
STRATEGIES="${STRATEGIES:-zero_shot few_shot cot}"
mkdir -p "$OUT" runs
echo "DENTEX(new) split=$SPLIT models=$MODELS out=$OUT GPU=${CUDA_VISIBLE_DEVICES:-unset}"
python -u dentex_vlm_comparison.py --split "$SPLIT" --strategy zero_shot --dry-run
for S in $STRATEGIES; do
  echo "===== START DENTEX(new) strategy=$S ====="
  python -u dentex_vlm_comparison.py --split "$SPLIT" --cases 0 --strategy "$S" \
    --models $MODELS --out-dir "$OUT" --tag "${S}_${SPLIT}_new"
done
echo "DENTEX_NEW_ALL_DONE"
