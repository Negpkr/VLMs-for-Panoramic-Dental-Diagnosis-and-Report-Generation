#!/usr/bin/env bash
# Sequential full matrix: Tufts 1000, then DENTEX qed 752 (755 on disk minus 3 held-out exemplars).
# GPU 1 only. Does not start itself.
#
# Frozen protocol on both:
#   pooled patients × 32 slots
#   parse findings / Missing teeth only (Total-count lines are not scored)
#   Table A headline = Macro F1
#   DENTEX Table B = four independent OvR + Macro-4 F1
#
# When you say run all:
#   cd /home/s222393187/Dental
#   nohup bash runs/run_all.sh > runs/all_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo $! > runs/ALL.pid
set -euo pipefail
cd /home/s222393187/Dental
source activate_env.sh
export CUDA_VISIBLE_DEVICES=1
export SPLIT="${SPLIT:-qed}"

echo "===== JOB 1/2 TUFTS full 1000 × 3 strategies  GPU=${CUDA_VISIBLE_DEVICES} ====="
bash runs/run_tufts_strategies.sh
echo "===== JOB 2/2 DENTEX split=$SPLIT × 3 strategies ====="
bash runs/run_dentex_strategies.sh
echo "ALL_JOBS_DONE"
