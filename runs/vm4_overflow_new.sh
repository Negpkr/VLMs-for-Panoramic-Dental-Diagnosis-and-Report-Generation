#!/usr/bin/env bash
# VM4 -> overflow / heavy model. RadFM (~14B) is the slowest; by default VM4 runs
# RadFM only across all three tasks so it does not bottleneck VM1-3. Override MODELS/TASK.
#   TASK=tufts MODELS=radfm bash runs/vm4_overflow_new.sh
set -euo pipefail; cd /home/s222393187/Dental
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODELS="${MODELS:-radfm}"
TASK="${TASK:-all}"
run_one(){ case "$1" in
  tufts) bash runs/run_tufts_new.sh;; dentex) bash runs/run_dentex_new.sh;; rq3) bash runs/run_rq3_new.sh;;
esac; }
{ if [[ "$TASK" == "all" ]]; then for x in tufts dentex rq3; do run_one "$x"; done; else run_one "$TASK"; fi; } \
  > runs/vm4_overflow_new_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > runs/VM4.pid; echo "VM4 overflow started pid=$(cat runs/VM4.pid) TASK=$TASK MODELS=$MODELS"
