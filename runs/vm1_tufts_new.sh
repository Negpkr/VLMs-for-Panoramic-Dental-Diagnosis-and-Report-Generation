#!/usr/bin/env bash
# VM1 -> Tufts, all five new VLMs. Start ONLY when told:  bash runs/vm1_tufts_new.sh
set -euo pipefail; cd /home/s222393187/Dental
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
nohup bash runs/run_tufts_new.sh > runs/vm1_tufts_new_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > runs/VM1.pid; echo "VM1 Tufts(new) started pid=$(cat runs/VM1.pid)"
