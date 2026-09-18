#!/usr/bin/env bash
# VM2 -> DENTEX QED, all five new VLMs.
set -euo pipefail; cd /home/s222393187/Dental
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
nohup bash runs/run_dentex_new.sh > runs/vm2_dentex_new_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > runs/VM2.pid; echo "VM2 DENTEX(new) started pid=$(cat runs/VM2.pid)"
