#!/usr/bin/env bash
# VM3 -> RQ3 PR-Reports, all five new VLMs.
set -euo pipefail; cd /home/s222393187/Dental
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
nohup bash runs/run_rq3_new.sh > runs/vm3_rq3_new_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > runs/VM3.pid; echo "VM3 RQ3(new) started pid=$(cat runs/VM3.pid)"
