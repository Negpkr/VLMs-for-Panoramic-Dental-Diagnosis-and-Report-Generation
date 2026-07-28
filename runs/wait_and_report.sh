#!/usr/bin/env bash
set -euo pipefail
cd /home/s222393187/Dental
source activate_env.sh >/dev/null
LOG=runs/comparison_launch_20260728_233321.log
PID=$(cat runs/COMPARISON.pid)
echo "[watchdog] waiting for PID $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "[watchdog] process ended; checking log"
if grep -q COMPARISON_DONE "$LOG"; then
  echo "[watchdog] success — generating report"
  python comparison_report.py
  echo "[watchdog] REPORT_DONE"
else
  echo "[watchdog] COMPARISON_DONE not found — check $LOG"
  tail -40 "$LOG"
fi
