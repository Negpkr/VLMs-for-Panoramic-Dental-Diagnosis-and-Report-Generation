#!/usr/bin/env bash
# Live monitor for the full notebook run
echo "Watching /home/s222393187/Dental/runs/CURRENT_STATUS.txt (Ctrl+C to stop)"
echo "Full log: ls -t /home/s222393187/Dental/runs/full_run_*.log | head -1"
echo
while true; do
  clear
  echo "=== $(date) ==="
  echo "--- status ---"
  cat /home/s222393187/Dental/runs/CURRENT_STATUS.txt 2>/dev/null || echo "(no status yet)"
  echo
  PID=$(cat /home/s222393187/Dental/runs/CURRENT_RUN.pid 2>/dev/null)
  if [ -n "$PID" ] && ps -p "$PID" >/dev/null 2>&1; then
    echo "--- process ---"
    ps -p "$PID" -o pid,etime,pcpu,pmem,cmd
  else
    echo "--- process ---"
    echo "NOT RUNNING (finished or crashed). Check log for FINISHED/FATAL."
  fi
  echo
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
  echo
  echo "--- last 15 log lines ---"
  LOG=$(ls -t /home/s222393187/Dental/runs/full_run_*.log 2>/dev/null | head -1)
  [ -n "$LOG" ] && tail -15 "$LOG"
  sleep 5
done
