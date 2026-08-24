#!/usr/bin/env bash
# Phase-aware watchdog for run_all_reruns.sh. Exits (re-invoking the assistant)
# on a NEW milestone line, orchestrator exit, or disk <70G. The assistant pings
# the user and re-arms it. $1 = orchestrator pid.
cd "$(dirname "$0")/.."
ORCH="${1:?need orch pid}"
LOG=logs/run_all_reruns.log
freeG(){ df --output=avail -BG / | tail -1 | tr -dc '0-9'; }
mile(){ grep -cE "gate:|DONE\.|HARD FAIL|gate FAILED|run_all_reruns DONE" "$LOG" 2>/dev/null || echo 0; }
BASE=$(mile)
while true; do
  if ! kill -0 "$ORCH" 2>/dev/null; then echo "ORCH_GONE"; exit 0; fi
  cur=$(mile); if [ "${cur:-0}" -gt "${BASE:-0}" ]; then echo "MILESTONE"; exit 0; fi
  f=$(freeG); if [ "${f:-999}" -lt 70 ]; then echo "DISK_LOW ${f}G"; exit 0; fi
  sleep 60
done
