#!/usr/bin/env bash
# Watch the post-orchestrator phase (3c run -> dGeDi Stage-1 rerun). Exits
# (re-invoking the assistant) on a NEW milestone across the relevant logs, or
# disk <70G. Milestones: 3c DONE, dGeDi smoke gate result, dGeDi DONE, HARD FAIL.
cd "$(dirname "$0")/.."
freeG(){ df --output=avail -BG / | tail -1 | tr -dc '0-9'; }
mile(){ cat logs/run_stage3_3c.log logs/run_stage1_dgedi.log logs/chain_stage1dgedi.log 2>/dev/null \
  | grep -cE "run_stage3_3c DONE|smoke gate:|run_stage1_dgedi DONE|HARD FAIL|exited rc="; }
BASE=$(mile)
while true; do
  cur=$(mile); if [ "${cur:-0}" -gt "${BASE:-0}" ]; then echo "MILESTONE"; exit 0; fi
  f=$(freeG); if [ "${f:-999}" -lt 70 ]; then echo "DISK_LOW ${f}G"; exit 0; fi
  # stop if nothing is driving the pipeline anymore
  pgrep -f "run_stage3_3c.sh|chain_stage1dgedi_after_3c.sh|run_stage1_dgedi.sh" >/dev/null 2>&1 || { echo "ALL_DRIVERS_GONE"; exit 0; }
  sleep 60
done
