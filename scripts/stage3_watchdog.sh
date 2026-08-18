#!/usr/bin/env bash
# Watchdog for the Stage-3 v2 orchestrator (pid passed as $1). Polls every 5 min
# and EXITS (which re-invokes the assistant) only on an actionable event:
#   - orchestrator process exits (crash or normal DONE)
#   - disk free drops below 80 GB
#   - the FoundationPose /tmp mesh leak regresses (>800 .obj files)
# Otherwise it appends a heartbeat to logs/stage3_watchdog.log and keeps waiting.
cd "$(dirname "$0")/.."
ORCH="${1:-83348}"
LOG=logs/stage3_watchdog.log
ts(){ date -Is; }
echo "[$(ts)] watchdog start (orch pid=$ORCH)" >>"$LOG"
while true; do
  # 1. orchestrator gone?
  if ! kill -0 "$ORCH" 2>/dev/null; then
    echo "[$(ts)] EXIT reason=ORCH_GONE" >>"$LOG"; echo "ORCH_GONE"; exit 0
  fi
  # 2. disk pressure? (GB free on the main fs)
  freeG=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ -n "$freeG" ] && [ "$freeG" -lt 80 ]; then
    echo "[$(ts)] EXIT reason=DISK_LOW free=${freeG}G" >>"$LOG"; echo "DISK_LOW ${freeG}G"; exit 0
  fi
  # 3. FP /tmp leak regression (only if FP container is up)
  fp=$(docker compose ps -q foundationpose 2>/dev/null)
  tmpobj=0
  if [ -n "$fp" ]; then
    tmpobj=$(docker exec "$fp" sh -c 'ls /tmp/*.obj 2>/dev/null | wc -l' 2>/dev/null | tr -dc '0-9')
    tmpobj=${tmpobj:-0}
    if [ "$tmpobj" -gt 800 ]; then
      echo "[$(ts)] EXIT reason=TMP_LEAK n=${tmpobj}" >>"$LOG"; echo "TMP_LEAK ${tmpobj}"; exit 0
    fi
  fi
  phase=$(grep -aE "=====" logs/orchestrator_full.log 2>/dev/null | tail -1 | sed 's/.*===== //; s/ =====.*//')
  outs=$(ls -1 object_retrieval/results_bop_stage3_v2/ 2>/dev/null | tr '\n' ' ')
  echo "[$(ts)] alive free=${freeG}G fp_tmp=${tmpobj} phase='${phase}' outs='${outs}'" >>"$LOG"
  sleep 300
done
