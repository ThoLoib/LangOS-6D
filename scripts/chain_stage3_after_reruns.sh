#!/usr/bin/env bash
# =============================================================================
# chain_stage3_after_reruns.sh — shell-level chainer so the full
# Stage 1 -> Stage 2 -> Stage 3(3c) sequence runs with NO assistant in the loop
# (the fix for the idle-gap failure mode).
#
# Waits for the run_all_reruns.sh orchestrator (pid $1) to exit, then:
#   * if it finished successfully ("run_all_reruns DONE" in its log) -> launch
#     run_stage3_3c.sh (Stage-3 next-best-non-GT).
#   * otherwise (hard fail / killed) -> do NOT run 3c; log and stop so the
#     failure is reviewed, not papered over.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
ORCH="${1:?need orchestrator pid}"
LOG=logs/chain_stage3.log
ts(){ date -Is; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

log "chainer armed on orchestrator pid $ORCH; will run Stage-3 3c after it succeeds"
while kill -0 "$ORCH" 2>/dev/null; do sleep 60; done
log "orchestrator pid $ORCH exited"

if grep -q "run_all_reruns DONE" logs/run_all_reruns.log 2>/dev/null; then
  log "Stage 1+2 succeeded -> launching run_stage3_3c.sh"
  bash scripts/run_stage3_3c.sh
  rc=$?
  log "run_stage3_3c.sh exited rc=$rc"
else
  log "Stage 1+2 did NOT report DONE (hard fail/killed) -> NOT running 3c; leaving for review"
fi
log "chainer finished"
