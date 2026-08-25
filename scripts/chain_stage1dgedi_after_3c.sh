#!/usr/bin/env bash
# =============================================================================
# chain_stage1dgedi_after_3c.sh — run the Stage-1 dGeDi geometry arm AFTER the
# Stage-3 3c run releases the GPU (no assistant in the loop).
#
# The Stage-1 dGeDi arm was skipped by the orchestrator (health-gate bug, now
# fixed). It is independent of 3c, but must not contend for the GPU, so it
# waits for the 3c runner (pid $1 = the chain_stage3_after_reruns.sh process
# that is blocking on run_stage3_3c.sh) to exit, then launches
# run_stage1_dgedi.sh (which is itself smoke-gated on real dGeDi registration).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
WAITPID="${1:?need the 3c runner pid to wait on}"
LOG=logs/chain_stage1dgedi.log
ts(){ date -Is; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

log "armed: will run Stage-1 dGeDi arm after 3c runner pid $WAITPID exits"
while kill -0 "$WAITPID" 2>/dev/null; do sleep 60; done
log "3c runner pid $WAITPID exited (GPU released) -> launching run_stage1_dgedi.sh"
bash scripts/run_stage1_dgedi.sh
rc=$?
log "run_stage1_dgedi.sh exited rc=$rc"
log "chainer finished"
