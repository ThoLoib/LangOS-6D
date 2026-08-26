#!/usr/bin/env bash
# =============================================================================
# chain_isolated_mi3dor.sh — autonomous tail of the GPU queue.
#
# Order of the shared GPU: dGeDi (running) -> siglipfix -> [this] isolated GPU
# arms -> MI3DOR CAD-fix rerun.  This chainer:
#   Phase 0  waits until the siglipfix run finishes (GPU released).
#   Phase 1  runs the GPU-side isolated arms (siglip-only + ulip2-cross-only),
#            appending into the isolated results dir the CPU phase already began.
#   Phase 2  waits for the MI3DOR CAD download to finish, then re-runs Stage-2.
#
# $1 = the MI3DOR CAD-download pid to wait on before Phase 2 (optional).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
DL_PID="${1:-}"
SIGLOG=logs/run_stage1_siglip.log
LOG=logs/chain_isolated_mi3dor.log
ts(){ date -Is; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
log "armed: siglipfix-gate -> isolated GPU arms -> MI3DOR ulipfix (dl pid=${DL_PID:-none})"

# --- Phase 0: gate on siglipfix completion (proceed on DONE or HARD FAIL) -----
while true; do
  if [ -f "$SIGLOG" ] && grep -q "run_stage1_siglip DONE" "$SIGLOG"; then
    log "siglipfix DONE detected -> GPU free"; break; fi
  if [ -f "$SIGLOG" ] && grep -q "HARD FAIL" "$SIGLOG"; then
    log "siglipfix HARD FAIL -> proceeding anyway (siglip-only may recompute)"; break; fi
  sleep 60
done

# --- Phase 1: isolated GPU arms (siglip-only, ulip2-cross-only) ---------------
log "Phase 1: run_stage1_isolated.sh gpu"
bash scripts/run_stage1_isolated.sh gpu; log "isolated gpu rc=$?"

# --- Phase 2: MI3DOR CAD-fix rerun (needs the download complete) --------------
if [ -n "$DL_PID" ]; then
  log "Phase 2: waiting for MI3DOR CAD download pid $DL_PID"
  while kill -0 "$DL_PID" 2>/dev/null; do sleep 30; done
  log "download pid $DL_PID exited"
fi
log "Phase 2: run_mi3dor_ulipfix.sh"
bash scripts/run_mi3dor_ulipfix.sh; log "mi3dor ulipfix rc=$?"
log "chain_isolated_mi3dor DONE"
