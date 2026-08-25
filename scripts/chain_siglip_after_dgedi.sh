#!/usr/bin/env bash
# Run the fair SigLIP rerun AFTER the dGeDi Stage-1 run releases the GPU (no
# assistant in the loop). $1 = the run_stage1_dgedi.sh pid to wait on.
set -uo pipefail
cd "$(dirname "$0")/.."
WAITPID="${1:?need the dGeDi runner pid}"
LOG=logs/chain_siglip.log
ts(){ date -Is; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
log "armed: SigLIP rerun after dGeDi runner pid $WAITPID exits"
while kill -0 "$WAITPID" 2>/dev/null; do sleep 60; done
log "dGeDi runner pid $WAITPID exited (GPU released) -> launching run_stage1_siglip.sh"
bash scripts/run_stage1_siglip.sh
log "chain_siglip finished (rc=$?)"
