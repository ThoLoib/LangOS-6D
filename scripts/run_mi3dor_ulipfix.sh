#!/usr/bin/env bash
# =============================================================================
# run_mi3dor_ulipfix.sh — re-run Stage-2 MI3DOR with the CAD meshes present, so
# the ULIP-2 shape channel is no longer empty.
#
# The prior fixed-weights run logged "[mi3dor] Using 0 CAD meshes after category
# filter": object_database/MI3DOR/model/test/ was absent, so the shape gallery
# (id source = mesh glob) was empty and S_shape contributed nothing — the
# reported ~0.699 fusion was CLIP+DINO only, never the intended 3-way.  With the
# meshes downloaded, the gallery repopulates and (gallery ULIP/DINO caches are
# already local) only the ULIP query images are encoded.  New output dir; the
# ULIP=0 run is left intact for comparison.  Same command as the fixedw run.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_mi3dor_ulipfix.log
ts(){ date -Is; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
S2OUT=results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix

log "===== MI3DOR CAD-fix rerun START -> object_retrieval/$S2OUT ====="
NMESH=$(python3 -c "import glob;print(len(glob.glob('object_database/MI3DOR/model/test/*/*.obj')))")
log "CAD meshes matching test/*/*.obj: $NMESH (was 0 in the ULIP=0 run)"
if [ "$NMESH" -lt 500 ]; then
  log "ABORT: only $NMESH CAD meshes — download looks incomplete (expected a few thousand)"; exit 3
fi

docker compose run --rm oscar bash -lc \
  "cd /app/object_retrieval && PYTHONHASHSEED=0 MI3DOR_MODES=partial \
   MI3DOR_RESULT_FOLDER=$S2OUT MI3DOR_DINO_POOLING=mean \
   python3 -u retrieval_mi3dor_eval_oscarplus.py" > logs/stage2_mean_ulipfix.log 2>&1
rc=$?; log "mi3dor eval rc=$rc"

log "sanity — CAD-mesh count + any headline rows:"
grep -iE "Using [0-9]+ CAD meshes|gallery:|E7|ULIP|NN_|FT|fus" logs/stage2_mean_ulipfix.log | tail -25 | tee -a "$LOG"
"$RC" copy "object_retrieval/$S2OUT" "$REMOTE/object_retrieval/$S2OUT" \
  --transfers 16 --checkers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true
log "synced object_retrieval/$S2OUT"
log "===== MI3DOR CAD-fix rerun DONE (rc=$rc) ====="
