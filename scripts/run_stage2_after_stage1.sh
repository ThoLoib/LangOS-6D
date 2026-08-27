#!/usr/bin/env bash
# =============================================================================
# run_stage2_after_stage1.sh — auto-runs the two Stage-2 additions once the
# Stage-1 folder (run_stage1_full.sh) finishes:
#   1. MI3DOR full-mesh arm (A4 transfer) at the corrected 42v+k5 config.
#   2. MI3DOR cross-mode fusion-weight sweep -> ternary heatmap CSV
#      (with a BASE self-check: FT@(0.3,0.4,0.3) must ~= 0.682, else it aborts).
# Both reuse the pinned MI3DOR driver config (42 views, k=5, mean, 0.3/0.4/0.3).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_stage2_after_stage1.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"
S2OUT=results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix

# --- 1. wait for the Stage-1 consolidated folder to finish -------------------
log "waiting for Stage-1 (run_stage1_full DONE) ..."
for i in $(seq 1 5760); do            # up to ~5h
  grep -q "run_stage1_full DONE" logs/run_stage1_full.log 2>/dev/null && break
  sleep 30
done
if grep -q "run_stage1_full DONE" logs/run_stage1_full.log 2>/dev/null; then
  log "Stage-1 finished — starting Stage-2 additions."
else
  log "WARN: Stage-1 not marked done after wait; proceeding (Stage-2 is independent of the Stage-1 caches)."
fi

# --- 2. Phase 1: MI3DOR full-mesh arm (A4 transfer) -------------------------
log ">>> Phase 1: MI3DOR full-mesh arm (MI3DOR_MODES=fullmesh) -> $S2OUT/fullmesh/"
docker compose run --rm oscar bash -lc \
  "cd /app/object_retrieval && PYTHONHASHSEED=0 MI3DOR_MODES=fullmesh \
   MI3DOR_RESULT_FOLDER=$S2OUT MI3DOR_DINO_POOLING=mean \
   python3 -u retrieval_mi3dor_eval_oscarplus.py" > logs/stage2_fullmesh.log 2>&1
log "    Phase 1 rc=$?"
grep -iE "gallery:|Using [0-9]+ CAD|NN_accuracy|FT_mean|clip_dino_ulip_full|ulip_only" logs/stage2_fullmesh.log | tail -14 | tee -a "$LOG"

# --- 3. Phase 2: cross-mode weight sweep (heatmap CSV) ----------------------
log ">>> Phase 2: MI3DOR cross-mode weight sweep (BASE self-check inside)"
docker compose run --rm oscar bash -lc \
  "cd /app/object_retrieval && PYTHONHASHSEED=0 MI3DOR_DINO_POOLING=mean \
   python3 -u mi3dor_weight_sweep.py" > logs/stage2_wsweep.log 2>&1
log "    Phase 2 rc=$?"
grep -iE "SELF-CHECK|ABORT|optimum|wrote .* points|cached .* queries" logs/stage2_wsweep.log | tail -8 | tee -a "$LOG"

# --- 4. sync both to Drive --------------------------------------------------
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy "object_retrieval/$S2OUT" "$REMOTE/object_retrieval/$S2OUT" --transfers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true
"$RC" copy "object_retrieval/results_mi3dor_wsweep" "$REMOTE/object_retrieval/results_mi3dor_wsweep" --transfers 8 --stats 0 >>logs/rclone_reruns.log 2>&1 || true
log "===== run_stage2_after_stage1 DONE ====="
