#!/usr/bin/env bash
# =============================================================================
# run_stage1_full.sh — ONE consolidated Stage-1 results folder at the corrected
# comparable config: 42 shape views (SHAPE_AGG_VIEWS=42) + ulip_view_topk=5,
# matching DINOv2 (42 views, k=5) and Stage-2/MI3DOR.
#
# The partial-view .npz are not on this machine, so each shape encoder's gallery
# is force-loaded from its Drive cache. The force-load takes ONE cache, and the
# three encoders have incompatible dims (ULIP-col 1280 / uni3d 1024 / xyz 512),
# so we run in phases — one force-cache per phase — all into the SAME dir with
# --resume. Final folder holds every non-alias arm.
#   $1 = pid of the already-running colored core run to wait on (optional).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
OUT=object_retrieval/results_shrec18_v2_stage1_42v_k5
IMG=object_images/shrec18_v2
COL="$IMG/.ulip_partial_cache_c3b88090d599c522.pt"   # ULIP-2 colored, 1280-d
XYZ="$IMG/.ulip_partial_cache_641102dfbaf4e90c.pt"   # ULIP-2 xyz,    512-d
UNI="$IMG/.ulip_partial_cache_eabcf9b9096553c9.pt"   # Uni3D,        1024-d
S1COMMON="--data-root eval/datasets/shrec18/shrec18_full --images-dir $IMG --desc-file object_database/shrec18_v2/descriptions_attributes.json"
LOG=logs/run_stage1_full.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"; mkdir -p "$OUT/_cache"

CORE_PID="${1:-}"
if [ -n "$CORE_PID" ]; then
  log "waiting for colored-core run (pid $CORE_PID) ..."
  while kill -0 "$CORE_PID" 2>/dev/null; do sleep 20; done
  log "colored-core run finished."
fi

# wait for the xyz + uni3d cache downloads
log "waiting for shape-cache downloads ..."
for i in $(seq 1 240); do
  grep -q "ALL_CACHE_DL_DONE" logs/dl_shape_caches.log 2>/dev/null && break
  sleep 15
done
for c in "$COL" "$XYZ" "$UNI"; do
  sz=$(stat -c%s "$c" 2>/dev/null || echo 0)
  log "cache $(basename "$c"): $sz bytes"
  [ "$sz" -lt 100000000 ] && { log "ABORT: $c missing/small"; exit 3; }
done

run_phase(){  # $1=label $2=force-cache $3=arms $4=extra-flags
  local label="$1" cache="$2" arms="$3" extra="${4:-}"
  log ">>> phase $label  (cache=$(basename "$cache") flags='$extra')"
  # STAGE1_GEOMETRY_BACKEND=dgedi: use the dGeDi HTTP service (the one that is
  # actually up + cross-stage-comparable) — the default "gedi" probes a legacy
  # service that isn't running and SILENTLY SKIPS the aligned geometry arms.
  # No-op for the non-geometry phases.
  docker compose run --rm -e SHREC_FORCE_PARTIAL_CACHE="/app/$cache" -e SHREC_DINO_POOLING=mean \
    -e STAGE1_GEOMETRY_BACKEND=dgedi oscar bash -lc \
    "cd /app && PYTHONHASHSEED=0 SHREC_DINO_POOLING=mean SHREC_FORCE_PARTIAL_CACHE=/app/$cache \
     STAGE1_GEOMETRY_BACKEND=dgedi \
     python3 -u experiments/experiment1_shrec18_stage1.py --ablations $arms $S1COMMON \
     --results-root $OUT --resume --allow-partial-gallery $extra" \
    > "logs/stage1_full_$label.log" 2>&1
  log "    phase $label rc=$? ; force-load: $(grep -c 'FORCE-loaded' logs/stage1_full_$label.log) ; skipped-geo: $(grep -c 'skipping GeDi-signal' logs/stage1_full_$label.log)"
}

# --- Phase A: everything ULIP-2-COLORED or no-shape (non-geometry) -----------
COLORED_ARMS="E1a_text_only,E1_view_only,E1_shape_only,E1b_text_view,E1_oscar_cascade,E1c_full_fusion,E1d_clip_pruned,\
E2b_fullmesh_shape_only,E4_siglip,E4_siglip_only,E6_rrf,E7_ulip2_cross_shape_only,\
O2_clip_threshold,O2_clip_threshold_cal,O2_visual_first,O4_V8,O4_V16,O4_V32,\
A2_view_only_V8,A2_view_only_V16,A2_view_only_V32,A2_view_only_V42,\
A7_shape_only_V8,A7_shape_only_V16,A7_shape_only_V32,A7_shape_only_V42,A7f_full_fusion_shape_V42"
run_phase A_colored "$COL" "$COLORED_ARMS"

# --- Phase B: Uni3D shape (A3 alternative) -----------------------------------
run_phase B_uni3d "$UNI" "E7_uni3d,E7_uni3d_shape_only"

# --- Phase C: XYZ-only ULIP-2 (A6 / O5) --------------------------------------
run_phase C_xyz "$XYZ" "O5_xyz_only,O5_xyz_shape_only"

# --- Phase D: geometry re-ranking (colored shape + dGeDi) --------------------
# Ensure the dGeDi service is up ON THE SHREC GALLERY (unit-diameter, scale-
# invariant handling) before the geometry arms — matches run_stage1_dgedi.sh.
log "ensuring dGeDi service on the SHREC gallery (.dgedi_gallery_shrec) ..."
DGEDI_CACHE_DIR=.dgedi_gallery_shrec docker compose up -d dgedi >/dev/null 2>&1
for _ in $(seq 1 60); do
  [ "$(docker inspect --format '{{.State.Health.Status}}' "$(docker compose ps -q dgedi 2>/dev/null)" 2>/dev/null)" = healthy ] && break
  sleep 5
done
log "dGeDi health: $(docker inspect --format '{{.State.Health.Status}}' "$(docker compose ps -q dgedi 2>/dev/null)" 2>/dev/null)"
run_phase D_geom "$COL" \
  "E2_fitness,E2_chamfer_unaligned,E2_chamfer_ransac,E2_chamfer_icp,E2_both,O1c_gedi_post_fusion,O1e_gedi_with_base" \
  "--with-geometry --geom-k 50"   # PIN K=50 (C2/BASE depth); without it the experiment
                                  # auto-picks K from the hit@K curve (came out K=100, 2x cost)

# --- Final: consolidated summary + Drive sync --------------------------------
log "===== consolidated Stage-1 grid (42 views, k=5) ====="
if [ -f "$OUT/stage1_summary.csv" ]; then
  awk -F, 'NR==1{next}{printf "  %-28s nDCG=%s\n",$1,$5}' "$OUT/stage1_summary.csv" | sort | tee -a "$LOG"
  n=$(awk -F, 'END{print NR-1}' "$OUT/stage1_summary.csv"); log "TOTAL arms in folder: $n"
fi
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy "$OUT" "$REMOTE/$OUT" --transfers 16 --checkers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true
log "===== run_stage1_full DONE -> $OUT ====="
