#!/usr/bin/env bash
# =============================================================================
# run_stage1_siglip.sh — re-run the Stage-1 E4 SigLIP appearance arm FAIRLY,
# after the pooler_output fix (commit f50a844f). The old E4_siglip=0.5245 used
# SigLIP's degenerate patch-0 token (no CLS token exists); the fix routes it
# through SigLIP's MAP-head pooler_output.
#
# Forces ONLY the SigLIP appearance channel to recompute: copies the reusable
# Tier-1 score caches (base = CLIP+DINO+ULIP, ULIP, Uni3D) from the mean-only
# run but NOT scores_siglip.pt, so the siglip pass re-encodes the gallery+queries
# with the fixed encoder (fresh _map-keyed .siglip_cache). Writes a NEW dir; the
# mean-only result is untouched. Runs E4_siglip + E1c_full_fusion (DINOv2 ref).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_stage1_siglip.log
ts(){ date -Is; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
sync_dir(){ "$RC" copy "$1" "$REMOTE/$1" --transfers 16 --checkers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true; log "synced $1"; }

S1COMMON="--data-root eval/datasets/shrec18/shrec18_full \
  --images-dir object_images/shrec18_v2 \
  --desc-file object_database/shrec18_v2/descriptions_attributes.json"
SRC=object_retrieval/results_shrec18_v2_stage1_mean_mean_only
OUT=object_retrieval/results_shrec18_v2_stage1_mean_siglipfix

log "===== run_stage1_siglip START ====="
mkdir -p "$OUT/_cache"
# Reusable Tier-1 caches — copy everything EXCEPT scores_siglip.pt (force fair recompute).
cp -n "$SRC/_cache/scores_base.pt"      "$OUT/_cache/" 2>/dev/null || true
cp -n "$SRC/_cache/scores_ulip_"*.pt    "$OUT/_cache/" 2>/dev/null || true
cp -n "$SRC/_cache/scores_uni3d.pt"     "$OUT/_cache/" 2>/dev/null || true
[ -f "$OUT/_cache/scores_siglip.pt" ] && { log "WARN removing stale scores_siglip.pt"; rm -f "$OUT/_cache/scores_siglip.pt"; }

log "run E4_siglip (fresh MAP-pooled SigLIP) + E1c_full_fusion (DINOv2 ref)"
docker compose run --rm oscar bash -lc \
  "cd /app && PYTHONHASHSEED=0 SHREC_DINO_POOLING=mean \
   python3 -u experiments/experiment1_shrec18_stage1.py \
   --ablations E4_siglip E1c_full_fusion $S1COMMON \
   --results-root $OUT --resume --allow-partial-gallery" > logs/stage1_siglipfix.log 2>&1
rc=$?; if [ $rc -ne 0 ]; then log "HARD FAIL: SigLIP rerun exited $rc"; exit "$rc"; fi

log "results (new MAP-pooled SigLIP vs old patch-0 SigLIP 0.5245, DINOv2 0.5889):"
grep -E "^E4_siglip,|^E1c_full_fusion," "$OUT/stage1_summary.csv" 2>/dev/null | tee -a "$LOG"
sync_dir "$OUT"
log "===== run_stage1_siglip DONE -> $OUT ====="
