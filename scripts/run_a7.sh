#!/usr/bin/env bash
# =============================================================================
# run_a7.sh — ULIP-2 shape view-count sweep (A7), isolated, using the partial-
# view gallery cache force-loaded from Drive (the raw *_partial.npz are not on
# this machine). $1 = the cache-download pid to wait on (optional).
#
# The cache holds all 42 per-view embeddings per CAD; shape_agg_views pools the
# first N at score time, so one 42-view cache serves V8/16/32/42 with no
# re-encode. V16 (= E1_shape_only) must reproduce 0.5256 — that is the validation
# that the force-loaded cache is the right one (ULIP-2 colored ViT-g).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
DL_PID="${1:-}"
LOG=logs/run_a7.log
ts(){ date -Is; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
CACHE=object_images/shrec18_v2/.ulip_partial_cache_c3b88090d599c522.pt

if [ -n "$DL_PID" ]; then
  log "waiting for partial-cache download pid $DL_PID ..."
  while kill -0 "$DL_PID" 2>/dev/null; do sleep 15; done
fi
SZ=$(stat -c%s "$CACHE" 2>/dev/null || echo 0)
log "partial cache present: $SZ bytes"
if [ "$SZ" -lt 700000000 ]; then log "ABORT: cache too small ($SZ) — download incomplete"; exit 3; fi

OUT=object_retrieval/results_shrec18_v2_stage1_mean_a7
mkdir -p "$OUT/_cache"
S1COMMON="--data-root eval/datasets/shrec18/shrec18_full \
  --images-dir object_images/shrec18_v2 \
  --desc-file object_database/shrec18_v2/descriptions_attributes.json"
ARMS="E1_shape_only,A7_shape_only_V8,A7_shape_only_V16,A7_shape_only_V32,A7_shape_only_V42"

log "running A7 (force-loaded partial cache; shape passes computed fresh) ..."
docker compose run --rm -e SHREC_FORCE_PARTIAL_CACHE="/app/$CACHE" oscar bash -lc \
  "cd /app && PYTHONHASHSEED=0 SHREC_DINO_POOLING=mean \
   python3 -u experiments/experiment1_shrec18_stage1.py \
   --ablations $ARMS $S1COMMON --results-root $OUT --resume --allow-partial-gallery" \
  > logs/stage1_a7.log 2>&1
rc=$?; log "A7 experiment rc=$rc"

log "sanity — did it use the partial cache (not full-mesh)?"
grep -iE "force-loaded|Fallback auf full|Keine partial|full mesh" logs/stage1_a7.log | head -3 | tee -a "$LOG"
log "A7 view sweep (V16 must == 0.5256):"
grep -E "^(E1_shape_only|A7_shape_only)" "$OUT/stage1_summary.csv" 2>/dev/null \
  | awk -F, '{printf "  %-22s nDCG=%s\n",$1,$5}' | tee -a "$LOG"
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy "$OUT" "$REMOTE/$OUT" --transfers 16 --checkers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true
log "===== A7 DONE (rc=$rc) ====="
