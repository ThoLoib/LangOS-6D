#!/usr/bin/env bash
# =============================================================================
# run_all_reruns.sh — self-driving Stage-1 -> Stage-2 reruns (Stage-3 added later).
#
# Runs unattended, no assistant-in-the-loop between phases (the fix for the
# 2026-08-20 idle gap). Phases:
#   0. wait for the SHREC dGeDi gallery to finish building
#   1. STAGE 1: repoint dgedi to the SHREC gallery -> dGeDi SMOKE GATE ->
#        if PASS: full mean+dGeDi run at K=50, then derive K=20 and K=5 from the
#                 cached geometry_scores (nested top-K, cheap)
#        if FAIL: fall back to the validated mean-only run (no geometry) and flag
#   2. STAGE 2: MI3DOR fused-arm rerun (mean pooling, corrected 0.3/0.4/0.3
#        weights, partial shape source) into a NEW dir
# Disk-guarded, syncs each result dir to Drive, logs to logs/run_all_reruns.log.
# A watchdog (armed separately) re-pings the assistant on completion / halt.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_all_reruns.log
ts(){ date -Is; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
freeG(){ df --output=avail -BG / | tail -1 | tr -dc '0-9'; }
guard(){ local f; f=$(freeG); if [ "${f:-0}" -lt 60 ]; then log "HARD FAIL: disk ${f}G < 60G — aborting"; exit 3; fi; log "disk ${f}G free — ok"; }
sync_dir(){ "$RC" copy "$1" "$REMOTE/$1" --transfers 16 --checkers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true; log "synced $1"; }
wait_healthy(){ for _ in $(seq 1 120); do
  [ "$(docker inspect --format '{{.State.Health.Status}}' "$(docker compose ps -q "$1" 2>/dev/null)" 2>/dev/null)" = healthy ] && return 0
  sleep 5; done; return 1; }

S1COMMON="--data-root eval/datasets/shrec18/shrec18_full \
  --images-dir object_images/shrec18_v2 \
  --desc-file object_database/shrec18_v2/descriptions_attributes.json"
S1OUT=object_retrieval/results_shrec18_v2_stage1_mean_dgedi
SMOKE=object_retrieval/results_shrec18_v2_stage1_smoke

log "===== run_all_reruns START ====="; guard

# ---------------------------------------------------------------- Phase 0 ----
log "Phase 0: waiting for SHREC dGeDi gallery (3308 descriptors)"
until [ "$(ls object_retrieval/.dgedi_gallery_shrec/*.npz 2>/dev/null | wc -l)" -ge 3308 ]; do sleep 60; done
[ -f object_retrieval/.dgedi_gallery_shrec/diameters.json ] || { log "HARD FAIL: diameters.json missing"; exit 2; }
log "Phase 0 done: gallery = $(ls object_retrieval/.dgedi_gallery_shrec/*.npz | wc -l) descriptors"

# ---------------------------------------------------------------- Phase 1 ----
guard
log "Phase 1: repoint dgedi service to the SHREC gallery (.dgedi_gallery_shrec)"
DGEDI_CACHE_DIR=.dgedi_gallery_shrec docker compose up -d --force-recreate dgedi >/dev/null 2>&1
if ! wait_healthy dgedi; then log "HARD FAIL: dgedi not healthy after repoint"; exit 4; fi
NG=$(docker compose run --rm oscar bash -lc \
  "python3 -c \"import urllib.request,json;print(json.load(urllib.request.urlopen('http://dgedi:5061/health'))['n_gallery'])\"" 2>/dev/null | tr -dc '0-9')
log "dgedi /health n_gallery=${NG:-?} (want 3308)"

# --- dGeDi SMOKE GATE (E2_both, 20 queries) ---
log "Phase 1: dGeDi smoke (E2_both K=5, 20 queries)"
rm -f "$SMOKE/_cache/geometry_scores.jsonl"          # fresh dGeDi geometry, not stale GeDi
docker compose run --rm oscar bash -lc \
  "cd /app && PYTHONHASHSEED=0 SHREC_DINO_POOLING=mean STAGE1_GEOMETRY_BACKEND=dgedi \
   python3 -u experiments/experiment1_shrec18_stage1.py --ablations E2_both \
   --limit-queries 20 --with-geometry --geom-k 5 $S1COMMON \
   --results-root $SMOKE --resume --allow-partial-gallery" > logs/smoke_stage1_dgedi.log 2>&1 || true
GATE=$(python3 -c "
import json,os
p='$SMOKE/_cache/geometry_scores.jsonl'; n=ok=0
if os.path.isfile(p):
    for l in open(p):
        try: r=json.loads(l)
        except: continue
        n+=1
        if not r.get('failed') and (r.get('fitness') or 0)>0: ok+=1
frac = ok/n if n else 0.0
print('PASS' if (n>=40 and frac>=0.30) else 'FAIL', n, ok, round(frac,3))
")
log "dGeDi smoke gate: $GATE  (need >=40 pairs, >=30% registered)"
DGEDI_OK=0; case "$GATE" in PASS*) DGEDI_OK=1;; esac

if [ "$DGEDI_OK" = 1 ]; then
  log "Phase 1: dGeDi gate PASSED -> full mean+dGeDi run, K sweep 50/20/5"
  # K=50 computes the geometry (expensive); K=20/5 re-rank from its cache.
  for K in 50 20 5; do
    OUT="${S1OUT}_k${K}"; guard
    if [ "$K" != 50 ]; then
      mkdir -p "$OUT/_cache"
      cp -n "${S1OUT}_k50/_cache/"*.pt "$OUT/_cache/" 2>/dev/null || true
      cp -n "${S1OUT}_k50/_cache/geometry_scores.jsonl" "$OUT/_cache/" 2>/dev/null || true
      cp -n "${S1OUT}_k50/_cache/"*.json "$OUT/_cache/" 2>/dev/null || true
    else
      mkdir -p "$OUT/_cache"
      cp -n object_retrieval/results_shrec18_v2_stage1/_cache/scores_ulip_*.pt "$OUT/_cache/" 2>/dev/null || true
      cp -n object_retrieval/results_shrec18_v2_stage1/_cache/scores_siglip.pt "$OUT/_cache/" 2>/dev/null || true
      cp -n object_retrieval/results_shrec18_v2_stage1/_cache/scores_uni3d.pt "$OUT/_cache/" 2>/dev/null || true
    fi
    log "Phase 1: Stage-1 mean+dGeDi K=$K -> $OUT"
    docker compose run --rm oscar bash -lc \
      "cd /app && PYTHONHASHSEED=0 SHREC_DINO_POOLING=mean STAGE1_GEOMETRY_BACKEND=dgedi \
       python3 -u experiments/experiment1_shrec18_stage1.py --all --with-geometry \
       --geom-k $K $S1COMMON --results-root $OUT --resume --allow-partial-gallery" \
      > "logs/stage1_mean_dgedi_k${K}.log" 2>&1
    rc=$?; if [ $rc -ne 0 ]; then log "HARD FAIL: Stage-1 K=$K exited $rc"; exit "$rc"; fi
    sync_dir "$OUT"
  done
else
  log "Phase 1: dGeDi gate FAILED -> fallback: mean-only Stage-1 (no geometry). dGeDi flagged for manual fix."
  OUT="${S1OUT%_dgedi}_mean_only"; mkdir -p "$OUT/_cache"
  cp -n object_retrieval/results_shrec18_v2_stage1/_cache/scores_ulip_*.pt "$OUT/_cache/" 2>/dev/null || true
  cp -n object_retrieval/results_shrec18_v2_stage1/_cache/scores_siglip.pt "$OUT/_cache/" 2>/dev/null || true
  cp -n object_retrieval/results_shrec18_v2_stage1/_cache/scores_uni3d.pt "$OUT/_cache/" 2>/dev/null || true
  docker compose run --rm oscar bash -lc \
    "cd /app && PYTHONHASHSEED=0 SHREC_DINO_POOLING=mean \
     python3 -u experiments/experiment1_shrec18_stage1.py --all $S1COMMON \
     --results-root $OUT --resume --allow-partial-gallery" > logs/stage1_mean_only.log 2>&1
  rc=$?; if [ $rc -ne 0 ]; then log "HARD FAIL: Stage-1 mean-only exited $rc"; exit "$rc"; fi
  sync_dir "$OUT"
fi
log "Phase 1 (Stage 1) DONE."

# ---------------------------------------------------------------- Phase 2 ----
guard
S2OUT=results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw
log "Phase 2: Stage-2 MI3DOR fused-arm rerun (mean, partial, fixed weights) -> $S2OUT"
docker compose run --rm oscar bash -lc \
  "cd /app/object_retrieval && PYTHONHASHSEED=0 MI3DOR_MODES=partial \
   MI3DOR_RESULT_FOLDER=$S2OUT MI3DOR_DINO_POOLING=mean \
   python3 -u retrieval_mi3dor_eval_oscarplus.py" > logs/stage2_mean_fixedw.log 2>&1
rc=$?; if [ $rc -ne 0 ]; then log "HARD FAIL: Stage-2 exited $rc"; exit "$rc"; fi
sync_dir "object_retrieval/$S2OUT"
log "Phase 2 (Stage 2) DONE."

log "===== run_all_reruns DONE (Stage 1 + Stage 2). Stage 3 handled separately. ====="
