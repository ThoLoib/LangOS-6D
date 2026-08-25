#!/usr/bin/env bash
# =============================================================================
# run_stage1_dgedi.sh — the Stage-1 dGeDi geometry arm ONLY (the arm that the
# 2026-08-24 orchestrator run had to skip: it fell back to mean-only because a
# health-gate bug probed the GeDi service instead of dGeDi. Fixed in
# experiment1_shrec18_stage1.py::gedi_available; this re-runs just the geometry
# arm so Stage 1's mean+dGeDi cells get filled).
#
# Gated: a fresh 20-query dGeDi smoke must actually REGISTER (>=30% pairs) —
# this is the real test of the SHREC scale-invariance handling, which never ran
# before because geometry was skipped. Only on PASS does the full K=50/20/5 run
# proceed (K=20/5 derived from the K=50 geometry cache). On FAIL it stops and
# leaves the failure for review (never caches bogus geometry).
#
# Writes to NEW dirs (results_shrec18_v2_stage1_mean_dgedi_k{50,20,5}); the
# mean-only Stage-1 result already on disk is untouched.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_stage1_dgedi.log
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

log "===== run_stage1_dgedi START ====="; guard

# Ensure the dgedi service is up on the SHREC gallery.
log "ensure dgedi on the SHREC gallery (.dgedi_gallery_shrec)"
DGEDI_CACHE_DIR=.dgedi_gallery_shrec docker compose up -d dgedi >/dev/null 2>&1
if ! wait_healthy dgedi; then log "HARD FAIL: dgedi not healthy"; exit 4; fi

# --- dGeDi SMOKE GATE (real geometry now that the health gate is fixed) ---
log "dGeDi smoke (E2_both K=5, 20 queries) — fresh geometry"
rm -f "$SMOKE/_cache/geometry_scores.jsonl"
docker compose run --rm oscar bash -lc \
  "cd /app && PYTHONHASHSEED=0 SHREC_DINO_POOLING=mean STAGE1_GEOMETRY_BACKEND=dgedi \
   python3 -u experiments/experiment1_shrec18_stage1.py --ablations E2_both \
   --limit-queries 20 --with-geometry --geom-k 5 $S1COMMON \
   --results-root $SMOKE --resume --allow-partial-gallery" > logs/stage1_dgedi_smoke.log 2>&1 || true
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
case "$GATE" in
  PASS*) ;;
  *) log "HARD FAIL: dGeDi smoke still fails (scale/registration) — geometry NOT run. Review logs/stage1_dgedi_smoke.log"; exit 5;;
esac

# --- FULL mean+dGeDi run, K sweep 50/20/5 (K=20/5 re-rank the K=50 cache) ---
for K in 50 20 5; do
  OUT="${S1OUT}_k${K}"; guard
  mkdir -p "$OUT/_cache"
  if [ "$K" != 50 ]; then
    cp -n "${S1OUT}_k50/_cache/"*.pt "$OUT/_cache/" 2>/dev/null || true
    cp -n "${S1OUT}_k50/_cache/geometry_scores.jsonl" "$OUT/_cache/" 2>/dev/null || true
    cp -n "${S1OUT}_k50/_cache/"*.json "$OUT/_cache/" 2>/dev/null || true
  else
    cp -n object_retrieval/results_shrec18_v2_stage1/_cache/scores_ulip_*.pt "$OUT/_cache/" 2>/dev/null || true
    cp -n object_retrieval/results_shrec18_v2_stage1/_cache/scores_siglip.pt "$OUT/_cache/" 2>/dev/null || true
    cp -n object_retrieval/results_shrec18_v2_stage1/_cache/scores_uni3d.pt "$OUT/_cache/" 2>/dev/null || true
  fi
  log "Stage-1 mean+dGeDi K=$K -> $OUT"
  docker compose run --rm oscar bash -lc \
    "cd /app && PYTHONHASHSEED=0 SHREC_DINO_POOLING=mean STAGE1_GEOMETRY_BACKEND=dgedi \
     python3 -u experiments/experiment1_shrec18_stage1.py --all --with-geometry \
     --geom-k $K $S1COMMON --results-root $OUT --resume --allow-partial-gallery" \
    > "logs/stage1_mean_dgedi_k${K}.log" 2>&1
  rc=$?; if [ $rc -ne 0 ]; then log "HARD FAIL: Stage-1 K=$K exited $rc"; exit "$rc"; fi
  sync_dir "$OUT"
done
log "===== run_stage1_dgedi DONE (K=50/20/5) ====="
