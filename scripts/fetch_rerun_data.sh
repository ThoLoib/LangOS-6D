#!/usr/bin/env bash
# =============================================================================
# fetch_rerun_data.sh — render-only downloads for the Stage-1/2 reruns.
#
# Pulls ONLY what the reruns actually read that isn't local:
#   - MI3DOR gallery renders  (*.png)  -> rebuild the mean-DINO gallery cache
#   - MI3DOR query images              -> compute query embeddings (uncached)
#   - shrec18_v2 gallery renders (*.png) -> rebuild mean-DINO (queries = same set)
# Skips partials (ULIP gallery cache already on Drive), CADs, CLS-DINO/SigLIP.
# Small gallery caches (clip + ulip .pt) are fetched separately in the rerun
# phase after verifying their exact paths. Disk-guarded (halt if <100G free).
#
# GPU-free: safe to run alongside 3b_cross_geo. Writes to NEW dataset dirs, no
# conflict with the BOP galleries the running eval reads.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
RC="${RCLONE:-$HOME/apps/rclone/rclone}"
REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
ts(){ date -Is; }; log(){ echo "[$(ts)] $*"; }
freeG(){ df --output=avail -BG / | tail -1 | tr -dc '0-9'; }
guard(){ local f; f=$(freeG); if [ "${f:-0}" -lt 100 ]; then
  log "HARD FAIL: disk ${f}G < 100G — aborting"; exit 3; fi; log "disk ${f}G free — ok"; }

pull(){ # src  dst  [extra rclone args...]
  local src="$1" dst="$2"; shift 2
  log "pull $src -> $dst"
  "$RC" copy "$REMOTE/$src" "$dst" --transfers 16 --checkers 16 \
    --stats-one-line --stats 60s "$@" 2>&1 | tail -4
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then log "HARD FAIL: rclone exit $rc on $src"; exit "$rc"; fi
  guard
}

guard
pull "object_images/MI3DOR"            object_images/MI3DOR            --include "*.png"
pull "eval/datasets/mi3dor/image/test" eval/datasets/mi3dor/image/test
pull "object_images/shrec18_v2"        object_images/shrec18_v2        --include "*.png"
log "===== fetch_rerun_data DONE (renders + MI3DOR queries) ====="
log "MI3DOR renders: $(find object_images/MI3DOR -name '*.png' 2>/dev/null | wc -l) png"
log "MI3DOR queries: $(find eval/datasets/mi3dor/image/test -name '*.png' -o -name '*.jpg' 2>/dev/null | wc -l) imgs"
log "shrec renders:  $(find object_images/shrec18_v2 -name '*.png' 2>/dev/null | wc -l) png"
