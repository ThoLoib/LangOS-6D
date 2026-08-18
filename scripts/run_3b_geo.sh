#!/usr/bin/env bash
# =============================================================================
# run_3b_geo.sh — the paired 3b(cross+geo) run.
#
# The main orchestrator (run_stage3.sh) runs GT -> 3b(cross, no-geo). This adds
# the geometry-verified twin: 3b with the dGeDi repo-config re-rank (top-5),
# same cross query, SAME --gt-records, so D_posed / Delta / F-score are directly
# comparable with/without geometry per dataset (motivated by 3a: geo helped
# retrieval on LM-O + T-LESS-cross, hurt YCB-V — does that carry to pose?).
#
# It waits for the main run to finish (so no oscar/FP/dgedi contention), reuses
# the already-up FP + dgedi services, writes to a NEW dir (3b_cross_geo), and
# syncs. Launch in the background; on exit the assistant is re-invoked to ping.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
ts(){ date -Is; }
log(){ echo "[$(ts)] $*"; }

OUTROOT="${OUTROOT:-results_bop_stage3_v2}"
REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
GEO="--dgedi --dgedi-repo --dgedi-top-k 5"
GTREC="/app/object_retrieval/$OUTROOT/gt/combined_gt.json"
DONE_MARK="Stage-3 v2 DONE"

wait_healthy(){ local h=""; for _ in $(seq 1 90); do
  h=$(docker inspect --format '{{.State.Health.Status}}' "$(docker compose ps -q "$1")" 2>/dev/null)
  [ "$h" = "healthy" ] && break; sleep 4; done; log "$1: ${h:-unknown}"; }

sync_now(){ "$HOME/apps/rclone/rclone" copy "object_retrieval/$OUTROOT" \
  "$REMOTE/object_retrieval/$OUTROOT" --transfers 16 --checkers 16 \
  --stats-one-line --stats 0 >>logs/rclone_stage3.log 2>&1 || true; }

# 1. wait for the main orchestrator to finish its GT + 3b(cross)
log "waiting for main run to reach '$DONE_MARK' ..."
until grep -qa "$DONE_MARK" logs/orchestrator_full.log 2>/dev/null; do sleep 120; done
log "main run done. Verifying GT records exist ..."
if [ ! -f "object_retrieval/$OUTROOT/gt/combined_gt.json" ]; then
  log "HARD FAIL: GT records missing at $OUTROOT/gt/combined_gt.json — aborting 3b+geo"; exit 2
fi

# 2. ensure FP + dgedi are up (no-recreate = leave a healthy one untouched)
log "ensuring foundationpose + dgedi are up"
docker compose up -d --no-recreate foundationpose dgedi >/dev/null 2>&1
wait_healthy foundationpose
wait_healthy dgedi

# 3. run 3b(cross+geo) into a NEW dir, paired against the same GT records
log "===== 3b cross+geo -> $OUTROOT/3b_cross_geo ====="
docker compose run --rm oscar bash -lc \
  "cd /app/object_retrieval && PYTHONHASHSEED=0 python3 -u eval_bop_pose.py \
   --datasets all --mode 3b --output $OUTROOT/3b_cross_geo \
   --gt-records $GTREC $GEO" 2>&1 | tee logs/stage3_3b_cross_geo.log
rc=${PIPESTATUS[0]}
sync_now
if [ "$rc" -ne 0 ]; then log "HARD FAIL: 3b+geo exited $rc"; exit "$rc"; fi
log "===== 3b cross+geo DONE (synced) ====="
