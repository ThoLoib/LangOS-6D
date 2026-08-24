#!/usr/bin/env bash
# =============================================================================
# run_stage3_3c.sh — Stage-3 "3c" next-best-non-GT diagnostic.
#
# Reuses the stored 3a_cross ranking (gallery = G_proxy ∪ all target CADs) and
# poses, per query, the highest-ranked candidate that is NOT the exact target —
# the best available stand-in from the richer 3a gallery. D_posed + paired Delta
# (vs the exact-CAD gt benchmark) + provenance (real target CAD vs G_proxy),
# decomposing 3b's substitution cost into "gallery too sparse" vs "substitution
# inherently lossy". Retrieval is free (reused); only FoundationPose + D_sym run.
#
# Ordering: launched AFTER Stage 1+2 (run_all_reruns.sh) so it never contends
# for the GPU with the reruns. Needs the foundationpose service + the on-disk
# 3a_cross rankings and gt records (already present in results_bop_stage3_v2).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_stage3_3c.log
ts(){ date -Is; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
freeG(){ df --output=avail -BG / | tail -1 | tr -dc '0-9'; }
guard(){ local f; f=$(freeG); if [ "${f:-0}" -lt 60 ]; then log "HARD FAIL: disk ${f}G < 60G — aborting"; exit 3; fi; log "disk ${f}G free — ok"; }
sync_dir(){ "$RC" copy "$1" "$REMOTE/$1" --transfers 16 --checkers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true; log "synced $1"; }
wait_healthy(){ for _ in $(seq 1 120); do
  [ "$(docker inspect --format '{{.State.Health.Status}}' "$(docker compose ps -q "$1" 2>/dev/null)" 2>/dev/null)" = healthy ] && return 0
  sleep 5; done; return 1; }

FROM3A=results_bop_stage3_v2/3a_cross
GT=results_bop_stage3_v2/gt/combined_gt.json
OUTROOT=results_bop_stage3_v2
COMMON="--mode 3c --from-3a $FROM3A --gt-records $GT"

log "===== run_stage3_3c START ====="; guard
[ -f "object_retrieval/$GT" ] || { log "HARD FAIL: gt records $GT missing"; exit 2; }
[ -d "object_retrieval/$FROM3A/ycbv_stage3a" ] || { log "HARD FAIL: 3a rankings $FROM3A missing"; exit 2; }

log "bring up foundationpose service"
docker compose up -d foundationpose >/dev/null 2>&1
if ! wait_healthy foundationpose; then log "HARD FAIL: foundationpose not healthy"; exit 4; fi

# --- SMOKE GATE: 5 ycbv instances, must produce finite D_sym on >=1 ---
SMOKE=$OUTROOT/3c_smoke
log "3c smoke (ycbv, 5 targets) -> $SMOKE"
docker compose run --rm oscar bash -lc \
  "cd /app/object_retrieval && PYTHONHASHSEED=0 python3 -u eval_bop_pose.py \
   $COMMON --datasets ycbv --max-targets 5 --output $SMOKE" \
  > logs/stage3_3c_smoke.log 2>&1 || true
GATE=$(python3 -c "
import json,os
p='object_retrieval/$SMOKE/combined_stage3c.json'
if not os.path.isfile(p): print('FAIL nofile'); raise SystemExit
d=json.load(open(p)); ds=d.get('dsym',{})
n=ds.get('n_estimated',0); mean=ds.get('d_sym_mean')
ok = n>=1 and mean is not None and mean==mean and mean>0
print(('PASS' if ok else 'FAIL'), 'n_estimated=%s d_sym_mean=%s'%(n,mean))
")
log "3c smoke gate: $GATE"
case "$GATE" in PASS*) ;; *) log "HARD FAIL: 3c smoke gate failed"; exit 5;; esac

# --- FULL 3c run over all three datasets ---
guard
OUT=$OUTROOT/3c_cross
log "3c full run (all datasets) -> $OUT"
docker compose run --rm oscar bash -lc \
  "cd /app/object_retrieval && PYTHONHASHSEED=0 python3 -u eval_bop_pose.py \
   $COMMON --datasets all --output $OUT" \
  > logs/stage3_3c_full.log 2>&1
rc=$?; if [ $rc -ne 0 ]; then log "HARD FAIL: 3c full run exited $rc"; exit "$rc"; fi
sync_dir "object_retrieval/$OUT"
log "===== run_stage3_3c DONE -> $OUT ====="
