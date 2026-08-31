#!/usr/bin/env bash
# =============================================================================
# run_stage3_fullmesh_gated.sh — 3a full-mesh (Pfadfix 2026-08-28) und danach
# 3b/3c fuer JEDEN nachgeholten Arm, der die bisher beste getestete Config
# schlaegt (ULIP-2 cross ohne Geometrie, R@1 0.482).
# Laeuft strikt nach dem Geometrie-Redo — nie zwei GPU-Jobs gleichzeitig.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/run_stage3_fullmesh.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"
OUT=results_bop_stage3_v2; GT=$OUT/gt/combined_gt.json; BASE=0.482

log "warte auf das Geometrie-Redo ..."
for i in $(seq 1 2880); do
  grep -q "run_stage3_geo_redo DONE" logs/run_stage3_geo_redo.log 2>/dev/null && break; sleep 30; done
log "Geometrie-Redo fertig — weiter."

s3(){ log ">>> $1"
  docker compose run --rm ${3:-} oscar bash -lc \
    "cd /app/object_retrieval && PYTHONHASHSEED=0 ${4:-} \
     python3 -u eval_bop_pose.py --datasets all --mode $2 --output $OUT/$1" \
    > "logs/stage3_$1.log" 2>&1; log "    $1 rc=$?"; }
r1(){ python3 -c "
import json,os
f='object_retrieval/$OUT/$1/combined_stage3a.json'
print('%.4f'%((json.load(open(f)).get('overall') or json.load(open(f))).get('recall@1',0)) if os.path.isfile(f) else '')" 2>/dev/null; }

# --- 3a full-mesh (mit Absorb-Kontrolle) ------------------------------------
s3 3a_fullmesh "3a --fullmesh"
grep -h "\[stage3\]\[fullmesh\]" logs/stage3_3a_fullmesh.log 2>/dev/null | tee -a "$LOG"

# --- Gate: welche nachgeholten Arme schlagen die bisher beste Config? -------
log "===== Gate gegen ULIP-2 cross ohne Geometrie (R@1 $BASE) ====="
WINNERS=""
for arm in 3a_fullmesh 3a_cross_geo_distance 3a_cross_geo_fitness 3a_cross_geo_borda \
           3a_pc_geo_distance 3a_pc_geo_fitness; do
  v=$(r1 "$arm"); [ -z "$v" ] && { log "  $arm: kein Ergebnis"; continue; }
  if awk -v x="$v" -v b="$BASE" 'BEGIN{exit !(x+0>b+0)}'; then
    log "  $arm: R@1=$v  > $BASE  -> Pose"; WINNERS="$WINNERS $arm"
  else log "  $arm: R@1=$v  <= $BASE"; fi
done

# --- 3b/3c fuer die Gewinner ------------------------------------------------
for arm in $WINNERS; do
  case "$arm" in
    3a_fullmesh)            F="--fullmesh";;
    3a_pc_geo_*)            F="--pc-query --dgedi --dgedi-repo --dgedi-top-k 5";;
    3a_cross_geo_*)         F="--dgedi --dgedi-repo --dgedi-top-k 5";;
  esac
  SIG=""; case "$arm" in *_distance) SIG=distance;; *_fitness) SIG=fitness;; *_borda) SIG=borda;; esac
  ENV=""; PRE=""; [ -n "$SIG" ] && { ENV="-e STAGE3_GEO_SIGNAL=$SIG"; PRE="STAGE3_GEO_SIGNAL=$SIG"; }
  n="${arm#3a_}"
  s3 "3b_$n" "3b $F --gt-records $GT" "$ENV" "$PRE"
  s3 "3c_$n" "3c $F --from-3a $OUT/$arm" "$ENV" "$PRE"
done
[ -z "$WINNERS" ] && log "kein Arm schlaegt $BASE -> keine zusaetzlichen Pose-Laeufe"

RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy "object_retrieval/$OUT" "$REMOTE/object_retrieval/$OUT" --transfers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true
log "===== run_stage3_fullmesh DONE ====="
