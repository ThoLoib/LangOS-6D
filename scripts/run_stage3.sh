#!/usr/bin/env bash
# =============================================================================
# run_stage3.sh — Stage-3 BOP evaluation orchestrator (revised concept 2026-08-17)
#
# Runs, in order, against the shared union gallery:
#   1. GT exact-CAD FoundationPose benchmark  (mode gt -> D_posed_gt)
#   2. Four 3a RETRIEVAL variants (no pose):
#        A  pc-query,   no geometry
#        A+ pc-query,   + dGeDi geometry (repo config, K=5)
#        B  cross-query, no geometry
#        B+ cross-query, + dGeDi geometry (repo config, K=5)
#   3. Pick the best 3a variant by combined Recall@1
#   4. 3b with that config (+ gt-records) -> D_posed + Delta = D_posed - D_posed_gt
#
# Memory hygiene (the last run filled the disk via 95k orphaned /tmp meshes in
# the FP container — fixed in estimater.py): gedi stays DOWN throughout; FP is
# force-recreated (drops any stale writable layer + reloads the estimater fix)
# and only up for gt/3b; dgedi only for the +geometry variants.
#
# Usage:
#   MAX=20 bash scripts/run_stage3.sh    # smoke (~20 targets/dataset)
#   bash scripts/run_stage3.sh           # full test_targets_bop19
#
# Env:
#   MAX      targets per dataset (0 = all). Default 0.
#   OUTROOT  results root. Default results_bop_stage3_v2 (smoke: _smoke suffix).
#   REMOTE   rclone remote for sync. Default gdrive:Masterthesis/OSCAR
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
ts() { date -Is; }

MAX="${MAX:-0}"
SUFFIX=""; [ "$MAX" != "0" ] && SUFFIX="_smoke"
OUTROOT="${OUTROOT:-results_bop_stage3_v2${SUFFIX}}"
REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
MT=""; [ "$MAX" != "0" ] && MT="--max-targets $MAX"
GEO="--dgedi --dgedi-repo --dgedi-top-k 5"
PYSEED="PYTHONHASHSEED=0"

echo "[$(ts)] Stage-3 v2 | MAX=$MAX | OUTROOT=$OUTROOT"

run_oscar() {   # run a python cmd inside the oscar container, log to file
  local log="$1"; shift
  docker compose run --rm oscar bash -lc \
    "cd /app/object_retrieval && $PYSEED python3 -u eval_bop_pose.py $*" \
    2>&1 | tee "logs/$log"
}

wait_healthy() {  # $1 = service
  local h="" ; for _ in $(seq 1 90); do
    h=$(docker inspect --format '{{.State.Health.Status}}' \
        "$(docker compose ps -q "$1")" 2>/dev/null)
    [ "$h" = "healthy" ] && break; sleep 4
  done
  echo "[$(ts)] $1: ${h:-unknown}"
}

# --- background gdrive sync (skips unchanged files; no duplicate uploads) -----
sync_now() { "$HOME/apps/rclone/rclone" copy "$OUTROOT" \
  "$REMOTE/object_retrieval/$OUTROOT" --transfers 16 --checkers 16 \
  --stats-one-line --stats 0 >>logs/rclone_stage3.log 2>&1 || true; }

echo "[$(ts)] gedi DOWN (never needed here)"
docker compose stop gedi >/dev/null 2>&1 || true

# ============================================================================
# PHASE 2 first (retrieval, no FP): 4x 3a variants. dgedi up for +geo variants.
# ============================================================================
echo "[$(ts)] recreate dgedi (reload server + reclaim writable layer)"
docker compose up -d --force-recreate dgedi >/dev/null 2>&1
wait_healthy dgedi
echo "[$(ts)] FP DOWN during 3a (retrieval only, saves GPU)"
docker compose stop foundationpose >/dev/null 2>&1 || true

declare -A A_R1
run_variant() {  # name  outdir  extra-flags
  local name="$1" out="$2"; shift 2
  echo "[$(ts)] ===== 3a variant $name -> $OUTROOT/$out ====="
  run_oscar "stage3_3a_${name}${SUFFIX}.log" \
    --datasets all --mode 3a $MT --output "$OUTROOT/$out" "$@"
  sync_now
  local f="object_retrieval/$OUTROOT/$out/combined_stage3a.json"
  A_R1[$name]=$(python3 -c "import json;print(json.load(open('$f')).get('recall@1',-1))" 2>/dev/null || echo -1)
  echo "[$(ts)] $name combined Recall@1 = ${A_R1[$name]}"
}

run_variant pc      3a_pc        --pc-query
run_variant pc_geo  3a_pc_geo    --pc-query $GEO
run_variant cross   3a_cross
run_variant cross_geo 3a_cross_geo $GEO

# --- pick best variant by combined Recall@1 ---
BEST=$(python3 - <<PY
best=None; bv=-1
for n,v in {"pc":${A_R1[pc]:--1},"pc_geo":${A_R1[pc_geo]:--1},"cross":${A_R1[cross]:--1},"cross_geo":${A_R1[cross_geo]:--1}}.items():
    if v>bv: bv,best=v,n
print(best)
PY
)
echo "[$(ts)] BEST 3a variant = $BEST  (R@1: pc=${A_R1[pc]} pc_geo=${A_R1[pc_geo]} cross=${A_R1[cross]} cross_geo=${A_R1[cross_geo]})"
BEST_FLAGS=""
case "$BEST" in
  pc)        BEST_FLAGS="--pc-query" ;;
  pc_geo)    BEST_FLAGS="--pc-query $GEO" ;;
  cross)     BEST_FLAGS="" ;;
  cross_geo) BEST_FLAGS="$GEO" ;;
esac

# ============================================================================
# PHASE 1: GT exact-CAD FoundationPose benchmark (needs FP; no gallery).
# ============================================================================
echo "[$(ts)] recreate foundationpose (reload estimater fix + drop 109GB layer)"
docker compose up -d --force-recreate foundationpose >/dev/null 2>&1
wait_healthy foundationpose
echo "[$(ts)] ===== GT benchmark -> $OUTROOT/gt ====="
run_oscar "stage3_gt${SUFFIX}.log" \
  --datasets all --mode gt $MT --output "$OUTROOT/gt"
sync_now
GTREC="/app/object_retrieval/$OUTROOT/gt/combined_gt.json"

# ============================================================================
# PHASE 3: 3b with the best 3a config, paired against the GT benchmark.
# ============================================================================
echo "[$(ts)] ===== 3b ($BEST) -> $OUTROOT/3b_${BEST} ====="
run_oscar "stage3_3b_${BEST}${SUFFIX}.log" \
  --datasets all --mode 3b $MT --output "$OUTROOT/3b_${BEST}" \
  --gt-records "$GTREC" $BEST_FLAGS
sync_now

echo "[$(ts)] ===== Stage-3 v2 DONE -> $OUTROOT (best 3a=$BEST) ====="
