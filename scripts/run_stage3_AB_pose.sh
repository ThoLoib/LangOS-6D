#!/usr/bin/env bash
# =============================================================================
# Weekend Stage-3: Run A (ULIP-pc) then Run B (ULIP-cross / image query),
# both FULL FUSION (0.3/0.4/0.3), NO geometry, each Stage 3a AND 3b WITH pose.
#
#   Run A  results_bop_stage3_ulippc    shape arm = ULIP-2 pc-query (--pc-query)
#   Run B  results_bop_stage3_ulipcross shape arm = ULIP-2 image/cross (default)
#
# The ONLY difference is the shape-arm query modality (point cloud vs image),
# so this is the clean pc-vs-cross comparison at correct full-fusion weights.
# Neither run uses geometry, so dgedi/gedi stay DOWN (frees GPU for
# FoundationPose -> no OOM). Run C (best-of-{A,B} + dGeDi geometry) is set up
# SEPARATELY afterwards, once A vs B is decided.
#
# Bare --output names (driver runs from /app/object_retrieval). results_* is
# gitignored; per-dataset files written as each finishes -> resumable.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
ts() { date -Is; }

echo "[$(ts)] stopping gedi + dgedi (neither run uses geometry; frees GPU for FP)"
docker compose stop gedi dgedi >/dev/null 2>&1 || true
docker compose up -d foundationpose >/dev/null 2>&1

run_cfg() {                         # $1 = output dir   $2 = extra flags
  local OUT="$1"; local EXTRA="$2"
  for MODE in 3a 3b; do
    echo "[$(ts)] ===== $OUT  mode=$MODE  flags='$EXTRA' ====="
    docker compose run --rm oscar bash -lc \
      "cd /app/object_retrieval && python3 -u eval_bop_pose.py \
         --datasets all --mode $MODE --pose $EXTRA --output $OUT" \
      || echo "[$(ts)] WARN: $OUT mode=$MODE exited non-zero (continuing)"
  done
}

echo "[$(ts)] ##### RUN A: ULIP-pc, full fusion, pose #####"
run_cfg results_bop_stage3_ulippc "--pc-query"

echo "[$(ts)] ##### RUN B: ULIP-cross (image query), full fusion, pose #####"
run_cfg results_bop_stage3_ulipcross ""

echo "[$(ts)] ===== A + B DONE. Compare, then set up Run C (best + dGeDi geometry). ====="
python3 - <<'PY' || true
import json, os
def r1(p):
    try: return json.load(open(p)).get("recall@1")
    except Exception: return None
a=r1("object_retrieval/results_bop_stage3_ulippc/combined_stage3a.json")
b=r1("object_retrieval/results_bop_stage3_ulipcross/combined_stage3a.json")
print(f"[compare] combined 3a Recall@1 — pc(A)={a}  cross(B)={b}")
if a is not None and b is not None:
    print(f"[compare] best retrieval = {'pc (A)' if a>=b else 'cross (B)'}")
PY
