#!/usr/bin/env bash
# =============================================================================
# Run C: best-of-{A,B} retrieval + dGeDi at its REPO config, K=5, 3a+3b, pose.
#
# dGeDi repo config = 6000 keypoints / 10k RANSAC iters / + ICP (demo.py),
# via `--dgedi --dgedi-repo --dgedi-top-k 5`. Purpose: document at full scale
# (with pose / D_sym) that the faithful dGeDi geometry re-rank does NOT help
# Stage-3 instance retrieval (our subset test: 6000 kp @ K=5 = 0.470 vs fused
# 0.600 on ycbv). Retrieval arm auto-picked = whichever of A (pc) / B (cross)
# had the higher combined 3a Recall@1.
#
# LAUNCH ONLY AFTER Run A + Run B (run_stage3_AB_pose.sh) have finished --
# it needs their combined_stage3a.json to pick the arm, and it shares the GPU.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
ts() { date -Is; }

A=object_retrieval/results_bop_stage3_ulippc/combined_stage3a.json
B=object_retrieval/results_bop_stage3_ulipcross/combined_stage3a.json
if [ ! -f "$A" ] || [ ! -f "$B" ]; then
  echo "[$(ts)] ERROR: Run A and/or Run B combined_stage3a.json missing -> run A/B first."
  exit 1
fi
BEST=$(python3 - "$A" "$B" <<'PY'
import json, sys
def r1(p):
    try: return json.load(open(p)).get("recall@1")
    except Exception: return None
a, b = r1(sys.argv[1]), r1(sys.argv[2])
print(f"pc {a} {b}" if (a is not None and (b is None or a >= b)) else f"cross {a} {b}")
PY
)
ARM=$(echo "$BEST" | awk '{print $1}')
echo "[$(ts)] A(pc) vs B(cross) combined 3a R@1: $(echo "$BEST" | awk '{print $2" vs "$3}') -> arm=$ARM"
FLAGS=""; [ "$ARM" = "pc" ] && FLAGS="--pc-query"
OUT="results_bop_stage3_runC_${ARM}_dgedirepo"

echo "[$(ts)] gedi down; recreate dgedi (reloads server code + diameters); foundationpose up"
docker compose stop gedi >/dev/null 2>&1 || true
docker compose up -d --force-recreate dgedi >/dev/null 2>&1
docker compose up -d foundationpose >/dev/null 2>&1
for _ in $(seq 1 60); do
  h=$(docker inspect --format '{{.State.Health.Status}}' "$(docker compose ps -q dgedi)" 2>/dev/null)
  [ "$h" = "healthy" ] && break; sleep 4
done
echo "[$(ts)] dgedi: ${h:-unknown}"

for MODE in 3a 3b; do
  echo "[$(ts)] ===== Run C  $OUT  mode=$MODE  (dGeDi repo 6000kp/10k/+ICP, K=5) ====="
  docker compose run --rm oscar bash -lc \
    "cd /app/object_retrieval && python3 -u eval_bop_pose.py \
       --datasets all --mode $MODE --pose $FLAGS \
       --dgedi --dgedi-repo --dgedi-top-k 5 --output $OUT" \
    || echo "[$(ts)] WARN: Run C mode=$MODE exited non-zero (continuing)"
done
echo "[$(ts)] ===== Run C DONE -> $OUT ====="
