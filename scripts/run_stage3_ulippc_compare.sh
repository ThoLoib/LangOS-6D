#!/usr/bin/env bash
# =============================================================================
# Stage-3 comparison: ULIP-pc full-fusion, WITHOUT vs WITH co-scaled dGeDi.
# Two configs, each Stage 3a AND 3b, WITH pose (3b is pose-defined -> D_sym).
#
#   Run A  results_bop_stage3_ulippc         ULIP-pc, full fusion, no geometry
#   Run B  results_bop_stage3_ulippc_dgedi   + co-scaled dGeDi E2_both
#                                             (both_borda = RANSAC fitness +
#                                              trimmed chamfer d_ransac), K=20
#
# ULIP-pc mode  = default ULIP gallery (NO --uni3d) + partial pc-query
#                 (--pc-query: driver back-projects depth+GT-mask+RGB -> cloud,
#                  ULIP encode_pointcloud). topk_softmax view-agg unchanged.
# Full fusion   = CLIP-text 0.3 + DINOv2 0.4 + shape 0.3 (config default).
# dGeDi         = HTTP service :5061, candidate-diameter co-scaling (verified
#                 1316 diameters loaded), self-normalized gallery descriptors.
#
# Output names are BARE (no object_retrieval/ prefix): the driver runs from
# /app/object_retrieval, so a prefixed name double-nests (prior bug).
# results_* is gitignored. Per-dataset result files are written as each
# dataset finishes, so a crash resumes at dataset granularity.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
ts() { date -Is; }

echo "[$(ts)] ensuring services (dgedi, foundationpose) up"
docker compose up -d dgedi foundationpose >/dev/null 2>&1

run_cfg() {                         # $1 = output dir   $2 = extra flags
  local OUT="$1"; local EXTRA="$2"
  for MODE in 3a 3b; do
    echo "[$(ts)] ===== $OUT  mode=$MODE  flags='$EXTRA' ====="
    docker compose run --rm oscar bash -lc \
      "cd /app/object_retrieval && python3 -u eval_bop_pose.py \
         --datasets all --mode $MODE --pose --pc-query $EXTRA --output $OUT" \
      || echo "[$(ts)] WARN: $OUT mode=$MODE exited non-zero (continuing)"
  done
}

echo "[$(ts)] ##### RUN A: ULIP-pc, no geometry #####"
run_cfg results_bop_stage3_ulippc ""

echo "[$(ts)] ##### RUN B: ULIP-pc + co-scaled dGeDi (both_borda, K=20) #####"
run_cfg results_bop_stage3_ulippc_dgedi "--dgedi --dgedi-top-k 20"

echo "[$(ts)] ===== ALL DONE ====="
