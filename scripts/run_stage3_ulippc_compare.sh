#!/usr/bin/env bash
# =============================================================================
# Stage-3 comparison: ULIP-pc FULL FUSION, WITHOUT vs WITH co-scaled dGeDi.
# Two configs, each Stage 3a AND 3b, WITH pose (3b is pose-defined -> D_sym).
#
#   Run A  results_bop_stage3_ulippc         ULIP-pc, full fusion, no geometry
#   Run B  results_bop_stage3_ulippc_dgedi   same + co-scaled dGeDi (K=20)
#
# Full fusion = CLIP 0.3 + DINO 0.4 + shape 0.3 (now set explicitly in
# stage3_gallery._base_cfg; audit P0.1 — the old default was 0/0.5/0.5).
# ULIP-pc mode = default ULIP gallery (NO --uni3d) + --pc-query.
#
# GPU / OOM (learned the hard way): FoundationPose starves when other CUDA
# services co-reside on the 24 GB GPU. So:
#   * gedi is NEVER needed here -> stopped up front.
#   * dgedi is only needed for Run B -> kept DOWN during Run A, and
#     force-recreated before Run B so it reloads the corrected diameters.json
#     (audit P0.4 + 5.3: the service caches diameters at startup only).
#
# Output names are BARE (driver runs from /app/object_retrieval; a prefixed
# name double-nests). results_* is gitignored; per-dataset files are written as
# each dataset finishes -> resumable at dataset granularity.
#
# PREREQUISITE (one-time, run before this): regenerate diameters.json via
#   docker compose run --rm dgedi python3 /oscar/dgedi_service/compute_diameters.py \
#     --manifest .../manifest.json --out .../diameters.json
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
ts() { date -Is; }

echo "[$(ts)] stopping gedi (never used here; frees ~5 GB for FoundationPose)"
docker compose stop gedi >/dev/null 2>&1 || true

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

echo "[$(ts)] ##### RUN A: ULIP-pc full fusion, no geometry (dgedi DOWN) #####"
docker compose stop dgedi >/dev/null 2>&1 || true
docker compose up -d foundationpose >/dev/null 2>&1
run_cfg results_bop_stage3_ulippc ""

echo "[$(ts)] ##### RUN B: ULIP-pc full fusion + co-scaled dGeDi (K=20) #####"
echo "[$(ts)] recreating dgedi so it reloads the corrected diameters.json"
docker compose up -d --force-recreate dgedi >/dev/null 2>&1
docker compose up -d foundationpose >/dev/null 2>&1
# wait (<=3 min) for dgedi to finish loading gallery + diameters
for _ in $(seq 1 60); do
  docker compose logs dgedi 2>&1 | grep -q "\[dgedi\] ready:" && break
  sleep 3
done
docker compose logs dgedi 2>&1 | grep "\[dgedi\] ready:" | tail -1
run_cfg results_bop_stage3_ulippc_dgedi "--dgedi --dgedi-top-k 20"

echo "[$(ts)] ===== ALL DONE ====="
