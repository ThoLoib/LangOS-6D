#!/usr/bin/env bash
# =============================================================================
# run_stage1_isolated.sh — single-channel (ISOLATED) Stage-1 ablation arms.
#
# Several Stage-1 ablations were only ever measured *inside fusion* (the ablated
# channel toggled while text+view still contribute), which masks the isolated
# effect of the changed component.  This script re-runs each such comparison
# with the ablated channel scored ALONE (all weight on it, no fusion), so the
# single variable is directly comparable:
#
#   E2b   partial-view  vs  full-mesh          S_shape reference (shape-only)
#   E4    DINOv2        vs  SigLIP             appearance (view-only, MAP-pooled)
#   E7    ULIP-2(pc) vs Uni3D(pc) vs ULIP-2(cross)  shape encoder / query-mode
#   O5    XYZ+RGB       vs  XYZ-only           query point cloud (shape-only)
#
# Two phases, split by resource:
#   cpu  — arms whose per-channel score cache already exists (fullmesh/xyz/uni3d
#          + the pc/dino baselines + free aliases): pure CPU derivation, GPU
#          hidden so it never contends with a running geometry job.
#   gpu  — E4_siglip_only (needs the MAP-pooled scores_siglip.pt from the
#          siglipfix run) + E7_ulip2_cross_shape_only (encodes the query images
#          through ULIP-2's image tower; light GPU).  Run after siglipfix frees
#          the GPU.  Appends into the SAME results dir with --resume.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
PHASE="${1:?usage: run_stage1_isolated.sh cpu|gpu}"
LOG=logs/run_stage1_isolated.log
ts(){ date -Is; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"

S1COMMON="--data-root eval/datasets/shrec18/shrec18_full \
  --images-dir object_images/shrec18_v2 \
  --desc-file object_database/shrec18_v2/descriptions_attributes.json"
SRC_MEAN=object_retrieval/results_shrec18_v2_stage1_mean_mean_only
SRC_SIGLIP=object_retrieval/results_shrec18_v2_stage1_mean_siglipfix
OUT=object_retrieval/results_shrec18_v2_stage1_mean_isolated

# --ablations takes ONE comma-separated string (E1,E4,O4_V8), never space-sep.
CPU_ARMS="E1_view_only,E1_shape_only,E2b_partial_shape_only,E2b_fullmesh_shape_only,E4_dino_only,E7_ulip2_pc_shape_only,E7_uni3d_shape_only,O5_xyzrgb_shape_only,O5_xyz_shape_only"
GPU_ARMS="E4_siglip_only,E7_ulip2_cross_shape_only"

mkdir -p "$OUT/_cache"

case "$PHASE" in
  cpu)
    log "===== isolated CPU phase START -> $OUT ====="
    # Reusable per-channel score caches (built by the mean-only run). Copying
    # them makes the pass runner return from cache BEFORE building any pipeline
    # (no GPU, no model load), so these arms are pure CPU rank derivations.
    for f in scores_base.pt scores_ulip_pc_rgb.pt scores_ulip_pc_fullmesh.pt \
             scores_ulip_pc_xyz.pt scores_uni3d.pt; do
      cp -n "$SRC_MEAN/_cache/$f" "$OUT/_cache/" 2>/dev/null \
        && log "cached $f" || log "WARN missing $SRC_MEAN/_cache/$f"
    done
    log "arms: $CPU_ARMS"
    CUDA_VISIBLE_DEVICES="" docker compose run --rm -e CUDA_VISIBLE_DEVICES="" oscar bash -lc \
      "cd /app && PYTHONHASHSEED=0 SHREC_DINO_POOLING=mean \
       python3 -u experiments/experiment1_shrec18_stage1.py \
       --ablations $CPU_ARMS $S1COMMON \
       --results-root $OUT --resume --allow-partial-gallery" > logs/stage1_isolated_cpu.log 2>&1
    rc=$?; log "isolated CPU phase rc=$rc"
    ;;
  gpu)
    log "===== isolated GPU phase START -> $OUT ====="
    # E4_siglip_only must use the fair MAP-pooled SigLIP scores from siglipfix.
    if cp -n "$SRC_SIGLIP/_cache/scores_siglip.pt" "$OUT/_cache/" 2>/dev/null; then
      log "cached scores_siglip.pt from siglipfix (fair MAP-pooled)"
    else
      log "WARN no siglipfix scores_siglip.pt — E4_siglip_only will recompute on GPU (still fair post-f50a844f)"
    fi
    log "arms: $GPU_ARMS"
    docker compose run --rm oscar bash -lc \
      "cd /app && PYTHONHASHSEED=0 SHREC_DINO_POOLING=mean \
       python3 -u experiments/experiment1_shrec18_stage1.py \
       --ablations $GPU_ARMS $S1COMMON \
       --results-root $OUT --resume --allow-partial-gallery" > logs/stage1_isolated_gpu.log 2>&1
    rc=$?; log "isolated GPU phase rc=$rc"
    ;;
  *) log "unknown phase '$PHASE'"; exit 2;;
esac

log "current isolated summary rows:"
grep -E "^(E1_view_only|E1_shape_only|E2b_.*shape_only|E4_dino_only|E4_siglip_only|E7_.*shape_only|O5_.*shape_only)," \
  "$OUT/stage1_summary.csv" 2>/dev/null | tee -a "$LOG"
"$RC" copy "$OUT" "$REMOTE/$OUT" --transfers 16 --checkers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true
log "synced $OUT"
log "===== isolated $PHASE phase DONE (rc=${rc:-?}) ====="
