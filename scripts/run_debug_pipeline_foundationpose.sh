#!/usr/bin/env bash
# =============================================================================
# Thesis-aligned pipeline run (methodology chapter defaults)
#
# Step A:  GroundingDINO + SAM2.1 → mask (largest CC + 5×5 dilation)
# Step B1: CLIP (S_text) + DINOv2 top-5 softmax (S_view) + ULIP-2 PC (S_shape)
#          → weighted sum fusion (0.3 / 0.4 / 0.3)
# Step B2: GeDi geometry re-ranking (RANSAC inlier count) on top-5 fused
# Step C:  Scale estimation (ICP, B2 transform reuse) → FoundationPose
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

python3.11 -m pipeline.run_pipeline \
    --rgb   eval/datasets/ycbv_gso/test/000049/rgb/000001.png \
    --depth eval/datasets/ycbv_gso/test/000049/depth/000001.png \
    --camera eval/datasets/ycbv_gso/test/000049/scene_camera.json \
    --prompt "I need the round tuna can" \
    --descriptions object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json \
    --reference_images object_images/ycbv_gso/ \
    --cad_models object_database/ycbv_gso/ \
    --ulip_repo /ulip \
    --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt \
    --ulip_mode pc \
    --ulip-partial-views \
    --pose_method foundationpose \
    --geometry-reranking \
    --geometry-reranking-signal both \
    --geometry-reranking-top-k 5 \
    --output debug_output/thesis_default \
    --debug-viz \
    "$@"
