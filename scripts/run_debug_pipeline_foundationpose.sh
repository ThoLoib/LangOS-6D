#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

python3.11 -m pipeline.run_pipeline \
    --rgb   eval/datasets/ycbv_gso/test/000051/rgb/000001.png \
    --depth eval/datasets/ycbv_gso/test/000051/depth/000001.png \
    --camera eval/datasets/ycbv_gso/test/000051/scene_camera.json \
    --prompt "I need the scissor" \
    --descriptions object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json \
    --reference_images object_images/ycbv_gso/ \
    --cad_models object_database/ycbv_gso/ \
    --ulip_repo /ulip \
    --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt \
    --ulip_mode pc \
    --ulip-partial-views \
    --pose_method foundationpose \
    --output debug_output/pc-mode \
    --debug-viz \
    --until-step 8 \
    --skip_steps 3 \
    --gt-bbox-compensation \
    --scale-gate \
    --scale-gate-min 0.8 \
    --scale-gate-max 1.2 \
    --ulip-rotation-eval \
    --ulip-rotation-eval-top-k 5 \
    --ulip-rotation-eval-weight 1.0 \
    "$@"
