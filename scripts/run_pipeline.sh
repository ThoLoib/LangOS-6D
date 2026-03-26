#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
python3.11 -m pipeline.run_pipeline \
    --rgb eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
    --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
    --prompt "mustard bottle" \
    --descriptions object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json \
    --reference_images object_images/ycbv_gso/ \
    --cad_models object_database/ycbv_gso/ \
    --output pipeline_output \
    "$@"
