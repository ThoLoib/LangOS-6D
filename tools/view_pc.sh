#!/usr/bin/env bash
# Usage: tools/view_pc.sh <path-to-any.npz> [output.html]
# Renders any point cloud .npz (must contain a "points" array, optionally
# "colors") to an interactive HTML file and opens it in your browser.
set -euo pipefail
cd "$(dirname "$0")/.."

NPZ="${1:?usage: view_pc.sh <path-to.npz> [output.html]}"
OUT="${2:-object_images/pc_viewer.html}"

docker run --rm -v "$(pwd)":/app -w /app tholoi/oscar-plus \
  bash -lc "python3 tools/pc_viewer.py '$NPZ' '$OUT'"

xdg-open "$OUT" >/dev/null 2>&1 || echo "Open manually: file://$(pwd)/$OUT"
