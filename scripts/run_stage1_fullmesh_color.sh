#!/usr/bin/env bash
# =============================================================================
# run_stage1_fullmesh_color.sh — die zwei Stage-1-Full-Mesh-Arme nachholen.
#
# Der erste Versuch (in run_fullmesh_color_redo.sh) scheiterte an fehlenden
# Pfadangaben: der Default images_dir des Treibers ist object_images/shrec18
# (leer), die Renderings liegen in shrec18_v2. Ergebnis war
# "rendered: 0 -> scored gallery: 0". Hier stehen dieselben Flags wie in
# run_stage1_full.sh.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/run_stage1_fullmesh_color.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"

IMG=object_images/shrec18_v2
OUT=results_shrec18_v2_stage1_42v_k5
COMMON="--data-root eval/datasets/shrec18/shrec18_full --images-dir $IMG \
        --desc-file object_database/shrec18_v2/descriptions_attributes.json \
        --results-root $OUT"

for _ in $(seq 1 90); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${u:-99999}" -lt 12000 ] && break; sleep 20
done
log "GPU used=${u:-?} MiB"

log ">>> E2b_fullmesh + E2b_fullmesh_shape_only (mit Farbe aus der Textur)"
docker compose run --rm oscar bash -lc \
  "cd /app && PYTHONHASHSEED=0 SHREC_DINO_POOLING=mean \
   python3 -u experiments/experiment1_shrec18_stage1.py \
   --ablations E2b_fullmesh,E2b_fullmesh_shape_only --overwrite $COMMON" \
  > logs/stage1_fullmesh_color.log 2>&1
log "    rc=$?"

python3 - <<'PY' | tee -a "$LOG"
import json, os
B = "object_retrieval/results_shrec18_v2_stage1_42v_k5"
print(f"  {'Arm':<32}{'nDCG':>9}{'vorher':>9}")
for arm, old in [("E2b_fullmesh", None), ("E2b_fullmesh_shape_only", 0.4858),
                 ("E1c_full_fusion", 0.5868), ("E1_shape_only", 0.5353)]:
    f = os.path.join(B, arm, "metrics_summary.json")
    v = json.load(open(f)).get("nDCG") if os.path.isfile(f) else None
    print(f"  {arm:<32}{(f'{v:.4f}' if v else '—'):>9}"
          f"{(f'{old:.4f}' if old else '—'):>9}")
print("\n  Vergleichspaar: E2b_fullmesh_shape_only gegen E1_shape_only (0.5353).")
PY
log "===== FERTIG ====="
