#!/usr/bin/env bash
# =============================================================================
# run_fullmesh_color_redo.sh — Full-Mesh-Arme mit farbigen Mesh-Samples neu.
#
# Hintergrund: sample_pointcloud_from_mesh() las nur `face_colors`. Texturierte
# Meshes (SHREC'18, GSO, YCB-V) tragen ihre Farbe aber in einer Bilddatei plus
# UV-Koordinaten; dort ist face_colors None, und der farbige ULIP-2-Backbone
# bekam Nullen. Behoben ueber to_color() -> vertex_colors (Details im AI_LOG).
#
# WICHTIG — die Caches MUESSEN weg. Der Fingerprint (_get_cache_path) hasht nur
# Config-Flags und DATEIGROESSEN, nicht die Farbwerte. `ulip2_use_colors` war
# schon vorher True, der Hash ist also unveraendert: ohne Beiseiteraeumen
# wuerden die alten, farblosen Caches wieder getroffen und der ganze Lauf waere
# wirkungslos. Sie werden verschoben, nicht geloescht.
#
# Laeufe:
#   Stage 3   3a_pc, 3a_cross                 (Bezugspunkte, liefern arm_ranks)
#             3a_pc_fullmesh, 3a_cross_fullmesh
#   Stage 1   E2b_fullmesh, E2b_fullmesh_shape_only
#
# Jeder Stage-3-Lauf schreibt jetzt `arm_ranks` je Query — damit faellt der
# ISOLIERTE Shape-Kanal (ulip_only_full) gratis mit ab, ohne eigene Laeufe mit
# Gewichten (0,0,1).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_fullmesh_color_redo.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"
OUT=results_bop_stage3_v2
STASH=".ulip_cache_stash_$(date +%Y%m%d_%H%M)"

gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }
wait_gpu(){ for _ in $(seq 1 90); do [ "$(gpu_used)" -lt 12000 ] && return 0; sleep 20; done; }

# ---- 0. Alte Full-Mesh-Caches beiseite ------------------------------------
log "===== 0/3  Full-Mesh-Caches beiseiteraeumen -> $STASH ====="
mkdir -p "$STASH"
n=0
for f in object_database/gso/.ulip_cache_*.pt \
         object_database/ycbv/.ulip_cache_*.pt \
         object_database/housecat6d/.ulip_cache_*.pt \
         eval/datasets/shrec18/shrec18_full/cad/.ulip_cache_*.pt; do
  [ -f "$f" ] || continue
  d="$STASH/$(dirname "$f")"; mkdir -p "$d"; mv "$f" "$d/" && n=$((n+1))
done
log "    $n Cache-Dateien verschoben (Partial-Caches bleiben unberuehrt)"

# ---- 1. Stage 3 ------------------------------------------------------------
s3(){ log ">>> $1"; wait_gpu
  docker compose run --rm oscar bash -lc \
    "cd /app/object_retrieval && PYTHONHASHSEED=0 \
     python3 -u eval_bop_pose.py --datasets all --mode $2 --output $OUT/$1" \
    > "logs/stage3_$1.log" 2>&1
  log "    $1 rc=$?"
  grep -h "\[stage3\]\[fullmesh\]" "logs/stage3_$1.log" 2>/dev/null | tee -a "$LOG"
  python3 -c "
import json, os
f='object_retrieval/$OUT/$1/combined_stage3a.json'
if os.path.isfile(f):
    d=json.load(open(f)); o=d.get('overall') or d
    print(f\"    R@1={o.get('recall@1',0):.4f}  MRR={o.get('mrr',0):.4f}\")" | tee -a "$LOG"; }

log "===== 1/3  Stage 3 — Bezugspunkte und Full-Mesh ====="
s3 3a_pc_v2            "3a --pc-query"
s3 3a_cross_v2         "3a"
s3 3a_pc_fullmesh_v2   "3a --fullmesh --pc-query"
s3 3a_cross_fullmesh_v2 "3a --fullmesh"

# ---- 2. Isolierter Shape-Kanal aus den arm_ranks ---------------------------
log "===== 2/3  Isolierter Shape-Kanal (ulip_only_full) ====="
python3 - <<'PY' | tee -a "$LOG"
import json, glob, os
import numpy as np
B = "object_retrieval/results_bop_stage3_v2"
ARMS = ["clip_dino_ulip_full", "ulip_only_full", "dino_only_full", "clip_only"]
print(f"  {'Arm':<24}{'fusioniert':>12}{'Shape allein':>14}{'DINO allein':>13}{'CLIP allein':>13}")
for run in ["3a_pc_v2", "3a_pc_fullmesh_v2", "3a_cross_v2", "3a_cross_fullmesh_v2"]:
    recs = []
    for f in glob.glob(os.path.join(B, run, "*_stage3a", "records.json")):
        recs += json.load(open(f))
    if not recs:
        print(f"  {run:<24}  (keine Records)"); continue
    row = []
    for arm in ARMS:
        r = [x["arm_ranks"][arm] for x in recs
             if x.get("arm_ranks", {}).get(arm) is not None]
        row.append(f"{np.mean(np.array(r) == 1):.4f}" if r else "—")
    print(f"  {run:<24}" + "".join(f"{v:>13}" for v in row))
print("\n  Erwartung: der partial-vs-fullmesh-Abstand ist SPALTE 'Shape allein'")
print("  groesser als fusioniert — Text und Erscheinung federn ihn dort zu 70% ab.")
print("  Stage-1-Gegenstueck (SHREC, pc): partial 0.5353 vs full-mesh 0.4858.")
PY

# ---- 3. Stage 1 ------------------------------------------------------------
log "===== 3/3  Stage 1 — Full-Mesh-Arme ====="
wait_gpu
docker compose run --rm oscar bash -lc \
  "cd /app && PYTHONHASHSEED=0 SHREC_DINO_POOLING=mean \
   python3 -u experiments/experiment1_shrec18_stage1.py \
   --ablations E2b_fullmesh,E2b_fullmesh_shape_only --overwrite \
   --results-root results_shrec18_v2_stage1_42v_k5" \
  > logs/stage1_fullmesh_color.log 2>&1
log "    stage1 rc=$?"
tail -20 logs/stage1_fullmesh_color.log | tee -a "$LOG"

log "===== FERTIG ====="
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy "object_retrieval/$OUT" "$REMOTE/object_retrieval/$OUT" \
  --transfers 16 --stats 0 >> logs/rclone_reruns.log 2>&1 || true
