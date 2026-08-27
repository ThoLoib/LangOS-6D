#!/usr/bin/env bash
# =============================================================================
# run_remaining_chain.sh — EINE sequenzielle Kette fuer alles Verbleibende.
#
# Ersetzt die drei unabhaengigen Waiter (legacy / stage3 / geo), die auf
# verschiedene Marker horchten und deshalb GLEICHZEITIG feuern konnten —
# genau das hat am 2026-08-27 zu CUDA-OOM gefuehrt (drei GPU-Jobs auf einer
# 24-GB-Karte). Hier laeuft jeder Schritt strikt nach dem vorigen.
#
# Reihenfolge:
#   1. MI3DOR OSCAR-Legacy (V=8, tau=0.37, fullmesh)   — Publikationsvergleich
#   2. Stage-3 3a: OSCAR-Baseline + full-mesh          — Uni3D gestrichen
#   3. Stage-3 3b/3c fuer full-mesh, NUR falls es ULIP-2-cross schlaegt
#   4. Stage-3 3a-Geometrie: Distanz / Fitness / Borda GETRENNT (wie Stage-1 C1)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_remaining_chain.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"
OUT=results_bop_stage3_v2
GT=results_bop_stage3_v2/gt
BASE=0.482          # ULIP-2 cross R@1 — die Latte fuer den Pose-Lauf

gpu_free(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }
wait_gpu(){ for _ in $(seq 1 60); do [ "$(gpu_free)" -lt 9000 ] && return 0; sleep 20; done; }

# ---- 1. MI3DOR OSCAR-Legacy (V=8) ------------------------------------------
log "===== 1/4  MI3DOR OSCAR-Legacy V=8 ====="; wait_gpu
S2OUT=results_mi3dor_oscar_legacy_v8
docker compose run --rm oscar bash -lc \
  "cd /app/object_retrieval && PYTHONHASHSEED=0 MI3DOR_MODES=fullmesh \
   MI3DOR_NUM_VIEWS=8 MI3DOR_DINO_POOLING=mean MI3DOR_RESULT_FOLDER=$S2OUT \
   python3 -u retrieval_mi3dor_eval_oscarplus.py" > logs/mi3dor_oscar_legacy.log 2>&1
log "    rc=$?"
S="object_retrieval/$S2OUT/fullmesh/metrics_summary_topk_15.json"
[ -f "$S" ] && python3 -c "
import json;d=json.load(open('$S'));v=d['variants']
for k in ['oscar_maxview','oscar_softmax','clip_dino_ulip_full']:
    m=v.get(k,{})
    print(f\"    {k:<24} NN={m.get('NN_accuracy',0):.2f} FT={m.get('FT_mean',0):.3f} NDCG2R={m.get('nDCG@2R_mean',0):.3f}\")
print('    -> Legacy-Baseline = oscar_maxview (CLIP-tau + DINO Hard-Max, 8 Views)')" | tee -a "$LOG"

# ---- 2. Stage-3 3a ----------------------------------------------------------
s3(){ log ">>> $1"; wait_gpu
  docker compose run --rm ${3:-} oscar bash -lc \
    "cd /app/object_retrieval && PYTHONHASHSEED=0 ${4:-} \
     python3 -u eval_bop_pose.py --datasets all --mode ${2} --output $OUT/$1" \
    > "logs/stage3_$1.log" 2>&1
  log "    $1 rc=$?"; }
r1(){ python3 -c "
import json,os
f='object_retrieval/$OUT/$1/combined_stage3a.json'
print((json.load(open(f)).get('overall') or json.load(open(f))).get('recall@1','') if os.path.isfile(f) else '')" 2>/dev/null; }
sane(){ awk -v x="$1" 'BEGIN{exit !(x+0>0.05 && x+0<0.95)}'; }

log "===== 2/4  Stage-3 3a (Uni3D gestrichen: Stage 1 zeigt ULIP-2 >= Uni3D) ====="
s3 3a_oscar    "3a --oscar-baseline"
s3 3a_fullmesh "3a --fullmesh"
OSC=$(r1 3a_oscar); FUL=$(r1 3a_fullmesh)
log "3a R@1 — OSCAR=$OSC | full-mesh=$FUL   (ULIP-2 cross BASE=$BASE)"

# ---- 3. Pose, gated ---------------------------------------------------------
log "===== 3/4  Pose (gated) ====="
if sane "$OSC"; then s3 3b_oscar "3b --oscar-baseline --gt-records $GT"
else log "OSCAR 3a R@1=$OSC unplausibel — kein Pose-Lauf"; fi
if sane "$FUL" && awk -v x="$FUL" -v b="$BASE" 'BEGIN{exit !(x+0>b+0)}'; then
  log "full-mesh ($FUL) schlaegt cross ($BASE) -> Pose + 3c"
  s3 3b_fullmesh "3b --fullmesh --gt-records $GT"
  s3 3c_fullmesh "3c --fullmesh --from-3a $OUT/3a_fullmesh"
else log "full-mesh ($FUL) schlaegt cross ($BASE) nicht -> kein zusaetzlicher Pose-Lauf"; fi

# ---- 4. Geometrie: Distanz / Fitness / Borda getrennt -----------------------
log "===== 4/4  3a-Geometrie, Kriterien getrennt (wie Stage-1 C1) ====="
for SIG in distance fitness borda; do
  s3 "3a_cross_geo_$SIG" "3a --dgedi --dgedi-repo --dgedi-top-k 5" "-e STAGE3_GEO_SIGNAL=$SIG" "STAGE3_GEO_SIGNAL=$SIG"
done
for SIG in distance fitness; do
  s3 "3a_pc_geo_$SIG" "3a --dgedi --dgedi-repo --dgedi-top-k 5 --pc-query" "-e STAGE3_GEO_SIGNAL=$SIG" "STAGE3_GEO_SIGNAL=$SIG"
done

log "===== ZUSAMMENFASSUNG ====="
python3 - <<'PY' | tee -a "$LOG"
import json, os
B="object_retrieval/results_bop_stage3_v2"
def r1(p):
    f=os.path.join(B,p,"combined_stage3a.json")
    if not os.path.isfile(f): return "—"
    d=json.load(open(f)); v=(d.get('overall') or d).get('recall@1')
    return f"{v:.3f}" if isinstance(v,(int,float)) else "—"
print(f"  {'Modus':<8}{'ohne Geo':>10}{'Distanz':>10}{'Fitness':>10}{'Borda':>10}")
for name,ng,pre in [("cross","3a_cross","3a_cross_geo"),("pc","3a_pc","3a_pc_geo")]:
    print(f"  {name:<8}{r1(ng):>10}{r1(pre+'_distance'):>10}{r1(pre+'_fitness'):>10}{r1(pre+'_borda'):>10}")
print(f"\n  OSCAR-Baseline (E5): R@1 {r1('3a_oscar')} | full-mesh: {r1('3a_fullmesh')}")
print("  Stage-1 C1 zum Vergleich: Distanz 0.6405 > Borda 0.6362 > Fitness 0.6251 (nDCG)")
PY
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
for d in "$OUT" "$S2OUT"; do
  "$RC" copy "object_retrieval/$d" "$REMOTE/object_retrieval/$d" --transfers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true
done
log "===== run_remaining_chain DONE ====="
