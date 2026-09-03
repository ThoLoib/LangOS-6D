#!/usr/bin/env bash
# =============================================================================
# run_stage3_fullmesh_pose.sh — Pose fuer den neuen besten Retrieval-Arm.
#
# Ausloeser: nach dem Farb-Fix erreicht cross+full-mesh R@1 0.5151 und schlaegt
# damit die eingefrorene Konfiguration (0.4818). Die Vereinbarung
# (AGREEMENTS.md, 2026-08-xx) lautet: schlaegt ein nachgeholter Arm die bisher
# getesteten, werden 3b und 3c auch mit ihm gefahren.
#
#   1. 3b_cross_fullmesh   Proxy-Pose + D_sym, gepaart gegen dieselbe GT-Referenz
#   2. 3c_cross_fullmesh   Zerlegung: echtes Fremd-CAD gegen Proxy-Gallery
#
# Die farbigen Full-Mesh-Caches liegen seit dem 01.09. an ihrem Platz; der
# Stash .ulip_cache_stash_* enthaelt nur noch die alten farblosen Fassungen und
# darf NICHT zurueckgespielt werden.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_stage3_fullmesh_pose.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"
OUT=results_bop_stage3_v2
GT=$OUT/gt/combined_gt.json
FROM=$OUT/3a_cross_fullmesh_v2

gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }
wait_gpu(){ for _ in $(seq 1 90); do [ "$(gpu_used)" -lt 12000 ] && return 0; sleep 20; done; }

# Vorpruefungen: beide Eingaben muessen DATEIEN bzw. Ordner sein. --gt-records
# einen Ordner zu uebergeben hat am 2026-08-28 einen Lauf gekostet.
[ -f "object_retrieval/$GT" ] || { log "ABBRUCH: $GT fehlt"; exit 3; }
[ -f "object_retrieval/$FROM/combined_stage3a.json" ] || {
  log "ABBRUCH: $FROM ohne combined_stage3a.json"; exit 3; }
log "Eingaben ok — GT $(du -h object_retrieval/$GT | cut -f1), 3a-Quelle $FROM"

s3(){ log ">>> $1"; wait_gpu
  docker compose run --rm oscar bash -lc \
    "cd /app/object_retrieval && PYTHONHASHSEED=0 \
     python3 -u eval_bop_pose.py --datasets all --mode $2 --output $OUT/$1" \
    > "logs/stage3_$1.log" 2>&1
  log "    $1 rc=$?"
  grep -h "\[stage3\]\[fullmesh\]" "logs/stage3_$1.log" 2>/dev/null | tee -a "$LOG"; }

log "===== 1/2  3b_cross_fullmesh ====="
s3 3b_cross_fullmesh "3b --fullmesh --gt-records $GT"

log "===== 2/2  3c_cross_fullmesh ====="
s3 3c_cross_fullmesh "3c --fullmesh --from-3a $FROM"

log "===== VERGLEICH ====="
python3 - <<'PY' | tee -a "$LOG"
import json, os
B = "object_retrieval/results_bop_stage3_v2"
def dsym(run, mode):
    f = os.path.join(B, run, f"combined_stage3{mode}.json")
    if not os.path.isfile(f): return None
    d = json.load(open(f))
    s = (d.get("overall") or d).get("dsym") or d.get("dsym") or {}
    de = (d.get("overall") or d).get("delta") or d.get("delta") or {}
    return s.get("d_sym_median"), de.get("delta_median"), s.get("coverage")

print(f"  {'3b-Arm':<26}{'D_sym Median':>14}{'Delta Median':>14}{'Deckung':>10}")
for run, lbl in [("3b_cross", "cross, partial (bisher)"),
                 ("3b_cross_fullmesh", "cross, full-mesh (neu)"),
                 ("3b_oscar", "OSCAR-Baseline")]:
    v = dsym(run, "b")
    if not v: print(f"  {lbl:<26}{'—':>14}"); continue
    print(f"  {lbl:<26}{v[0]:>11.2f} mm{v[1]:>11.2f} mm{v[2]:>10.3f}")
print("\n  GT-CAD-Referenz: 1.72 mm")

print(f"\n  {'3c-Zerlegung':<26}{'Median':>12}{'n':>8}")
for run, lbl in [("3c_cross", "cross, partial (bisher)"),
                 ("3c_cross_fullmesh", "cross, full-mesh (neu)")]:
    f = os.path.join(B, run, "combined_stage3c.json")
    if not os.path.isfile(f): print(f"  {lbl:<26}{'—':>12}"); continue
    d = json.load(open(f)); s = d.get("dsym") or {}
    p = d.get("provenance") or {}
    print(f"  {lbl:<26}{s.get('d_sym_median', 0):>9.2f} mm{s.get('n_estimated', 0):>8}")
    for k, t in [("target_cad", "echtes Fremd-CAD"), ("proxy", "Proxy-Gallery")]:
        e = p.get(k) or {}
        if e: print(f"      {t:<22}{e.get('d_sym_median', 0):>9.2f} mm{e.get('n', 0):>8}")
print("\n  Entscheidend: schlaegt full-mesh auch in der POSE, aendert sich die")
print("  empfohlene Konfiguration der Arbeit, nicht nur die Retrieval-Zahl.")
PY

RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy "object_retrieval/$OUT" "$REMOTE/object_retrieval/$OUT" \
  --transfers 16 --stats 0 >> logs/rclone_reruns.log 2>&1 || true
log "===== FERTIG ====="
