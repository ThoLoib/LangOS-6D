#!/usr/bin/env bash
# =============================================================================
# run_stage1_cross_fullmesh.sh — die fehlende vierte Zelle in Stage 1.
#
#   Query-Modus x Gallery-Repraesentation, isolierter Shape-Kanal (nDCG):
#                partial    full-mesh
#     pc          0.5353      0.4956
#     cross       0.4809      <- HIER
#
# Auf BOP ist genau diese Kombination der beste Arm (R@1 0.5151), auf SHREC
# fehlte sie bis 2026-09-03.
#
# NICHTS UEBERSCHREIBEN
# ---------------------
# Der Treiber schreibt am Ende best_config.json, stage1_summary.csv/.tex und
# stage1_summary_depth.csv/.tex NEU — abgeleitet aus NUR den gelaufenen Armen.
# Liefe er direkt in den kanonischen Ordner, waere die ueber 39 Arme gebildete
# Auswahl durch eine aus zweien ersetzt (am 02.09. genau so passiert).
# Deshalb: eigener Ergebnisordner, danach werden AUSSCHLIESSLICH die beiden
# Arm-Verzeichnisse hinuebergespiegelt. Vorher wird der kanonische Ordner
# gesichert, und es wird geprueft, dass die Zielverzeichnisse noch nicht
# existieren.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/run_stage1_cross_fullmesh.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"

CANON=object_retrieval/results_shrec18_v2_stage1_42v_k5
STAGING=object_retrieval/results_stage1_cross_fullmesh
ARMS="E7_ulip2_cross_fullmesh_shape_only,E7_ulip2_cross_fullmesh"
IMG=object_images/shrec18_v2

# ---- 0. Sicherung der Dateien, die der Aggregator anfassen wuerde ----------
BAK=".stage1_canon_backup_$(date +%Y%m%d_%H%M)"
mkdir -p "$BAK"
for f in best_config.json stage1_summary.csv stage1_summary.tex \
         stage1_summary_depth.csv stage1_summary_depth.tex; do
  [ -f "$CANON/$f" ] && cp -p "$CANON/$f" "$BAK/"
done
log "gesichert nach $BAK: $(ls "$BAK" | tr '\n' ' ')"

for a in ${ARMS//,/ }; do
  [ -e "$CANON/$a" ] && { log "ABBRUCH: $CANON/$a existiert bereits"; exit 3; }
done
log "Zielverzeichnisse frei — nichts wird ueberschrieben"

# ---- 1. Lauf in den EIGENEN Ordner ----------------------------------------
# Auf die Pose-Kette warten, nicht nur auf die GPU-Belegung. Eine reine
# Speicherschwelle mit Zeitlimit startet nach Ablauf TROTZDEM — und zwei
# GPU-Jobs auf einer 24-GB-Karte waren am 2026-08-27 der OOM.
if pgrep -f "run_stage3_fullmesh_pose.sh" >/dev/null 2>&1; then
  log "warte auf run_stage3_fullmesh_pose.sh ..."
  while pgrep -f "run_stage3_fullmesh_pose.sh" >/dev/null 2>&1; do sleep 120; done
  log "Pose-Kette beendet."
fi
for _ in $(seq 1 180); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${u:-99999}" -lt 12000 ] && break; sleep 20
done
log "GPU used=${u:-?} MiB"

log ">>> $ARMS"
docker compose run --rm oscar bash -lc \
  "cd /app && PYTHONHASHSEED=0 SHREC_DINO_POOLING=mean \
   python3 -u experiments/experiment1_shrec18_stage1.py \
   --ablations $ARMS \
   --data-root eval/datasets/shrec18/shrec18_full --images-dir $IMG \
   --desc-file object_database/shrec18_v2/descriptions_attributes.json \
   --results-root $STAGING" \
  > logs/stage1_cross_fullmesh.log 2>&1
log "    rc=$?"

# ---- 2. NUR die Arm-Verzeichnisse spiegeln ---------------------------------
ok=1
for a in ${ARMS//,/ }; do
  if [ -f "$STAGING/$a/metrics_summary.json" ]; then
    docker compose run --rm --no-deps oscar bash -lc \
      "cp -r /app/$STAGING/$a /app/$CANON/" >/dev/null 2>&1
    log "    uebernommen: $a"
  else
    log "    FEHLT: $a — nicht uebernommen"; ok=0
  fi
done
[ "$ok" -eq 1 ] || log "WARNUNG: nicht alle Arme erzeugt, Staging bleibt liegen"

# ---- 3. Gegenpruefung: Aggregat-Dateien unveraendert? ----------------------
log "===== Pruefung: wurde etwas ueberschrieben? ====="
for f in "$BAK"/*; do
  b=$(basename "$f")
  if cmp -s "$f" "$CANON/$b"; then log "    $b unveraendert"
  else log "    !! $b GEAENDERT — aus $BAK zurueckspielen"; fi
done

log "===== MATRIX (isolierter Shape-Kanal, nDCG) ====="
python3 - <<'PY' | tee -a "$LOG"
import json, os
B = "object_retrieval/results_shrec18_v2_stage1_42v_k5"
def nd(a):
    f = os.path.join(B, a, "metrics_summary.json")
    if not os.path.isfile(f): return None
    m = json.load(open(f))
    return m["metrics"]["nDCG"], m.get("metrics_depth", {}).get("NN_sub")
cells = {("pc", "partial"): "E1_shape_only",
         ("pc", "full-mesh"): "E2b_fullmesh_shape_only",
         ("cross", "partial"): "E7_ulip2_cross_shape_only",
         ("cross", "full-mesh"): "E7_ulip2_cross_fullmesh_shape_only"}
print(f"  {'':<8}{'partial':>20}{'full-mesh':>20}")
for mode in ("pc", "cross"):
    row = ""
    for rep in ("partial", "full-mesh"):
        v = nd(cells[(mode, rep)])
        row += (f"{v[0]:.4f} / {v[1]:.4f}".rjust(20) if v else "—".rjust(20))
    print(f"  {mode:<8}{row}")
print("  (nDCG / NN_sub)")
f = nd("E7_ulip2_cross_fullmesh")
if f: print(f"\n  fusioniert cross x full-mesh: nDCG {f[0]:.4f}  NN_sub {f[1]:.4f}")
print("  Vergleich fusioniert: BASE 0.5868/0.3413 · E2b_fullmesh 0.5935/0.3598")
PY
log "===== FERTIG ====="
