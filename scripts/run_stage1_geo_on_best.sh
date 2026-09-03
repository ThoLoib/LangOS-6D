#!/usr/bin/env bash
# =============================================================================
# run_stage1_geo_on_best.sh — Geometrie auf dem staerksten Arm OHNE Geometrie.
#
# Bisher sitzen alle Geometrie-Arme auf dem BASE-Ranking (E1c_full_fusion,
# 0.5868). Seit dem Farb-Fix ist E2b_fullmesh mit 0.5935 vorn, und der noch
# laufende Arm E7_ulip2_cross_fullmesh koennte ihn ablösen. Welcher Arm die
# Grundlage bildet, entscheidet dieses Skript deshalb SELBST — aus den
# tatsaechlichen Ergebnissen, nicht aus einer Annahme.
#
# Die interessante Frage dahinter: ein besseres Ausgangsranking hebt die Latte,
# die das Re-Ranking ueberspringen muss. Auf BOP hat die Geometrie daran
# scheitert (sie war informativ, aber schwaecher als der Score, den sie
# ersetzt). Auf SHREC gewinnt sie bisher +0.054 nDCG — gegen ein SCHWAECHERES
# Ausgangsranking. Schrumpft der Gewinn auf dem staerkeren Arm, bestaetigt das
# den Mechanismus auf einem zweiten Datensatz.
#
# dGeDi-GALLERY
# -------------
# Der Dienst bedient derzeit die BOP-Gallery (1316). SHREC braucht 3308 aus
# .dgedi_gallery_shrec. Ohne Umschaltung laeuft der Lauf ins Leere — genau der
# Fehler vom 2026-08-28, der 17 Stunden gekostet hat. Deshalb: umschalten,
# Groesse PRUEFEN, rechnen, danach zurueckschalten.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/run_stage1_geo_on_best.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"

CANON=object_retrieval/results_shrec18_v2_stage1_42v_k5
STAGING=object_retrieval/results_stage1_geo_on_best
IMG=object_images/shrec18_v2

# ---- 0. Auf die vorherigen Ketten warten ----------------------------------
for chain in run_stage3_fullmesh_pose.sh run_stage1_cross_fullmesh.sh; do
  if pgrep -f "$chain" >/dev/null 2>&1; then
    log "warte auf $chain ..."
    while pgrep -f "$chain" >/dev/null 2>&1; do sleep 120; done
  fi
done
log "Vorketten beendet."

# ---- 1. Gate: wer ist der staerkste Arm OHNE Geometrie? -------------------
BEST=$(python3 - <<'PY'
import json, os
B = "object_retrieval/results_shrec18_v2_stage1_42v_k5"
# Kandidaten sind genau die Arme, fuer die ein Geometrie-Gegenstueck definiert
# ist. Andere Arme kaemen als Grundlage nicht in Frage, ohne neue Specs.
CAND = {"E1c_full_fusion": "E2_chamfer_ransac",       # existiert bereits
        "E2b_fullmesh": "E2b_fullmesh_geo",
        "E7_ulip2_cross_fullmesh": "E7_ulip2_cross_fullmesh_geo"}
best, arm = -1.0, None
for a in CAND:
    f = os.path.join(B, a, "metrics_summary.json")
    if not os.path.isfile(f):
        continue
    v = json.load(open(f))["metrics"]["nDCG"]
    print(f"# {a:<28}{v:.4f}", flush=True)
    if v > best:
        best, arm = v, a
print(f"{arm}|{CAND[arm]}|{best:.4f}")
PY
)
echo "$BEST" | grep '^#' | tee -a "$LOG"
LINE=$(echo "$BEST" | tail -1)
BASE_ARM=${LINE%%|*}; GEO_ARM=$(echo "$LINE" | cut -d'|' -f2); BASE_ND=${LINE##*|}
log "staerkster Arm ohne Geometrie: $BASE_ARM (nDCG $BASE_ND) -> Geometrie-Arm $GEO_ARM"

if [ "$GEO_ARM" = "E2_chamfer_ransac" ]; then
  log "Das ist der BASE — die Geometrie darauf ist bereits gerechnet (0.6405). Nichts zu tun."
  exit 0
fi
[ -e "$CANON/$GEO_ARM" ] && { log "ABBRUCH: $CANON/$GEO_ARM existiert bereits"; exit 3; }

# ---- 2. dGeDi auf die SHREC-Gallery umschalten ---------------------------
log "===== dGeDi auf SHREC umschalten ====="
DGEDI_CACHE_DIR=.dgedi_gallery_shrec docker compose up -d --force-recreate dgedi >/dev/null 2>&1
for _ in $(seq 1 30); do
  n=$(docker compose run --rm --no-deps oscar bash -lc \
      "cd /app/object_retrieval && python3 -c \"from dgedi_bridge import dgedi_health as h; d=h() or {}; print('NGAL=%s' % d.get('n_gallery',0))\"" 2>/dev/null \
      | grep -oE 'NGAL=[0-9]+' | tail -1 | cut -d= -f2)
  [ "${n:-0}" -ge 3000 ] && break; sleep 10
done
log "dGeDi n_gallery=${n:-0}"
[ "${n:-0}" -lt 3000 ] && { log "ABBRUCH: falsche Gallery (erwartet 3308)"; exit 3; }

# ---- 3. Rechnen, in den EIGENEN Ordner ------------------------------------
# --geom-k 50 ist gepinnt (C2/BASE-Tiefe). Ohne die Angabe waehlt der Treiber
# K selbst und der Vergleich mit E2_chamfer_ransac waere hinfaellig.
log ">>> $GEO_ARM (Grundlage $BASE_ARM, K=50)"
docker compose run --rm oscar bash -lc \
  "cd /app && PYTHONHASHSEED=0 SHREC_DINO_POOLING=mean \
   python3 -u experiments/experiment1_shrec18_stage1.py \
   --ablations $GEO_ARM --with-geometry --geom-k 50 \
   --data-root eval/datasets/shrec18/shrec18_full --images-dir $IMG \
   --desc-file object_database/shrec18_v2/descriptions_attributes.json \
   --results-root $STAGING" \
  > logs/stage1_geo_on_best.log 2>&1
log "    rc=$?"

# ---- 4. Nur das Arm-Verzeichnis spiegeln ---------------------------------
if [ -f "$STAGING/$GEO_ARM/metrics_summary.json" ]; then
  docker compose run --rm --no-deps oscar bash -lc \
    "cp -r /app/$STAGING/$GEO_ARM /app/$CANON/" >/dev/null 2>&1
  log "    uebernommen: $GEO_ARM"
else
  log "    FEHLT: $GEO_ARM — nicht uebernommen"
fi

# ---- 5. dGeDi zurueck auf die BOP-Gallery --------------------------------
log "===== dGeDi zurueck auf BOP ====="
docker compose up -d --force-recreate dgedi >/dev/null 2>&1
sleep 20

# ---- 6. Vergleich ---------------------------------------------------------
log "===== GEOMETRIE-GEWINN, alt gegen neu ====="
python3 - "$BASE_ARM" "$GEO_ARM" <<'PY' | tee -a "$LOG"
import json, os, sys
B = "object_retrieval/results_shrec18_v2_stage1_42v_k5"
base_arm, geo_arm = sys.argv[1], sys.argv[2]
def m(a):
    f = os.path.join(B, a, "metrics_summary.json")
    if not os.path.isfile(f): return None
    d = json.load(open(f))
    return d["metrics"]["nDCG"], d.get("metrics_depth", {}).get("NN_sub")
print(f"  {'Grundlage':<30}{'nDCG':>9}{'NN_sub':>9}{'-> mit Geometrie':>20}")
for b, g in [("E1c_full_fusion", "E2_chamfer_ransac"), (base_arm, geo_arm)]:
    vb, vg = m(b), m(g)
    if not vb: continue
    tail = (f"{vg[0]:.4f} / {vg[1]:.4f}" if vg else "—")
    print(f"  {b:<30}{vb[0]:>9.4f}{(vb[1] or 0):>9.4f}{tail:>20}")
    if vg:
        print(f"  {'   Gewinn durch Geometrie':<30}{vg[0]-vb[0]:>+9.4f}"
              f"{(vg[1] or 0)-(vb[1] or 0):>+9.4f}")
print("\n  Erwartung: der Gewinn faellt auf dem staerkeren Ausgangsranking KLEINER aus.")
print("  Genau daran ist die Geometrie auf BOP gescheitert — sie war informativ,")
print("  aber schwaecher als der Fusions-Score, den sie ersetzt.")
PY
log "===== FERTIG ====="
