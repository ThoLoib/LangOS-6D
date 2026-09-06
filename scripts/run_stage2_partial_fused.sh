#!/usr/bin/env bash
# =============================================================================
# run_stage2_partial_fused.sh — die fehlende Zelle der Stage-2-A4-Matrix:
# MI3DOR mit *partieller* Gallery-Repraesentation UND korrekten Fusionsgewichten.
#
# Warum es diesen Lauf braucht
# ----------------------------
# partial-vs-full-mesh liegt auf MI3DOR bisher nur ISOLIERT vor:
#   * ..._dinomean/{partial,fullmesh}  — partial griff wirklich, aber die
#     Fusionsgewichte waren (0, 0.5, 0.5) -> fusionierte Arme unbrauchbar.
#   * ..._fixedw/partial               — Gewichte korrekt, aber der CAD-Glob
#     fand 0 Meshes -> ULIP-Kanal leer (ulip_only NN=0.00).
#   * ..._ulipfix/partial              — Gewichte + CAD-Glob korrekt, aber es
#     gibt keine *_partial.npz mehr: build_pipeline fiel still auf full-mesh
#     zurueck (ulip2_use_partial_views=False im geschriebenen Config-Block).
# Dieser Lauf schliesst die Luecke ueber den vorhandenen Partial-Cache.
#
# Die eine kritische Variable
# ---------------------------
# SHREC_FORCE_PARTIAL_CACHE ist NICHT SHREC-spezifisch — es wirkt in
# eval_common.build_pipeline fuer jeden Datensatz und umgeht Discovery plus
# Fingerprint der fehlenden *_partial.npz. Der Cache haelt alle 42 Views je
# Objekt im 1280-d-Raum des FARBIGEN ULIP-2 (identisch zum Produktionslauf).
# Ohne die Variable laeuft dieser Pass still als full-mesh — genau der Fehler,
# der ..._ulipfix/partial unbrauchbar gemacht hat.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_stage2_partial_fused.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"

OUT=results_mi3dor_oscarplus_v2_tau037_dinomean_partialforce
CACHE=object_images/MI3DOR/.ulip_partial_cache_f6bcf93bb6c92c68.pt
REF=object_retrieval/results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix/fullmesh/metrics_summary_topk_15.json

# --- 0. Vorbedingungen ------------------------------------------------------
if [ ! -f "$CACHE" ]; then
  log "ABBRUCH: Partial-Cache fehlt: $CACHE"; exit 1
fi
if [ -d "object_retrieval/$OUT" ]; then
  log "ABBRUCH: $OUT existiert bereits — nichts ueberschreiben."; exit 1
fi
log "Partial-Cache: $CACHE ($(du -h "$CACHE" | cut -f1))"

# --- 1. Lauf ----------------------------------------------------------------
log ">>> MI3DOR partial (erzwungener Cache), Gewichte 0.3/0.4/0.3 -> $OUT/partial/"
docker compose run --rm \
  -e SHREC_FORCE_PARTIAL_CACHE="/app/$CACHE" \
  oscar bash -lc \
  "cd /app/object_retrieval && PYTHONHASHSEED=0 MI3DOR_MODES=partial \
   MI3DOR_RESULT_FOLDER=$OUT MI3DOR_DINO_POOLING=mean \
   python3 -u retrieval_mi3dor_eval_oscarplus.py" \
  > logs/stage2_partial_fused.log 2>&1
RC=$?
log "    rc=$RC"

# --- 2. Ergebnis pruefen, nicht den Rueckgabewert ---------------------------
# Drei unabhaengige Pruefungen, weil jede allein schon einmal getaeuscht hat:
#   a) der Cache wurde WIRKLICH erzwungen (Logzeile),
#   b) der geschriebene Config-Block sagt ulip2_use_partial_views=True,
#   c) die Zahlen unterscheiden sich vom Full-Mesh-Lauf (kein stiller Fallback).
SUM="object_retrieval/$OUT/partial/metrics_summary_topk_15.json"
if [ ! -f "$SUM" ]; then
  log "FEHLGESCHLAGEN: kein metrics_summary geschrieben."; exit 1
fi
grep -q "FORCE-loaded" logs/stage2_partial_fused.log \
  && log "OK  (a) Partial-Cache erzwungen" \
  || log "WARNUNG (a) keine FORCE-loaded-Zeile — Pass lief vermutlich als full-mesh!"
grep -q "no partial PCs found" logs/stage2_partial_fused.log \
  && log "WARNUNG stiller Fallback auf full-mesh im Log gefunden!"

python3 - "$SUM" "$REF" <<'PY' | tee -a "$LOG"
import json, sys
s = json.load(open(sys.argv[1])); r = json.load(open(sys.argv[2]))
c = s["config"]
print("OK  (b) ulip2_use_partial_views=%s  weights=(%s, %s, %s)  n=%s"
      % (c["ulip2_use_partial_views"], c["weight_clip"], c["weight_dino"],
         c["weight_ulip"], s["num_queries"]))
same = 0
for arm in s["variants"]:
    a, b = s["variants"][arm], r["variants"][arm]
    if abs(a["NN_accuracy"] - b["NN_accuracy"]) < 1e-9:
        same += 1
    print("    %-24s NN=%6.2f  FT=%.4f   (full-mesh: NN=%6.2f  FT=%.4f)"
          % (arm, a["NN_accuracy"], a["FT_mean"], b["NN_accuracy"], b["FT_mean"]))
shape_id = abs(s["variants"]["ulip_only_full"]["NN_accuracy"]
               - r["variants"]["ulip_only_full"]["NN_accuracy"]) < 1e-9
print("OK  (c) Shape-Arm unterscheidet sich vom Full-Mesh-Lauf: %s"
      % ("NEIN — stiller Fallback!" if shape_id else "ja"))
PY

log "===== run_stage2_partial_fused DONE ====="
