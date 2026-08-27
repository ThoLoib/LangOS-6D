#!/usr/bin/env bash
# =============================================================================
# run_mi3dor_oscar_legacy.sh — OSCAR-Legacy-Baseline auf MI3DOR: die publizierte
# Kaskade (CLIP-tau=0.37-Pruning -> DINOv2 Best-View) bei **8 Views**, gescort
# mit UNSEREM Evaluator und UNSERER Gallery.
#
# Warum so (siehe docs/OSCAR_LEGACY_COMPARISON.md):
#   * Pullis Evaluator wendet die CLIP-Shortlist NICHT auf das Ranking an — ihre
#     Zahlen sind reines DINOv2-Retrieval. Wir reproduzieren daher nicht ihren
#     Code, sondern den publizierten MECHANISMUS (Kaskade) bei V=8.
#   * Gallery/Views unterscheiden sich (sie: 1817 Objekte x 1 View; wir: 3848 x 42
#     auf 8 getrimmt). Das wird im Text ausgewiesen, nicht wegdefiniert.
#
# OSCAR hat KEINEN Shape-Kanal -> die CLIP/DINO-Arme sind shape-modus-unabhaengig,
# also MI3DOR_MODES=fullmesh (halbiert die Laufzeit). Der DINO-Cache haelt alle 42
# Views, step4._apply_view_limit() trimmt auf 8 -> kein Re-Encoding.
#
# Relevanter Arm im Ergebnis: `oscar_maxview` (CLIP-tau-Shortlist + DINO Hard-Max)
# = die treue OSCAR-Kaskade. Zusaetzlich fallen `oscar_softmax` und
# `clip_pruned_dino_ulip` an (View-Aggregations- bzw. Shape-Variante).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_mi3dor_oscar_legacy.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"
OUT=results_mi3dor_oscar_legacy_v8

# --- auf Stage-2 warten (nicht zwei GPU-Jobs gleichzeitig) ------------------
log "warte auf Stage-2 (run_stage2_after_stage1 DONE) ..."
for i in $(seq 1 2880); do   # bis zu 24h
  grep -q "run_stage2_after_stage1 DONE" logs/run_stage2_after_stage1.log 2>/dev/null && break
  sleep 30
done
grep -q "run_stage2_after_stage1 DONE" logs/run_stage2_after_stage1.log 2>/dev/null \
  && log "Stage-2 fertig — starte OSCAR-Legacy-Lauf." \
  || log "WARN: Stage-2 nicht als fertig markiert; starte trotzdem."

log ">>> OSCAR-Legacy: V=8, tau=0.37, fullmesh-only -> object_retrieval/$OUT"
docker compose run --rm oscar bash -lc \
  "cd /app/object_retrieval && PYTHONHASHSEED=0 \
   MI3DOR_MODES=fullmesh MI3DOR_NUM_VIEWS=8 MI3DOR_DINO_POOLING=mean \
   MI3DOR_RESULT_FOLDER=$OUT \
   python3 -u retrieval_mi3dor_eval_oscarplus.py" > logs/mi3dor_oscar_legacy.log 2>&1
rc=$?; log "    rc=$rc"

# --- Sanity: hat es wirklich 8 Views benutzt? -------------------------------
log "Sanity — View-Limit im Log:"
grep -iE "view limit|num_views|views per|applied view" logs/mi3dor_oscar_legacy.log | head -3 | tee -a "$LOG"

# --- Ergebnisse ausgeben ----------------------------------------------------
S="object_retrieval/$OUT/fullmesh/metrics_summary_topk_15.json"
if [ -f "$S" ]; then
  log "===== OSCAR-Legacy (V=8) — alle Arme ====="
  python3 - <<PY | tee -a "$LOG"
import json
d=json.load(open("$S"))
v=d.get("variants",{})
print(f"  {'Arm':<26}{'NN':>7}{'FT':>8}{'ST':>8}{'F1':>8}{'nDCG@2R':>9}{'mAP':>8}{'ANMRR':>8}")
for k,m in v.items():
    print(f"  {k:<26}{m.get('NN_accuracy',0):>7.2f}{m.get('FT_mean',0):>8.3f}"
          f"{m.get('ST_mean',0):>8.3f}{m.get('F1_mean',0):>8.3f}"
          f"{m.get('nDCG@2R_mean',0):>9.3f}{m.get('mAP',0):>8.3f}{m.get('ANMRR_mean',0):>8.3f}")
print("\n  -> OSCAR-Legacy-Baseline = 'oscar_maxview' (CLIP-tau + DINO Hard-Max)")
print("  -> Publiziert (Pulli Table 1): NN 89.4 | FT 0.708 | ST 0.850 | F 0.238 | DCG 0.844 | ANMRR 0.205")
print("     ACHTUNG: andere Gallery (1817x1 View vs 3848x8) + ihr Evaluator nutzt die")
print("     CLIP-Shortlist nicht -> ihre Zahlen sind DINO-only. Nur als Kontext zitieren.")
PY
fi
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy "object_retrieval/$OUT" "$REMOTE/object_retrieval/$OUT" --transfers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true
log "===== run_mi3dor_oscar_legacy DONE (rc=$rc) ====="
