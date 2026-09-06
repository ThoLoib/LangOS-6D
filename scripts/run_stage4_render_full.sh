#!/usr/bin/env bash
# =============================================================================
# run_stage4_render_full.sh — die render-Stufe als VOLLERHEBUNG nachziehen.
#
# Warum
# -----
# Alle uebrigen Onboarding-Stufen sind ueber alle 59 Ziel-CADs gemessen; nur
# `render` ruhte auf n=5 — ausgerechnet der Posten, der ueber die Haelfte der
# Onboarding-Zeit ausmacht. Die Begruendung war die Laufzeit, sie haelt der
# Nachrechnung aber nicht stand: 59 x (14,45 + 34,68) s sind rund 48 Minuten.
#
# Warum gewartet wird
# -------------------
# Blender rendert mit CYCLES auf CUDA. Parallel zu einem anderen GPU-Job
# gemessen waere die Zahl wertlos — die Konkurrenz verfaelscht genau die
# Groesse, die gemessen werden soll. Das Skript wartet deshalb, bis die GPU
# frei ist, und prueft das doppelt: kein OSCAR-Eval-Prozess mehr, und der
# belegte Speicher unter der Schwelle der Dienste im Leerlauf.
#
# Das n=5-Ergebnis wird NICHT ueberschrieben — es bleibt als Vergleich liegen.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results_stage4
LOG=logs/run_stage4_render_full.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"

BLENDER="${BLENDER:-/home/tessa/Cap3D/captioning_pipeline/blender-3.4.1-linux-x64/blender}"
OUT=results_stage4/onboarding_render_n59.json
REF=results_stage4/onboarding_render.json
IDLE_MIB=${IDLE_MIB:-9000}     # dgedi + foundationpose im Leerlauf ~6,7 GiB

# --- 0. Vorbedingungen ------------------------------------------------------
[ -x "$BLENDER" ] || { log "ABBRUCH: Blender nicht unter $BLENDER"; exit 1; }
[ -f "$OUT" ] && { log "ABBRUCH: $OUT existiert bereits — nichts ueberschreiben."; exit 1; }

# --- 1. Auf die freie GPU warten -------------------------------------------
log "warte auf freie GPU (laufender Eval-Prozess + Speicherbelegung) ..."
for i in $(seq 1 720); do            # bis zu 12 h
  RUNNING=$(pgrep -fc "retrieval_mi3dor_eval_oscarplus|eval_bop_pose|experiment1_shrec18" 2>/dev/null || echo 0)
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  USED=${USED:-99999}
  if [ "$RUNNING" -eq 0 ] && [ "$USED" -lt "$IDLE_MIB" ]; then
    log "GPU frei (belegt ${USED} MiB, kein Eval-Prozess) — starte nach 60 s Nachlauf."
    sleep 60
    break
  fi
  [ $((i % 20)) -eq 0 ] && log "  ... noch belegt (${USED} MiB, $RUNNING Prozesse)"
  sleep 60
done

RUNNING=$(pgrep -fc "retrieval_mi3dor_eval_oscarplus|eval_bop_pose|experiment1_shrec18" 2>/dev/null || echo 0)
[ "$RUNNING" -eq 0 ] || { log "ABBRUCH: nach 12 h laeuft immer noch ein Eval-Prozess."; exit 1; }

# --- 2. Vollerhebung --------------------------------------------------------
log ">>> render ueber ALLE 59 CADs, 16 + 42 Views -> $OUT"
log "    erwartete Dauer ~48 min"
python3 experiments/experiment4_onboarding.py --stages render \
  --num-views 16,42 --blender "$BLENDER" \
  --work-dir "$HOME/.stage4_render_work_n59" \
  --out "$OUT" > logs/stage4_render_n59.log 2>&1
log "    rc=$?"

# --- 3. Ergebnis pruefen, nicht den Rueckgabewert ---------------------------
[ -f "$OUT" ] || { log "FEHLGESCHLAGEN: keine Ergebnisdatei."; exit 1; }
python3 - "$OUT" "$REF" <<'PY' | tee -a "$LOG"
import json, sys, os
n = json.load(open(sys.argv[1]))
o = json.load(open(sys.argv[2])) if os.path.isfile(sys.argv[2]) else None
bad = False
print("  %-8s %5s %10s %10s %10s   %s" % ("Views","n","Median","IQR","p95","vorher (n=5)"))
for v, b in sorted(n["by_views"].items(), key=lambda kv: int(kv[0])):
    s = b.get("per_step", {}).get("render", {})
    ref = (o or {}).get("by_views", {}).get(v, {}).get("per_step", {}).get("render", {})
    if s.get("n", 0) < 55:
        bad = True
    print("  %-8s %5s %9.2fs %9.2fs %9.2fs   %s" % (
        v, s.get("n"), s.get("median", 0), s.get("iqr", 0), s.get("p95", 0),
        ("%.2fs  ->  Abweichung %+.2fs (%+.1f %%)" % (
            ref["median"], s["median"] - ref["median"],
            100 * (s["median"] - ref["median"]) / ref["median"]) if ref else "—")))
# der eigentliche Zweck: haelt die alte Zahl?
print()
if bad:
    print("  WARNUNG: weniger als 55 Objekte gemessen — keine Vollerhebung.")
else:
    print("  OK — Vollerhebung ueber alle 59 CADs.")
PY

log "===== run_stage4_render_full DONE ====="
