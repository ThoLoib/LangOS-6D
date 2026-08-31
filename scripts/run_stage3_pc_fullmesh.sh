#!/usr/bin/env bash
# =============================================================================
# run_stage3_pc_fullmesh.sh — die fehlende Zelle der 2x2-Matrix.
#
#            | partial Gallery       | full-mesh Gallery
#   ---------+-----------------------+---------------------------
#   cross Q  | 3a_cross      0.4818  | 3a_fullmesh  (run_stage3_rest)
#   pc    Q  | 3a_pc         0.4636  | 3a_pc_fullmesh  <- HIER
#
# Warum diese und nicht nur die cross-Variante: SHREC'18 ist durchgehend
# pc-Modus, der Stage-1-Befund "partial > full-mesh" (0.5353 vs 0.4858) ist
# also ein pc-Vergleich. Nur pc-vs-pc ist das like-for-like-Gegenstueck auf BOP.
#
# Haengt sich an run_stage3_rest.sh an (kein Parallelbetrieb -> kein OOM;
# der 2026-08-27-Fehler waren drei gleichzeitige GPU-Jobs auf einer 24-GB-Karte).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/run_stage3_pc_fullmesh.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"
OUT=results_bop_stage3_v2

# --- auf die laufende Kette warten (PID ODER Prozessname, falls sie neu startet)
log "warte auf run_stage3_rest.sh ..."
while pgrep -f "run_stage3_rest.sh" >/dev/null 2>&1; do sleep 120; done
log "Kette beendet."

# --- GPU muss frei sein, sonst OOM
for _ in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${used:-99999}" -lt 9000 ] && break
  sleep 20
done
log "GPU used=${used:-?} MiB"

log ">>> 3a_pc_fullmesh  (pc-Query + full-mesh Gallery, keine Geometrie)"
docker compose run --rm oscar bash -lc \
  "cd /app/object_retrieval && PYTHONHASHSEED=0 \
   python3 -u eval_bop_pose.py --datasets all --mode 3a --fullmesh --pc-query \
   --output $OUT/3a_pc_fullmesh" > logs/stage3_3a_pc_fullmesh.log 2>&1
log "    rc=$?"

# Beleg, dass die Gallery WIRKLICH full-mesh war (Stage-2 fiel hier still zurueck)
grep -h "\[stage3\]\[fullmesh\]" logs/stage3_3a_pc_fullmesh.log 2>/dev/null | tee -a "$LOG"

log "===== 2x2 Gallery-Repraesentation x Query-Modus (R@1) ====="
python3 - <<'PY' | tee -a "$LOG"
import json, os
B = "object_retrieval/results_bop_stage3_v2"
def r1(p):
    f = os.path.join(B, p, "combined_stage3a.json")
    if not os.path.isfile(f): return None
    d = json.load(open(f)); return (d.get('overall') or d).get('recall@1')
def s(v): return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
rows = [("cross", "3a_cross", "3a_fullmesh"), ("pc", "3a_pc", "3a_pc_fullmesh")]
print(f"  {'Query':<8}{'partial':>10}{'full-mesh':>12}{'Delta':>10}")
for name, a, b in rows:
    va, vb = r1(a), r1(b)
    d = f"{vb-va:+.4f}" if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else "—"
    print(f"  {name:<8}{s(va):>10}{s(vb):>12}{d:>10}")
print("\n  Stage-1 (SHREC, pc-Modus): partial 0.5353 vs full-mesh 0.4858 -> +0.0495 fuer partial")
print("  Erwartung: die pc-Zeile ist das like-for-like-Gegenstueck dazu.")
PY

RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy "object_retrieval/$OUT/3a_pc_fullmesh" \
      "$REMOTE/object_retrieval/$OUT/3a_pc_fullmesh" --transfers 16 --stats 0 \
      >> logs/rclone_reruns.log 2>&1 || true
log "===== run_stage3_pc_fullmesh DONE ====="
