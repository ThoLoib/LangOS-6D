#!/usr/bin/env bash
# =============================================================================
# run_stage4_full.sh — die vollstaendige Stage-4-Messung, beide Seiten.
#
#   1. Query-Latenz  ycbv, 50 Queries, 16 und 42 Views, MIT Pose
#   2. Query-Latenz  dieselbe Kette MIT Geometrie (K=5), separat gemessen
#   3. Onboarding    alle 59 Ziel-CADs, 16 und 42 Views, volle Kette
#                    + Render auf dem Host + Invalidierungsaufschlag
#
# Alle bisherigen Onboarding-Zahlen stammen aus mehreren Teillaeufen, weil
# `partial`, `embed_clip` und das echte Cache-Anhaengen erst am 2026-09-01
# messbar wurden. Dieser Lauf ersetzt sie durch EINE zusammenhaengende Messung.
#
# Sequenziell, weil zwei GPU-Jobs auf einer 24-GB-Karte am 2026-08-27 den OOM
# ausgeloest haben.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/run_stage4_full.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"
mkdir -p results_stage4

wait_gpu(){ for _ in $(seq 1 90); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${u:-99999}" -lt 12000 ] && return 0; sleep 20; done; }

log "===== 1/3  Query-Latenz (mit Pose) ====="
wait_gpu
bash scripts/stage4_query.sh -d ycbv -n 50 -v 16,42 \
  -o results_stage4/query_latency_ycbv.json >> "$LOG" 2>&1
log "    rc=$?"

log "===== 2/3  Query-Latenz mit Geometrie (K=5, ohne Pose) ====="
# Getrennt gemessen: die Geometrie kostet ~5,5 s je Anfrage und wuerde die
# uebrigen Posten in der Anteilsrechnung sonst unsichtbar machen. dGeDi steht
# auf der BOP-Gallery — fuer Stage 4 richtig, das ist der BOP-Query-Pfad.
wait_gpu
bash scripts/stage4_query.sh -d ycbv -n 25 -v 42 --geometry --no-pose \
  -o results_stage4/query_latency_ycbv_geo.json >> "$LOG" 2>&1
log "    rc=$?"

log "===== 3/3  Onboarding, alle 59 CADs ====="
wait_gpu
bash scripts/stage4_onboarding.sh -v 16,42 --render-objects 5 \
  -o results_stage4/onboarding.json >> "$LOG" 2>&1
log "    rc=$?"

log "===== ZUSAMMENFASSUNG ====="
python3 - <<'PY' | tee -a "$LOG"
import json, os
R = "results_stage4"
def load(n):
    f = os.path.join(R, n)
    return json.load(open(f)) if os.path.isfile(f) else None
def fmt(s): return f"{s/60:.1f} min" if s >= 60 else (f"{s:.2f} s" if s >= 1 else f"{s*1000:.0f} ms")

q, g, o = load("query_latency_ycbv.json"), load("query_latency_ycbv_geo.json"), load("onboarding.json")
r = load("onboarding_render.json")
if q:
    print("  QUERY je Anfrage (warm, mit Pose)")
    for v in [str(x) for x in q["views"]]:
        b = q["by_views"][v]
        tot = b["per_query_total_s"]["median"]
        print(f"    {v:>3} Views: {fmt(tot):>9}   " + " · ".join(
            f"{k} {s['median']*1000:.0f}ms" for k, s in b["per_step"].items()
            if s.get("n") and k != "retrieval_total"))
if g:
    v = str(g["views"][0]); st = g["by_views"][v]["per_step"].get("geometry", {})
    if st: print(f"\n  GEOMETRIE (K=5): {fmt(st['median'])} je Anfrage")
if o:
    print("\n  ONBOARDING je CAD")
    for v in [str(x) for x in o["view_counts"]]:
        b = o["by_views"][v]; tot = b["per_object_total_s"]["median"]
        rv = (r or {}).get("by_views", {}).get(v, {}).get("per_step", {}).get("render", {}).get("median", 0)
        print(f"    {v:>3} Views: {fmt(tot + rv):>9} (davon Render {fmt(rv)})   " + " · ".join(
            f"{k} {s['median']*1000:.0f}ms" for k, s in b["per_step"].items() if s.get("n")))
    inv = o.get("invalidation") or {}
    if inv.get("extrapolated_full_reencode_min"):
        print(f"\n  Inkrementell gegen Invalidierung: "
              f"{fmt(o['by_views'][str(max(o['view_counts']))]['per_object_total_s']['median'])}"
              f" gegen {inv['extrapolated_full_reencode_min']:.1f} min "
              f"({inv['gallery_size']} Objekte)")
PY
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy results_stage4 "$REMOTE/results_stage4" --transfers 8 --stats 0 \
  >> logs/rclone_reruns.log 2>&1 || true
log "===== FERTIG ====="
