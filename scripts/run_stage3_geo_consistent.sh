#!/usr/bin/env bash
# =============================================================================
# run_stage3_geo_consistent.sh — 3a-Geometriearme mit dem STAGE-1-KONSISTENTEN
# Rangkriterium (reine ausgerichtete Distanz statt Borda).
#
# Warum: Stage-1 C1 weist die Distanz als bestes Geometriesignal aus
# (0.6405 vs 0.6362 Borda vs 0.6251 fitness), Stage 3 nutzte bisher Borda.
# Der Unterschied ist bei K=5 klein (~0.002 nDCG in Stage-1-Groessenordnung),
# aber die Methodik soll ueber die Stufen konsistent sein.
#
# Nebenbei schreibt der Lauf die ROHWERTE (fitness, d_ransac) in die Records
# -> kuenftige Kriterienwechsel sind Tier-2-Ableitungen ohne Neuregistrierung.
#
# Nur 3a (Retrieval). 3b_geo wird NICHT neu gerechnet: dessen Aussage lautet
# ohnehin "Geometrie in der Pose-Stufe abschalten", und ein konsistenter
# Negativbefund aendert die Schlussfolgerung nicht (im Text so ausweisen).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_stage3_geo_consistent.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"
OUT=results_bop_stage3_v2

log "warte auf die Stage-3-Kette (run_stage3_e5 DONE) ..."
for i in $(seq 1 2880); do
  grep -q "run_stage3_e5 DONE" logs/run_stage3_e5.log 2>/dev/null && break
  sleep 30
done
grep -q "run_stage3_e5 DONE" logs/run_stage3_e5.log 2>/dev/null \
  && log "Stage-3-Kette fertig — starte konsistente Geometriearme." \
  || log "WARN: Stage-3-Kette nicht als fertig markiert; starte trotzdem."

# dGeDi muss laufen (auf der BOP-Gallery, wie die bestehenden Geo-Arme)
docker compose up -d dgedi >/dev/null 2>&1
log "dGeDi: $(docker inspect --format '{{.State.Health.Status}}' "$(docker compose ps -q dgedi 2>/dev/null)" 2>/dev/null)"

run(){  # $1=label  $2=extra-flags
  log ">>> $1 (STAGE3_GEO_SIGNAL=distance)"
  docker compose run --rm -e STAGE3_GEO_SIGNAL=distance oscar bash -lc \
    "cd /app/object_retrieval && PYTHONHASHSEED=0 STAGE3_GEO_SIGNAL=distance \
     python3 -u eval_bop_pose.py --datasets all --mode 3a --dgedi --dgedi-repo \
     --dgedi-top-k 5 $2 --output $OUT/$1" > "logs/stage3_$1.log" 2>&1
  log "    $1 rc=$?"
  f="object_retrieval/$OUT/$1/combined_stage3a.json"
  [ -f "$f" ] && python3 -c "
import json;d=json.load(open('$f'));m=d.get('overall',d)
print(f\"    R@1={m.get('recall@1')} R@5={m.get('recall@5')} MRR={m.get('mrr')}\")
g=d.get('geometry_coverage');print(f'    geometry_coverage: {g}')" | tee -a "$LOG"
}

# Beide Geometriesignale GETRENNT — wie Stage-1 C1 (Distanz vs. Fitness), plus
# Borda als Referenz auf die bisherige Stage-3-Variante. Alle drei aus DERSELBEN
# Registrierung: der Lauf schreibt die Rohwerte (fitness, d_ransac) in die
# Records, die beiden weiteren Kriterien sind danach Tier-2-Ableitungen.
for SIG in distance fitness borda; do
  log ">>> cross · Kriterium=$SIG"
  docker compose run --rm -e STAGE3_GEO_SIGNAL=$SIG oscar bash -lc \
    "cd /app/object_retrieval && PYTHONHASHSEED=0 STAGE3_GEO_SIGNAL=$SIG \
     python3 -u eval_bop_pose.py --datasets all --mode 3a --dgedi --dgedi-repo \
     --dgedi-top-k 5 --output $OUT/3a_cross_geo_$SIG" > "logs/stage3_3a_cross_geo_$SIG.log" 2>&1
  log "    cross/$SIG rc=$?"
done
for SIG in distance fitness; do
  log ">>> pc · Kriterium=$SIG"
  docker compose run --rm -e STAGE3_GEO_SIGNAL=$SIG oscar bash -lc \
    "cd /app/object_retrieval && PYTHONHASHSEED=0 STAGE3_GEO_SIGNAL=$SIG \
     python3 -u eval_bop_pose.py --datasets all --mode 3a --dgedi --dgedi-repo \
     --dgedi-top-k 5 --pc-query --output $OUT/3a_pc_geo_$SIG" > "logs/stage3_3a_pc_geo_$SIG.log" 2>&1
  log "    pc/$SIG rc=$?"
done

log "===== Vergleich Borda (alt) vs Distanz (konsistent) ====="
python3 - <<'PY' | tee -a "$LOG"
import json,os
B="object_retrieval/results_bop_stage3_v2"
def r1(p):
    f=os.path.join(B,p,"combined_stage3a.json")
    if not os.path.isfile(f): return None
    d=json.load(open(f)); v=(d.get('overall',d)).get('recall@1')
    return f"{v:.3f}" if isinstance(v,(int,float)) else "—"
print(f"  {'Modus':<8}{'ohne Geo':>10}{'Distanz':>10}{'Fitness':>10}{'Borda':>10}{'Borda(alt)':>12}")
for name,nogeo,pre in [("cross","3a_cross","3a_cross_geo"),("pc","3a_pc","3a_pc_geo")]:
    print(f"  {name:<8}{r1(nogeo):>10}{r1(pre+'_distance'):>10}"
          f"{r1(pre+'_fitness'):>10}{r1(pre+'_borda'):>10}{r1(pre):>12}")
print("\n  Stage-1 C1 zum Vergleich: Distanz 0.6405 > Borda 0.6362 > Fitness 0.6251 (nDCG)")
PY
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy "object_retrieval/$OUT" "$REMOTE/object_retrieval/$OUT" --transfers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true
log "===== run_stage3_geo_consistent DONE ====="
