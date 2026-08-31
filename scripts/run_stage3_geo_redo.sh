#!/usr/bin/env bash
# =============================================================================
# run_stage3_geo_redo.sh — 3a-Geometrie neu, nachdem der erste Versuch (17 h)
# ins Leere lief: der dGeDi-Dienst hing noch am SHREC-Cache (n_gallery=3308),
# also fand er fuer keinen BOP-Kandidaten Deskriptoren -> 0/12284 Registrierungen
# und die "Geometrie"-Arme reproduzierten exakt das Ranking ohne Geometrie.
#
# Konsequenz: VOR jedem Lauf wird die Gallery-Groesse geprueft und bei falschem
# Wert abgebrochen, statt stundenlang Nullergebnisse zu produzieren.
# Zusaetzlich nach dem ersten Lauf ein Deckungs-Gate (>50 % angewandt).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/run_stage3_geo_redo.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"
OUT=results_bop_stage3_v2

check_gallery(){
  # Marker-Zeile statt roher Ziffernfilterung: der CUDA-Banner enthaelt selbst
  # Zahlen und wuerde sonst mitgelesen (Fehler beim ersten Versuch).
  n=$(docker compose run --rm --no-deps oscar bash -lc \
      "cd /app/object_retrieval && python3 -c \"from dgedi_bridge import dgedi_health as h; print('NGAL=%d' % h().get('n_gallery',0))\"" 2>/dev/null \
      | grep -oE 'NGAL=[0-9]+' | tail -1 | cut -d= -f2)
  log "dGeDi n_gallery=$n"
  if [ "${n:-0}" -lt 1000 ] || [ "${n:-0}" -gt 2000 ]; then
    log "ABBRUCH: n_gallery=$n ist nicht die BOP-Gallery (erwartet ~1316).";
    log "  Fix: docker compose up -d --force-recreate dgedi   (ohne DGEDI_CACHE_DIR)"; exit 3
  fi
}
check_gallery

run(){ log ">>> $1"
  docker compose run --rm -e STAGE3_GEO_SIGNAL=$2 oscar bash -lc \
    "cd /app/object_retrieval && PYTHONHASHSEED=0 STAGE3_GEO_SIGNAL=$2 \
     python3 -u eval_bop_pose.py --datasets all --mode 3a --dgedi --dgedi-repo \
     --dgedi-top-k 5 $3 --output $OUT/$1" > "logs/stage3_$1.log" 2>&1
  log "    $1 rc=$?"
  python3 -c "
import json,os
f='object_retrieval/$OUT/$1/combined_stage3a.json'
if os.path.isfile(f):
    d=json.load(open(f)); g=d.get('geometry_coverage',{}) or {}
    ap,tot=g.get('n_geo_applied',0),g.get('n_dgedi_query',1)
    print(f'    Deckung: {ap}/{tot} ({100*ap/max(tot,1):.1f} %)  R@1={(d.get(\"overall\") or d).get(\"recall@1\"):.3f}')
    raise SystemExit(0 if ap>0.5*tot else 9)" | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" = 9 ] && { log "ABBRUCH: Geometrie griff bei <50 % — nicht weiterrechnen."; exit 4; }
}

run 3a_cross_geo_distance distance ""
run 3a_cross_geo_fitness  fitness  ""
run 3a_cross_geo_borda    borda    ""
run 3a_pc_geo_distance    distance "--pc-query"
run 3a_pc_geo_fitness     fitness  "--pc-query"

# 3b OSCAR nachholen — beim ersten Versuch war --gt-records ein Verzeichnis
log ">>> 3b_oscar (gt-records als DATEI)"
docker compose run --rm oscar bash -lc \
  "cd /app/object_retrieval && PYTHONHASHSEED=0 python3 -u eval_bop_pose.py \
   --datasets all --mode 3b --oscar-baseline \
   --gt-records $OUT/gt/combined_gt.json --output $OUT/3b_oscar" > logs/stage3_3b_oscar.log 2>&1
log "    3b_oscar rc=$?"

log "===== ZUSAMMENFASSUNG ====="
python3 - <<'PY' | tee -a "$LOG"
import json, os
B="object_retrieval/results_bop_stage3_v2"
def r1(p):
    f=os.path.join(B,p,"combined_stage3a.json")
    if not os.path.isfile(f): return "—"
    v=(json.load(open(f)).get('overall') or json.load(open(f))).get('recall@1')
    return f"{v:.3f}" if isinstance(v,(int,float)) else "—"
print(f"  {'Modus':<8}{'ohne Geo':>10}{'Distanz':>10}{'Fitness':>10}{'Borda':>10}")
for n,ng,pre in [("cross","3a_cross","3a_cross_geo"),("pc","3a_pc","3a_pc_geo")]:
    print(f"  {n:<8}{r1(ng):>10}{r1(pre+'_distance'):>10}{r1(pre+'_fitness'):>10}{r1(pre+'_borda'):>10}")
print("\n  Stage-1 C1: Distanz 0.6405 > Borda 0.6362 > Fitness 0.6251 (nDCG)")
PY
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy "object_retrieval/$OUT" "$REMOTE/object_retrieval/$OUT" --transfers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true
log "===== run_stage3_geo_redo DONE ====="
