#!/usr/bin/env bash
# =============================================================================
# run_stage3_rest.sh — der verbleibende Stage-3-Rest in EINER Kette.
#
# Vorgeschichte: das Deckungs-Gate der vorigen Kette brach ab, nachdem ich den
# (redundanten) Borda-Arm absichtlich gekillt hatte — 0 % Deckung sah wie ein
# Fehllauf aus. Hier gibt es deshalb KEINEN harten Abbruch mehr; die Deckung
# wird nur protokolliert, und der Borda-Arm ist von vornherein nicht enthalten.
#
#   1. 3a pc-Geometrie: Distanz, Fitness   (Borda liegt als Altlauf vor: 0.413)
#   2. 3b_oscar                            (E5-Pose; --gt-records als DATEI)
#   3. 3a_fullmesh                         (Pfadfix 2026-08-28)
#   4. 3b/3c fuer jeden Arm, der 0.482 schlaegt
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/run_stage3_rest.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"
OUT=results_bop_stage3_v2; GT=$OUT/gt/combined_gt.json; BASE=0.482

# Gallery-Vorabpruefung bleibt (sie hat den 17-h-Leerlauf verhindert)
n=$(docker compose run --rm --no-deps oscar bash -lc \
   "cd /app/object_retrieval && python3 -c \"from dgedi_bridge import dgedi_health as h; print('NGAL=%d'%h().get('n_gallery',0))\"" 2>/dev/null \
   | grep -oE 'NGAL=[0-9]+' | tail -1 | cut -d= -f2)
log "dGeDi n_gallery=$n"
[ "${n:-0}" -lt 1000 ] && { log "ABBRUCH: falsche Gallery"; exit 3; }

s3(){ log ">>> $1"
  docker compose run --rm ${3:-} oscar bash -lc \
    "cd /app/object_retrieval && PYTHONHASHSEED=0 ${4:-} \
     python3 -u eval_bop_pose.py --datasets all --mode $2 --output $OUT/$1" \
    > "logs/stage3_$1.log" 2>&1
  log "    $1 rc=$?"
  python3 -c "
import json,os
f='object_retrieval/$OUT/$1/combined_stage3a.json'
if os.path.isfile(f):
    d=json.load(open(f)); g=d.get('geometry_coverage') or {}
    ap,t=g.get('n_geo_applied'),g.get('n_dgedi_query',1)
    cov=f'  Geo {100*ap/max(t,1):.1f}%' if ap is not None else ''
    print(f\"    R@1={(d.get('overall') or d).get('recall@1'):.4f}{cov}\")" | tee -a "$LOG"; }
r1(){ python3 -c "
import json,os
f='object_retrieval/$OUT/$1/combined_stage3a.json'
print('%.4f'%((json.load(open(f)).get('overall') or json.load(open(f))).get('recall@1',0)) if os.path.isfile(f) else '')" 2>/dev/null; }

log "===== 1/4  pc-Geometrie (Distanz, Fitness) ====="
s3 3a_pc_geo_distance "3a --dgedi --dgedi-repo --dgedi-top-k 5 --pc-query" "-e STAGE3_GEO_SIGNAL=distance" "STAGE3_GEO_SIGNAL=distance"
s3 3a_pc_geo_fitness  "3a --dgedi --dgedi-repo --dgedi-top-k 5 --pc-query" "-e STAGE3_GEO_SIGNAL=fitness"  "STAGE3_GEO_SIGNAL=fitness"

log "===== 2/4  3b_oscar (E5-Pose) ====="
s3 3b_oscar "3b --oscar-baseline --gt-records $GT"

log "===== 3/4  3a_fullmesh ====="
s3 3a_fullmesh "3a --fullmesh"
grep -h "\[stage3\]\[fullmesh\]" logs/stage3_3a_fullmesh.log 2>/dev/null | tee -a "$LOG"

log "===== 4/4  Gate gegen ULIP-2 cross ohne Geometrie ($BASE) ====="
for arm in 3a_fullmesh 3a_pc_geo_distance 3a_pc_geo_fitness; do
  v=$(r1 "$arm"); [ -z "$v" ] && { log "  $arm: kein Ergebnis"; continue; }
  if awk -v x="$v" -v b="$BASE" 'BEGIN{exit !(x+0>b+0)}'; then
    log "  $arm: R@1=$v > $BASE -> Pose"
    case "$arm" in
      3a_fullmesh) F="--fullmesh"; E=""; P="";;
      *_distance)  F="--pc-query --dgedi --dgedi-repo --dgedi-top-k 5"; E="-e STAGE3_GEO_SIGNAL=distance"; P="STAGE3_GEO_SIGNAL=distance";;
      *_fitness)   F="--pc-query --dgedi --dgedi-repo --dgedi-top-k 5"; E="-e STAGE3_GEO_SIGNAL=fitness";  P="STAGE3_GEO_SIGNAL=fitness";;
    esac
    n="${arm#3a_}"
    s3 "3b_$n" "3b $F --gt-records $GT" "$E" "$P"
    s3 "3c_$n" "3c $F --from-3a $OUT/$arm" "$E" "$P"
  else log "  $arm: R@1=$v <= $BASE"; fi
done

log "===== ZUSAMMENFASSUNG 3a ====="
python3 - <<'PY' | tee -a "$LOG"
import json,os
B="object_retrieval/results_bop_stage3_v2"
def g(p):
    f=os.path.join(B,p,"combined_stage3a.json")
    if not os.path.isfile(f): return "—"
    d=json.load(open(f)); c=(d.get('geometry_coverage') or {})
    ap,t=c.get('n_geo_applied'),c.get('n_dgedi_query',1)
    v=(d.get('overall') or d).get('recall@1')
    return f"{v:.3f}" + (f" ({100*ap/max(t,1):.0f}%)" if ap is not None else "")
print(f"  {'':<10}{'ohne Geo':>12}{'Borda':>12}{'Distanz':>12}{'Fitness':>12}")
for m,ng,pre in [("cross","3a_cross","3a_cross_geo"),("pc","3a_pc","3a_pc_geo")]:
    print(f"  {m:<10}{g(ng):>12}{g(pre):>12}{g(pre+'_distance'):>12}{g(pre+'_fitness'):>12}")
print(f"\n  E5 OSCAR: {g('3a_oscar')}   full-mesh: {g('3a_fullmesh')}")
print("  Stage-1 C1: Distanz 0.6405 > Borda 0.6362 > Fitness 0.6251 (nDCG)")
PY
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy "object_retrieval/$OUT" "$REMOTE/object_retrieval/$OUT" --transfers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true
log "===== run_stage3_rest DONE ====="
