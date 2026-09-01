#!/usr/bin/env bash
# =============================================================================
# stage4_query.sh — Query-Latenz messen. EIN Aufruf, eine Tabelle.
#
#   bash scripts/stage4_query.sh                    # ycbv, 50 Queries, 16 + 42 Views
#   bash scripts/stage4_query.sh -d lmo -n 30       # anderer Datensatz
#   bash scripts/stage4_query.sh --no-pose          # nur Retrieval
#   bash scripts/stage4_query.sh --geometry         # mit dGeDi-Rerank bei K=5
#
# Gemessene Kette (pipeline/run_pipeline.py, Schritte 1-8 + B2):
#   io -> segment (GroundingDINO+SAM2.1) -> pointcloud -> encode_query
#      -> clip -> dino -> ulip -> fusion -> [geometry] -> pose
#
# Schritt 7 (Skalenbestimmung) fehlt bewusst: er wurde als eigenstaendige
# Komponente verworfen und laeuft auch in der Stage-3-Konfiguration nicht.
#
# Kalt und warm werden getrennt ausgewiesen — die Modelle einmal zu laden kostet
# ein Vielfaches einer Query.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

DS=ycbv; NQ=50; VIEWS="16,42"; EXTRA=""; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    -d|--dataset) DS="$2"; shift 2;;
    -n|--n-queries) NQ="$2"; shift 2;;
    -v|--views) VIEWS="$2"; shift 2;;
    --no-pose) EXTRA="$EXTRA --no-pose"; shift;;
    --geometry) EXTRA="$EXTRA --geometry"; shift;;
    --proxy-only) EXTRA="$EXTRA --proxy-only"; shift;;
    -o|--out) OUT="$2"; shift 2;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "Unbekannte Option: $1"; exit 2;;
  esac
done
[ -z "$OUT" ] && OUT="results_stage4/query_latency_${DS}.json"
mkdir -p logs results_stage4

echo "== Stage 4b — Query-Latenz ================================"
echo "   Datensatz: $DS | Queries: $NQ | Views: $VIEWS${EXTRA:+ |$EXTRA}"
echo

docker compose run --rm oscar bash -lc \
  "cd /app && PYTHONHASHSEED=0 python3 -u experiments/experiment4_query_latency.py \
   --dataset $DS --n-queries $NQ --views $VIEWS $EXTRA --out $OUT" \
  2>&1 | tee "logs/stage4_query_${DS}.log" \
  | grep -E "^\[stage4\]|^  [A-Za-z_]+ +[0-9]|SUMME|Ende zu Ende|^===|^    [a-z_]+ +[0-9]"

echo
python3 - "$OUT" <<'PY'
import json, os, sys
f = sys.argv[1]
if not os.path.isfile(f):
    sys.exit("keine Ergebnisse unter " + f)
d = json.load(open(f))
Q = d.get("stage1_quality_ndcg", {})
views = [str(v) for v in d["views"]]

def fmt(s):
    return f"{s:.3f} s" if s >= 1 else f"{s*1000:.1f} ms"

# retrieval_total ist eine Klammer um clip/dino/ulip/fusion — nicht mitzaehlen.
CONTAINER = {"retrieval_total"}
print("== QUERY je Anfrage (warm) ================================")
print(f"  {'Schritt':<16}" + "".join(f"{v + ' Views':>14}" for v in views))
seen = []
for v in views:
    for k in d["by_views"][v]["per_step"]:
        if k not in seen:
            seen.append(k)
tot = {v: 0.0 for v in views}
for k in seen:
    per = {v: d["by_views"][v]["per_step"].get(k, {}).get("median", 0.0) for v in views}
    mark = "  (Klammer)" if k in CONTAINER else ""
    print(f"  {k:<16}" + "".join(f"{fmt(per[v]):>14}" for v in views) + mark)
    if k not in CONTAINER:
        for v in views:
            tot[v] += per[v]
print("  " + "-" * (16 + 14 * len(views)))
print(f"  {'ENDE ZU ENDE':<16}" + "".join(f"{fmt(tot[v]):>14}" for v in views))

base = max(views, key=int)
print(f"\n  {'Views':>6}{'Kosten':>10}{'nDCG (Stage 1)':>18}")
for v in sorted(views, key=int):
    q = Q.get(v)
    print(f"  {v:>6}{100*tot[v]/tot[base] if tot[base] else 0:>9.0f}%"
          f"{(f'{q:.4f}' if q else '—'):>18}")

nd = sum(d["by_views"][v].get("n_no_detection", 0) or 0 for v in views)
print(f"\n  Gallery {d['gallery_size']} | gemessene Queries "
      f"{d['by_views'][base]['per_query_total_s'].get('n')} | ohne Detektion {nd}")
print("  Kaltstart, einmalig: "
      + ", ".join(f"{k} {v:.1f} s" for k, v in (d.get("cold_start_s") or {}).items()))
print(f"\n  Rohdaten: {f}")
PY
