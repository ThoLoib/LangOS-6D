#!/usr/bin/env bash
# =============================================================================
# stage4_onboarding.sh — Onboarding-Latenz messen. EIN Aufruf, eine Tabelle.
#
#   bash scripts/stage4_onboarding.sh                  # alle 59 CADs, 16 + 42 Views
#   bash scripts/stage4_onboarding.sh -n 5             # nur 5 CADs (Schnelltest)
#   bash scripts/stage4_onboarding.sh -v 42            # nur eine View-Zahl
#   bash scripts/stage4_onboarding.sh --no-render      # ohne Blender
#
# Warum ein Wrapper: die Kette ist auf zwei Umgebungen verteilt. Blender liegt
# unter /home/tessa/… und ist NICHT ins Compose gemountet, laeuft also auf dem
# Host; Encoder und LLaVA brauchen den Container. Das Skript fuehrt beides
# nacheinander aus und legt die Ergebnisse zusammen.
#
# Gemessene Kette (docs/PREPROCESSING.md §1):
#   mesh -> render -> partial -> describe -> embed(dino|clip|ulip) -> cache
# SYNC und VERIFY (rclone auf Drive) sind bewusst NICHT enthalten: sie sind
# Netz- und Infrastrukturzeit, keine Eigenschaft der Pipeline.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

NOBJ=0                # 0 = alle 59
VIEWS="16,42"
DO_RENDER=1
NRENDER=3             # Blender ist teuer; wenige Objekte reichen fuer Median+IQR
OUT="results_stage4/onboarding.json"
BLENDER="${BLENDER:-/home/tessa/Cap3D/captioning_pipeline/blender-3.4.1-linux-x64/blender}"

while [ $# -gt 0 ]; do
  case "$1" in
    -n|--max-objects) NOBJ="$2"; shift 2;;
    -v|--views)       VIEWS="$2"; shift 2;;
    --no-render)      DO_RENDER=0; shift;;
    --render-objects) NRENDER="$2"; shift 2;;
    -o|--out)         OUT="$2"; shift 2;;
    -h|--help)        sed -n '2,20p' "$0"; exit 0;;
    *) echo "Unbekannte Option: $1"; exit 2;;
  esac
done

mkdir -p logs results_stage4
LIM=""; [ "$NOBJ" -gt 0 ] && LIM="--max-objects $NOBJ"

echo "== Stage 4a — Onboarding =================================="
echo "   CADs: $([ "$NOBJ" -gt 0 ] && echo "$NOBJ" || echo "alle 59") | Views: $VIEWS"
echo

# ---- 1. Container: mesh, partial, describe, embed, Cache-Anhaengen ---------
docker compose run --rm oscar bash -lc \
  "cd /app && PYTHONHASHSEED=0 python3 -u experiments/experiment4_onboarding.py \
   --stages mesh,partial,describe,embed --reuse-renders --num-views $VIEWS \
   $LIM --measure-invalidation --out $OUT" 2>&1 | tee logs/stage4_onboarding.log \
  | grep -E "^\[stage4\]|^  [A-Za-z_]+ +[0-9]|SUMME|Gesamt je Objekt|^===|Ladezeit|^    [a-z_]+ +[0-9]"

# ---- 2. Host: Blender ------------------------------------------------------
if [ "$DO_RENDER" -eq 1 ]; then
  if [ -x "$BLENDER" ]; then
    echo; echo "== Render-Stufe (Host, $NRENDER Objekte) =================="
    python3 experiments/experiment4_onboarding.py --stages render \
      --num-views "$VIEWS" --max-objects "$NRENDER" --blender "$BLENDER" \
      --work-dir "$HOME/.stage4_render_work" \
      --out results_stage4/onboarding_render.json 2>&1 \
      | tee logs/stage4_render.log | grep -E "^  render +[0-9]|Gesamt je Objekt|Views ---"
  else
    echo; echo "WARNUNG: Blender nicht unter $BLENDER — render uebersprungen."
    echo "         Mit BLENDER=/pfad/zu/blender neu aufrufen."
  fi
fi

# ---- 3. Gesamttabelle ------------------------------------------------------
echo
python3 - "$OUT" results_stage4/onboarding_render.json <<'PY'
import json, os, sys
main, rend = sys.argv[1], sys.argv[2]
if not os.path.isfile(main):
    sys.exit("keine Ergebnisse unter " + main)
d = json.load(open(main))
r = json.load(open(rend)) if os.path.isfile(rend) else None
Q = d.get("stage1_quality_ndcg", {})

def fmt(s):
    return f"{s/60:.2f} min" if s >= 60 else (f"{s:.2f} s" if s >= 1 else f"{s*1000:.1f} ms")

views = [str(v) for v in d["view_counts"]]
print("== ONBOARDING je CAD ======================================")
print(f"  {'Schritt':<18}" + "".join(f"{v + ' Views':>14}" for v in views))
steps, totals = [], {v: 0.0 for v in views}
if r:
    steps.append(("render", {v: (r["by_views"].get(v, {}).get("per_step", {})
                                 .get("render", {}).get("median", 0.0)) for v in views}))
seen = []
for v in views:
    for k in d["by_views"][v]["per_step"]:
        if k not in seen:
            seen.append(k)
for k in seen:
    steps.append((k, {v: d["by_views"][v]["per_step"].get(k, {}).get("median", 0.0)
                      for v in views}))
for name, per in steps:
    print(f"  {name:<18}" + "".join(f"{fmt(per[v]):>14}" for v in views))
    for v in views:
        totals[v] += per[v]
print("  " + "-" * (18 + 14 * len(views)))
print(f"  {'GESAMT':<18}" + "".join(f"{fmt(totals[v]):>14}" for v in views))

base = max(views, key=lambda x: int(x))
print(f"\n  {'Views':>6}{'Kosten':>10}{'nDCG (Stage 1)':>18}")
for v in sorted(views, key=int):
    share = 100 * totals[v] / totals[base] if totals[base] else 0
    q = Q.get(v)
    print(f"  {v:>6}{share:>9.0f}%{(f'{q:.4f}' if q else '—'):>18}")

inv = d.get("invalidation") or {}
if inv.get("extrapolated_full_reencode_min"):
    print(f"\n  Inkrementell (dieses Objekt):        {fmt(totals[base])}")
    print(f"  Was der aktuelle Cache erzwingt:     "
          f"{inv['extrapolated_full_reencode_min']:.1f} min "
          f"({inv['gallery_size']} Objekte neu encodiert)")
ml = d.get("model_load_once_s") or {}
if ml:
    print("\n  Einmalige Modell-Ladezeit (keine Onboarding-Kosten): "
          + ", ".join(f"{k} {v:.1f} s" for k, v in ml.items()))
print(f"\n  Rohdaten: {main}" + (f" + {rend}" if r else ""))
PY
