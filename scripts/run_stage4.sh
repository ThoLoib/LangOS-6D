#!/usr/bin/env bash
# =============================================================================
# run_stage4.sh — beide Latenzexperimente, streng sequenziell.
#
# Sequenziell, weil sich sonst zwei GPU-Jobs eine 24-GB-Karte teilen — genau
# der Fehler vom 2026-08-27 (drei parallele Waiter -> CUDA-OOM).
#
#   1. Query-Latenz, ycbv, mit Pose, 16 gegen 42 Views
#   2. Onboarding, 16 gegen 42 Views
#
# Zur render-Stufe: Blender liegt unter /home/tessa/blender/..., also AUSSERHALB
# des Container-Mounts (docker-compose bindet nur . und die beiden Encoder-Repos
# ein). Die render-Stufe laeuft deshalb als dritter Schritt auf dem HOST, wo
# Blender erreichbar ist; --stages render braucht kein torch. Schritt 2 nutzt
# solange --reuse-renders, damit describe und embed auf echten Daten messen.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results_stage4
LOG=logs/run_stage4.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"

NQ="${NQ:-50}"            # Queries je View-Zahl
NOBJ="${NOBJ:-59}"        # Ziel-CADs (0 = alle 59)
VIEWS="${VIEWS:-16,42}"

gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }
wait_gpu(){ for _ in $(seq 1 90); do [ "$(gpu_used)" -lt 12000 ] && return 0; sleep 20; done; }

run(){ # name, args...
  local name="$1"; shift
  log ">>> $name"; wait_gpu
  docker compose run --rm oscar bash -lc \
    "cd /app && PYTHONHASHSEED=0 python3 -u $*" \
    > "logs/stage4_$name.log" 2>&1
  log "    $name rc=$?  (GPU $(gpu_used) MiB)"
  tr '\r' '\n' < "logs/stage4_$name.log" | grep -E "^  [A-Za-z_]+ +[0-9]|Ende zu Ende|Gesamt je Objekt|SUMME" \
    | tail -16 | tee -a "$LOG"
}

log "===== 1/2  Query-Latenz (ycbv, mit Pose, Views $VIEWS) ====="
run query_latency \
  "experiments/experiment4_query_latency.py --dataset ycbv --n-queries $NQ \
   --warmup 2 --views $VIEWS --out results_stage4/query_latency_ycbv.json"

log "===== 2/2  Onboarding (Views $VIEWS) ====="
run onboarding \
  "experiments/experiment4_onboarding.py --stages mesh,describe,embed \
   --reuse-renders --num-views $VIEWS $([ "$NOBJ" -gt 0 ] && echo "--max-objects $NOBJ") \
   --measure-invalidation --out results_stage4/onboarding.json"

log "===== 3/3  Render-Stufe auf dem HOST (Blender nicht im Container) ====="
BLENDER="${BLENDER:-/home/tessa/Cap3D/captioning_pipeline/blender-3.4.1-linux-x64/blender}"
if [ -x "$BLENDER" ]; then
  log ">>> render  ($BLENDER)"
  # Wenige Objekte: Blender kostet Minuten pro CAD, und die Verteilung ueber
  # 8 reale Meshes reicht fuer Median und IQR der Renderzeit.
  # Eigenes Arbeitsverzeichnis: Schritt 2 laeuft im Container und legt seine
  # Ordner als root an, in die der Host-Prozess nicht schreiben darf
  # (PermissionError im ersten Lauf).
  python3 experiments/experiment4_onboarding.py --stages render \
    --num-views "$VIEWS" --max-objects "${NRENDER:-8}" --blender "$BLENDER" \
    --work-dir "${RENDER_WORK:-$HOME/.stage4_render_work}" \
    --out results_stage4/onboarding_render.json \
    > logs/stage4_render.log 2>&1
  log "    render rc=$?"
  grep -E "^  render |Gesamt je Objekt" logs/stage4_render.log | tail -6 | tee -a "$LOG"
else
  log "WARNUNG: Blender nicht ausfuehrbar unter $BLENDER — render-Stufe uebersprungen."
fi

log "===== FERTIG ====="
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy results_stage4 "$REMOTE/results_stage4" --transfers 8 --stats 0 \
  >> logs/rclone_reruns.log 2>&1 || true
