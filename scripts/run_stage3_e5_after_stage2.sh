#!/usr/bin/env bash
# =============================================================================
# run_stage3_e5_after_stage2.sh — Stage-3 additions, auto after Stage-2.
#
# 3a (retrieval, cheap) on ALL THREE: OSCAR baseline (CLIP-τ-prune -> DINOv2
# cascade, no shape), Uni3D shape, full-mesh shape.
# 3b (pose, GPU-heavy) on OSCAR always; PLUS the single best of {Uni3D, full-
# mesh} — but ONLY if it out-retrieves the ULIP-2 cross BASE (R@1 0.482) in 3a.
# No point posing an encoder/reference that doesn't even retrieve better.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_stage3_e5.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"
OUT=results_bop_stage3_v2
GT=results_bop_stage3_v2/gt
BASE=0.482          # ULIP-2 cross R@1 (OSCAR+ 3a_cross) — the bar to beat

log "waiting for Stage-2 (run_stage2_after_stage1 DONE) ..."
for i in $(seq 1 720); do
  grep -q "run_stage2_after_stage1 DONE" logs/run_stage2_after_stage1.log 2>/dev/null && break
  sleep 30
done
grep -q "run_stage2_after_stage1 DONE" logs/run_stage2_after_stage1.log 2>/dev/null \
  && log "Stage-2 finished — starting Stage-3." \
  || log "WARN: Stage-2 not marked done; proceeding (Stage-3 is independent)."

s3(){  # $1=label  $2=extra-flags  $3=mode
  log ">>> $1 : eval_bop_pose --mode $3 $2 -> $OUT/$1"
  docker compose run --rm oscar bash -lc \
    "cd /app/object_retrieval && PYTHONHASHSEED=0 \
     python3 -u eval_bop_pose.py --datasets all --mode $3 $2 --output $OUT/$1" \
    > "logs/stage3_$1.log" 2>&1
  log "    $1 rc=$?"
}
r1(){ python3 -c "import json;d=json.load(open('$1'));m=d.get('overall',d);print(m.get('recall@1',''))" 2>/dev/null; }
sane(){ awk -v x="$1" 'BEGIN{exit !(x+0>0.05 && x+0<0.95)}'; }   # plausible R@1

# --- Phase R: 3a retrieval on all three -------------------------------------
s3 3a_oscar    "--oscar-baseline" 3a
s3 3a_uni3d    "--uni3d"          3a
s3 3a_fullmesh "--fullmesh"       3a
OSC=$(r1 "object_retrieval/$OUT/3a_oscar/combined_stage3a.json")
UNI=$(r1 "object_retrieval/$OUT/3a_uni3d/combined_stage3a.json")
FUL=$(r1 "object_retrieval/$OUT/3a_fullmesh/combined_stage3a.json")
log "3a R@1 — OSCAR=$OSC | Uni3D=$UNI | full-mesh=$FUL   (ULIP-2 cross BASE=$BASE, pc=0.464)"
grep -h "\[stage3\]\[fullmesh\]" logs/stage3_3a_fullmesh.log 2>/dev/null | tee -a "$LOG"

# --- pick the single best of {Uni3D, full-mesh} that beats the BASE ---------
WINNER=""; WFLAG=""; WSCORE=$BASE
if sane "$UNI" && awk -v x="$UNI" -v b="$WSCORE" 'BEGIN{exit !(x+0>b+0)}'; then
  WINNER=uni3d; WFLAG="--uni3d"; WSCORE=$UNI; fi
if sane "$FUL" && awk -v x="$FUL" -v b="$WSCORE" 'BEGIN{exit !(x+0>b+0)}'; then
  WINNER=fullmesh; WFLAG="--fullmesh"; WSCORE=$FUL; fi

# --- Phase P: pose. OSCAR always (if its 3a is sane); + winner if any -------
if sane "$OSC"; then
  s3 3b_oscar "--oscar-baseline --gt-records $GT" 3b
else
  log "OSCAR 3a R@1=$OSC not sane — skipping OSCAR pose; check logs/stage3_3a_oscar.log"
fi
if [ -n "$WINNER" ]; then
  log "best alt = $WINNER (3a R@1 $WSCORE > BASE $BASE) -> posing it"
  s3 "3b_$WINNER" "$WFLAG --gt-records $GT" 3b
else
  log "neither Uni3D ($UNI) nor full-mesh ($FUL) beats ULIP-2 cross ($BASE) in 3a -> no extra pose run"
fi

# --- Phase C: 3c substitution-cost decomposition for runs that beat cross ----
# 3c reuses the run's own 3a ranking (--from-3a) and poses the next-best-non-GT,
# so we can compare its foreignness-vs-substitution split to the BASE's ~50/50.
gt_beats_cross(){ sane "$1" && awk -v x="$1" -v b="$BASE" 'BEGIN{exit !(x+0>b+0)}'; }
if [ -n "$WINNER" ]; then
  log "3c decomposition on $WINNER (beats cross) -> $OUT/3c_$WINNER"
  s3 "3c_$WINNER" "$WFLAG --from-3a $OUT/3a_$WINNER" 3c
fi
if gt_beats_cross "$OSC"; then
  log "OSCAR (3a $OSC) beats cross ($BASE) -> 3c decomposition on OSCAR"
  s3 3c_oscar "--oscar-baseline --from-3a $OUT/3a_oscar" 3c
fi

for v in 3b_oscar 3b_uni3d 3b_fullmesh; do
  f="object_retrieval/$OUT/$v/combined_stage3b.json"
  [ -f "$f" ] && log "$v D_sym median=$(python3 -c "import json;print(json.load(open('$f')).get('dsym',{}).get('d_sym_median'))" 2>/dev/null) (OSCAR+ 3b=18.37)"
done
for v in 3c_oscar 3c_uni3d 3c_fullmesh; do
  f="object_retrieval/$OUT/$v/combined_stage3c.json"
  [ -f "$f" ] && log "$v D_sym median=$(python3 -c "import json;print(json.load(open('$f')).get('dsym',{}).get('d_sym_median'))" 2>/dev/null) (OSCAR+ 3c=15.34)"
done
RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy "object_retrieval/$OUT" "$REMOTE/object_retrieval/$OUT" --transfers 16 --stats 0 >>logs/rclone_reruns.log 2>&1 || true
log "===== run_stage3_e5 DONE ====="
