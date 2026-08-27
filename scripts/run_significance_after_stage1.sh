#!/usr/bin/env bash
# =============================================================================
# run_significance_after_stage1.sh — paired significance test (95% bootstrap CI
# + Wilcoxon) on the 42v+k5 Stage-1 folder once it finishes. Attaches a CI to
# every near-tie delta so we know which are real vs noise. nDCG + hit@1(NN_sub).
# Independent of Stage-2/3 — only needs the Stage-1 per-query records.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_significance.log
ts(){ date -Is; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
: > "$LOG"

log "waiting for Stage-1 (run_stage1_full DONE) ..."
for i in $(seq 1 5760); do
  grep -q "run_stage1_full DONE" logs/run_stage1_full.log 2>/dev/null && break
  sleep 30
done
grep -q "run_stage1_full DONE" logs/run_stage1_full.log 2>/dev/null \
  && log "Stage-1 finished — running significance test." \
  || log "WARN: Stage-1 not marked done; running anyway (needs the per-query JSONs)."

for M in nDCG NN_sub; do
  log ">>> paired significance on metric=$M"
  docker compose run --rm -e SIG_METRIC="$M" oscar bash -lc \
    "cd /app && python3 -u object_retrieval/paired_significance.py" \
    > "logs/significance_$M.log" 2>&1
  log "    metric=$M rc=$?"
  grep -E "^SIG |^    .*Δ|wrote" "logs/significance_$M.log" | tee -a "$LOG"
done

RC="${RCLONE:-$HOME/apps/rclone/rclone}"; REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"
"$RC" copy "object_retrieval/results_shrec18_v2_stage1_42v_k5" \
  "$REMOTE/object_retrieval/results_shrec18_v2_stage1_42v_k5" \
  --include "paired_significance_*.csv" --stats 0 >>logs/rclone_reruns.log 2>&1 || true
log "===== significance DONE ====="
