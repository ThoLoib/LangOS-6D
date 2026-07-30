#!/usr/bin/env bash
# Watchdog for the Stage-1 geometry ablations.
#
# Reports progress PER SIGNAL: a single "good pairs" count conflates the cheap
# chamfer_unaligned control (no GeDi, ~0.5s/query) with the expensive
# GeDi+RANSAC signals, so it says nothing useful about where the run is.
# Also reports gedi's state — a rising pair count alone once hid a 10h outage.
F=/home/tholoi/thesis/OSCAR/object_retrieval/results_shrec18_stage1/_cache/geometry_scores.jsonl
prev_results=""
while true; do
  run_st=$(docker inspect -f '{{.State.Status}}' stage1_geom 2>/dev/null)
  gedi_st=$(docker inspect -f '{{.State.Status}}/{{.State.Health.Status}}' oscar-gedi-1 2>/dev/null)
  gedi_restarts=$(docker inspect -f '{{.RestartCount}}' oscar-gedi-1 2>/dev/null)
  python3 - "$F" <<'PY'
import sys, json, collections
qs = collections.defaultdict(set)
failed = 0
try:
    for line in open(sys.argv[1], errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("failed"):
            failed += 1
            continue
        for f in ("fitness", "d_unaligned", "d_ransac", "d_icp"):
            if r.get(f) is not None:
                qs[f].add(r["qid"])
except FileNotFoundError:
    pass
parts = " ".join(f"{f}={len(qs[f])}" for f in
                 ("fitness", "d_unaligned", "d_ransac", "d_icp"))
print(f"[geom] {parts} failed={failed} /2101")
if failed:
    print(f"[geom] !! {failed} FAILED FITS — check gedi before trusting results")
PY
  echo "[geom] run=$run_st  gedi=$gedi_st restarts=$gedi_restarts"
  case "$gedi_st" in
    running/healthy) ;;
    *) echo "[geom] !! GEDI NOT HEALTHY ($gedi_st)" ;;
  esac
  cur=$(docker logs stage1_geom 2>&1 | grep -aE '^\[(run|aggregate)' || true)
  [ "$cur" != "$prev_results" ] && { echo "$cur" | tail -4; prev_results="$cur"; }
  [ "$run_st" != "running" ] && { echo "[geom] run container $run_st — watch ending"; exit 0; }
  sleep 1800
done
