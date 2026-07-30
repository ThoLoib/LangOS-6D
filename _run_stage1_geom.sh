#!/usr/bin/env bash
# Stage-1 geometry ablations (E2_*, O1c, O1d) — needs the gedi service up.
# Started DETACHED so the container survives if the launching shell dies
# (the earlier --rm client got killed mid-run and took the log plumbing
# with it, though the container itself kept going).
set -uo pipefail
cd /home/tholoi/thesis/OSCAR
mkdir -p logs
NAME=stage1_geom
LOG=logs/stage1_geom.log

docker rm -f "$NAME" >/dev/null 2>&1

echo "==== $(date '+%F %T') Stage-1 geometry START ====" | tee -a "$LOG"
docker compose run -d --name "$NAME" oscar \
    python3 -u experiments/experiment1_shrec18_stage1.py \
        --all --resume --with-geometry
echo "container: $NAME"

# Detached log pump: not tied to this shell's process tree.
setsid nohup docker logs -f "$NAME" >> "$LOG" 2>&1 < /dev/null &
disown
echo "log -> $LOG"
