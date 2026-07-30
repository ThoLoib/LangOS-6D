#!/usr/bin/env bash
# Duty-cycle the Stage-1 geometry run so it does not sit on the battery:
#   2 h running -> 30 min idle (containers down) -> repeat.
#
# Safe to interrupt at any point: pair scores are appended to
# geometry_scores.jsonl one record at a time and the loader skips a line
# truncated by a kill, so a stop costs at most the query in flight.
#
# gedi is stopped during the idle phase too — it holds the GeDi model on the
# GPU and draws power even when no requests are in flight.
#
# ---------------------------------------------------------------------------
# Why the WINDOWS docker.exe and `docker start`, not `docker compose run`:
#
# Docker Desktop's Resource Saver stops its WSL engine a few minutes after the
# last container exits — which is exactly what this script's idle phase
# creates.  When that happens Docker Desktop tears the `docker` CLI bind-mount
# out of the Ubuntu distro, so `/usr/bin/docker` vanishes (seen as both
# "Input/output error" and "No such file or directory" on 2026-07-28) and the
# next window cannot start.  Waking the engine does NOT restore that mount.
#
# The Windows binary lives at a fixed path, is reachable from WSL, and works
# whether or not the integration is up.  It cannot be used with `docker
# compose` here, though: the compose file's relative bind mounts would resolve
# against Windows paths instead of the WSL ones the containers actually use.
# So we reuse the ALREADY-CREATED containers — their mounts
# (/run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/Ubuntu/...) were set up
# from the Linux side and are preserved across start/stop.  `docker start`
# re-runs the container's original command, and the experiment is `--resume`,
# so each window picks up exactly where the last one stopped.
# ---------------------------------------------------------------------------
set -uo pipefail
cd /home/tholoi/thesis/OSCAR
mkdir -p logs
LOG=logs/geom_duty.log
OFF=${OFF:-1800}     # idle seconds
ON=${ON:-7200}       # running seconds
NAME=stage1_geom
GEDI=oscar-gedi-1
DOCKER="/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"

log() { echo "$(date '+%F %T') [duty] $*" >> "$LOG"; }
# docker.exe emits CRLF; strip it or every comparison silently fails.
d() { "$DOCKER" "$@" 2>&1 | tr -d '\r'; }
dq() { "$DOCKER" "$@" 2>/dev/null | tr -d '\r'; }

if [ ! -x "$DOCKER" ]; then
    log "!! $DOCKER not found — cannot drive Docker; exiting"
    exit 1
fi
for c in "$NAME" "$GEDI"; do
    if ! dq inspect -f '{{.Id}}' "$c" >/dev/null 2>&1; then
        log "!! container '$c' does not exist — recreate it with"
        log "   docker compose run -d --name $NAME oscar python3 -u experiments/experiment1_shrec18_stage1.py --all --resume --with-geometry"
        exit 1
    fi
done

log "duty cycle started (run ${ON}s / idle ${OFF}s) via docker.exe"

while true; do
    # ---- run phase -------------------------------------------------------
    log "starting $GEDI"
    d start "$GEDI" >> "$LOG" 2>&1
    st=""
    for _ in $(seq 1 36); do        # up to 3 min for the model to load
        st=$(dq inspect -f '{{.State.Health.Status}}' "$GEDI")
        [ "$st" = healthy ] && break
        sleep 5
    done
    log "gedi health=${st:-unknown}"
    if [ "$st" != healthy ]; then
        log "!! gedi not healthy — skipping this window rather than caching bogus fits"
        d stop "$GEDI" >> "$LOG" 2>&1
        log "idle ${OFF}s"
        sleep "$OFF"
        continue
    fi

    d start "$NAME" >> "$LOG" 2>&1
    log "run window ${ON}s (container $NAME)"

    # Poll instead of a flat sleep so a finished experiment ends the cycle
    # instead of being restarted every 2.5 h for no work.
    finished=0
    ec=1
    waited=0
    while [ "$waited" -lt "$ON" ]; do
        sleep 60
        waited=$((waited + 60))
        rst=$(dq inspect -f '{{.State.Status}}' "$NAME")
        if [ "$rst" != running ]; then
            ec=$(dq inspect -f '{{.State.ExitCode}}' "$NAME")
            log "run container $rst exit=${ec:-?} after ${waited}s"
            finished=1
            break
        fi
    done

    if [ "$finished" = 1 ] && [ "${ec:-1}" = 0 ]; then
        log "experiment completed — stopping gedi and ending duty cycle"
        d stop "$GEDI" >> "$LOG" 2>&1
        exit 0
    fi

    # ---- idle phase ------------------------------------------------------
    log "stopping for the idle phase"
    d stop "$NAME" >> "$LOG" 2>&1
    d stop "$GEDI" >> "$LOG" 2>&1
    log "idle ${OFF}s"
    sleep "$OFF"
done
