#!/usr/bin/env bash
# Upload the raw SHREC'18 distribution to Drive, kept separate from the
# mirrored repo tree (OSCAR/eval/... holds derived caches, not raw data).
#
# `copy`, never `sync` — sync would delete remote files that are absent
# locally.  Resumable: re-running skips whatever already matches.
set -uo pipefail
SRC=/home/tholoi/thesis/OSCAR/eval/datasets/shrec18/shrec18_full
DST=gdrive:Masterthesis/OSCAR/raw_datasets/shrec18_full
LOG=/home/tholoi/thesis/OSCAR/logs/upload_shrec_raw.log

mkdir -p "$(dirname "$LOG")"
echo "=== $(date '+%F %T') upload START  $SRC -> $DST ===" >> "$LOG"

rclone copy "$SRC" "$DST" \
    --transfers 12 \
    --checkers 24 \
    --drive-chunk-size 64M \
    --fast-list \
    --stats 60s \
    --stats-one-line \
    --log-file "$LOG" \
    --log-level INFO

rc=$?
echo "=== $(date '+%F %T') upload FINISHED rc=$rc ===" >> "$LOG"
exit $rc
