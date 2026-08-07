#!/usr/bin/env bash
# Duty-cycle-aware watchdog.  Unlike _geom_watch.sh this must NOT exit when
# the run container is down — that is a normal idle phase now.  It ends only
# when the duty loop itself is gone.
#
# Uses the WINDOWS docker.exe for the same reason _geom_duty.sh does: the
# Linux `docker` CLI disappears from the distro whenever Docker Desktop's
# Resource Saver stops the engine during an idle phase, which would otherwise
# make this watchdog report "down" for everything and hide a real failure.
#
# Emits per-signal progress (an aggregate count conflates the cheap
# chamfer_unaligned control with the ~40 s/query aligned signals) plus the
# duty phase, gedi restarts, and any failure signal.
F=/home/tholoi/thesis/OSCAR/object_retrieval/results_shrec18_stage1/_cache/geometry_scores.jsonl
DUTYLOG=/home/tholoi/thesis/OSCAR/logs/geom_duty.log
DOCKER="/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
dq() { "$DOCKER" "$@" 2>/dev/null | tr -d '\r'; }

# Warn on the DELTA, not on the level.  Genuine registration failures are
# real signal and stay cached forever (a sparse near-planar query that RANSAC
# cannot certify is a property of the data, not a fault), so a warning keyed
# on "failed > 0" fires every tick forever and trains the reader to ignore
# it — the same trap as the curl healthcheck that could never pass.  What
# actually matters is failures *starting to grow*, which is what a dead
# service looks like.
prev_failed=${PREV_FAILED:-0}

while true; do
    duty_pid=$(pgrep -f _geom_duty.sh | head -1)
    run_st=$(dq inspect -f '{{.State.Status}}' stage1_geom)
    gedi_st=$(dq inspect -f '{{.State.Status}}' oscar-gedi-1)
    gedi_r=$(dq inspect -f '{{.RestartCount}}' oscar-gedi-1)
    phase=$(grep -aE '\[duty\]' "$DUTYLOG" 2>/dev/null | tail -1 | sed 's/.*\[duty\] //')

    python3 - "$F" <<'PY'
import sys, json, collections
# Collapse to the LAST record per (qid, cad) — the same rule the engine's
# cache uses.  Counting raw lines double-counts: a genuinely-failed pair
# stores d_ransac/d_icp = null, the `missing` check treats null as "not
# computed", so every new run window legitimately retries those fits and
# appends another failed record.  That retry is deliberate (it is what lets
# a transient GeDi outage self-heal), so the watchdog must dedupe rather
# than the engine stop retrying.
last = {}
try:
    for line in open(sys.argv[1], errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        last[(r["qid"], r["cad"])] = r
except FileNotFoundError:
    pass

qs = collections.defaultdict(set)
failed_pairs = 0
failed_qs = set()
for (qid, _cad), r in last.items():
    if r.get("failed"):
        failed_pairs += 1
        failed_qs.add(qid)
        continue
    for f in ("fitness", "d_unaligned", "d_ransac", "d_icp"):
        if r.get(f) is not None:
            qs[f].add(qid)

parts = " ".join(f"{f}={len(qs[f])}" for f in
                 ("fitness", "d_unaligned", "d_ransac", "d_icp"))
print(f"[geom] {parts} failed={len(failed_qs)}q/{failed_pairs}pairs /2101")
with open("/tmp/_geom_failed_count", "w") as fh:
    fh.write(str(len(failed_qs)))
PY

    failed_now=$(cat /tmp/_geom_failed_count 2>/dev/null || echo 0)
    if [ "$failed_now" -gt "$prev_failed" ]; then
        echo "[geom] !! FAILING QUERIES GREW ${prev_failed} -> ${failed_now}" \
             "— check gedi is alive before trusting these pairs"
    fi
    prev_failed=$failed_now

    echo "[duty] ${phase:-unknown} | run=${run_st:-down} gedi=${gedi_st:-down} restarts=${gedi_r:-0}"

    # A skipped window is a real failure signal, not noise — surface it.
    if grep -qa "skipping this window" <<<"$phase"; then
        echo "[duty] !! window SKIPPED — check docker/gedi; no work done this cycle"
    fi

    if [ -z "$duty_pid" ]; then
        echo "[duty] duty loop is GONE — no further windows will start"
        tail -3 "$DUTYLOG" 2>/dev/null
        exit 0
    fi
    sleep 1800
done
