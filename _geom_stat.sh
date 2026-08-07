#!/usr/bin/env bash
F=${1:-/home/tholoi/thesis/OSCAR}/object_retrieval/results_shrec18_stage1/_cache/geometry_scores.jsonl
stat -c '%s bytes  mtime=%y' "$F"
python3 - "$F" <<'PY'
import sys, json, collections
qs = collections.defaultdict(set)   # signal-field -> queries
n = failed = 0
for line in open(sys.argv[1], errors="replace"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    n += 1
    if r.get("failed"):
        failed += 1
        continue
    for f in ("fitness", "d_unaligned", "d_ransac", "d_icp"):
        if r.get(f) is not None:
            qs[f].add(r["qid"])
print(f"records={n} failed={failed}")
for f in ("fitness", "d_unaligned", "d_ransac", "d_icp"):
    print(f"  {f:<14} {len(qs[f]):>5} queries")
PY
