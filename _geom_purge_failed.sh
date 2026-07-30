#!/usr/bin/env bash
# The GeDi service was OOM-killed mid-run; every pair attempted afterwards was
# cached as {"failed": true}.  Those records are permanent poison (the B2
# failure policy ranks failed candidates last), so drop them and keep only the
# successful fits.  Original is backed up first.
set -euo pipefail
ROOT="${1:-/home/tholoi/thesis/OSCAR}"
F="$ROOT/object_retrieval/results_shrec18_stage1/_cache/geometry_scores.jsonl"
cp -a "$F" "$F.bak_$(date +%Y%m%d_%H%M%S)"
python3 - "$F" <<'PY'
import sys, json
p = sys.argv[1]
keep, drop = [], 0
for line in open(p, errors="replace"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        drop += 1
        continue
    if r.get("failed"):
        drop += 1
    else:
        keep.append(line)
open(p, "w").write("\n".join(keep) + ("\n" if keep else ""))
qs = {json.loads(l)["qid"] for l in keep}
print(f"kept {len(keep)} good pairs ({len(qs)} queries), dropped {drop} failed")
PY
