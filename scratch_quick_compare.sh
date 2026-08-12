#!/usr/bin/env bash
set -uo pipefail
cd /home/thomas/thesis/OSCAR
N=50
run(){ echo "=== RUN $1 ==="; docker compose run --rm -T oscar bash -lc \
  "cd /app/object_retrieval && python3 -u eval_bop_pose.py --datasets ycbv --mode 3a $2 --max-targets $N --output $3" \
  2>&1 | grep -E "Recall@1|dGeDi geometry re-rank ON|\|gallery\| =" | tail -3; }
run "ULIP-image" "--dgedi" "rq_ulip_img"
run "ULIP-pc"    "--pc-query --dgedi" "rq_ulip_pc"
run "Uni3D-pc"   "--uni3d --dgedi" "rq_uni3d_pc"
echo "=== TABLE (ycbv, N=$N targets) ==="
python3 - <<PY
import json
base=0.645
print(f"{'config':12s} {'no-geo R@1':>10} {'+dGeDi R@1':>10} {'R@5 pre/post':>14} {'MRR post':>9}")
for lbl,d in [("ULIP-image","rq_ulip_img"),("ULIP-pc","rq_ulip_pc"),("Uni3D-pc","rq_uni3d_pc")]:
  try:
    s=json.load(open(f"object_retrieval/{d}/ycbv_stage3a/summary.json")); pg=s.get("pre_geometry",{})
    print(f"{lbl:12s} {pg.get('recall@1',0):10.3f} {s['recall@1']:10.3f} {pg.get('recall@5',0):6.3f}/{s['recall@5']:.3f}   {s['mrr']:8.3f}")
  except Exception as e: print(lbl,"ERR",e)
print(f"(full-run ULIP-image baseline ref: R@1 {base})")
PY
echo "=== DONE ==="
