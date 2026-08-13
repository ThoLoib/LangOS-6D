#!/usr/bin/env python3
"""
Build BOP-format pose result CSVs from a Stage-3 results dir, so the OFFICIAL
object-balanced BOP-AR can be computed by bop_toolkit (audit P0.8) — instead of
the internal pooled `bop_ar()`. The official evaluator recomputes VSD/MSSD/MSPD
itself, so it also sidesteps the internal VSD path for the headline number.

Reads each `<results>/<ds>_stage3a/records.json` (needs the poses stored by
eval_bop_pose: `oracle_R/oracle_t`, and `top1_R/top1_t` when `top1_is_exact`)
and writes, per dataset:
  * oracle_<ds>-<split>.csv     — GT CAD posed by FoundationPose (pose ceiling)
  * retrieved_<ds>-<split>.csv  — retrieved-exact top-1 pose (top1 == GT obj)

Then run the official evaluator, e.g.:
  python3 third_party/bop_toolkit/scripts/eval_bop19_pose.py \
      --renderer_type=vispy \
      --results_path <out> --eval_path <out>/eval \
      --result_filenames oracle_ycbv-test.csv,oracle_tless-test.csv,oracle_lmo-test.csv

Run:
  python3 object_retrieval/build_bop_csv.py --results <results_dir> --out <csv_dir>
"""
import argparse
import json
import os

# BOP split name per dataset (test split used by test_targets_bop19).
_SPLIT = {"ycbv": "test", "tless": "test", "lmo": "test",
          "itodd": "test", "hb": "test"}


def _rows_from_records(records, which):
    """which='oracle' | 'retrieved'. Yield BOP result rows."""
    for r in records:
        if which == "oracle":
            R, t = r.get("oracle_R"), r.get("oracle_t")
            score = r.get("oracle_pose_conf", 1.0)
        else:  # retrieved-exact only: same obj_id geometry as GT
            if not r.get("top1_is_exact"):
                continue
            R, t = r.get("top1_R"), r.get("top1_t")
            score = r.get("top1_pose_conf", 1.0)
        if R is None or t is None:
            continue
        yield (r["scene_id"], r["im_id"], r["obj_id"], float(score), R, t)


def _write_csv(path, rows):
    n = 0
    with open(path, "w") as f:
        f.write("scene_id,im_id,obj_id,score,R,t,time\n")
        for sid, im, obj, score, R, t in rows:
            Rs = " ".join(f"{v:.9f}" for v in R)
            ts = " ".join(f"{v:.9f}" for v in t)
            f.write(f"{sid},{im},{obj},{score:.6f},{Rs},{ts},-1\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results_bop_stage3_* dir")
    ap.add_argument("--out", required=True, help="output dir for BOP CSVs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    written = []
    for sub in sorted(os.listdir(args.results)):
        if not sub.endswith("_stage3a"):
            continue                      # BOP-AR is a 3a metric (exact CAD)
        ds = sub[:-len("_stage3a")]
        split = _SPLIT.get(ds, "test")
        recs = json.load(open(os.path.join(args.results, sub, "records.json")))
        for which, prefix in (("oracle", "oracle"), ("retrieved", "retrieved")):
            rows = list(_rows_from_records(recs, which))
            if not rows:
                continue
            path = os.path.join(args.out, f"{prefix}_{ds}-{split}.csv")
            n = _write_csv(path, rows)
            written.append((os.path.basename(path), n))
            print(f"[bop-csv] {os.path.basename(path)}: {n} poses", flush=True)

    print(f"[bop-csv] wrote {len(written)} files -> {args.out}")
    print("[bop-csv] then run third_party/bop_toolkit/scripts/eval_bop19_pose.py "
          "on the oracle_*/retrieved_* filenames.")


if __name__ == "__main__":
    main()
