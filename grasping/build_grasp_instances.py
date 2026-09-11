#!/usr/bin/env python3
"""Build the frozen instance lists of the Stage-5 proxy-grasp study.

For every case in `proxy_grasp_cases.py` this collects the Stage-3 3b
instances (scene, image, gt_idx) in which the case's FIXED proxy was the
retrieved top-1, joins the BOP visibility annotation and the paired
exact-CAD FoundationPose record, and writes everything the experiment needs
into `proxy_grasp_instances.json` — including the Stage-3 poses, so the grasp
study can also run on the archived Stage-3 output (`--pose-source stage3`).

Inputs (the user provides them; nothing is downloaded):
    --records-dir   results_bop_stage3_v2/3b_cross   (<ds>_stage3b/records.json)
    --gt-records    results_bop_stage3_v2/gt/combined_gt.json
                    (final_results/stage3/gt_combined_gt.json is the same file)

    python3 grasping/build_grasp_instances.py \
        --records-dir object_retrieval/results_bop_stage3_v2/3b_cross \
        --gt-records  final_results/stage3/gt_combined_gt.json

Pure Python: runs on a host without numpy / the sim stack.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _ROOT)

from grasping.proxy_grasp_cases import ALL_CASES  # noqa: E402

# BOP test roots (kept local to stay numpy-free; sim_scene.BOP_DATASETS is the
# sim-side twin of this table).
_TEST_ROOTS = {
    "ycbv": ["eval/datasets/ycbv/test", "eval/datasets/ycbv_gso/test"],
    "tless": ["eval/datasets/tless/test_primesense"],
    "lmo": ["eval/datasets/lmo/test"],
}


def _test_root(ds: str) -> str:
    for c in _TEST_ROOTS[ds]:
        p = os.path.join(_ROOT, c)
        if os.path.isdir(p) and os.listdir(p):
            return p
    return os.path.join(_ROOT, _TEST_ROOTS[ds][0])


_INFO_CACHE: dict = {}


def _visib(ds: str, scene: int, im: int, gt_idx: int):
    """visib_fract from BOP scene_gt_info.json (None if the file is absent)."""
    key = (ds, scene)
    if key not in _INFO_CACHE:
        p = os.path.join(_test_root(ds), f"{scene:06d}", "scene_gt_info.json")
        try:
            _INFO_CACHE[key] = json.load(open(p))
        except OSError:
            _INFO_CACHE[key] = None
    info = _INFO_CACHE[key]
    try:
        return round(float(info[str(im)][gt_idx]["visib_fract"]), 4)
    except (TypeError, KeyError, IndexError):
        return None


def _r(v, nd):
    return [round(float(x), nd) for x in v]


def main():
    ap = argparse.ArgumentParser(description="freeze the proxy-grasp instance lists")
    ap.add_argument("--records-dir",
                    default=os.path.join(_ROOT, "object_retrieval/results_bop_stage3_v2/3b_cross"))
    ap.add_argument("--gt-records",
                    default=os.path.join(_ROOT, "final_results/stage3/gt_combined_gt.json"))
    ap.add_argument("--out", default=os.path.join(_THIS, "proxy_grasp_instances.json"))
    args = ap.parse_args()

    gt_by_key = {}
    if os.path.isfile(args.gt_records):
        for r in json.load(open(args.gt_records)).get("all_records", []):
            gt_by_key[(r["dataset"], r["scene_id"], r["im_id"], r["obj_id"], r["gt_idx"])] = r
        print(f"[build] {len(gt_by_key)} exact-CAD (gt) records from {args.gt_records}")
    else:
        print(f"[build] WARNING: no gt records at {args.gt_records} — Stage-3 gt poses omitted")

    recs_by_ds = {}
    for ds in ("ycbv", "tless", "lmo"):
        p = os.path.join(args.records_dir, f"{ds}_stage3b", "records.json")
        if not os.path.isfile(p):
            sys.exit(f"[build] missing {p} — point --records-dir at the 3b_cross run folder")
        d = json.load(open(p))
        recs_by_ds[ds] = d["records"] if isinstance(d, dict) else d
        print(f"[build] {ds}: {len(recs_by_ds[ds])} 3b records")

    out = {"built": _dt.datetime.now().isoformat(timespec="seconds"),
           "records_dir": os.path.relpath(args.records_dir, _ROOT),
           "gt_records": os.path.relpath(args.gt_records, _ROOT) if gt_by_key else None,
           "cases": {}}
    print(f"\n{'case':<8}{'proxy (fixed)':<58}{'n_obj':>6}{'n_top1':>7}{'share':>7}"
          f"{'pMed':>6}{'gt':>5}  scenes")
    for c in ALL_CASES:
        ds, oid = c["dataset"], c["obj_id"]
        rs = [r for r in recs_by_ds[ds] if r["obj_id"] == oid]
        sel = [r for r in rs if r.get("top1") == c["proxy"] and r.get("d_posed") is not None]
        insts = []
        for r in sorted(sel, key=lambda r: (r["scene_id"], r["im_id"], r["gt_idx"])):
            k = (ds, r["scene_id"], r["im_id"], oid, r["gt_idx"])
            g = gt_by_key.get(k)
            inst = dict(scene=r["scene_id"], im=r["im_id"], gt_idx=r["gt_idx"],
                        visib=_visib(ds, r["scene_id"], r["im_id"], r["gt_idx"]),
                        diameter=r.get("diameter"),
                        fp_proxy=dict(d=r["d_posed"], conf=r.get("top1_pose_conf"),
                                      R=_r(r["top1_R"], 6), t=_r(r["top1_t"], 3)))
            if g is not None and "R" in g:
                inst["fp_gt"] = dict(d=g.get("d_posed_gt"), conf=g.get("pose_conf"),
                                     R=_r(g["R"], 6), t=_r(g["t"], 3))
            insts.append(inst)
        pmed = statistics.median(i["fp_proxy"]["d"] for i in insts) if insts else None
        scenes = Counter(i["scene"] for i in insts)
        out["cases"][c["key"]] = dict(
            dataset=ds, obj_id=oid, proxy=c["proxy"], n_obj=len(rs), n_top1=len(insts),
            share=(round(len(insts) / len(rs), 4) if rs else None),
            p_med=(round(pmed, 2) if pmed is not None else None),
            n_gt_paired=sum(1 for i in insts if "fp_gt" in i),
            scenes={str(s): n for s, n in sorted(scenes.items())}, instances=insts)
        print(f"{c['key']:<8}{c['proxy'][:57]:<58}{len(rs):>6}{len(insts):>7}"
              f"{(100*len(insts)/max(len(rs),1)):>6.0f}%{(pmed or 0):>6.1f}"
              f"{sum(1 for i in insts if 'fp_gt' in i):>5}  {dict(sorted(scenes.items()))}")

    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    os.replace(tmp, args.out)
    n = sum(len(v["instances"]) for v in out["cases"].values())
    print(f"\n[build] {n} instances for {len(out['cases'])} cases -> {args.out} "
          f"({os.path.getsize(args.out)//1024} KB)")


if __name__ == "__main__":
    main()
