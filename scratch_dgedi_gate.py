"""
dGeDi as a conservative PLAUSIBILITY GATE (not a re-rank): demote only candidates
whose geometry is incompatible with the observation (RANSAC fitness below a
threshold / no correspondences), keep the fused order otherwise. Tests whether the
"gate not reorder" lesson from the scale gate transfers to geometry, and whether
it generalizes (ycbv/tless/lmo) or stays conditional.

Uses the fast 512kp/5k dGeDi (deployable). Saves per-candidate fitness/d_ransac so
the threshold sweep is fully offline afterward.

Run (oscar container; dgedi UP): docker compose run --rm oscar \
    python3 scratch_dgedi_gate.py --datasets ycbv,tless,lmo --n 150 --k 20
"""
import argparse, json, os, sys
import numpy as np, httpx

_REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_REPO, "object_retrieval"))
sys.path.insert(0, os.path.join(_REPO, "object_retrieval")); sys.path.insert(0, _REPO)

from eval_common import run_query, fusion_ranking, crop_by_bbox
from query_cloud import backproject_masked
from stage3_gallery import assemble_gallery, TARGET_DATASETS
from eval_bop_pose import (_matching_instances, _bbox_of, _pad_bbox, _pose_inputs,
                           _cam_entry, load_bop_targets, DATASET_TEST, _THIS_DIR)
from PIL import Image

DGEDI_URL = "http://dgedi:5061"


def dgedi_geo(qc, cand):
    payload = {"query_points": np.asarray(qc, np.float32).tolist(),
               "candidate_ids": list(cand), "ransac_threshold": 0.03,
               "trim_ratio": 0.1, "ransac_keypoints": 512, "ransac_max_iter": 5000}
    try:
        r = httpx.post(f"{DGEDI_URL}/rerank", json=payload, timeout=300)
        return r.json().get("results", {})
    except Exception:
        return {}


def rank_of(order, tgt): return order.index(tgt) + 1 if tgt in order else None
def M(rs):
    n = len(rs); return (sum(1 for r in rs if r == 1)/n, sum(1 for r in rs if r and r <= 5)/n)


def gate_ranks(data, thresh, K):
    rk = []
    for d in data:
        top = d["top"][:K]; keep = []; drop = []
        for o in top:
            g = d["geo"].get(o, {})
            fit = g.get("ransac_fitness", 0.0) if g.get("ok") else 0.0
            (keep if fit >= thresh else drop).append(o)
        rk.append(rank_of(keep + drop + d["top"][K:], d["tgt"]))
    return rk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ycbv"); ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=20); args = ap.parse_args()
    print("[setup] gallery ...", flush=True)
    G = assemble_gallery(TARGET_DATASETS)
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = G.components()
    cfg = G.eval_cfg; top_k = len(G.gallery_ids) + 5; clip_rows = len(clip_retr._desc_labels)

    all_data = {}
    for ds in args.datasets.split(","):
        ds_test = DATASET_TEST[ds]
        targets = load_bop_targets(os.path.join(_THIS_DIR, ds_test["targets"]))[:args.n]
        test_root = os.path.join(_THIS_DIR, ds_test["test_root"])
        data = []
        for t in targets:
            sid, im, oid = t["scene_id"], t["im_id"], t["obj_id"]
            sdir = os.path.join(test_root, f"{sid:06d}")
            rgbp = os.path.join(sdir, "rgb", f"{im:06d}.png")
            if not os.path.isfile(rgbp): rgbp = rgbp[:-4] + ".jpg"
            if not os.path.isfile(rgbp): continue
            rgb = Image.open(rgbp).convert("RGB"); rgb_np = np.asarray(rgb, np.uint8)
            cam = _cam_entry(sdir, im); tgt = f"{ds}/obj_{oid:06d}"
            for gi, gt, info in _matching_instances(sdir, im, oid):
                bb = _bbox_of(info)
                if bb is None: continue
                roi = crop_by_bbox(rgb, _pad_bbox(bb, rgb.width, rgb.height))
                depth_m, _dmm, mask, K = _pose_inputs(sdir, im, gi, cam)
                qc, qcol = backproject_masked(depth_m, mask, K, rgb=rgb_np)
                if len(qc) < 64: continue
                qemb = shape_m.encode_pointcloud(qc, colors=qcol)
                out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m, roi, cfg,
                                ulip_query_emb=qemb, dino_full_top_k=top_k,
                                ulip_full_top_k=top_k, clip_full_top_k=clip_rows)
                fused = fusion_ranking(out["fused_full"])
                top = [o for o, _ in fused[:args.k]]
                geo = dgedi_geo(qc, top)
                data.append({"tgt": tgt, "top": [o for o, _ in fused],
                             "geo": {o: {"ok": bool(geo.get(o, {}).get("ok")),
                                         "ransac_fitness": float(geo.get(o, {}).get("ransac_fitness", 0.0)),
                                         "d_ransac": float(geo.get(o, {}).get("d_ransac", 0.0))} for o in top}})
        all_data[ds] = data
        fr1 = sum(1 for d in data if d["top"][0] == d["tgt"]) / len(data)
        print(f"\n=== {ds} dGeDi GATE, N={len(data)}, K={args.k} ===")
        print(f"fused R@1={fr1:.3f}")
        print(f"{'fit_thresh':>10} {'R@1':>7} {'R@5':>7}")
        for th in (0.0, 0.02, 0.05, 0.08, 0.12, 0.20):
            r1, r5 = M(gate_ranks(data, th, args.k))
            print(f"{th:10.2f} {r1:7.3f} {r5:7.3f}")
    fn = os.path.join(_REPO, "scratch_dgedi_gate_" + "_".join(all_data) + ".json")
    json.dump(all_data, open(fn, "w"))
    print(f"\n[saved] -> {fn}")


if __name__ == "__main__":
    main()
