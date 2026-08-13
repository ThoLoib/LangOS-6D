"""
Does the fast 512-keypoint dGeDi re-rank match the repo-faithful 6000-keypoint
one on RETRIEVAL QUALITY? (Latency already settled: ~1.9s vs ~10s at K=20.)

Same queries, same fused top-K, same both_borda combine (fitness + trimmed
chamfer). The ONLY difference is the /rerank RANSAC budget:
  * fast    ransac_keypoints=512,  ransac_max_iter=5000   (our deployable setting)
  * repo    ransac_keypoints=6000, ransac_max_iter=10000  (dGeDi demo.py)
Full-fusion ULIP-pc retrieval (weights 0.3/0.4/0.3), ycbv.

Run (oscar container; dgedi up):
    docker compose run --rm oscar python3 scratch_dgedi_kp_quality.py --n 150 --k 20
"""
import argparse, os, sys, time
import numpy as np
import httpx

_REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_REPO, "object_retrieval"))
sys.path.insert(0, os.path.join(_REPO, "object_retrieval"))
sys.path.insert(0, _REPO)

from eval_common import run_query, fusion_ranking, crop_by_bbox
from query_cloud import backproject_masked
from stage3_gallery import assemble_gallery, TARGET_DATASETS
from stage3_metrics import rank_of_target
from eval_bop_pose import (_matching_instances, _bbox_of, _pad_bbox, _pose_inputs,
                           _cam_entry, load_bop_targets, DATASET_TEST, _THIS_DIR, _geo_rerank)
from PIL import Image

DGEDI_URL = "http://dgedi:5061"


def rerank_kp(query_points, cand_ids, nkp, max_iter):
    """/rerank with an explicit keypoint/iteration budget -> geo dict + elapsed."""
    payload = {"query_points": np.asarray(query_points, np.float32).tolist(),
               "candidate_ids": list(cand_ids),
               "ransac_threshold": 0.03, "trim_ratio": 0.1,
               "ransac_keypoints": int(nkp), "ransac_max_iter": int(max_iter)}
    t = time.time()
    r = httpx.post(f"{DGEDI_URL}/rerank", json=payload, timeout=900)
    dt = time.time() - t
    return r.json().get("results", {}), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    print("[setup] assembling ULIP gallery (full fusion 0.3/0.4/0.3) ...", flush=True)
    G = assemble_gallery(TARGET_DATASETS)
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = G.components()
    cfg = G.eval_cfg
    top_k = len(G.gallery_ids) + 5
    clip_rows = len(clip_retr._desc_labels)

    ds = "ycbv"; ds_test = DATASET_TEST[ds]
    targets = load_bop_targets(os.path.join(_THIS_DIR, ds_test["targets"]))[:args.n]
    test_root = os.path.join(_THIS_DIR, ds_test["test_root"])

    ranks = {"pre": [], "kp512": [], "kp6000": []}
    t_acc = {"kp512": 0.0, "kp6000": 0.0}; n_pairs = 0
    for t in targets:
        sid, im, oid = t["scene_id"], t["im_id"], t["obj_id"]
        sdir = os.path.join(test_root, f"{sid:06d}")
        rgbp = os.path.join(sdir, "rgb", f"{im:06d}.png")
        if not os.path.isfile(rgbp): rgbp = rgbp[:-4] + ".jpg"
        if not os.path.isfile(rgbp): continue
        rgb = Image.open(rgbp).convert("RGB"); rgb_np = np.asarray(rgb, np.uint8)
        cam = _cam_entry(sdir, im)
        tgt = f"{ds}/obj_{oid:06d}"
        for gi, gt, info in _matching_instances(sdir, im, oid):
            bb = _bbox_of(info)
            if bb is None: continue
            roi = crop_by_bbox(rgb, _pad_bbox(bb, rgb.width, rgb.height))
            depth_m, _dmm, mask, K = _pose_inputs(sdir, im, gi, cam)
            qc, qcol = backproject_masked(depth_m, mask, K, rgb=rgb_np)
            if len(qc) < 64: continue
            qemb = shape_m.encode_pointcloud(qc, colors=qcol)      # ULIP pc-query
            out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m, roi, cfg,
                            ulip_query_emb=qemb, dino_full_top_k=top_k,
                            ulip_full_top_k=top_k, clip_full_top_k=clip_rows)
            fused = fusion_ranking(out["fused_full"])
            cand = [o for o, _ in fused[:args.k]]
            ranks["pre"].append(rank_of_target(fused, tgt))
            g512, dt512 = rerank_kp(qc, cand, 512, 5000)
            g6000, dt6000 = rerank_kp(qc, cand, 6000, 10000)
            t_acc["kp512"] += dt512; t_acc["kp6000"] += dt6000; n_pairs += len(cand)
            ranks["kp512"].append(rank_of_target(_geo_rerank(fused, g512, args.k), tgt))
            ranks["kp6000"].append(rank_of_target(_geo_rerank(fused, g6000, args.k), tgt))

    def summ(rs):
        n = len(rs)
        return (sum(1 for r in rs if r == 1)/n,
                sum(1 for r in rs if r and r <= 5)/n,
                sum((1.0/r if r else 0) for r in rs)/n)

    print(f"\n=== ycbv ULIP-pc full-fusion, dGeDi keypoint quality, "
          f"N={len(ranks['pre'])}, K={args.k} ===")
    print(f"{'variant':22s} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    for name, lab in (("pre", "fused (no geo)"),
                      ("kp512", "+dGeDi 512kp/5k"),
                      ("kp6000", "+dGeDi 6000kp/10k")):
        r1, r5, mrr = summ(ranks[name])
        print(f"{lab:22s} {r1:6.3f} {r5:6.3f} {mrr:6.3f}")
    if n_pairs:
        print(f"\n[latency] 512kp: {t_acc['kp512']/n_pairs*1000:.0f} ms/pair | "
              f"6000kp: {t_acc['kp6000']/n_pairs*1000:.0f} ms/pair "
              f"(n_pairs={n_pairs})")
    # agreement: how often do the two settings pick the SAME top-1?
    same = sum(1 for a, b in zip(ranks["kp512"], ranks["kp6000"])
               if (a == 1) == (b == 1))
    print(f"[agreement] top-1 hit agrees on {same}/{len(ranks['kp512'])} queries")


if __name__ == "__main__":
    main()
