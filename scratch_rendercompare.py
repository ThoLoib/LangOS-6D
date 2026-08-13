"""
Render-and-compare re-rank prototype: does "how well does this posed CAD explain
the observation" re-rank the fused top-K better than fusion alone?

Two verification scores per candidate (posed by FoundationPose with its OWN CAD):
  * fp_conf    — FoundationPose confidence (render-and-compare WITH visibility,
                 in feature+RGB space; reuses existing infra).
  * chamfer    — symmetric trimmed point distance between the posed CAD and the
                 observed cloud (naive geometry, NO visibility handling).
Re-rank the fused top-K by each; compare R@1 to fused. If fp_conf helps but
chamfer doesn't, visibility matters -> build the fast visibility-aware version.

Slow (FP per candidate) -> small subset, isolates the SIGNAL, not the speed.
Run (oscar container, foundationpose UP): docker compose run --rm oscar \
    python3 scratch_rendercompare.py --datasets ycbv --n 40 --k 5
"""
import argparse, os, sys, time
import numpy as np
from scipy.spatial import cKDTree

_REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_REPO, "object_retrieval"))
sys.path.insert(0, os.path.join(_REPO, "object_retrieval")); sys.path.insert(0, _REPO)

from eval_common import run_query, fusion_ranking, crop_by_bbox
from query_cloud import backproject_masked
from stage3_gallery import assemble_gallery, TARGET_DATASETS
from eval_bop_pose import (_matching_instances, _bbox_of, _pad_bbox, _pose_inputs,
                           _cam_entry, load_bop_targets, DATASET_TEST, _THIS_DIR,
                           estimate_pose, sample_surface_mm)
from PIL import Image

_CAD = {}


def cad_mm(mesh_path, units_m):
    if mesh_path not in _CAD:
        _CAD[mesh_path] = sample_surface_mm(mesh_path, units_m=units_m)
    return _CAD[mesh_path]


def trimmed_nn(a, b, trim=0.2):
    d, _ = cKDTree(b).query(a, k=1); d.sort()
    keep = max(1, int(len(d) * (1 - trim)))
    return float(d[:keep].mean())


def rank_of(order, tgt): return order.index(tgt) + 1 if tgt in order else None
def Mm(rs):
    n = len(rs); return (sum(1 for r in rs if r == 1)/n, sum(1 for r in rs if r and r <= 5)/n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ycbv"); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--refine-iter", type=int, default=5); args = ap.parse_args()
    print("[setup] gallery ...", flush=True)
    G = assemble_gallery(TARGET_DATASETS)
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = G.components()
    cfg = G.eval_cfg; top_k = len(G.gallery_ids) + 5
    clip_rows = len(clip_retr._desc_labels); id2mesh = G.id_to_pose_mesh

    for ds in args.datasets.split(","):
        ds_test = DATASET_TEST[ds]
        targets = load_bop_targets(os.path.join(_THIS_DIR, ds_test["targets"]))[:args.n]
        test_root = os.path.join(_THIS_DIR, ds_test["test_root"])
        R = {m: [] for m in ("fused", "fp_conf", "chamfer")}; tsum = 0.0; npair = 0
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
                obs_mm = qc.astype(np.float64) * 1000.0
                qemb = shape_m.encode_pointcloud(qc, colors=qcol)
                out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m, roi, cfg,
                                ulip_query_emb=qemb, dino_full_top_k=top_k,
                                ulip_full_top_k=top_k, clip_full_top_k=clip_rows)
                fused = fusion_ranking(out["fused_full"])
                top = [o for o, _ in fused[:args.k]]
                R["fused"].append(rank_of([o for o, _ in fused], tgt))
                conf = {}; cham = {}
                for o in top:
                    path, units = id2mesh.get(o, (None, False))
                    if not path or not os.path.isfile(path):
                        conf[o] = -1.0; cham[o] = 1e9; continue
                    try:
                        _t0 = time.time()
                        Rt, tt, c = estimate_pose(path, rgb_np, depth_m, mask, K,
                                                  mesh_units_m=units, refine_iter=args.refine_iter)
                        tsum += time.time() - _t0; npair += 1
                        posed = cad_mm(path, units) @ np.asarray(Rt).T + np.asarray(tt).reshape(3)
                        conf[o] = float(c)
                        cham[o] = trimmed_nn(obs_mm, posed) + trimmed_nn(posed, obs_mm)
                    except Exception:
                        conf[o] = -1.0; cham[o] = 1e9
                by_conf = sorted(top, key=lambda o: conf[o], reverse=True) + [o for o, _ in fused[args.k:]]
                by_cham = sorted(top, key=lambda o: cham[o]) + [o for o, _ in fused[args.k:]]
                R["fp_conf"].append(rank_of(by_conf, tgt))
                R["chamfer"].append(rank_of(by_cham, tgt))
        print(f"\n=== {ds} render-and-compare re-rank, N={len(R['fused'])}, K={args.k} ===")
        print(f"{'variant':10s} {'R@1':>7} {'R@5':>7}")
        for m in ("fused", "fp_conf", "chamfer"):
            r1, r5 = Mm(R[m]); print(f"{m:10s} {r1:7.3f} {r5:7.3f}")
        if npair:
            print(f"[latency] FP(refine_iter={args.refine_iter}) {tsum/npair*1000:.0f} ms/candidate "
                  f"-> K={args.k} ~ {tsum/npair*args.k:.1f} s/query")


if __name__ == "__main__":
    main()
