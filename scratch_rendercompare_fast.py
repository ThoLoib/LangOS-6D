"""
QUICK render-and-compare: replace FoundationPose with a fast metric coarse pose
(FPFH + RANSAC + ICP, true metric scale so size errors are still caught), then
the same symmetric trimmed chamfer between posed CAD and observed cloud.

Directly comparable to scratch_rendercompare.py (same ycbv subset): fused 0.525,
FP-pose chamfer 0.600. Question: does a FAST metric pose keep the gain?

No FP / no dgedi service -- pure Open3D. Run (oscar container):
    docker compose run --rm oscar python3 scratch_rendercompare_fast.py --n 40 --k 5
"""
import argparse, os, sys, time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

_REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_REPO, "object_retrieval"))
sys.path.insert(0, os.path.join(_REPO, "object_retrieval")); sys.path.insert(0, _REPO)

from eval_common import run_query, fusion_ranking, crop_by_bbox
from query_cloud import backproject_masked
from stage3_gallery import assemble_gallery, TARGET_DATASETS
from eval_bop_pose import (_matching_instances, _bbox_of, _pad_bbox, _pose_inputs,
                           _cam_entry, load_bop_targets, DATASET_TEST, _THIS_DIR,
                           sample_surface_mm)
from PIL import Image

_CAD = {}
VOX = 0.006     # 6 mm


def cad_m(path, units):
    if path not in _CAD:
        _CAD[path] = sample_surface_mm(path, units_m=units).astype(np.float64) / 1000.0
    return _CAD[path]


def _prep(pts):
    p = o3d.geometry.PointCloud(); p.points = o3d.utility.Vector3dVector(pts)
    p = p.voxel_down_sample(VOX)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*2, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        p, o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*5, max_nn=100))
    return p, f


def fast_pose(obs_m, cadp):
    """T mapping obs (camera frame) -> CAD frame, via FPFH-RANSAC + ICP (metric)."""
    src, sf = _prep(obs_m); tgt, tf = _prep(cadp)
    if len(src.points) < 10 or len(tgt.points) < 10:
        return None
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, sf, tf, mutual_filter=True, max_correspondence_distance=VOX*1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3, checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOX*1.5)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(60000, 0.999))
    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, VOX*2, res.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return np.asarray(icp.transformation)


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
    ap.add_argument("--k", type=int, default=5); args = ap.parse_args()
    print("[setup] gallery ...", flush=True)
    G = assemble_gallery(TARGET_DATASETS)
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = G.components()
    cfg = G.eval_cfg; top_k = len(G.gallery_ids) + 5
    clip_rows = len(clip_retr._desc_labels); id2mesh = G.id_to_pose_mesh

    for ds in args.datasets.split(","):
        ds_test = DATASET_TEST[ds]
        targets = load_bop_targets(os.path.join(_THIS_DIR, ds_test["targets"]))[:args.n]
        test_root = os.path.join(_THIS_DIR, ds_test["test_root"])
        R = {m: [] for m in ("fused", "chamfer")}; tsum = 0.0; npair = 0
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
                obs_m = qc.astype(np.float64)
                qemb = shape_m.encode_pointcloud(qc, colors=qcol)
                out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m, roi, cfg,
                                ulip_query_emb=qemb, dino_full_top_k=top_k,
                                ulip_full_top_k=top_k, clip_full_top_k=clip_rows)
                fused = fusion_ranking(out["fused_full"])
                top = [o for o, _ in fused[:args.k]]
                R["fused"].append(rank_of([o for o, _ in fused], tgt))
                cham = {}
                for o in top:
                    path, units = id2mesh.get(o, (None, False))
                    if not path or not os.path.isfile(path):
                        cham[o] = 1e9; continue
                    cp = cad_m(path, units)
                    t0 = time.time(); T = fast_pose(obs_m, cp); dt = time.time() - t0
                    tsum += dt; npair += 1
                    if T is None:
                        cham[o] = 1e9; continue
                    obs_in_cad = obs_m @ T[:3, :3].T + T[:3, 3]
                    cham[o] = trimmed_nn(obs_in_cad, cp) + trimmed_nn(cp, obs_in_cad)
                by_cham = sorted(top, key=lambda o: cham[o]) + [o for o, _ in fused[args.k:]]
                R["chamfer"].append(rank_of(by_cham, tgt))
        print(f"\n=== {ds} FAST render-and-compare, N={len(R['fused'])}, K={args.k} ===")
        print(f"{'variant':10s} {'R@1':>7} {'R@5':>7}")
        for m in ("fused", "chamfer"):
            r1, r5 = Mm(R[m]); print(f"{m:10s} {r1:7.3f} {r5:7.3f}")
        if npair: print(f"[latency] fast_pose {tsum/npair*1000:.0f} ms/candidate "
                        f"-> K={args.k} ~ {tsum/npair*args.k:.2f} s/query")


if __name__ == "__main__":
    main()
