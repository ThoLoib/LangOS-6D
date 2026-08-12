"""
Focused GeDi vs dGeDi geometry re-rank comparison on ycbv (ULIP-pc retrieval).
Same queries + same fused top-K for both, so the ONLY difference is the geometry
descriptor. Both correctly CO-SCALED (query normalized by each candidate's
scale). GeDi uses its own regime (unit max-radius, voxel 0.01, r_lrf 0.5,
RANSAC 0.02); dGeDi uses the service (co-scale by candidate diameter).

Run (oscar container): python3 scratch_gedi_vs_dgedi.py --n 30 --k 10
"""
import argparse, base64, os, sys
import numpy as np
import httpx
import open3d as o3d

# gallery/data paths in stage3_gallery are relative to object_retrieval/ -> cd there
_REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_REPO, "object_retrieval"))
sys.path.insert(0, os.path.join(_REPO, "object_retrieval"))
from eval_common import run_query, fusion_ranking, crop_by_bbox
from query_cloud import backproject_masked
from stage3_gallery import assemble_gallery, TARGET_DATASETS, DATASET_LAYOUT, split_id
from stage3_metrics import rank_of_target
from dgedi_bridge import dgedi_rerank
from eval_bop_pose import (_matching_instances, _bbox_of, _pad_bbox, _pose_inputs,
                           _cam_entry, load_bop_targets, DATASET_TEST, _THIS_DIR, _geo_rerank)
from PIL import Image

GEDI_URL = "http://gedi:5060"


# ---- GeDi geometry (correct co-scaling) ------------------------------------
def sample_mesh_m(path, units_m, n=10000):
    m = o3d.io.read_triangle_mesh(path)
    if m.is_empty():
        p = o3d.io.read_point_cloud(path); pts = np.asarray(p.points)
    else:
        m.compute_vertex_normals(); pts = np.asarray(m.sample_points_uniformly(n).points)
    pts = pts * (1.0 if units_m else 0.001)   # -> metres
    return pts.astype(np.float64)


def max_radius(pts):
    c = pts.mean(0); return float(np.linalg.norm(pts - c, axis=1).max()) or 1.0


def norm_voxel(pts, scale, voxel=0.01):
    """center, ÷scale (-> ~unit max-radius), voxel-downsample."""
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector((pts - pts.mean(0)) / scale)
    p = p.voxel_down_sample(voxel)
    return np.asarray(p.points, dtype=np.float32)


def gedi_desc(pts, n_kp=512):
    if len(pts) < 100:
        return None, None
    b64 = base64.b64encode(np.ascontiguousarray(pts).tobytes()).decode()
    try:
        r = httpx.post(f"{GEDI_URL}/compute_descriptors",
                       json={"points": b64, "num_keypoints": min(n_kp, len(pts)), "seed": 0},
                       timeout=120)
        j = r.json()
    except Exception:
        return None, None
    if j.get("num_keypoints", 0) == 0:
        return None, None
    kp = pts[np.array(j["keypoint_indices"], int)]
    desc = np.frombuffer(base64.b64decode(j["descriptors"]), np.float32).reshape(-1, j["dim"])
    return kp, desc


def _feat(d):
    f = o3d.pipelines.registration.Feature(); f.resize(d.shape[1], d.shape[0]); f.data = d.T.copy(); return f


def ransac_chamfer(kp_q, d_q, kp_t, d_t, thr=0.02, trim=0.1):
    pq = o3d.geometry.PointCloud(); pq.points = o3d.utility.Vector3dVector(kp_q.astype(np.float64))
    pt = o3d.geometry.PointCloud(); pt.points = o3d.utility.Vector3dVector(kp_t.astype(np.float64))
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pq, pt, _feat(d_q), _feat(d_t), mutual_filter=True, max_correspondence_distance=thr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3, checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                              o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(thr)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999))
    if len(res.correspondence_set) == 0:
        return {"ok": False}
    T = np.asarray(res.transformation)
    qa = kp_q @ T[:3, :3].T + T[:3, 3]
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(kp_t).query(qa, k=1); dist.sort()
    keep = max(1, int(round(len(dist) * (1 - trim))))
    return {"ok": True, "ransac_fitness": float(res.fitness), "d_ransac": float(dist[:keep].mean())}


def gedi_rerank(q_cloud_m, cand_ids, id_to_mesh, cand_cache):
    geo = {}
    for cid in cand_ids:
        path, units_m = id_to_mesh.get(cid, (None, False))
        if not path or not os.path.isfile(path):
            geo[cid] = {"ok": False}; continue
        if cid not in cand_cache:                       # candidate desc cached (unit by own radius)
            cpts = sample_mesh_m(path, units_m); r = max_radius(cpts)
            cand_cache[cid] = (gedi_desc(norm_voxel(cpts, r)), r)
        (kp_t, d_t), r = cand_cache[cid]
        if kp_t is None:
            geo[cid] = {"ok": False}; continue
        kp_q, d_q = gedi_desc(norm_voxel(q_cloud_m, r))   # query co-scaled by candidate radius
        if kp_q is None:
            geo[cid] = {"ok": False}; continue
        geo[cid] = ransac_chamfer(kp_q, d_q, kp_t, d_t)
    return geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    print("[setup] assembling ULIP gallery ...", flush=True)
    G = assemble_gallery(TARGET_DATASETS)             # ULIP (default encoder)
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = G.components()
    cfg = G.eval_cfg
    top_k = len(G.gallery_ids) + 5
    clip_rows = len(clip_retr._desc_labels)
    ds = "ycbv"; ds_test = DATASET_TEST[ds]
    targets = load_bop_targets(os.path.join(_THIS_DIR, ds_test["targets"]))[:args.n]
    test_root = os.path.join(_THIS_DIR, ds_test["test_root"])

    ranks = {"pre": [], "gedi": [], "dgedi": []}
    cand_cache = {}
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
            qemb = shape_m.encode_pointcloud(qc, colors=qcol)     # ULIP pc-query
            out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m, roi, cfg,
                            ulip_query_emb=qemb, dino_full_top_k=top_k,
                            ulip_full_top_k=top_k, clip_full_top_k=clip_rows)
            fused = fusion_ranking(out["fused_full"])
            cand = [o for o, _ in fused[:args.k]]
            ranks["pre"].append(rank_of_target(fused, tgt))
            # GeDi
            g = gedi_rerank(qc, cand, G.id_to_pose_mesh, cand_cache)
            ranks["gedi"].append(rank_of_target(_geo_rerank(fused, g, args.k), tgt))
            # dGeDi (service, already co-scaled)
            dg = dgedi_rerank(qc, cand)
            ranks["dgedi"].append(rank_of_target(_geo_rerank(fused, dg, args.k), tgt) if dg else rank_of_target(fused, tgt))

    def summ(rs):
        n = len(rs); r1 = sum(1 for r in rs if r == 1)/n; r5 = sum(1 for r in rs if r and r <= 5)/n
        mrr = sum((1.0/r if r else 0) for r in rs)/n
        return r1, r5, mrr
    print(f"\n=== ycbv ULIP-pc, N_instances={len(ranks['pre'])}, K={args.k} ===")
    print(f"{'variant':16s} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    for name in ("pre", "gedi", "dgedi"):
        r1, r5, mrr = summ(ranks[name])
        print(f"{('fused(no-geo)' if name=='pre' else '+'+name):16s} {r1:6.3f} {r5:6.3f} {mrr:6.3f}")


if __name__ == "__main__":
    main()
