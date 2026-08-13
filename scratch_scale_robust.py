"""
Scale gate with OUTLIER-ROBUST observed extent. The raw max-min bbox is inflated
by stray depth pixels (true-target sf p90 ~7.4), which falsely gates the correct
CAD. Compare observed-extent estimators:
  raw   = max - min
  pct   = p98 - p2 per axis
  sor   = statistical-outlier-removal (voxel 5mm, nb=20, std=2.0) then max-min
Gate: keep candidates with sf<=1+tol, demote the rest. sf = median(2-largest
sorted obs / cad ratio).

Saves scratch_scale_data2.json {tgt, top, cad_ext(per top id), obs_raw/pct/sor}
so tol/method tuning is fully OFFLINE afterward.

Run: docker compose run --rm oscar python3 scratch_scale_robust.py --n 150 --k 20
"""
import argparse, json, os, sys
import numpy as np
import open3d as o3d

_REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_REPO, "object_retrieval"))
sys.path.insert(0, os.path.join(_REPO, "object_retrieval"))
sys.path.insert(0, _REPO)

from eval_common import run_query, fusion_ranking, crop_by_bbox
from query_cloud import backproject_masked
from stage3_gallery import assemble_gallery, TARGET_DATASETS
from stage3_metrics import rank_of_target
from eval_bop_pose import (_matching_instances, _bbox_of, _pad_bbox, _pose_inputs,
                           _cam_entry, load_bop_targets, DATASET_TEST, _THIS_DIR)
from PIL import Image

_CAD = {}


def cad_ext_m(cid, id_to_mesh):
    if cid in _CAD: return _CAD[cid]
    path, units_m = id_to_mesh.get(cid, (None, False))
    ext = None
    if path and os.path.isfile(path):
        try:
            m = o3d.io.read_triangle_mesh(path)
            if not m.is_empty():
                ext = (np.asarray(m.get_axis_aligned_bounding_box().get_extent(), float)
                       * (1.0 if units_m else 0.001))
        except Exception:
            ext = None
    _CAD[cid] = ext
    return ext


def obs_extents(qc):
    """raw / percentile / SOR extents (metres) for one query cloud."""
    raw = (qc.max(0) - qc.min(0)).astype(float)
    pct = (np.percentile(qc, 98, 0) - np.percentile(qc, 2, 0)).astype(float)
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(qc.astype(np.float64))
    p = p.voxel_down_sample(0.005)
    if len(p.points) >= 20:
        p, _ = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    a = np.asarray(p.points)
    sor = (a.max(0) - a.min(0)).astype(float) if len(a) else raw
    return {"raw": raw, "pct": pct, "sor": sor}


def sf(obs, cad):
    if cad is None or float(np.max(cad)) < 1e-6: return None
    o = np.sort(obs)[::-1][:2]; c = np.sort(cad)[::-1][:2]
    c = np.where(c > 1e-6, c, 1.0)
    return float(np.median(o / c))


def gate_ranks(data, method, tol, K):
    ranks = []
    for d in data:
        top = d["top"][:K]
        sfs = [sf(np.array(d["obs_" + method]), np.array(d["cad_ext"][o]) if d["cad_ext"].get(o) else None)
               for o in top]
        keep = [o for o, s in zip(top, sfs) if (s is None or s <= 1 + tol)]
        drop = [o for o, s in zip(top, sfs) if not (s is None or s <= 1 + tol)]
        order = keep + drop + d["top"][K:]
        ranks.append(order.index(d["tgt"]) + 1 if d["tgt"] in order else None)
    return ranks


def r1r5(rs):
    n = len(rs)
    return sum(1 for r in rs if r == 1)/n, sum(1 for r in rs if r and r <= 5)/n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--datasets", default="ycbv")
    args = ap.parse_args()
    print("[setup] gallery ...", flush=True)
    G = assemble_gallery(TARGET_DATASETS)
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = G.components()
    cfg = G.eval_cfg; top_k = len(G.gallery_ids) + 5
    clip_rows = len(clip_retr._desc_labels); id_to_mesh = G.id_to_pose_mesh

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
                ext = obs_extents(qc)
                qemb = shape_m.encode_pointcloud(qc, colors=qcol)
                out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m, roi, cfg,
                                ulip_query_emb=qemb, dino_full_top_k=top_k,
                                ulip_full_top_k=top_k, clip_full_top_k=clip_rows)
                fused = fusion_ranking(out["fused_full"])
                top = [o for o, _ in fused[:args.k]]
                cadext = {o: (cad_ext_m(o, id_to_mesh).tolist()
                              if cad_ext_m(o, id_to_mesh) is not None else None) for o in top}
                data.append({"tgt": tgt, "top": [o for o, _ in fused], "cad_ext": cadext,
                             "obs_raw": ext["raw"].tolist(), "obs_pct": ext["pct"].tolist(),
                             "obs_sor": ext["sor"].tolist()})
        all_data[ds] = data
        fused_r1 = sum(1 for d in data if d["top"][0] == d["tgt"]) / len(data)
        print(f"\n=== {ds} robust scale gate, N={len(data)}, K={args.k} ===")
        print(f"fused R@1 = {fused_r1:.3f}")
        print(f"{'method':6s} {'tol':>5} {'R@1':>7} {'R@5':>7}")
        for method in ("pct", "sor"):
            best = (0, None)
            for tol in (0.10, 0.15, 0.20, 0.25, 0.30):
                r1, r5 = r1r5(gate_ranks(data, method, tol, args.k))
                print(f"{method:6s} {tol:5.2f} {r1:7.3f} {r5:7.3f}")
                if r1 > best[0]: best = (r1, tol)
            print(f"  -> best {method}: R@1={best[0]:.3f} @ tol={best[1]}")
    fn = os.path.join(_REPO, "scratch_scale_data_" + "_".join(all_data) + ".json")
    with open(fn, "w") as f:
        json.dump(all_data, f)
    print(f"\n[saved] {sum(len(v) for v in all_data.values())} queries -> {fn}")


if __name__ == "__main__":
    main()
