"""
Re-rank the fused top-K by the NAIVE SCALE CHECK (step7.estimate_fast logic)
instead of dGeDi geometry. Rationale: geometry co-scales absolute size away,
but the observed object's metric size is a strong INSTANCE discriminator for
same-shape / different-size CADs. Uses metric depth (metres) vs candidate CAD
bbox (converted to metres).

scale_factor = median( sort(obs_bbox)[:2] / sort(cad_bbox_m)[:2] )   # 2 largest
  ~1 for the correct instance at the right size (partial view -> <=1).

Compares, on ycbv full-fusion ULIP-pc:
  fused (no rerank)  vs  +scale (pure: rank top-K by |sf-1|)  vs
  +scale (borda: mean-rank of fused & scale-closeness).

Also SAVES per-query data (fused top-K, obs bbox, target) + a cad-bbox cache to
scratch_scale_data.json so the re-ranking rule can be refined offline (CPU only).

Run (oscar container; NO dgedi needed):
    docker compose run --rm oscar python3 scratch_scale_check.py --n 150 --k 20
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

_CAD_BBOX = {}      # id -> (3,) metres, cached


def cad_bbox_m(cid, id_to_mesh):
    if cid in _CAD_BBOX:
        return _CAD_BBOX[cid]
    path, units_m = id_to_mesh.get(cid, (None, False))
    ext = None
    if path and os.path.isfile(path):
        try:
            m = o3d.io.read_triangle_mesh(path)
            if not m.is_empty():
                ext = np.asarray(m.get_axis_aligned_bounding_box().get_extent(),
                                 float) * (1.0 if units_m else 0.001)  # -> metres
        except Exception:
            ext = None
    _CAD_BBOX[cid] = ext
    return ext


def scale_factor(obs_bbox, cad_ext):
    """step7.estimate_fast: median ratio of the 2 largest sorted axes."""
    if cad_ext is None or float(np.max(cad_ext)) < 1e-6:
        return None
    o = np.sort(obs_bbox)[::-1][:2]
    c = np.sort(cad_ext)[::-1][:2]
    c = np.where(c > 1e-6, c, 1.0)
    return float(np.median(o / c))


def borda(fused_ids, key_vals):
    """mean rank of (fused order, ascending key_vals) over the same id set."""
    n = len(fused_ids)
    fused_rank = {oid: i for i, oid in enumerate(fused_ids)}
    order = sorted(range(n), key=lambda i: key_vals[i])
    scale_rank = {fused_ids[order[j]]: j for j in range(n)}
    mr = {oid: (fused_rank[oid] + scale_rank[oid]) / 2 for oid in fused_ids}
    return sorted(fused_ids, key=lambda o: mr[o])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    print("[setup] assembling ULIP gallery (full fusion) ...", flush=True)
    G = assemble_gallery(TARGET_DATASETS)
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = G.components()
    cfg = G.eval_cfg
    top_k = len(G.gallery_ids) + 5
    clip_rows = len(clip_retr._desc_labels)
    id_to_mesh = G.id_to_pose_mesh

    ds = "ycbv"; ds_test = DATASET_TEST[ds]
    targets = load_bop_targets(os.path.join(_THIS_DIR, ds_test["targets"]))[:args.n]
    test_root = os.path.join(_THIS_DIR, ds_test["test_root"])

    ranks = {"pre": [], "pure": [], "borda": []}
    saved = []
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
            obs_bbox = (qc.max(0) - qc.min(0)).astype(float)     # metres
            qemb = shape_m.encode_pointcloud(qc, colors=qcol)
            out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m, roi, cfg,
                            ulip_query_emb=qemb, dino_full_top_k=top_k,
                            ulip_full_top_k=top_k, clip_full_top_k=clip_rows)
            fused = fusion_ranking(out["fused_full"])
            top = [o for o, _ in fused[:args.k]]
            ranks["pre"].append(rank_of_target(fused, tgt))

            # scale factor per candidate; |sf-1| is the re-rank key (inf if no bbox)
            sfs = [scale_factor(obs_bbox, cad_bbox_m(o, id_to_mesh)) for o in top]
            key = [abs(s - 1.0) if s is not None else 1e9 for s in sfs]
            pure = [top[i] for i in np.argsort(key, kind="stable")] + \
                   [o for o, _ in fused[args.k:]]
            bord = borda(top, key) + [o for o, _ in fused[args.k:]]
            ranks["pure"].append(rank_of_target([(o, 0) for o in pure], tgt))
            ranks["borda"].append(rank_of_target([(o, 0) for o in bord], tgt))

            saved.append({"tgt": tgt, "obs_bbox": obs_bbox.tolist(),
                          "top": top, "sf": [None if s is None else round(s, 4) for s in sfs]})

    def summ(rs):
        n = len(rs)
        return (sum(1 for r in rs if r == 1)/n,
                sum(1 for r in rs if r and r <= 5)/n,
                sum((1.0/r if r else 0) for r in rs)/n)

    print(f"\n=== ycbv ULIP-pc full-fusion, NAIVE SCALE re-rank, N={len(ranks['pre'])}, K={args.k} ===")
    print(f"{'variant':22s} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    for name, lab in (("pre", "fused (no rerank)"),
                      ("pure", "+scale (pure |sf-1|)"),
                      ("borda", "+scale (borda)")):
        r1, r5, mrr = summ(ranks[name])
        print(f"{lab:22s} {r1:6.3f} {r5:6.3f} {mrr:6.3f}")
    with open(os.path.join(_REPO, "scratch_scale_data.json"), "w") as f:
        json.dump(saved, f)
    print(f"\n[saved] {len(saved)} queries -> scratch_scale_data.json (for offline refinement)")


if __name__ == "__main__":
    main()
