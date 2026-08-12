"""
GeDi geometry re-rank on ycbv (ULIP-pc retrieval) using the EXACT Stage-1 recipe.

Faithful to experiments/stage1_reproduce.py:geometry_rerank — reuses the PRODUCTION
GeometryReRanker (pipeline/step_b2_geometry_reranking.py), so nothing about the GeDi
usage is re-implemented:
  * query cloud  = center + UNIT-normalize (pts -= mean; pts /= max_radius), normals@0.08
  * candidate CAD = NATIVE mesh scale, 10000 sampled pts, NO normalization
  * GeDi service  = no normalization, fixed r_lrf=0.5, voxel_size default (0.005)
  * signal        = chamfer_ransac, all_aligned=True, combined by both_borda
This is the *scale-inconsistent* Stage-1 regime (unit query vs native-scale CAD) on
purpose — the point is to reproduce what Stage-1 actually did, not the co-scaled fix.

The fused retrieval + both_borda combine + metrics are identical to
scratch_gedi_vs_dgedi.py, so the ONLY difference vs that script's "+gedi" is the
cloud preparation (Stage-1 unit/native/dense vs co-scaled/sparse).

Run (oscar container, gedi service up):
    docker compose up -d gedi
    docker compose run --rm oscar python3 scratch_gedi_stage1.py --n 50 --k 20
"""
import argparse, os, sys
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
                           _cam_entry, load_bop_targets, DATASET_TEST, _THIS_DIR, _geo_rerank)
from pipeline.config import PipelineConfig
from pipeline.step6_fusion import FusedCandidate
from pipeline.step_b2_geometry_reranking import GeometryReRanker
from PIL import Image


def unit_query_pcd(qc):
    """Stage-1 query prep: center + unit-normalize + estimate normals@0.08."""
    pts = np.asarray(qc, dtype=np.float64)
    pts = pts - pts.mean(0)
    r = float(np.linalg.norm(pts, axis=1).max())
    if r > 0:
        pts = pts / r
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    return pcd


def gedi_rerank_stage1(reranker, qc, cand_ids, id_to_mesh, k):
    """Stage-1-faithful GeDi geometry via the production GeometryReRanker.

    Returns a geo dict {id: {ok, ransac_fitness, d_ransac}} in the same shape the
    dgedi bridge returns, so _geo_rerank(...) applies the identical both_borda."""
    cands = []
    for cid in cand_ids:
        path, _units = id_to_mesh.get(cid, (None, False))
        cands.append(FusedCandidate(object_id=cid, fused_score=0.0,
                                    cad_model_path=path or ""))
    pcd = unit_query_pcd(qc)
    res = reranker.rerank(cands, pcd, signal="chamfer_ransac",
                          all_aligned=True, query_id=None)
    geo = {}
    for c in res.candidates:
        ok = not c.registration_failed and np.isfinite(c.d_ransac)
        geo[c.object_id] = {"ok": ok,
                            "ransac_fitness": float(c.ransac_fitness),
                            "d_ransac": float(c.d_ransac) if ok else float("inf")}
    return geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    print("[setup] assembling ULIP gallery ...", flush=True)
    G = assemble_gallery(TARGET_DATASETS)
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = G.components()
    cfg = G.eval_cfg
    top_k = len(G.gallery_ids) + 5
    clip_rows = len(clip_retr._desc_labels)

    # Production reranker, Stage-1 config: default voxel_size, both_borda via all_aligned.
    gcfg = PipelineConfig(geometry_reranking_top_k=args.k,
                          geometry_reranking_signal="chamfer_ransac",
                          geometry_skip_icp=True,
                          gedi_cache_dir=os.path.join(_REPO, "object_retrieval",
                                                      "results_stage3_gedi_stage1", "_gedi"))
    os.makedirs(gcfg.gedi_cache_dir, exist_ok=True)
    reranker = GeometryReRanker(gcfg)
    print(f"[setup] voxel_size={gcfg.voxel_size} (max_corr_dist={ (gcfg.voxel_size or 0.005)*1.5 }); "
          f"gedi available={reranker._get_gedi().available}", flush=True)

    ds = "ycbv"; ds_test = DATASET_TEST[ds]
    targets = load_bop_targets(os.path.join(_THIS_DIR, ds_test["targets"]))[:args.n]
    test_root = os.path.join(_THIS_DIR, ds_test["test_root"])

    ranks = {"pre": [], "gedi": []}
    for t in targets:
        sid, im, oid = t["scene_id"], t["im_id"], t["obj_id"]
        sdir = os.path.join(test_root, f"{sid:06d}")
        rgbp = os.path.join(sdir, "rgb", f"{im:06d}.png")
        if not os.path.isfile(rgbp):
            rgbp = rgbp[:-4] + ".jpg"
        if not os.path.isfile(rgbp):
            continue
        rgb = Image.open(rgbp).convert("RGB"); rgb_np = np.asarray(rgb, np.uint8)
        cam = _cam_entry(sdir, im)
        tgt = f"{ds}/obj_{oid:06d}"
        for gi, gt, info in _matching_instances(sdir, im, oid):
            bb = _bbox_of(info)
            if bb is None:
                continue
            roi = crop_by_bbox(rgb, _pad_bbox(bb, rgb.width, rgb.height))
            depth_m, _dmm, mask, K = _pose_inputs(sdir, im, gi, cam)
            qc, qcol = backproject_masked(depth_m, mask, K, rgb=rgb_np)
            if len(qc) < 64:
                continue
            qemb = shape_m.encode_pointcloud(qc, colors=qcol)        # ULIP pc-query
            out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m, roi, cfg,
                            ulip_query_emb=qemb, dino_full_top_k=top_k,
                            ulip_full_top_k=top_k, clip_full_top_k=clip_rows)
            fused = fusion_ranking(out["fused_full"])
            cand = [o for o, _ in fused[:args.k]]
            ranks["pre"].append(rank_of_target(fused, tgt))
            geo = gedi_rerank_stage1(reranker, qc, cand, G.id_to_pose_mesh, args.k)
            ranks["gedi"].append(rank_of_target(_geo_rerank(fused, geo, args.k), tgt))

    def summ(rs):
        n = len(rs)
        r1 = sum(1 for r in rs if r == 1) / n
        r5 = sum(1 for r in rs if r and r <= 5) / n
        mrr = sum((1.0 / r if r else 0) for r in rs) / n
        return r1, r5, mrr

    print(f"\n=== ycbv ULIP-pc, Stage-1-style GeDi, N_instances={len(ranks['pre'])}, K={args.k} ===")
    print(f"{'variant':18s} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    for name in ("pre", "gedi"):
        r1, r5, mrr = summ(ranks[name])
        label = "fused(no-geo)" if name == "pre" else "+gedi(stage1)"
        print(f"{label:18s} {r1:6.3f} {r5:6.3f} {mrr:6.3f}")


if __name__ == "__main__":
    main()
