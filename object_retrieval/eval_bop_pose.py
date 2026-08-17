"""
eval_bop_pose.py
================
Stage-3 BOP evaluation for OSCAR+, per ``STAGE3_EVALUATION_CONCEPT.md``
(revised 2026-08-17). Three independent settings over the same RGB-D BOP
queries (YCB-V, T-LESS, LM-O; GT visible bbox + mask + 6D pose, so retrieval
and pose are isolated from segmentation):

    --mode 3a   exact CAD in the gallery -> RETRIEVAL ONLY
                Recall@1/5/10, MRR, geometry coverage, mean #registered.
                FoundationPose is NOT run in 3a.

    --mode gt   exact-CAD FoundationPose benchmark (the "GT run"):
                FoundationPose with the GROUND-TRUTH target CAD ->
                D_posed_gt = D_sym(T_gt·P_T, T_hat·P_T)  (P_P == P_T).
                No retrieval; the reference D_sym for the Delta pairing.

    --mode 3b   proxy-only gallery -> retrieve top-1 proxy, FoundationPose it ->
                D_posed = D_sym(T_gt·P_T, T_hat·P_P), and (with --gt-records)
                the paired substitution cost Delta = D_posed - D_posed_gt.

Pose quality is reported as D_sym (mm + /diameter) and F-score at 1% and 5% of
the target diameter, for BOTH the gt benchmark and 3b — the two are directly
comparable on one scale. Official BOP-AR (VSD/MSSD/MSPD) is descoped from the
headline (user decision 2026-08-17); raw estimated poses are stored per
instance so it can be derived later.

Determinism: all explicit RNGs are seeded (see ``_seed_everything``). Two
sources are NOT bit-reproducible and are documented, not silenced:
  * FoundationPose's pose-hypothesis sampling / refinement is stochastic on GPU;
    we fix refine_iter and store the returned pose, but repeated calls can
    differ slightly.
  * open3d RANSAC in the dGeDi service is seeded server-side where the open3d
    build supports it; older builds ignore the seed (documented in DETERMINISM).

How to run (inside the oscar container, from object_retrieval/):
    python3 eval_bop_pose.py --datasets all --mode 3a                 # retrieval
    python3 eval_bop_pose.py --datasets all --mode 3a --pc-query      # pc mode
    python3 eval_bop_pose.py --datasets all --mode 3a --dgedi --dgedi-repo
    python3 eval_bop_pose.py --datasets all --mode gt                 # FP benchmark
    python3 eval_bop_pose.py --datasets all --mode 3b \
        --gt-records results_bop_stage3_gt/combined_gt.json
"""

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OSCAR_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _OSCAR_ROOT not in sys.path:
    sys.path.insert(0, _OSCAR_ROOT)

from eval_common import run_query, fusion_ranking, crop_by_bbox
from query_cloud import backproject_masked
from dgedi_bridge import dgedi_rerank, dgedi_health
from stage3_gallery import assemble_gallery, TARGET_DATASETS, UNI3D_OVERRIDES
from stage3_metrics import (rank_of_target, summarize_retrieval,
                            sample_surface_mm, d_sym, summarize_dsym,
                            summarize_delta, instance_key)
from pipeline.foundationpose_bridge import call_foundationpose

logger = logging.getLogger(__name__)

# FoundationPose runs in its own container on the compose network. It works in
# METRES; BOP is millimetres — so depth px*depth_scale/1000 -> m, BOP meshes
# (models_eval, mm) pass scale=0.001, and the returned translation *1000 -> mm.
FP_URL = "http://foundationpose:5050/estimate_pose"
_M_TO_MM = 1000.0

# Minimum points in the query partial cloud for the shape/geometry arms; below
# this, skip pc-query encode + dGeDi (degrade to the appearance arms only).
MIN_CLOUD_PTS = 64


def _seed_everything(seed: int = 0):
    """Seed every explicit RNG the eval touches. FoundationPose (separate
    container, GPU) and — on older open3d builds — RANSAC remain stochastic;
    that residual is documented, not hidden."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ============================================================================
# Per-dataset BOP query layout (test scenes + targets)
# ============================================================================
DATASET_TEST = {
    "ycbv":  dict(test_root="../eval/datasets/ycbv/test",
                  targets="../eval/datasets/ycbv/test_targets_bop19.json"),
    "tless": dict(test_root="../eval/datasets/tless/test_primesense",
                  targets="../eval/datasets/tless/test_targets_bop19.json"),
    "lmo":   dict(test_root="../eval/datasets/lmo/test",
                  targets="../eval/datasets/lmo/test_targets_bop19.json"),
}


# ============================================================================
# BOP loaders
# ============================================================================

def load_bop_targets(path):
    with open(path) as f:
        return json.load(f)


def _load_scene_json(scene_dir, name, im_id):
    p = os.path.join(scene_dir, name)
    if not os.path.isfile(p):
        return []
    with open(p) as f:
        return json.load(f).get(str(im_id), [])


def _matching_instances(scene_dir, im_id, obj_id):
    """All (gt_idx, gt, gt_info) instances of obj_id in an image (inst_count>1
    handled). gt_idx indexes scene_gt — it names the mask_visib file."""
    gts = _load_scene_json(scene_dir, "scene_gt.json", im_id)
    infos = _load_scene_json(scene_dir, "scene_gt_info.json", im_id)
    out = []
    for i, g in enumerate(gts):
        if g.get("obj_id") == obj_id:
            info = infos[i] if i < len(infos) else {}
            out.append((i, g, info))
    return out


def _bbox_of(info):
    b = info.get("bbox_visib") or info.get("bbox_obj")
    if not b or b[2] <= 0 or b[3] <= 0:   # w,h must be positive
        return None
    return b


def _pad_bbox(bbox, img_w, img_h, min_size=16):
    """Grow a tiny (heavily-occluded) bbox to a minimum size, centred and
    clamped to the image (a 1px-thin crop crashes the HF image processor)."""
    x, y, w, h = (float(v) for v in bbox)
    cx, cy = x + w / 2.0, y + h / 2.0
    w, h = max(w, min_size), max(h, min_size)
    x = min(max(0.0, cx - w / 2.0), max(0.0, img_w - w))
    y = min(max(0.0, cy - h / 2.0), max(0.0, img_h - h))
    return [x, y, min(w, img_w), min(h, img_h)]


# ============================================================================
# Pose inputs + FoundationPose call
# ============================================================================

def _cam_entry(scene_dir, im_id):
    p = os.path.join(scene_dir, "scene_camera.json")
    with open(p) as f:
        return json.load(f)[str(im_id)]


def _gt_pose(gt):
    """(R 3x3, t 3) from a scene_gt entry — BOP camera frame, mm."""
    R = np.array(gt["cam_R_m2c"], float).reshape(3, 3)
    t = np.array(gt["cam_t_m2c"], float).reshape(3)
    return R, t


def _pose_inputs(scene_dir, im_id, gt_idx, cam):
    """Full-frame depth (metres for FP), mask, K for one instance."""
    im6 = f"{im_id:06d}"
    depth_raw = np.array(Image.open(os.path.join(scene_dir, "depth", f"{im6}.png")))
    depth_m = depth_raw.astype(np.float32) * float(cam["depth_scale"]) / _M_TO_MM
    mask_p = os.path.join(scene_dir, "mask_visib", f"{im6}_{gt_idx:06d}.png")
    mask = (np.array(Image.open(mask_p)) > 0).astype(np.uint8)
    K = np.array(cam["cam_K"], float).reshape(3, 3)
    return depth_m, mask, K


def estimate_pose(cad_path, rgb_np, depth_m, mask, K, mesh_units_m, refine_iter):
    """FoundationPose register() -> (R 3x3, t 3 in mm, conf). ``mesh_units_m``
    True if the mesh is already in metres (scale 1.0); False for BOP-mm meshes."""
    scale = 1.0 if mesh_units_m else (1.0 / _M_TO_MM)
    pose, conf = call_foundationpose(FP_URL, rgb=rgb_np, depth=depth_m, mask=mask,
                                     K=K, cad_path=cad_path, scale=scale,
                                     refine_iter=refine_iter)
    return pose[:3, :3], pose[:3, 3] * _M_TO_MM, float(conf)


def _models_eval_dir(dataset):
    return os.path.join(_THIS_DIR, "..", "eval", "datasets", dataset, "models_eval")


class _ModelCache:
    """Lazily loads BOP models_eval diameter + surface sample per obj_id (mm)."""
    def __init__(self, dataset):
        self.dir = _models_eval_dir(dataset)
        self.info = json.load(open(os.path.join(self.dir, "models_info.json")))
        self._c = {}

    def get(self, obj_id):
        if obj_id not in self._c:
            mp = os.path.join(self.dir, f"obj_{obj_id:06d}.ply")
            mi = self.info[str(obj_id)]
            self._c[obj_id] = dict(path=mp, diameter=float(mi["diameter"]),
                                   pts=sample_surface_mm(mp, units_m=False))
        return self._c[obj_id]


# ============================================================================
# Geometry re-rank (dGeDi, Stage-1 E2_both — Borda mean-rank of RANSAC fitness
# and trimmed Chamfer)
# ============================================================================

def _geo_rerank(fused_ranking, geo, top_k):
    """Re-rank the fused top-K by dGeDi geometry (Borda mean-rank of RANSAC
    fitness and trimmed distance). Failed/uncached candidates sort to the back
    of the shortlist; the tail past top_k is untouched."""
    head = fused_ranking[:top_k]
    tail = fused_ranking[top_k:]
    ids = [oid for oid, _ in head]
    NEG = float("-inf")

    def _sig(o, key, sign):
        g = geo.get(o)
        if not g or not g.get("ok"):
            return NEG
        return sign * float(g[key])

    fit = [_sig(o, "ransac_fitness", 1.0) for o in ids]
    dst = [_sig(o, "d_ransac", -1.0) for o in ids]

    def _ranks(vals):
        return np.argsort(np.argsort(-np.asarray(vals), kind="stable"),
                          kind="stable").astype(float)

    mean_rank = (_ranks(fit) + _ranks(dst)) / 2.0
    order = list(np.argsort(mean_rank, kind="stable"))
    head_re = [(ids[i], -float(mean_rank[i])) for i in order]
    return head_re + tail


# ============================================================================
# Per-object aggregation (concept doc requires per-object tables)
# ============================================================================

def _per_object(dsym_recs, value_key="d_sym"):
    """Group per-instance D_sym records by obj_id -> mean/median + n."""
    by = {}
    for r in dsym_recs:
        by.setdefault(r["obj_id"], []).append(r[value_key])
    out = {}
    for oid, vals in sorted(by.items()):
        a = np.array(vals, float)
        out[str(oid)] = {"n": int(a.size), "mean": float(a.mean()),
                         "median": float(np.median(a))}
    return out


# ============================================================================
# Mode `gt`: exact-CAD FoundationPose benchmark (D_posed_gt, P_P == P_T)
# ============================================================================

def _eval_gt_dataset(dataset, refine_iter, max_targets):
    models = _ModelCache(dataset)
    ds_test = DATASET_TEST[dataset]
    test_root = os.path.join(_THIS_DIR, ds_test["test_root"])
    targets = load_bop_targets(os.path.join(_THIS_DIR, ds_test["targets"]))
    if max_targets > 0:
        targets = targets[:max_targets]
    print(f"[stage3-gt] {dataset}: {len(targets)} BOP targets (FP with GT CAD)")

    records = []
    dsym_recs = []
    n_att = 0
    for t in tqdm(targets, desc=f"{dataset} gt"):
        scene_id, im_id, obj_id = t["scene_id"], t["im_id"], t["obj_id"]
        scene_dir = os.path.join(test_root, f"{scene_id:06d}")
        rgb_path = os.path.join(scene_dir, "rgb", f"{im_id:06d}.png")
        if not os.path.isfile(rgb_path):
            alt = rgb_path[:-4] + ".jpg"
            rgb_path = alt if os.path.isfile(alt) else rgb_path
        if not os.path.isfile(rgb_path):
            continue
        rgb_np = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        cam = _cam_entry(scene_dir, im_id)
        for gt_idx, gt, info in _matching_instances(scene_dir, im_id, obj_id):
            if _bbox_of(info) is None:
                continue
            depth_m, mask, K = _pose_inputs(scene_dir, im_id, gt_idx, cam)
            m = models.get(obj_id)
            R_gt, t_gt = _gt_pose(gt)
            n_att += 1
            rec = {"dataset": dataset, "scene_id": scene_id, "im_id": im_id,
                   "obj_id": obj_id, "gt_idx": gt_idx, "diameter": m["diameter"]}
            try:
                R_e, t_e, conf = estimate_pose(m["path"], rgb_np, depth_m, mask,
                                               K, mesh_units_m=False,
                                               refine_iter=refine_iter)
                ds = d_sym(m["pts"], R_gt, t_gt, m["pts"], R_e, t_e, m["diameter"])
                rec.update({"d_posed_gt": round(ds["d_sym"], 3),
                            "d_sym_norm": round(ds["d_sym_norm"], 4),
                            "fscore": ds["fscore"], "pose_conf": round(conf, 4),
                            "R": np.asarray(R_e).reshape(9).tolist(),
                            "t": np.asarray(t_e).reshape(3).tolist()})
                dsym_recs.append({**ds, "obj_id": obj_id,
                                  "_key": instance_key(rec)})
            except Exception as exc:
                logger.warning("FP-gt failed (%s im %s obj %s): %s",
                               scene_id, im_id, obj_id, exc)
                rec["failed"] = True
            records.append(rec)

    summary = {"dataset": dataset, "mode": "gt",
               "n_queries_evaluated": len(records),
               "dsym": summarize_dsym(dsym_recs, n_attempted=n_att),
               "per_object": _per_object(dsym_recs)}
    return {"summary": summary, "records": records, "dsym_recs": dsym_recs}


# ============================================================================
# Modes `3a` / `3b`: retrieval (+ 3b proxy pose + D_sym + Delta)
# ============================================================================

def _eval_retrieval_dataset(dataset, gallery, components, mode, max_targets,
                            refine_iter, prx_samples, gt_by_key,
                            use_uni3d=False, use_dgedi=False, dgedi_top_k=10,
                            use_pc_query=False, dgedi_repo=False):
    """3a: retrieval only. 3b: retrieval (proxy gallery) + FP top-1 + D_sym."""
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = components
    cfg = gallery.eval_cfg
    include_target = (mode == "3a")
    do_pose = (mode == "3b")
    G = len(gallery.gallery_ids)
    top_k = G + 5
    clip_rows = len(clip_retr._desc_labels)

    models = _ModelCache(dataset) if do_pose else None
    need_cloud = use_uni3d or use_pc_query or use_dgedi or do_pose

    ds_test = DATASET_TEST[dataset]
    test_root = os.path.join(_THIS_DIR, ds_test["test_root"])
    targets = load_bop_targets(os.path.join(_THIS_DIR, ds_test["targets"]))
    if max_targets > 0:
        targets = targets[:max_targets]
    print(f"[stage3] {dataset} {mode}: {len(targets)} BOP targets vs |gallery|={G}")

    ranks, fused_ranks, records, dsym_recs = [], [], [], []
    tgt_samples = {}
    n_missing_rgb = 0

    for t in tqdm(targets, desc=f"{dataset} {mode}"):
        scene_id, im_id, obj_id = t["scene_id"], t["im_id"], t["obj_id"]
        scene_dir = os.path.join(test_root, f"{scene_id:06d}")
        rgb_path = os.path.join(scene_dir, "rgb", f"{im_id:06d}.png")
        if not os.path.isfile(rgb_path):
            alt = rgb_path[:-4] + ".jpg"
            rgb_path = alt if os.path.isfile(alt) else rgb_path
        if not os.path.isfile(rgb_path):
            n_missing_rgb += 1
            continue
        rgb = Image.open(rgb_path).convert("RGB")
        rgb_np = np.asarray(rgb, dtype=np.uint8)
        cam = _cam_entry(scene_dir, im_id) if need_cloud else None
        target_nsid = f"{dataset}/obj_{obj_id:06d}"

        for gt_idx, gt, info in _matching_instances(scene_dir, im_id, obj_id):
            bbox = _bbox_of(info)
            if bbox is None:
                continue
            roi = crop_by_bbox(rgb, _pad_bbox(bbox, rgb.width, rgb.height))

            depth_m = mask = K = None
            if need_cloud:
                depth_m, mask, K = _pose_inputs(scene_dir, im_id, gt_idx, cam)

            q_cloud = q_colors = None
            if need_cloud and mask is not None:
                q_cloud, q_colors = backproject_masked(depth_m, mask, K, rgb=rgb_np)
                if len(q_cloud) < MIN_CLOUD_PTS:
                    q_cloud = q_colors = None

            ulip_q_emb = None
            pc_query_fallback = False
            if use_uni3d or use_pc_query:
                if q_cloud is not None:
                    try:
                        ulip_q_emb = shape_m.encode_pointcloud(q_cloud, colors=q_colors)
                    except Exception as exc:
                        logger.warning("pc-query encode failed (%s im %s obj %s): %s",
                                       scene_id, im_id, obj_id, exc)
                if ulip_q_emb is None:
                    pc_query_fallback = True

            out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m,
                            roi, cfg, ulip_query_emb=ulip_q_emb,
                            dino_full_top_k=top_k, ulip_full_top_k=top_k,
                            clip_full_top_k=clip_rows)
            fused_ranking = fusion_ranking(out["fused_full"])

            ranking = fused_ranking
            geo_applied = False
            dgedi_n_req = dgedi_n_ok = 0
            if use_dgedi and q_cloud is not None:
                cand_ids = [oid for oid, _ in fused_ranking[:dgedi_top_k]]
                _dg = ({"ransac_keypoints": 6000, "ransac_max_iter": 10000,
                        "use_icp": True} if dgedi_repo else {})
                geo = dgedi_rerank(q_cloud, cand_ids, **_dg)
                if geo:
                    dgedi_n_req = len(cand_ids)
                    dgedi_n_ok = sum(1 for v in geo.values() if v.get("ok"))
                    if dgedi_n_ok > 0:
                        ranking = _geo_rerank(fused_ranking, geo, dgedi_top_k)
                        geo_applied = True

            r = rank_of_target(ranking, target_nsid) if include_target else None
            if include_target:
                ranks.append(r)

            rec = {"dataset": dataset, "scene_id": scene_id, "im_id": im_id,
                   "obj_id": obj_id, "gt_idx": gt_idx, "target_id": target_nsid,
                   "target_rank": r,
                   # ranked shortlist with fused/geo scores (top-10, matching the
                   # deepest reported Recall@k). In 3b these are the proxies that
                   # displaced the removed exact target.
                   "top10": [{"id": oid, "score": round(s, 5)}
                             for oid, s in ranking[:10]]}
            if use_uni3d or use_pc_query:
                rec["pc_query_fallback"] = pc_query_fallback
            if use_dgedi:
                rec["fused_rank"] = (rank_of_target(fused_ranking, target_nsid)
                                     if include_target else None)
                rec["geo_applied"] = geo_applied
                rec["dgedi_n_requested"] = dgedi_n_req
                rec["dgedi_n_ok"] = dgedi_n_ok
                if include_target:
                    fused_ranks.append(rec["fused_rank"])

            # --- 3b: pose the RETRIEVED top-1 proxy, D_sym vs GT-posed target ---
            if do_pose and ranking:
                m = models.get(obj_id)
                R_gt, t_gt = _gt_pose(gt)
                top1 = ranking[0][0]
                rec["top1"] = top1
                rec["top1_is_exact"] = (top1 == target_nsid)
                tpath, tunits = gallery.id_to_pose_mesh.get(top1, (None, False))
                if tpath and os.path.isfile(tpath):
                    if obj_id not in tgt_samples:
                        tgt_samples[obj_id] = m["pts"]
                    try:
                        Rt, tt, conf = estimate_pose(tpath, rgb_np, depth_m, mask,
                                                     K, mesh_units_m=tunits,
                                                     refine_iter=refine_iter)
                        if top1 not in prx_samples:
                            prx_samples[top1] = sample_surface_mm(tpath, units_m=tunits)
                        ds = d_sym(tgt_samples[obj_id], R_gt, t_gt,
                                   prx_samples[top1], Rt, tt, m["diameter"])
                        rec.update({"d_posed": round(ds["d_sym"], 3),
                                    "d_sym_norm": round(ds["d_sym_norm"], 4),
                                    "fscore": ds["fscore"],
                                    "top1_pose_conf": round(conf, 4),
                                    "diameter": m["diameter"],
                                    "top1_R": np.asarray(Rt).reshape(9).tolist(),
                                    "top1_t": np.asarray(tt).reshape(3).tolist()})
                        key = instance_key(rec)
                        drec = {**ds, "obj_id": obj_id, "_key": key}
                        if gt_by_key and key in gt_by_key:
                            rec["delta"] = round(ds["d_sym"] - gt_by_key[key], 3)
                        dsym_recs.append(drec)
                    except Exception as exc:
                        logger.warning("FP top-1 pose failed (%s): %s", top1, exc)
                else:
                    logger.warning("top-1 mesh missing for %s", top1)

            records.append(rec)

    summary = _summarize(dataset, mode, G, records, ranks, dsym_recs,
                         n_missing_rgb, include_target, do_pose,
                         gt_by_key=gt_by_key,
                         fused_ranks=fused_ranks if use_dgedi else None)
    return {"summary": summary, "records": records, "ranks": ranks,
            "fused_ranks": fused_ranks, "dsym_recs": dsym_recs}


def _summarize(dataset, mode, G, records, ranks, dsym_recs, n_missing_rgb,
               include_target, do_pose, gt_by_key=None, fused_ranks=None):
    summary = {"dataset": dataset, "mode": mode, "gallery_size": G,
               "target_in_gallery": include_target,
               "n_queries_evaluated": len(records),
               "n_missing_rgb": n_missing_rgb}
    if include_target:
        summary.update(summarize_retrieval(ranks))
        if fused_ranks:
            summary["pre_geometry"] = summarize_retrieval(fused_ranks)
    if any(r and "pc_query_fallback" in r for r in records):
        n_pc = sum(1 for r in records if r and "pc_query_fallback" in r)
        summary["pc_query_fallback"] = {
            "n_fell_back": sum(1 for r in records if r and r.get("pc_query_fallback")),
            "n_pc_query": n_pc}
    if any(r and "geo_applied" in r for r in records):
        oks = [r.get("dgedi_n_ok", 0) for r in records if r and "geo_applied" in r]
        summary["geometry_coverage"] = {
            "n_geo_applied": sum(1 for r in records if r and r.get("geo_applied")),
            "n_dgedi_query": len(oks),
            "mean_n_registered": float(np.mean(oks)) if oks else 0.0}
    if do_pose:
        n_att = sum(1 for r in records if r and "top1" in r)
        summary["dsym"] = summarize_dsym(dsym_recs, n_attempted=n_att)
        summary["per_object"] = _per_object(dsym_recs)
        if gt_by_key:
            summary["delta"] = summarize_delta(dsym_recs, gt_by_key)
    return summary


def _print_summary(tag, s):
    print(f"\n[stage3] {tag} — {s['n_queries_evaluated']} queries")
    if "recall@1" in s:
        print(f"  Recall@1={s['recall@1']:.3f}  Recall@5={s['recall@5']:.3f}  "
              f"Recall@10={s['recall@10']:.3f}  MRR={s['mrr']:.3f}  "
              f"(found {s['n_target_found']}/{s['n_queries_evaluated']})")
        if "pre_geometry" in s:
            p = s["pre_geometry"]
            print(f"  pre-geometry: Recall@1={p['recall@1']:.3f} "
                  f"Recall@5={p['recall@5']:.3f} MRR={p['mrr']:.3f}")
    if "geometry_coverage" in s:
        g = s["geometry_coverage"]
        print(f"  geometry: applied {g['n_geo_applied']}/{g['n_dgedi_query']}, "
              f"mean #registered={g['mean_n_registered']:.2f}")
    if "dsym" in s and s["dsym"].get("n_estimated"):
        d = s["dsym"]
        line = (f"  D_sym mean={d['d_sym_mean']:.2f}mm median={d['d_sym_median']:.2f} "
                f"/diam={d['d_sym_norm_mean']:.3f} (n={d['n_estimated']}, "
                f"cov={d.get('coverage', 1.0):.2f})")
        if "fscore" in d:
            fs = " ".join(f"F@{k}={v['f']:.3f}" for k, v in d["fscore"].items())
            line += f"  {fs}"
        print(line)
    if "delta" in s and s["delta"].get("n_paired"):
        dl = s["delta"]
        print(f"  Delta mean={dl['delta_mean']:.2f}mm median={dl['delta_median']:.2f} "
              f"(paired n={dl['n_paired']})")


# ============================================================================
# Driver
# ============================================================================

def _load_gt_by_key(path):
    """Build instance_key -> D_posed_gt (mm) from a gt-benchmark records/combined
    file, so 3b can pair each proxy D_posed with its exact-CAD reference."""
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    recs = data.get("all_records") if isinstance(data, dict) else data
    if recs is None and isinstance(data, dict):
        recs = data.get("records", [])
    out = {}
    for r in (recs or []):
        if r.get("d_posed_gt") is None:
            continue
        k = (r.get("dataset"), r["scene_id"], r["im_id"], r["obj_id"], r["gt_idx"])
        out[k] = float(r["d_posed_gt"])
    return out


def run_stage3(datasets, mode="3a", max_targets=0,
               output_dir="results_bop_stage3", refine_iter=5,
               use_uni3d=False, use_dgedi=False, dgedi_top_k=10,
               use_pc_query=False, dgedi_repo=False, gt_records=None):
    for d in datasets:
        if d not in DATASET_TEST:
            raise ValueError(f"Unknown dataset {d}; choose {list(DATASET_TEST)}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*64}\nStage-3 {mode} — queries {datasets}\n{'='*64}")

    # ---- mode gt: no gallery, no retrieval — FP with the GT CAD ----
    if mode == "gt":
        per_dataset, all_records, all_dsym = {}, [], []
        for dataset in datasets:
            res = _eval_gt_dataset(dataset, refine_iter, max_targets)
            per_dataset[dataset] = res["summary"]
            rdir = os.path.join(output_dir, f"{dataset}_gt")
            os.makedirs(rdir, exist_ok=True)
            with open(os.path.join(rdir, "records.json"), "w") as f:
                json.dump(res["records"], f, indent=2)
            with open(os.path.join(rdir, "summary.json"), "w") as f:
                json.dump(res["summary"], f, indent=2)
            _print_summary(f"{dataset} gt", res["summary"])
            all_records += res["records"]
            all_dsym += res["dsym_recs"]
        combined = {"mode": "gt", "datasets": datasets,
                    "n_queries_evaluated": len(all_records),
                    "dsym": summarize_dsym(all_dsym, n_attempted=len(all_records)),
                    "per_dataset": per_dataset, "all_records": all_records}
        with open(os.path.join(output_dir, "combined_gt.json"), "w") as f:
            json.dump(combined, f, indent=2)
        _print_summary("COMBINED gt", combined)
        print(f"\n[stage3] saved -> {output_dir}")
        return combined

    # ---- modes 3a / 3b: assemble the union gallery once ----
    gt_by_key = _load_gt_by_key(gt_records) if mode == "3b" else {}
    if mode == "3b" and gt_records:
        print(f"[stage3] paired Delta against {len(gt_by_key)} gt records "
              f"from {gt_records}")
    target_datasets = TARGET_DATASETS if mode == "3a" else ()
    print(f"[stage3] assembling gallery (targets in gallery: "
          f"{list(target_datasets) or 'none (proxy-only)'})"
          f"{'  [shape arm: Uni3D]' if use_uni3d else ''}...")
    gallery = assemble_gallery(target_datasets=target_datasets,
                               extra_overrides=(UNI3D_OVERRIDES if use_uni3d else None))
    components = gallery.components()
    G = len(gallery.gallery_ids)
    print(f"[stage3] |gallery| = {G}  clip_rows = {len(components[1]._desc_labels)}")
    if use_dgedi:
        h = dgedi_health()
        print(f"[stage3] dGeDi geometry re-rank ON (top_k={dgedi_top_k}, "
              f"{'repo 6000kp/10k/+ICP' if dgedi_repo else 'fast 512kp/5k'}); "
              f"service: {h if h else 'UNREACHABLE — will degrade to fused'}")

    prx_samples = {}
    per_dataset = {}
    pooled = {"ranks": [], "dsym_recs": [], "all_records": []}
    for dataset in datasets:
        res = _eval_retrieval_dataset(
            dataset, gallery, components, mode, max_targets, refine_iter,
            prx_samples, gt_by_key, use_uni3d=use_uni3d, use_dgedi=use_dgedi,
            dgedi_top_k=dgedi_top_k, use_pc_query=use_pc_query,
            dgedi_repo=dgedi_repo)
        s = res["summary"]
        per_dataset[dataset] = s
        rdir = os.path.join(output_dir, f"{dataset}_stage{mode}")
        os.makedirs(rdir, exist_ok=True)
        with open(os.path.join(rdir, "records.json"), "w") as f:
            json.dump(res["records"], f, indent=2)
        with open(os.path.join(rdir, "summary.json"), "w") as f:
            json.dump(s, f, indent=2)
        _print_summary(f"{dataset} {mode}", s)
        pooled["all_records"] += res["records"]
        pooled["ranks"] += res["ranks"]
        pooled["dsym_recs"] += res["dsym_recs"]

    combined = _summarize("ALL", mode, G, pooled["all_records"], pooled["ranks"],
                          pooled["dsym_recs"], 0, mode == "3a", mode == "3b",
                          gt_by_key=gt_by_key,
                          fused_ranks=[r.get("fused_rank") for r in
                                       pooled["all_records"] if "fused_rank" in r]
                                      if use_dgedi else None)
    combined["datasets"] = datasets
    combined["per_dataset"] = per_dataset
    with open(os.path.join(output_dir, f"combined_stage{mode}.json"), "w") as f:
        json.dump(combined, f, indent=2)
    if len(datasets) > 1:
        _print_summary(f"COMBINED {mode}", combined)
    print(f"\n[stage3] saved -> {output_dir}")
    return combined


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Stage-3 BOP evaluation (OSCAR+)")
    ap.add_argument("--datasets", default="all",
                    help=f"comma-separated query datasets, or 'all' (= "
                         f"{','.join(TARGET_DATASETS)}).")
    ap.add_argument("--mode", choices=["3a", "gt", "3b"], default="3a",
                    help="3a=retrieval only; gt=FP with GT CAD (D_posed_gt); "
                         "3b=proxy pose + D_sym (+Delta with --gt-records).")
    ap.add_argument("--max-targets", type=int, default=0,
                    help="limit targets PER dataset (0 = all; for smoke tests)")
    ap.add_argument("--output", default="results_bop_stage3")
    ap.add_argument("--refine-iter", type=int, default=5,
                    help="FoundationPose refinement iterations (default 5)")
    ap.add_argument("--gt-records", default=None,
                    help="3b only: gt-benchmark records/combined JSON to pair "
                         "D_posed against for Delta.")
    ap.add_argument("--uni3d", action="store_true",
                    help="swap the shape arm ULIP-2 -> Uni3D (pc-query).")
    ap.add_argument("--pc-query", action="store_true",
                    help="point-cloud query for the shape arm (else ULIP-2 "
                         "image-cross query).")
    ap.add_argument("--dgedi", action="store_true",
                    help="add the dGeDi geometry re-rank of the fused top-K.")
    ap.add_argument("--dgedi-top-k", type=int, default=10,
                    help="fused shortlist depth re-ranked by dGeDi (default 10)")
    ap.add_argument("--dgedi-repo", action="store_true",
                    help="dGeDi repo config: 6000 kp / 10k iters / +ICP.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _seed_everything(args.seed)
    datasets = (list(TARGET_DATASETS) if args.datasets == "all"
                else [d.strip() for d in args.datasets.split(",") if d.strip()])

    logging.basicConfig(level=logging.WARNING)
    run_stage3(datasets, mode=args.mode, max_targets=args.max_targets,
               output_dir=args.output, refine_iter=args.refine_iter,
               use_uni3d=args.uni3d, use_dgedi=args.dgedi,
               dgedi_top_k=args.dgedi_top_k, use_pc_query=args.pc_query,
               dgedi_repo=args.dgedi_repo, gt_records=args.gt_records)


if __name__ == "__main__":
    main()
