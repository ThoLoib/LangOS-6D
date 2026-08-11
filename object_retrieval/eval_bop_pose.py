"""
eval_bop_pose.py
================
Stage-3 BOP evaluation for OSCAR+ (paired 3a / 3b), per
``Downloads/STAGE3_EVALUATION_CONCEPT.md``.

Query datasets: YCB-V, T-LESS, LM-O (RGB-D, GT visible bbox + mask + 6D pose).
Gallery: a multi-dataset union assembled by ``stage3_gallery`` —

    3a (exact CAD available): G_proxy ∪ G_target,d  → retrieval Recall@K/MRR
                              + pose BOP-AR (oracle & retrieved-exact)
    3b (proxy only):          G_proxy                → proxy pose + D_sym

Implemented: **Phase A** retrieval (Recall@1/5/10 + MRR) and **Phase B** 3a pose
(FoundationPose -> BOP-AR, oracle + conditional retrieved-exact). 3b proxy pose +
D_sym is Phase C. VSD needs a depth renderer (not yet wired) so BOP-AR is
currently MSSD/MSPD-only — reported with a note. Pose needs the foundationpose
compose service up.

In 3a the gallery is ONE big combined DB (G_proxy ∪ all target datasets); every
query dataset is scored against it and a combined summary is pooled.

How to run (inside the oscar container, from object_retrieval/):
    python3 eval_bop_pose.py --datasets all --mode 3a                    # retrieval, all queries
    python3 eval_bop_pose.py --datasets all --mode 3a --pose            # + BOP-AR
    python3 eval_bop_pose.py --datasets ycbv --mode 3a --max-targets 5  # single-dataset smoke
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OSCAR_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _OSCAR_ROOT not in sys.path:
    sys.path.insert(0, _OSCAR_ROOT)

from eval_common import run_query, fusion_ranking, crop_by_bbox
from stage3_gallery import assemble_gallery, TARGET_DATASETS
from stage3_metrics import (rank_of_target, summarize_retrieval,
                            load_bop_model_points, get_symmetries,
                            pose_errors, bop_ar,
                            sample_surface_mm, d_sym, summarize_dsym)
from pipeline.foundationpose_bridge import call_foundationpose


def _build_vsd_renderer(width, height):
    """pyrender+EGL depth renderer for VSD, or None if headless GL is
    unavailable (then BOP-AR degrades to MSSD/MSPD, flagged in the summary)."""
    try:
        from stage3_render import PyrenderDepthRenderer
        return PyrenderDepthRenderer(width, height)
    except Exception as exc:
        logger.warning("VSD renderer unavailable (%s); AR will be MSSD/MSPD only", exc)
        return None

logger = logging.getLogger(__name__)

# FoundationPose runs in its own container on the compose network. It works in
# METRES; BOP is millimetres — so depth px*depth_scale/1000 -> m, BOP meshes
# (models_eval, mm) pass scale=0.001, and the returned translation *1000 -> mm.
FP_URL = "http://foundationpose:5050/estimate_pose"
_M_TO_MM = 1000.0


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
    clamped to the image. A 1px-thin crop otherwise confuses the HF image
    processor's channel inference (it reads the size-1 axis as 1 channel and
    crashes on the 3-element mean). No-op for boxes already >= min_size."""
    x, y, w, h = (float(v) for v in bbox)
    cx, cy = x + w / 2.0, y + h / 2.0
    w, h = max(w, min_size), max(h, min_size)
    x = min(max(0.0, cx - w / 2.0), max(0.0, img_w - w))
    y = min(max(0.0, cy - h / 2.0), max(0.0, img_h - h))
    return [x, y, min(w, img_w), min(h, img_h)]


# ============================================================================
# Pose inputs + FoundationPose call (Phase B)
# ============================================================================

def _cam_entry(scene_dir, im_id):
    return _load_scene_json_dict(scene_dir, "scene_camera.json", im_id)


def _load_scene_json_dict(scene_dir, name, im_id):
    p = os.path.join(scene_dir, name)
    with open(p) as f:
        return json.load(f)[str(im_id)]


def _gt_pose(gt):
    """(R 3x3, t 3) from a scene_gt entry — BOP camera frame, mm."""
    R = np.array(gt["cam_R_m2c"], float).reshape(3, 3)
    t = np.array(gt["cam_t_m2c"], float).reshape(3)
    return R, t


def _pose_inputs(scene_dir, im_id, gt_idx, cam):
    """Full-frame depth (metres for FP, mm for VSD), mask, K for one instance."""
    im6 = f"{im_id:06d}"
    depth_raw = np.array(Image.open(os.path.join(scene_dir, "depth", f"{im6}.png")))
    depth_mm = depth_raw.astype(np.float32) * float(cam["depth_scale"])
    depth_m = depth_mm / _M_TO_MM
    mask_p = os.path.join(scene_dir, "mask_visib", f"{im6}_{gt_idx:06d}.png")
    mask = (np.array(Image.open(mask_p)) > 0).astype(np.uint8)
    K = np.array(cam["cam_K"], float).reshape(3, 3)
    return depth_m, depth_mm, mask, K


def estimate_pose(cad_path, rgb_np, depth_m, mask, K, mesh_units_m, refine_iter):
    """FoundationPose register() -> (R 3x3, t 3 in mm). ``mesh_units_m`` True if
    the mesh is already in metres (scale 1.0); False for BOP-mm meshes (0.001)."""
    scale = 1.0 if mesh_units_m else (1.0 / _M_TO_MM)
    pose, conf = call_foundationpose(FP_URL, rgb=rgb_np, depth=depth_m, mask=mask,
                                     K=K, cad_path=cad_path, scale=scale,
                                     refine_iter=refine_iter)
    return pose[:3, :3], pose[:3, 3] * _M_TO_MM, float(conf)


def _models_eval_dir(dataset):
    return os.path.join(_THIS_DIR, "..", "eval", "datasets", dataset, "models_eval")


class _ModelCache:
    """Lazily loads BOP models_eval points/symmetries/diameter per obj_id."""
    def __init__(self, dataset):
        self.dir = _models_eval_dir(dataset)
        self.info = json.load(open(os.path.join(self.dir, "models_info.json")))
        self._c = {}

    def get(self, obj_id):
        if obj_id not in self._c:
            mp = os.path.join(self.dir, f"obj_{obj_id:06d}.ply")
            mi = self.info[str(obj_id)]
            self._c[obj_id] = dict(pts=load_bop_model_points(mp),
                                   syms=get_symmetries(mi),
                                   diameter=float(mi["diameter"]),
                                   path=mp)
        return self._c[obj_id]


# ============================================================================
# Stage-3 retrieval (Phase A)
# ============================================================================

def _eval_dataset(dataset, gallery, components, mode, max_targets,
                  do_pose_3a, do_dsym_3b, refine_iter, prx_samples):
    """Retrieval (+pose) for ONE query dataset against the shared gallery.

    In 3a the gallery is the one big combined DB (proxies + ALL target datasets),
    so every query dataset is scored against the same index. Returns a per-dataset
    summary plus the raw accumulators so the caller can pool a combined summary.
    ``prx_samples`` is shared across datasets (proxy meshes are dataset-agnostic)."""
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = components
    cfg = gallery.eval_cfg
    include_target = (mode == "3a")
    G = len(gallery.gallery_ids)
    top_k = G + 5    # DINO/ULIP are per-object: this ranks the whole gallery.
    # CLIP.retrieve caps DESCRIPTION ROWS (42/object), not objects — so to let
    # the dedup reach all G objects, CLIP must be given the total row count.
    clip_rows = len(clip_retr._desc_labels)

    models = _ModelCache(dataset) if (do_pose_3a or do_dsym_3b) else None
    do_any_pose = do_pose_3a or do_dsym_3b

    ds_test = DATASET_TEST[dataset]
    test_root = os.path.join(_THIS_DIR, ds_test["test_root"])
    targets = load_bop_targets(os.path.join(_THIS_DIR, ds_test["targets"]))
    if max_targets > 0:
        targets = targets[:max_targets]
    print(f"[stage3] {dataset}: {len(targets)} BOP targets vs |gallery|={G}")

    ranks = []
    records = []
    oracle_recs = []      # one per GT instance (BOP-AR with the GT CAD)
    retr_recs = []        # only where top-1 == exact target (conditional AR)
    n_missing_rgb = 0

    _INF = dict(mssd=float("inf"), mspd=float("inf"))   # failed pose = miss
    vsd_renderer = None   # built lazily once we know the image size
    vsd_objs = set()      # obj_ids already added to the renderer
    dsym_recs = []        # 3b per-instance D_sym
    tgt_samples = {}      # obj_id  -> target surface points (mm), per-dataset

    for t in tqdm(targets, desc=f"{dataset} {mode}"):
        scene_id, im_id, obj_id = t["scene_id"], t["im_id"], t["obj_id"]
        scene_dir = os.path.join(test_root, f"{scene_id:06d}")
        rgb_path = os.path.join(scene_dir, "rgb", f"{im_id:06d}.png")
        if not os.path.isfile(rgb_path):
            # some datasets use .jpg
            alt = rgb_path[:-4] + ".jpg"
            rgb_path = alt if os.path.isfile(alt) else rgb_path
        if not os.path.isfile(rgb_path):
            n_missing_rgb += 1
            continue
        rgb = Image.open(rgb_path).convert("RGB")
        rgb_np = np.asarray(rgb, dtype=np.uint8)
        img_w = rgb.width
        cam = _cam_entry(scene_dir, im_id) if do_any_pose else None

        target_nsid = f"{dataset}/obj_{obj_id:06d}"

        for gt_idx, gt, info in _matching_instances(scene_dir, im_id, obj_id):
            bbox = _bbox_of(info)
            if bbox is None:
                continue
            roi = crop_by_bbox(rgb, _pad_bbox(bbox, rgb.width, rgb.height))

            out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m,
                            roi, cfg,
                            dino_full_top_k=top_k, ulip_full_top_k=top_k,
                            clip_full_top_k=clip_rows)
            ranking = fusion_ranking(out["fused_full"])   # [(nsid, score), ...]

            r = rank_of_target(ranking, target_nsid) if include_target else None
            if include_target:
                ranks.append(r)

            rec = {
                "scene_id": scene_id, "im_id": im_id, "obj_id": obj_id,
                "gt_idx": gt_idx, "target_id": target_nsid, "target_rank": r,
                "top5": [{"id": oid, "score": round(s, 5)}
                         for oid, s in ranking[:5]],
            }

            # --- 3a pose (BOP-AR): oracle GT CAD + conditional retrieved-exact ---
            if do_pose_3a:
                m = models.get(obj_id)
                R_gt, t_gt = _gt_pose(gt)
                depth_m, depth_mm, mask, K = _pose_inputs(scene_dir, im_id, gt_idx, cam)
                # build the VSD renderer once (first frame gives us H,W) and
                # register the object's GT CAD before rendering it
                if vsd_renderer is None:
                    vsd_renderer = _build_vsd_renderer(rgb.width, rgb.height)
                if vsd_renderer is not None and obj_id not in vsd_objs:
                    vsd_renderer.add_object(obj_id, m["path"])
                    vsd_objs.add(obj_id)
                try:
                    R_e, t_e, conf = estimate_pose(
                        m["path"], rgb_np, depth_m, mask, K,
                        mesh_units_m=False, refine_iter=refine_iter)
                    err = pose_errors(
                        R_e, t_e, R_gt, t_gt, K, m["pts"], m["syms"],
                        depth_test=(depth_mm if vsd_renderer is not None else None),
                        renderer=vsd_renderer, obj_id=obj_id,
                        diameter=m["diameter"])
                    rec["oracle_pose_conf"] = round(conf, 4)
                except Exception as exc:           # degrade: count as a miss
                    logger.warning("FP oracle failed (%s im %s obj %s): %s",
                                   scene_id, im_id, obj_id, exc)
                    err = _INF
                orec = dict(err, diameter=m["diameter"], img_w=img_w)
                oracle_recs.append(orec)
                rec["oracle_mssd"] = err["mssd"]
                rec["oracle_mspd"] = err["mspd"]
                # retrieved-exact: top-1 is the exact target -> same CAD/pose,
                # so reuse (mesh is identical); non-exact is handled by 3b.
                if ranking and ranking[0][0] == target_nsid:
                    retr_recs.append(orec)
                    rec["retrieved_exact"] = True

            # --- 3b: pose the top-1 PROXY at true metric size -> D_sym ---
            if do_dsym_3b and ranking:
                m = models.get(obj_id)          # target: diameter + surface
                R_gt, t_gt = _gt_pose(gt)
                depth_m, _dmm, mask, K = _pose_inputs(scene_dir, im_id, gt_idx, cam)
                top1 = ranking[0][0]
                rec["top1_proxy"] = top1
                ppath, punits = gallery.id_to_pose_mesh.get(top1, (None, False))
                if ppath and os.path.isfile(ppath):
                    if obj_id not in tgt_samples:
                        tgt_samples[obj_id] = sample_surface_mm(m["path"], units_m=False)
                    try:
                        Rp, tp, conf = estimate_pose(
                            ppath, rgb_np, depth_m, mask, K,
                            mesh_units_m=punits, refine_iter=refine_iter)
                        if top1 not in prx_samples:
                            prx_samples[top1] = sample_surface_mm(ppath, units_m=punits)
                        ds = d_sym(tgt_samples[obj_id], R_gt, t_gt,
                                   prx_samples[top1], Rp, tp, m["diameter"])
                        rec["d_sym"] = round(ds["d_sym"], 3)
                        rec["d_sym_norm"] = round(ds["d_sym_norm"], 4)
                        rec["proxy_pose_conf"] = round(conf, 4)
                        dsym_recs.append(ds)
                    except Exception as exc:      # degrade: drop this instance
                        logger.warning("FP proxy pose failed (%s): %s", top1, exc)
                else:
                    logger.warning("proxy mesh missing for %s", top1)

            records.append(rec)

    summary = _summarize(dataset, mode, G, records, ranks, oracle_recs,
                         retr_recs, dsym_recs, n_missing_rgb,
                         include_target, do_pose_3a, do_dsym_3b)
    return {"summary": summary, "records": records, "ranks": ranks,
            "oracle_recs": oracle_recs, "retr_recs": retr_recs,
            "dsym_recs": dsym_recs}


def _summarize(dataset, mode, G, records, ranks, oracle_recs, retr_recs,
               dsym_recs, n_missing_rgb, include_target, do_pose_3a, do_dsym_3b):
    summary = {"dataset": dataset, "mode": mode, "gallery_size": G,
               "target_in_gallery": include_target,
               "n_queries_evaluated": len(records),
               "n_missing_rgb": n_missing_rgb}
    if include_target:
        summary.update(summarize_retrieval(ranks))
    if do_pose_3a:
        summary["bop_ar_oracle"] = bop_ar(oracle_recs)
        summary["bop_ar_retrieved_exact"] = bop_ar(retr_recs)
        summary["n_exact_top1"] = len(retr_recs)
    if do_dsym_3b:
        summary["dsym"] = summarize_dsym(dsym_recs)
    return summary


def _print_summary(tag, s, include_target, do_pose_3a, do_dsym_3b):
    print(f"\n[stage3] {tag} — {s['n_queries_evaluated']} queries")
    if include_target and "recall@1" in s:
        print(f"  Recall@1={s['recall@1']:.3f}  Recall@5={s['recall@5']:.3f}  "
              f"Recall@10={s['recall@10']:.3f}  MRR={s['mrr']:.3f}  "
              f"(target found {s['n_target_found']}/{s['n_queries_evaluated']})")
    if do_pose_3a and "bop_ar_oracle" in s:
        ao, ar_ = s["bop_ar_oracle"], s["bop_ar_retrieved_exact"]
        vsd_o = f" / VSD {ao['ar_vsd']:.3f}" if ao.get("ar_vsd") is not None else ""
        print(f"  BOP-AR oracle          = {ao['ar']:.3f}  "
              f"(MSSD {ao['ar_mssd']:.3f} / MSPD {ao['ar_mspd']:.3f}{vsd_o}, n={ao['n_estimated']})")
        print(f"  BOP-AR retrieved-exact = {ar_['ar']:.3f}  (n_exact_top1={s['n_exact_top1']})")
        if ao.get("ar_note"):
            print(f"  note: {ao['ar_note']}")
    if do_dsym_3b and "dsym" in s and s["dsym"]["n_estimated"]:
        d = s["dsym"]
        print(f"  D_sym mean = {d['d_sym_mean']:.2f} mm  (median {d['d_sym_median']:.2f}, "
              f"/diam {d['d_sym_norm_mean']:.3f}, n={d['n_estimated']})")


def run_stage3(datasets, mode="3a", max_targets=0,
               output_dir="results_bop_stage3", do_pose=False, refine_iter=5):
    """Run Stage-3 over one or more query datasets against a SINGLE gallery.

    3a: gallery = G_proxy ∪ G_ycbv ∪ G_tless ∪ G_lmo (one big combined DB) —
        every query dataset retrieves against the same index, and a combined
        summary is pooled across all of them.
    3b: gallery = G_proxy only (exact targets removed)."""
    datasets = [d for d in datasets]
    for d in datasets:
        if d not in DATASET_TEST:
            raise ValueError(f"Unknown dataset {d}; choose {list(DATASET_TEST)}")
    include_target = (mode == "3a")
    do_pose_3a = do_pose and mode == "3a"
    do_dsym_3b = do_pose and mode == "3b"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*64}\nStage-3 {mode} — queries {datasets}\n{'='*64}")

    # --- assemble the single gallery ONCE ---
    target_datasets = TARGET_DATASETS if mode == "3a" else ()
    print(f"[stage3] assembling gallery (targets in gallery: "
          f"{list(target_datasets) or 'none (proxy-only)'})...")
    gallery = assemble_gallery(target_datasets=target_datasets)
    components = gallery.components()
    G = len(gallery.gallery_ids)
    print(f"[stage3] |gallery| = {G}  clip_rows = {len(components[1]._desc_labels)}")

    prx_samples = {}          # proxy nsid -> surface points (mm), shared
    per_dataset = {}
    pooled = {"records": 0, "ranks": [], "oracle_recs": [],
              "retr_recs": [], "dsym_recs": []}

    for dataset in datasets:
        res = _eval_dataset(dataset, gallery, components, mode, max_targets,
                            do_pose_3a, do_dsym_3b, refine_iter, prx_samples)
        s = res["summary"]
        per_dataset[dataset] = s
        result_dir = os.path.join(output_dir, f"{dataset}_stage{mode}")
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "records.json"), "w") as f:
            json.dump(res["records"], f, indent=2)
        with open(os.path.join(result_dir, "summary.json"), "w") as f:
            json.dump(s, f, indent=2)
        _print_summary(f"{dataset} {mode}", s, include_target, do_pose_3a, do_dsym_3b)
        pooled["records"] += len(res["records"])
        pooled["ranks"] += res["ranks"]
        pooled["oracle_recs"] += res["oracle_recs"]
        pooled["retr_recs"] += res["retr_recs"]
        pooled["dsym_recs"] += res["dsym_recs"]

    # --- combined summary pooled across all query datasets ---
    combined = _summarize("ALL", mode, G, [None] * pooled["records"],
                          pooled["ranks"], pooled["oracle_recs"],
                          pooled["retr_recs"], pooled["dsym_recs"], 0,
                          include_target, do_pose_3a, do_dsym_3b)
    combined["datasets"] = datasets
    combined["per_dataset"] = {d: per_dataset[d] for d in datasets}
    with open(os.path.join(output_dir, f"combined_stage{mode}.json"), "w") as f:
        json.dump(combined, f, indent=2)
    if len(datasets) > 1:
        _print_summary(f"COMBINED {mode}", combined, include_target,
                       do_pose_3a, do_dsym_3b)
    print(f"\n[stage3] saved -> {output_dir}")
    return combined


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Stage-3 BOP evaluation (OSCAR+)")
    ap.add_argument("--datasets", default="all",
                    help="comma-separated query datasets, or 'all' "
                         f"(= {','.join(TARGET_DATASETS)}). 3a scores them all "
                         "against one combined gallery.")
    ap.add_argument("--mode", choices=["3a", "3b"], default="3a")
    ap.add_argument("--max-targets", type=int, default=0,
                    help="limit targets PER dataset (0 = all; for smoke tests)")
    ap.add_argument("--output", default="results_bop_stage3")
    ap.add_argument("--pose", action="store_true",
                    help="run FoundationPose: in 3a -> BOP-AR (oracle + "
                         "retrieved-exact); in 3b -> proxy pose + D_sym. "
                         "Requires the foundationpose service.")
    ap.add_argument("--refine-iter", type=int, default=5,
                    help="FoundationPose refinement iterations (default 5)")
    args = ap.parse_args()

    datasets = (list(TARGET_DATASETS) if args.datasets == "all"
                else [d.strip() for d in args.datasets.split(",") if d.strip()])

    logging.basicConfig(level=logging.WARNING)
    run_stage3(datasets, mode=args.mode, max_targets=args.max_targets,
               output_dir=args.output, do_pose=args.pose,
               refine_iter=args.refine_iter)


if __name__ == "__main__":
    main()
