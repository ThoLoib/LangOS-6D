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

This file currently implements **Phase A**: retrieval Recall@1/5/10 + MRR for
3a (the only relevant item is the exact target CAD). Pose (BOP-AR) and 3b D_sym
are Phase B / C — see the clearly marked hooks below.

How to run (inside the oscar container, from object_retrieval/):
    python3 eval_bop_pose.py --dataset ycbv --mode 3a --max-targets 20
"""

import argparse
import json
import logging
import os
import sys

from PIL import Image
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OSCAR_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _OSCAR_ROOT not in sys.path:
    sys.path.insert(0, _OSCAR_ROOT)

from eval_common import run_query, fusion_ranking, crop_by_bbox
from stage3_gallery import assemble_gallery, TARGET_DATASETS
from stage3_metrics import rank_of_target, summarize_retrieval

logger = logging.getLogger(__name__)


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
    """All (gt, gt_info) instances of obj_id in an image (handles inst_count>1)."""
    gts = _load_scene_json(scene_dir, "scene_gt.json", im_id)
    infos = _load_scene_json(scene_dir, "scene_gt_info.json", im_id)
    out = []
    for i, g in enumerate(gts):
        if g.get("obj_id") == obj_id:
            info = infos[i] if i < len(infos) else {}
            out.append((g, info))
    return out


def _bbox_of(info):
    b = info.get("bbox_visib") or info.get("bbox_obj")
    if not b or b[2] <= 0 or b[3] <= 0:   # w,h must be positive
        return None
    return b


# ============================================================================
# Stage-3 retrieval (Phase A)
# ============================================================================

def run_stage3(dataset, mode="3a", max_targets=0,
               output_dir="results_bop_stage3"):
    if dataset not in DATASET_TEST:
        raise ValueError(f"Unknown dataset {dataset}; choose {list(DATASET_TEST)}")
    include_target = (mode == "3a")

    ds_test = DATASET_TEST[dataset]
    test_root = os.path.join(_THIS_DIR, ds_test["test_root"])
    targets_path = os.path.join(_THIS_DIR, ds_test["targets"])
    result_dir = os.path.join(output_dir, f"{dataset}_stage{mode}")
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n{'='*64}\nStage-3 {mode} — {dataset.upper()}  (retrieval phase)\n{'='*64}")

    # --- assemble the union gallery ---
    print(f"[stage3] assembling gallery (include_target={include_target})...")
    gallery = assemble_gallery(dataset, include_target=include_target)
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = gallery.components()
    cfg = gallery.eval_cfg
    G = len(gallery.gallery_ids)
    top_k = G + 5    # DINO/ULIP are per-object: this ranks the whole gallery.
    # CLIP.retrieve caps DESCRIPTION ROWS (42/object), not objects — so to let
    # the dedup reach all G objects, CLIP must be given the total row count.
    clip_rows = len(clip_retr._desc_labels)
    print(f"[stage3] |gallery| = {G}  clip_rows = {clip_rows}  "
          f"(target_in_gallery={include_target})")

    targets = load_bop_targets(targets_path)
    if max_targets > 0:
        targets = targets[:max_targets]
    print(f"[stage3] {len(targets)} BOP targets")

    ranks = []
    records = []
    n_missing_rgb = 0

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

        target_nsid = f"{dataset}/obj_{obj_id:06d}"

        for inst_idx, (gt, info) in enumerate(_matching_instances(
                scene_dir, im_id, obj_id)):
            bbox = _bbox_of(info)
            if bbox is None:
                continue
            roi = crop_by_bbox(rgb, bbox)

            out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m,
                            roi, cfg,
                            dino_full_top_k=top_k, ulip_full_top_k=top_k,
                            clip_full_top_k=clip_rows)
            ranking = fusion_ranking(out["fused_full"])   # [(nsid, score), ...]

            r = rank_of_target(ranking, target_nsid) if include_target else None
            if include_target:
                ranks.append(r)

            records.append({
                "scene_id": scene_id, "im_id": im_id, "obj_id": obj_id,
                "inst_idx": inst_idx, "target_id": target_nsid,
                "target_rank": r,
                "top5": [{"id": oid, "score": round(s, 5)}
                         for oid, s in ranking[:5]],
                # Phase B/C will add: gt_pose, oracle_pose, retrieved_pose,
                # bop_ar, d_sym.
            })

    # --- summary ---
    summary = {"dataset": dataset, "mode": mode,
               "gallery_size": G, "target_in_gallery": include_target,
               "n_queries_evaluated": len(records),
               "n_missing_rgb": n_missing_rgb}
    if include_target:
        summary.update(summarize_retrieval(ranks))

    with open(os.path.join(result_dir, "records.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(result_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[stage3] {dataset} {mode} — {len(records)} queries")
    if include_target:
        print(f"  Recall@1={summary['recall@1']:.3f}  "
              f"Recall@5={summary['recall@5']:.3f}  "
              f"Recall@10={summary['recall@10']:.3f}  "
              f"MRR={summary['mrr']:.3f}  "
              f"(target found {summary['n_target_found']}/{len(records)})")
    print(f"  saved -> {result_dir}")
    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Stage-3 BOP evaluation (OSCAR+)")
    ap.add_argument("--dataset", choices=list(DATASET_TEST), required=True)
    ap.add_argument("--mode", choices=["3a", "3b"], default="3a")
    ap.add_argument("--max-targets", type=int, default=0,
                    help="limit targets (0 = all; useful for smoke tests)")
    ap.add_argument("--output", default="results_bop_stage3")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    run_stage3(args.dataset, mode=args.mode, max_targets=args.max_targets,
               output_dir=args.output)


if __name__ == "__main__":
    main()
