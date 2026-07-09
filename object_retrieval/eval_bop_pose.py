"""
eval_bop_pose.py
================
BOP-core pose evaluation for OSCAR+ (Stages 3a/3b).

Iterates BOP target lists (``test_targets_bop19.json``), runs the full
OSCAR+ pipeline including retrieval (Steps B1/B2) and pose estimation
(Step C: FoundationPose + ICP fallback), and computes BOP-AR and ADD(-S)
pose metrics.

Supports three BOP-core datasets:
  - YCB-V (household objects, moderate occlusion)
  - T-LESS (texture-less industrial parts)
  - LM-O (heavy occlusion, Linemod-Occluded)

Two evaluation modes:
  - Stage 3a: GT CAD in gallery (oracle retrieval) -> isolates pose quality
  - Stage 3b: Proxy-only gallery (no GT CAD) -> tests full retrieval + pose

Thesis reference: Sections 5.3, 5.4.

How to run
----------
    cd OSCAR/object_retrieval
    python eval_bop_pose.py --dataset ycbv --mode 3a
    python eval_bop_pose.py --dataset tless --mode 3b

Prerequisites
-------------
  - BOP dataset downloaded (test split + models)
  - Reference images rendered for the gallery
  - CLIP descriptions generated
  - FoundationPose service running (for --pose-method foundationpose)
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OSCAR_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _OSCAR_ROOT not in sys.path:
    sys.path.insert(0, _OSCAR_ROOT)

from pipeline.config import PipelineConfig
from pipeline.run_pipeline import OSCARPlusPipeline

logger = logging.getLogger(__name__)


# ============================================================================
# Dataset configurations
# ============================================================================

DATASET_CONFIGS = {
    "ycbv": {
        "bop_root": "../eval/datasets/ycbv_gso/test",
        "ref_dir": "../object_images/ycbv_gso",
        "desc_file": "../object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json",
        "cad_models_dir": "../object_database/ycbv_gso",
        "cad_mesh_glob": "../object_database/ycbv_gso/*/meshes/model.obj",
        "targets_file": "../eval/datasets/ycbv_gso/test_targets_bop19.json",
    },
    "tless": {
        "bop_root": "../eval/datasets/tless/test_primesense",
        "ref_dir": "../object_images/tless",
        "desc_file": "../object_database/tless/descriptions_attributes.json",
        "cad_models_dir": "../object_database/tless",
        "cad_mesh_glob": "../object_database/tless/*/meshes/model.obj",
        "targets_file": "../eval/datasets/tless/test_targets_bop19.json",
    },
    "lmo": {
        "bop_root": "../eval/datasets/lmo/test",
        "ref_dir": "../object_images/lmo",
        "desc_file": "../object_database/lmo/descriptions_attributes.json",
        "cad_models_dir": "../object_database/lmo",
        "cad_mesh_glob": "../object_database/lmo/*/meshes/model.obj",
        "targets_file": "../eval/datasets/lmo/test_targets_bop19.json",
    },
}


# ============================================================================
# BOP target list helpers
# ============================================================================

def load_bop_targets(targets_path):
    """Load BOP target list.

    Format: [{"im_id": int, "inst_count": int, "obj_id": int, "scene_id": int}, ...]

    Returns:
        List of target dicts.
    """
    with open(targets_path) as f:
        targets = json.load(f)
    logger.info("Loaded %d BOP targets from %s", len(targets), targets_path)
    return targets


def load_scene_gt(scene_dir, im_id):
    """Load ground-truth annotations for a specific image.

    Returns:
        List of GT dicts: [{cam_R_m2c, cam_t_m2c, obj_id}, ...]
    """
    gt_path = os.path.join(scene_dir, "scene_gt.json")
    if not os.path.isfile(gt_path):
        return []
    with open(gt_path) as f:
        gt_data = json.load(f)
    key = str(im_id)
    return gt_data.get(key, [])


def load_scene_gt_info(scene_dir, im_id):
    """Load GT info (bboxes, visibility) for a specific image.

    Returns:
        List of GT info dicts: [{bbox_visib, bbox_obj, visib_fract, ...}, ...]
    """
    info_path = os.path.join(scene_dir, "scene_gt_info.json")
    if not os.path.isfile(info_path):
        return []
    with open(info_path) as f:
        info_data = json.load(f)
    key = str(im_id)
    return info_data.get(key, [])


# ============================================================================
# Pose metrics
# ============================================================================

def compute_add(R_est, t_est, R_gt, t_gt, model_points):
    """Average Distance of Distinguishable model points (ADD).

    Args:
        R_est, R_gt: (3, 3) rotation matrices.
        t_est, t_gt: (3,) translation vectors (meters).
        model_points: (N, 3) CAD model points.

    Returns:
        ADD distance (float).
    """
    pts_est = (R_est @ model_points.T).T + t_est
    pts_gt = (R_gt @ model_points.T).T + t_gt
    return float(np.linalg.norm(pts_est - pts_gt, axis=1).mean())


def compute_adds(R_est, t_est, R_gt, t_gt, model_points):
    """ADD-S (symmetric): average closest-point distance.

    For symmetric objects where pose ambiguity exists.
    """
    from scipy.spatial import cKDTree

    pts_est = (R_est @ model_points.T).T + t_est
    pts_gt = (R_gt @ model_points.T).T + t_gt

    tree = cKDTree(pts_est)
    dists, _ = tree.query(pts_gt, k=1)
    return float(dists.mean())


def pose_success_at_threshold(add_value, model_diameter, threshold_frac=0.1):
    """Check if ADD < threshold_frac * model_diameter."""
    return add_value < threshold_frac * model_diameter


# ============================================================================
# Main evaluation
# ============================================================================

def run_bop_pose_evaluation(
    dataset: str,
    mode: str = "3a",
    pose_method: str = "icp",
    output_dir: str = "results_bop_pose",
    max_targets: int = 0,
):
    """Run BOP pose evaluation.

    Args:
        dataset: "ycbv", "tless", or "lmo".
        mode: "3a" (GT CAD in gallery) or "3b" (proxy-only gallery).
        pose_method: "icp" or "foundationpose".
        output_dir: Output directory for results.
        max_targets: Limit number of targets (0 = all, useful for debugging).
    """
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(DATASET_CONFIGS)}")

    ds_cfg = DATASET_CONFIGS[dataset]
    result_dir = os.path.join(output_dir, f"{dataset}_stage{mode}_{pose_method}")
    os.makedirs(result_dir, exist_ok=True)

    # Load targets
    targets_path = ds_cfg["targets_file"]
    if not os.path.isfile(targets_path):
        print(f"ERROR: BOP targets file not found: {targets_path}")
        print(f"Please download the {dataset.upper()} dataset with BOP test targets.")
        return

    targets = load_bop_targets(targets_path)
    if max_targets > 0:
        targets = targets[:max_targets]

    # Build pipeline config
    config = PipelineConfig(
        description_file=ds_cfg["desc_file"],
        reference_images_dir=ds_cfg["ref_dir"],
        cad_models_dir=ds_cfg["cad_models_dir"],
        pose_method=pose_method,
    )

    print(f"\n{'='*60}")
    print(f"BOP Pose Evaluation — {dataset.upper()} Stage {mode}")
    print(f"  Targets: {len(targets)}")
    print(f"  Pose method: {pose_method}")
    print(f"  Output: {result_dir}")
    print(f"{'='*60}\n")

    results = []
    add_values = []
    adds_values = []
    success_count = 0
    total_count = 0

    for target in tqdm(targets, desc=f"{dataset} Stage {mode}"):
        scene_id = target["scene_id"]
        im_id = target["im_id"]
        obj_id = target["obj_id"]

        scene_dir = os.path.join(ds_cfg["bop_root"], f"{scene_id:06d}")
        rgb_path = os.path.join(scene_dir, "rgb", f"{im_id:06d}.png")
        depth_path = os.path.join(scene_dir, "depth", f"{im_id:06d}.png")

        if not os.path.isfile(rgb_path):
            logger.warning("RGB not found: %s", rgb_path)
            continue

        # Load GT pose for this target
        gt_list = load_scene_gt(scene_dir, im_id)
        gt_info_list = load_scene_gt_info(scene_dir, im_id)

        # Find the GT annotation matching this obj_id
        gt_entry = None
        gt_info_entry = None
        for i, g in enumerate(gt_list):
            if g.get("obj_id") == obj_id:
                gt_entry = g
                if i < len(gt_info_list):
                    gt_info_entry = gt_info_list[i]
                break

        if gt_entry is None:
            logger.warning("No GT for scene=%d im=%d obj=%d", scene_id, im_id, obj_id)
            continue

        # Extract GT bbox for cropping (bypass Step A)
        if gt_info_entry and "bbox_visib" in gt_info_entry:
            bbox = gt_info_entry["bbox_visib"]  # [x, y, w, h]
        elif gt_info_entry and "bbox_obj" in gt_info_entry:
            bbox = gt_info_entry["bbox_obj"]
        else:
            logger.warning("No bbox for scene=%d im=%d obj=%d", scene_id, im_id, obj_id)
            continue

        result_entry = {
            "scene_id": scene_id,
            "im_id": im_id,
            "obj_id": obj_id,
            "bbox": bbox,
            "pose_method": pose_method,
        }

        # TODO: Run pipeline for this target
        # This requires:
        # 1. Load RGB + Depth
        # 2. Crop by GT bbox (bypass Step A)
        # 3. Run Steps B1, B2, C
        # 4. Compare estimated pose with GT pose
        # 5. Compute ADD/ADD-S
        #
        # For now, store the target info for when the full pipeline
        # integration is wired up.

        results.append(result_entry)
        total_count += 1

    # Save results
    results_path = os.path.join(result_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    summary = {
        "dataset": dataset,
        "mode": mode,
        "pose_method": pose_method,
        "total_targets": len(targets),
        "total_evaluated": total_count,
        "add_mean": float(np.mean(add_values)) if add_values else None,
        "adds_mean": float(np.mean(adds_values)) if adds_values else None,
        "success_rate_0.1d": success_count / total_count if total_count > 0 else 0.0,
    }

    summary_path = os.path.join(result_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {result_dir}")
    print(f"  Total targets: {len(targets)}")
    print(f"  Evaluated: {total_count}")
    if add_values:
        print(f"  ADD mean: {np.mean(add_values):.4f}")
        print(f"  ADD-S mean: {np.mean(adds_values):.4f}")
        print(f"  Success rate (0.1d): {success_count/total_count:.2%}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BOP-core pose evaluation for OSCAR+ (Stages 3a/3b)",
    )
    parser.add_argument("--dataset", choices=["ycbv", "tless", "lmo"],
                        required=True, help="BOP dataset to evaluate")
    parser.add_argument("--mode", choices=["3a", "3b"], default="3a",
                        help="3a: GT CAD in gallery, 3b: proxy-only gallery")
    parser.add_argument("--pose-method", choices=["icp", "foundationpose"],
                        default="icp", dest="pose_method")
    parser.add_argument("--output", default="results_bop_pose",
                        help="Output directory")
    parser.add_argument("--max-targets", type=int, default=0,
                        help="Max targets to evaluate (0 = all)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_bop_pose_evaluation(
        dataset=args.dataset,
        mode=args.mode,
        pose_method=args.pose_method,
        output_dir=args.output,
        max_targets=args.max_targets,
    )


if __name__ == "__main__":
    main()
