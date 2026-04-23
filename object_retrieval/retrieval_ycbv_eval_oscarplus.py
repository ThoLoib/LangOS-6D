"""
retrieval_ycbv_eval_oscarplus.py
================================
OSCAR+ instance-level retrieval evaluation on YCBV-GSO (BOP format).

Iterates scenes → images → object instances, crops by visible bbox from
``scene_gt_info.json``, and runs the full OSCAR+ pipeline (CLIP + DINOv2 +
ULIP-2 + fusion).

How to run
----------
    cd OSCAR/object_retrieval
    python retrieval_ycbv_eval_oscarplus.py

Note: ULIP cross-mode runs on the fly (no pre-computed cache by default).
On a 6 GB GPU this may be slow (~10 s/query). To pre-compute ULIP embeddings,
save crops to disk first, then adapt ``precompute_ulip_query_embeddings.py``.
"""

import glob as _glob
import json
import os

from PIL import Image
from tqdm import tqdm

from eval_common import (
    EvalConfig,
    build_pipeline,
    crop_by_bbox,
    run_evaluation,
)

# ============================================================================
# CONFIG — edit these to match your environment
# ============================================================================
ref_dir       = "../object_images/ycbv_gso"
bop_root      = "../eval/datasets/ycbv_gso/test"
desc_file     = "../object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json"
cad_mesh_glob = "../object_database/ycbv_gso/*/meshes/model.obj"
result_folder = "results_ycbv_oscarplus"

cfg = EvalConfig(
    ref_dir=ref_dir,
    desc_file=desc_file,
    cad_mesh_glob=cad_mesh_glob,
    result_folder=result_folder,
)


# ============================================================================
# YCBV helpers
# ============================================================================

def _load_id_to_label():
    path = os.path.join(bop_root, "id_to_label.json")
    with open(path) as f:
        return json.load(f)


def _build_cad_mesh_items():
    """YCBV CAD layout: ``<name>/meshes/model.obj`` — obj_id is the
    grandparent directory name (e.g. ``003_cracker_box``)."""
    mesh_paths = sorted(_glob.glob(cad_mesh_glob))
    return [
        (os.path.basename(os.path.dirname(os.path.dirname(p))), p)
        for p in mesh_paths
    ]


def _make_query_factory():
    id_to_label = _load_id_to_label()
    scenes = sorted(
        s for s in os.listdir(bop_root)
        if os.path.isdir(os.path.join(bop_root, s))
    )

    def factory(k):
        for scene_id in tqdm(scenes, desc=f"Top-k={k} Scenes", unit="scene"):
            scene_dir = os.path.join(bop_root, scene_id)
            gt_path      = os.path.join(scene_dir, "scene_gt.json")
            gt_info_path = os.path.join(scene_dir, "scene_gt_info.json")
            rgb_dir      = os.path.join(scene_dir, "rgb")

            if not all(os.path.exists(p)
                       for p in (gt_path, gt_info_path, rgb_dir)):
                continue

            with open(gt_path) as f:
                scene_gt = json.load(f)
            with open(gt_info_path) as f:
                scene_gt_info = json.load(f)

            for img_id_str in tqdm(sorted(scene_gt.keys(), key=int),
                                   desc=scene_id, unit="img", leave=False):
                img_id = int(img_id_str)
                rgb_path = os.path.join(rgb_dir, f"{img_id:06d}.png")
                if not os.path.exists(rgb_path):
                    continue

                image = Image.open(rgb_path).convert("RGB")

                for inst_id, (gt_inst, info) in enumerate(
                    zip(scene_gt[img_id_str],
                        scene_gt_info[img_id_str])
                ):
                    label = id_to_label.get(str(gt_inst["obj_id"]))
                    if label is None:
                        continue

                    bbox = info.get("bbox_visib") or info.get("bbox_obj")
                    if not bbox:
                        continue

                    roi = crop_by_bbox(image, bbox)
                    if roi is None or min(roi.size) < 2:
                        continue

                    path_key = f"{scene_id}/{img_id:06d}_{inst_id:03d}"
                    yield (roi, label, path_key,
                           scene_id, f"{img_id:06d}_{inst_id:03d}")

    return factory


# ============================================================================
# Main
# ============================================================================

def main():
    cad_items = _build_cad_mesh_items()
    if not cad_items:
        print(f"[ycbv] WARNING: no CAD meshes found via {cad_mesh_glob}")

    components = build_pipeline(cfg, cad_mesh_items=cad_items or None)
    run_evaluation(cfg, lambda x: x, _make_query_factory(), components)


if __name__ == "__main__":
    main()
