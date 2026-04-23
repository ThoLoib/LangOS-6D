"""
retrieval_housecat6d_eval_oscarplus.py
======================================
OSCAR+ instance-level retrieval evaluation on HouseCat6D.

Iterates scenes → images → object instances, crops by instance mask from
``mask_visib/``, and runs the full OSCAR+ pipeline (CLIP + DINOv2 + ULIP-2 +
fusion).

How to run
----------
    cd OSCAR/object_retrieval
    python retrieval_housecat6d_eval_oscarplus.py

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
    crop_with_mask,
    run_evaluation,
)

# ============================================================================
# CONFIG — edit these to match your environment
# ============================================================================
ref_dir       = "../object_images/housecat6d"
bop_root      = "../eval/datasets/housecat6d/test"
desc_file     = "../object_database/housecat6d/descriptions_attributes.json"
# HouseCat6D CAD layout: <category>/<instance>.obj  (exclude bg/ scenes)
cad_mesh_glob = "../object_database/housecat6d/*/*.obj"
result_folder = "results_housecat6d_oscarplus"

# Categories that hold actual object meshes (not background scenes)
_EXCLUDE_CAD_DIRS = {"bg", "collision"}

cfg = EvalConfig(
    ref_dir=ref_dir,
    desc_file=desc_file,
    cad_mesh_glob=cad_mesh_glob,
    result_folder=result_folder,
)


# ============================================================================
# HouseCat6D helpers
# ============================================================================

def _load_id_to_label():
    path = os.path.join(bop_root, "id_to_label.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _build_cad_mesh_items():
    """HouseCat6D CAD layout: ``<category>/<instance>.obj`` — obj_id is the
    file stem (e.g. ``bottle-85_alcool``). Excludes bg/ and collision/."""
    mesh_paths = sorted(_glob.glob(cad_mesh_glob))
    items = []
    for p in mesh_paths:
        parent = os.path.basename(os.path.dirname(p))
        if parent in _EXCLUDE_CAD_DIRS:
            continue
        obj_id = os.path.splitext(os.path.basename(p))[0]
        items.append((obj_id, p))
    return items


def _make_query_factory():
    id_to_label = _load_id_to_label()
    scenes = sorted(
        s for s in os.listdir(bop_root)
        if os.path.isdir(os.path.join(bop_root, s))
    )

    def factory(k):
        for scene in tqdm(scenes, desc=f"Top-k={k} Scenes", unit="scene"):
            scene_dir = os.path.join(bop_root, scene)
            gt_path  = os.path.join(scene_dir, "scene_gt.json")
            mask_dir = os.path.join(scene_dir, "mask_visib")
            rgb_dir  = os.path.join(scene_dir, "rgb")

            if not all(os.path.exists(p)
                       for p in (gt_path, mask_dir, rgb_dir)):
                continue

            with open(gt_path) as f:
                gt_data = json.load(f)

            for img_id_str in tqdm(sorted(gt_data.keys(), key=int),
                                   desc=scene, unit="img", leave=False):
                img_id = int(img_id_str)
                rgb_path = os.path.join(rgb_dir, f"{img_id:06d}.png")
                if not os.path.exists(rgb_path):
                    continue

                image = Image.open(rgb_path).convert("RGB")

                for inst_id, obj in enumerate(gt_data[img_id_str]):
                    label = id_to_label.get(str(obj["obj_id"]))
                    if label is None:
                        continue

                    mask_path = os.path.join(
                        mask_dir, f"{img_id:06d}_{inst_id:06d}.png")
                    if not os.path.exists(mask_path):
                        continue

                    mask = Image.open(mask_path).convert("L")
                    roi = crop_with_mask(image, mask)
                    if roi is None:
                        continue

                    path_key = f"{scene}/{img_id:06d}_{inst_id:03d}"
                    yield (roi, label, path_key,
                           scene, f"{img_id:06d}_{inst_id:03d}")

    return factory


# ============================================================================
# Main
# ============================================================================

def main():
    cad_items = _build_cad_mesh_items()
    if not cad_items:
        print(f"[housecat6d] WARNING: no CAD meshes found via "
              f"{cad_mesh_glob} (excluding {_EXCLUDE_CAD_DIRS})")

    components = build_pipeline(cfg, cad_mesh_items=cad_items or None)
    run_evaluation(cfg, lambda x: x, _make_query_factory(), components)


if __name__ == "__main__":
    main()
