"""
retrieval_mi3dor_eval_oscarplus.py
=================================
OSCAR+ category-level retrieval evaluation on MI3DOR.

Mirrors the baseline ``retrieval_mi3dor_eval.py`` but drives CLIP + DINOv2 +
ULIP-2 + Score Fusion. All shared logic lives in ``eval_common.py``; this file
only contains MI3DOR-specific iteration, label mapping, and CONFIG.

How to run
----------
    cd OSCAR/object_retrieval
    python retrieval_mi3dor_eval_oscarplus.py

Outputs land in ``result_folder`` (see CONFIG). Per-query JSON records match
the baseline format (category, filename, gt, pred, clip_candidates,
matched_files) with ULIP and fusion fields added.
"""

import json
import os
import glob

from PIL import Image
from tqdm import tqdm

try:
    from eval_common import (
        EvalConfig,
        build_pipeline,
        load_ulip_query_cache,
        pre_encode_ulip_queries,
        run_evaluation,
    )
except ImportError:  # pragma: no cover - fallback for module execution
    from .eval_common import (
        EvalConfig,
        build_pipeline,
        load_ulip_query_cache,
        pre_encode_ulip_queries,
        run_evaluation,
    )

# ============================================================================
# CONFIG — edit these to match your environment
# ============================================================================
ref_dir               = "../object_images/MI3DOR"
bop_root              = "../eval/datasets/mi3dor/image/test"
desc_file             = "../object_database/MI3DOR/descriptions_attributes.json"
cad_mesh_glob         = "../object_database/MI3DOR/model/test/*/*.obj"
result_folder         = "results_mi3dor_oscarplus_f20_2"
ulip_query_cache_path = "ulip_query_cache_mi3dor.pt"


cfg = EvalConfig(
    ref_dir=ref_dir,
    desc_file=desc_file,
    cad_mesh_glob=cad_mesh_glob,
    result_folder=result_folder,
    topk=[15],
    clip_top_k=20,
    dino_top_k=9999,
    ulip2_top_k=9999,
    fusion_top_k=1,
    TOP_F=20,
    ulip_query_cache_path=ulip_query_cache_path,
)


# ============================================================================
# MI3DOR helpers
# ============================================================================

def to_category_label(object_id: str) -> str:
    """``airplane_test_0001`` -> ``airplane``."""
    if "_test" in object_id:
        return object_id.split("_test")[0]
    return object_id


def _get_categories():
    """Return categories that have CLIP descriptions."""
    all_cats = sorted(
        c for c in os.listdir(bop_root)
        if os.path.isdir(os.path.join(bop_root, c))
    )
    desc_cats = set()
    if desc_file and os.path.isfile(desc_file):
        with open(desc_file) as f:
            for k in json.load(f).keys():
                desc_cats.add(k.split("_test")[0] if "_test" in k else k)
    if desc_cats:
        cats = [c for c in all_cats if c in desc_cats]
        skipped = set(all_cats) - set(cats)
        if skipped:
            print(f"[mi3dor] {len(all_cats)} categories on disk, "
                  f"{len(cats)} have CLIP descriptions. "
                  f"Skipping: {sorted(skipped)}")
        return cats
    return all_cats


def _collect_filtered_cad_mesh_items(allowed_categories):
    """Collect CAD meshes only for categories that have descriptions."""
    allowed = set(allowed_categories)
    mesh_items = []
    for mesh_path in sorted(glob.glob(cad_mesh_glob)):
        obj_id = os.path.splitext(os.path.basename(mesh_path))[0]
        category = obj_id.split("_test")[0] if "_test" in obj_id else obj_id
        if category in allowed:
            mesh_items.append((obj_id, mesh_path))
    print(f"[mi3dor] Using {len(mesh_items)} CAD meshes after category filter")
    return mesh_items


def _collect_query_paths(categories):
    paths = []
    for cat in categories:
        cat_dir = os.path.join(bop_root, cat)
        for fname in sorted(os.listdir(cat_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(cat_dir, fname))
    return paths


def _make_query_factory(categories):
    def factory(k):
        for category in tqdm(categories,
                             desc=f"Top-k={k} Categories", unit="cat"):
            cat_dir = os.path.join(bop_root, category)
            query_files = sorted(
                f for f in os.listdir(cat_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            for fname in tqdm(query_files, desc=category,
                              unit="img", leave=False):
                img_path = os.path.join(cat_dir, fname)
                roi = Image.open(img_path).convert("RGB")
                yield roi, category, img_path, category, fname
    return factory


# ============================================================================
# Main
# ============================================================================

def main():
    categories = _get_categories()
    if not categories:
        raise RuntimeError(f"No categories found under {bop_root}")
    print(f"[mi3dor] Evaluating {len(categories)} categories")
    print(
        "[mi3dor] Both full-set and CLIP-pruned DINO/ULIP variants are "
        "computed in one run (six variants reported)."
    )

    cad_mesh_items = _collect_filtered_cad_mesh_items(categories)
    components = build_pipeline(cfg, cad_mesh_items=cad_mesh_items)
    _, _, _, _, shape_m = components

    ulip_cache = None
    if shape_m is not None:
        ulip_cache = load_ulip_query_cache(cfg.ulip_query_cache_path)
        if ulip_cache is None:
            print("[mi3dor] No pre-computed ULIP cache — encoding on the fly. "
                  "Run precompute_ulip_query_embeddings.py for faster eval.")
            all_paths = _collect_query_paths(categories)
            ulip_cache = pre_encode_ulip_queries(all_paths, shape_m)

    run_evaluation(cfg, to_category_label,
                   _make_query_factory(categories),
                   components, ulip_cache)


if __name__ == "__main__":
    main()
