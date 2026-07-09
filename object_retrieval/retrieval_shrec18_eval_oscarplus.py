"""
retrieval_shrec18_eval_oscarplus.py
====================================
OSCAR+ category-level retrieval evaluation on SHREC'18 ObjectNN+.

Stage 1 of the thesis evaluation: tunes and selects the best OSCAR+
configuration using the full ablation grid (E1-E7, O1-O5).

SHREC'18 ObjectNN+ supplies 2,101 RGB-D query crops from SceneNN scans
and a gallery of 3,308 ShapeNetSem indoor CAD models across 20 categories.
Ground-truth object crops are used (Step A is bypassed).

Reference:
    Pham et al., "SHREC'18 Track: RGB-D Object-to-CAD Retrieval", 2018.

Dataset structure expected
--------------------------
The SHREC'18 ObjectNN+ dataset should be placed under:

    eval/datasets/shrec18_objectnn/
        queries/                     # 2,101 RGB-D query crops
            <category>/              # e.g. "bathtub", "bed", "chair", ...
                <query_id>.png       # RGB crop
                <query_id>_depth.png # Depth crop (optional for retrieval)
        gallery/                     # ShapeNetSem subset metadata
            gallery.json             # {obj_id: {category: str, mesh_path: str}}
        gt_relevance.json            # {query_id: [relevant_obj_ids]}

Alternatively, the ObjectNN+ data from the SHREC'18 challenge page can be
used with the path variables below adjusted accordingly.

Reference images (rendered views of gallery CAD models) should be in:
    object_images/shrec18_objectnn/
        <obj_id>/
            view_0000.png ... view_NNNN.png

CLIP descriptions should be in:
    object_database/shrec18_objectnn/descriptions_attributes.json

CAD models should be in:
    object_database/shrec18_objectnn/model/<obj_id>.obj  (or .ply, .glb)

How to run
----------
    cd OSCAR/object_retrieval
    python retrieval_shrec18_eval_oscarplus.py

Outputs land in ``result_folder``. Per-query JSON records and metric
summaries follow the same format as the MI3DOR evaluation.
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
except ImportError:
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
# Dataset root for SHREC'18 ObjectNN+ queries
query_root            = "../eval/datasets/shrec18_objectnn/queries"
# Reference images (rendered CAD views)
ref_dir               = "../object_images/shrec18_objectnn"
# CLIP text descriptions
desc_file             = "../object_database/shrec18_objectnn/descriptions_attributes.json"
# CAD meshes for ULIP shape matching
cad_mesh_glob         = "../object_database/shrec18_objectnn/model/*.obj"
# Output folder
result_folder         = "results_shrec18_oscarplus"
# ULIP query embedding cache
ulip_query_cache_path = "ulip_query_cache_shrec18.pt"
# Ground-truth relevance mapping (SHREC'18 format)
gt_relevance_path     = "../eval/datasets/shrec18_objectnn/gt_relevance.json"


cfg = EvalConfig(
    ref_dir=ref_dir,
    desc_file=desc_file,
    cad_mesh_glob=cad_mesh_glob,
    result_folder=result_folder,
    topk=[15],
    clip_top_k=9999,       # Full-database scoring (thesis default for Stage 1)
    dino_top_k=9999,
    ulip2_top_k=9999,
    fusion_top_k=9999,
    TOP_F=20,
    fusion_method="weighted_sum",
    weight_clip=0.3,
    weight_dino=0.4,
    weight_ulip=0.3,
    ulip_query_cache_path=ulip_query_cache_path,
)


# ============================================================================
# SHREC'18 helpers
# ============================================================================

# Category label extraction depends on the naming convention.
# ObjectNN+ typically uses: <category>_<number> or just the directory name.

_gt_relevance = None  # lazy-loaded


def _load_gt_relevance():
    """Load ground-truth relevance mapping if available."""
    global _gt_relevance
    if _gt_relevance is not None:
        return _gt_relevance
    if os.path.isfile(gt_relevance_path):
        with open(gt_relevance_path) as f:
            _gt_relevance = json.load(f)
        print(f"[shrec18] Loaded GT relevance: {len(_gt_relevance)} queries.")
    else:
        _gt_relevance = {}
        print(f"[shrec18] WARNING: GT relevance file not found: {gt_relevance_path}")
    return _gt_relevance


def to_category_label(object_id: str) -> str:
    """Map an object_id to its category label.

    Handles common ObjectNN+ naming conventions:
        - ``bathtub_0001`` -> ``bathtub``
        - ``ShapeNetSem/bathtub/model_0001`` -> ``bathtub``
        - Plain category name -> as-is
    """
    # Strip path components if present
    name = os.path.basename(object_id)
    name = os.path.splitext(name)[0]

    # Try splitting at last underscore followed by digits
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]

    # Try parent directory as category
    parent = os.path.basename(os.path.dirname(object_id))
    if parent and parent != object_id:
        return parent

    return name


def _get_categories():
    """Return categories present in the query directory."""
    if not os.path.isdir(query_root):
        print(f"[shrec18] ERROR: query root not found: {query_root}")
        return []

    cats = sorted(
        c for c in os.listdir(query_root)
        if os.path.isdir(os.path.join(query_root, c))
    )

    # Filter to categories that have CLIP descriptions
    if desc_file and os.path.isfile(desc_file):
        with open(desc_file) as f:
            desc_keys = set(json.load(f).keys())
        desc_cats = set()
        for k in desc_keys:
            desc_cats.add(to_category_label(k))
        filtered = [c for c in cats if c in desc_cats]
        if len(filtered) < len(cats):
            skipped = set(cats) - set(filtered)
            print(f"[shrec18] {len(cats)} categories on disk, "
                  f"{len(filtered)} have CLIP descriptions. "
                  f"Skipping: {sorted(skipped)}")
            cats = filtered

    print(f"[shrec18] Categories: {len(cats)}")
    return cats


def query_factory(k):
    """Iterate over all SHREC'18 ObjectNN+ queries.

    Yields: (roi_image, gt_label, img_path, category, filename)

    The ``k`` parameter is passed by eval_common but not used for query
    iteration in SHREC'18 (all queries are always evaluated).
    """
    categories = _get_categories()
    if not categories:
        return

    for cat in categories:
        cat_dir = os.path.join(query_root, cat)
        # Find all RGB query images (exclude depth maps)
        img_files = sorted(
            f for f in os.listdir(cat_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
            and "_depth" not in f.lower()
        )

        for img_file in img_files:
            img_path = os.path.join(cat_dir, img_file)
            try:
                roi = Image.open(img_path).convert("RGB")
            except Exception as exc:
                print(f"[shrec18] WARNING: cannot load {img_path}: {exc}")
                continue

            gt_label = cat
            fname = os.path.splitext(img_file)[0]

            yield roi, gt_label, img_path, cat, fname


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("OSCAR+ Evaluation — SHREC'18 ObjectNN+ (Stage 1)")
    print("=" * 60)

    # Verify dataset exists
    if not os.path.isdir(query_root):
        print(f"\nERROR: SHREC'18 query directory not found: {query_root}")
        print("Please download the ObjectNN+ dataset and place it at:")
        print(f"  {os.path.abspath(query_root)}")
        print("\nExpected structure:")
        print("  queries/<category>/<query_id>.png")
        print("  queries/<category>/<query_id>_depth.png  (optional)")
        return

    categories = _get_categories()
    if not categories:
        print("ERROR: no valid categories found.")
        return

    # Count queries
    total_queries = 0
    all_img_paths = []
    for cat in categories:
        cat_dir = os.path.join(query_root, cat)
        imgs = [
            os.path.join(cat_dir, f) for f in os.listdir(cat_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
            and "_depth" not in f.lower()
        ]
        total_queries += len(imgs)
        all_img_paths.extend(imgs)

    print(f"\nDataset: {len(categories)} categories, {total_queries} queries")
    print(f"Gallery: {cad_mesh_glob}")
    print(f"References: {ref_dir}")
    print(f"Descriptions: {desc_file}")
    print(f"Output: {result_folder}\n")

    # Build pipeline
    components = build_pipeline(cfg)
    pipeline_cfg, clip_retr, dino_rer, fusion_mod, shape_m = components

    # ULIP query cache
    ulip_cache = load_ulip_query_cache(cfg.ulip_query_cache_path)
    if ulip_cache is None and shape_m is not None and all_img_paths:
        print("[shrec18] Pre-encoding ULIP query embeddings...")
        ulip_cache = pre_encode_ulip_queries(all_img_paths, shape_m)
        if cfg.ulip_query_cache_path:
            import torch
            torch.save(ulip_cache, cfg.ulip_query_cache_path)
            print(f"[shrec18] ULIP cache saved: {cfg.ulip_query_cache_path}")

    # Run evaluation
    summaries = run_evaluation(
        cfg, to_category_label, query_factory, components,
        ulip_cache=ulip_cache,
    )

    # Additional SHREC'18-specific metrics
    _load_gt_relevance()

    print("\n" + "=" * 60)
    print("SHREC'18 ObjectNN+ evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
