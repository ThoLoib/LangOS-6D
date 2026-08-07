#!/usr/bin/env python3
"""Minimal single-run MI3DOR (Stage-2) reproduction driver.

Edit the CONFIG block below, then run inside the oscar container:

    docker compose run --rm --no-deps oscar bash -lc \
        "cd /app && python3 -u experiments/mi3dor_reproduce.py"

Recreates ONE shape-mode / ONE depth result set for a specific config, using
the *production* pipeline (eval_common.build_pipeline + run_evaluation) — not a
re-implementation. It is the MI3DOR analogue of experiments/stage1_reproduce.py.

Unlike object_retrieval/retrieval_mi3dor_eval_oscarplus.py (which loops BOTH
shape modes and reports the full standard set), this runs exactly one config so
a single ablation cell is easy to reproduce. Still reports all six ranking arms
(clip_only / dino_only_full / ulip_only_full / dino_only_clip_pruned /
ulip_only_clip_pruned / clip_pruned_dino_ulip) for that config, because they are
derived from the same query pass at no extra cost.

See docs/MI3DOR_STAGE2_CONFIG.md for what every value means and why.
"""
import os
import sys

# ============================================================================
# CONFIG — edit these
# ============================================================================
# Repo root auto-detected from this file's location (<root>/experiments/…), so
# the same script works on the host and inside the container (/app). Override
# with an absolute path if your data lives elsewhere.
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- data paths (manual) ---
REF_DIR    = f"{REPO_ROOT}/object_images/MI3DOR"                              # gallery (42 views + caches)
BOP_ROOT   = f"{REPO_ROOT}/eval/datasets/mi3dor/image/test"                  # query images
DESC_FILE  = f"{REPO_ROOT}/object_database/MI3DOR/descriptions_attributes.json"
CAD_GLOB   = f"{REPO_ROOT}/object_database/MI3DOR/model/test/*/*.obj"        # full meshes
OUTPUT_DIR = f"{REPO_ROOT}/object_retrieval/results_mi3dor_singlerun"        # where metrics land
ULIP_QUERY_CACHE = f"{REPO_ROOT}/object_retrieval/ulip_query_cache_mi3dor.pt"  # reused if present

# --- the one run's ablation config ---
SHAPE_MODE   = "partial"        # "partial" (per-view clouds) | "fullmesh" (sampled mesh)
TOPK         = 15               # reported retrieval depth
TOP_F        = 20               # precision/recall/F1 depth
CLIP_PRUNE_K = 20               # CLIP shortlist fallback depth for the OSCAR arms
# CLIP shortlist S' for the pruned / OSCAR arms (Pulli et al. arXiv:2601.07333):
#   "threshold" -> S' = {o : sim_text >= CLIP_TAU}, top-CLIP_PRUNE_K fallback
#   "topk"      -> S' = CLIP top-CLIP_PRUNE_K (legacy)
CLIP_PRUNE_MODE = "threshold"
CLIP_TAU        = 0.37

# channels / fusion
FUSION_METHOD = "weighted_sum"  # "weighted_sum" | "rank_fusion" (RRF, E6)
WEIGHTS       = (0.30, 0.40, 0.30)   # (clip, dino, ulip) — BASE

# view aggregation (both channels top-k soft-max)
NUM_VIEWS = 42
DINO_TOPK = 5;  DINO_TAU = 0.5
ULIP_TOPK = 5;  ULIP_TAU = 0.5       # 5 = equalised to DINO (the 2026-08-05 run); 8 = pipeline default

# encoders (usually leave as-is)
APPEARANCE      = "dinov2"       # "dinov2" | "siglip" (E4)
SHAPE_ENCODER   = "ulip2"        # "ulip2"  | "uni3d"  (E7)
ULIP_USE_COLORS = True           # xyzrgb (True) | xyz-only (False, O5)

# scope
LIMIT_CATEGORIES = None          # e.g. ["airplane", "car"] for a fast smoke; None = all 21
# ============================================================================

# The production modules live in object_retrieval/ and import `pipeline.*`
# relative to the repo root, so both dirs must be importable.
sys.path.insert(0, os.path.join(REPO_ROOT, "object_retrieval"))
sys.path.insert(0, REPO_ROOT)
os.chdir(os.path.join(REPO_ROOT, "object_retrieval"))   # caches/paths expect this cwd

import numpy as np
import torch
import eval_common as ec               # defines EvalConfig, build_pipeline, run_evaluation
import retrieval_mi3dor_eval_oscarplus as mev   # reuse its dataset readers


def main():
    # Determinism: ULIP samples the query cloud stochastically; seed so repeat
    # runs are identical (same reason as stage1_reproduce.py).
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Point the reused dataset readers at OUR paths (they use module globals).
    mev.ref_dir       = REF_DIR
    mev.bop_root      = BOP_ROOT
    mev.desc_file     = DESC_FILE
    mev.cad_mesh_glob = CAD_GLOB

    cfg = ec.EvalConfig(
        ref_dir=REF_DIR,
        desc_file=DESC_FILE,
        cad_mesh_glob=CAD_GLOB,
        result_folder=os.path.join(OUTPUT_DIR, SHAPE_MODE),
        topk=[TOPK],
        TOP_F=TOP_F,
        clip_top_k=CLIP_PRUNE_K,
        clip_prune_mode=CLIP_PRUNE_MODE,
        clip_tau=CLIP_TAU,
        clip_fallback_k=CLIP_PRUNE_K,
        dino_top_k=9999,          # *_full arms score the whole gallery
        ulip2_top_k=9999,
        fusion_top_k=9999,
        fusion_method=FUSION_METHOD,
        weight_clip=WEIGHTS[0],
        weight_dino=WEIGHTS[1],
        weight_ulip=WEIGHTS[2],
        ulip2_use_partial_views=(SHAPE_MODE == "partial"),
        ulip_query_cache_path=ULIP_QUERY_CACHE,
        # View-aggregation + encoder knobs are PipelineConfig fields, applied
        # via pipeline_overrides in build_pipeline() (they are not EvalConfig
        # top-level fields).
        pipeline_overrides={
            "num_views": NUM_VIEWS,
            "dino_view_aggregation": "topk_softmax",
            "dino_view_topk": DINO_TOPK,
            "dino_view_temperature": DINO_TAU,
            "ulip_view_aggregation": "topk_softmax",
            "ulip_view_topk": ULIP_TOPK,
            "ulip_view_temperature": ULIP_TAU,
            "ulip2_use_colors": ULIP_USE_COLORS,
            "appearance_encoder": APPEARANCE,
            "shape_encoder": SHAPE_ENCODER,
            # Stage-2 is retrieval only — no B2 geometry re-ranking (this eval
            # path never calls step_b2 regardless, but be explicit).
            "geometry_reranking_enabled": False,
        },
    )

    print(f"[mi3dor-repro] mode={SHAPE_MODE}  topk={TOPK}  "
          f"weights={WEIGHTS}  ulip_topk={ULIP_TOPK}  appearance={APPEARANCE}  "
          f"shape={SHAPE_ENCODER}")

    # --- stage the data readers (same calls as the full driver's main()) ---
    mev._quarantine_foreign_views(REF_DIR)
    all_categories = mev._get_categories()          # gallery is ALWAYS the full set
    if not all_categories:
        raise RuntimeError(f"No categories found under {BOP_ROOT}")

    # LIMIT_CATEGORIES restricts only which QUERIES are evaluated; the gallery
    # stays the full database so retrieval semantics (and therefore per-category
    # metrics) match a full run. Limiting the gallery too would make retrieval
    # trivially in-category.
    if LIMIT_CATEGORIES:
        keep = set(LIMIT_CATEGORIES)
        query_categories = [c for c in all_categories if c in keep]
        if not query_categories:
            raise RuntimeError(f"LIMIT_CATEGORIES {LIMIT_CATEGORIES} matched none "
                               f"of {all_categories}")
    else:
        query_categories = all_categories
    print(f"[mi3dor-repro] gallery categories: {len(all_categories)} | "
          f"query categories: {len(query_categories)}")

    cad_items = mev._collect_filtered_cad_mesh_items(all_categories)   # FULL gallery

    # --- build production pipeline + (load or compute) ULIP query cache ---
    components = ec.build_pipeline(cfg, cad_mesh_items=cad_items)
    _, _, _, _, shape_m = components
    ulip_cache = ec.load_ulip_query_cache(cfg.ulip_query_cache_path)
    if ulip_cache is None and shape_m is not None:
        print("[mi3dor-repro] no ULIP query cache — encoding query images once.")
        ulip_cache = ec.pre_encode_ulip_queries(
            mev._collect_query_paths(query_categories), shape_m)

    # --- run the single config; writes results + summary into result_folder ---
    ec.run_evaluation(cfg, mev.to_category_label,
                      mev._make_query_factory(query_categories),
                      components, ulip_cache)

    print(f"[mi3dor-repro] done -> {cfg.result_folder}/"
          f"{{results,metrics_summary}}_topk_{TOPK}.json")


if __name__ == "__main__":
    main()
