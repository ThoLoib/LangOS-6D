"""
eval_common.py
==============
Shared evaluation infrastructure for OSCAR+ retrieval experiments.

Dataset-specific scripts provide:
  - An EvalConfig with paths and knobs
  - A to_label_fn(object_id) -> label function
  - A query_factory(k) -> iterator of (roi, gt_label, img_path, category, fname)

This module provides everything else: pipeline init, query processing,
metrics, CSV streaming, and JSON summaries.
"""

import glob as _glob
import json
import os
import sys
from dataclasses import dataclass, field
from math import log2
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OSCAR_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _OSCAR_ROOT not in sys.path:
    sys.path.insert(0, _OSCAR_ROOT)

from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.step2_pointcloud import PointCloudResult  # noqa: E402
from pipeline.step3_clip_retrieval import CLIPRetriever  # noqa: E402
from pipeline.step4_dino_reranking import DINOReRanker  # noqa: E402
from pipeline.step5_shape_matching import ShapeMatcher  # noqa: E402
from pipeline.step6_fusion import ScoreFusion  # noqa: E402

RANKING_KEYS = (
    "clip_only",
    "dino_only_full",
    "ulip_only_full",
    "dino_only_clip_pruned",
    "ulip_only_clip_pruned",
    "clip_pruned_dino_ulip",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Pipeline + output settings shared across datasets."""

    ref_dir: str = ""
    desc_file: str = ""
    cad_mesh_glob: str = ""
    result_folder: str = "results"

    topk: List[int] = field(default_factory=lambda: [15])
    TOP_F: int = 20

    clip_top_k: int = 9999
    dino_top_k: int = 9999
    ulip2_top_k: int = 9999
    fusion_top_k: int = 9999
    fusion_method: str = "weighted_sum"
    weight_clip: float = 0
    weight_dino: float = 0.5
    weight_ulip: float = 0.5

    ulip_repo_path: str = "/ulip"
    ulip2_checkpoint: str = "/ulip/checkpoints/ulip2_pointbert_10k.pt"
    ulip2_mode: str = "cross"
    ulip2_use_partial_views: bool = False
    ulip_query_cache_path: str = ""


# ---------------------------------------------------------------------------
# Metric helpers (verbatim from retrieval_mi3dor_eval.py)
# ---------------------------------------------------------------------------

def dcg_at_k(rels, k):
    dcg = 0.0
    for i in range(min(len(rels), k)):
        dcg += rels[i] / log2(i + 2)
    return dcg


def ideal_dcg_at_k(n_relevant, k):
    ideal = 0.0
    for i in range(min(n_relevant, k)):
        ideal += 1.0 / log2(i + 2)
    return ideal


def average_precision_from_binary(rels):
    rels = np.asarray(rels, dtype=np.int32)
    if rels.sum() == 0:
        return 0.0
    precisions = []
    cum = 0
    for i, r in enumerate(rels, start=1):
        if r:
            cum += 1
            precisions.append(cum / i)
    return float(np.mean(precisions)) if precisions else 0.0


def compute_anmrr(ranks, num_rel, K):
    if num_rel == 0:
        return None
    if not ranks:
        avr = K + 1
    else:
        padded = ranks + [K + 1] * max(0, num_rel - len(ranks))
        avr = float(np.mean(padded))
    denom = K - (num_rel + 1) / 2.0
    if denom <= 0:
        return 0.0
    return (avr - (num_rel + 1) / 2.0) / denom


def mean_ignore_nan(xs):
    xs = np.array(xs, dtype=float)
    xs = xs[~np.isnan(xs)]
    return float(xs.mean()) if xs.size > 0 else float("nan")


# ---------------------------------------------------------------------------
# Incremental metric accumulators (constant memory)
# ---------------------------------------------------------------------------

def make_accum():
    return {
        "nn_correct": 0, "ft": 0.0, "st": 0.0, "f1": 0.0,
        "ndcg": 0.0, "ap": [], "anmrr": [], "count": 0,
    }


def update_accum(accum, ids_scores, gt_label, to_label_fn, top_f):
    full_labels = [to_label_fn(oid) for oid, _ in ids_scores]
    if not full_labels:
        return
    rels = np.array([1 if lab == gt_label else 0 for lab in full_labels], dtype=int)
    num_rel = int(rels.sum())
    if num_rel == 0:
        return
    accum["count"] += 1
    if full_labels[0] == gt_label:
        accum["nn_correct"] += 1
    k_ft = num_rel
    k_st = min(2 * num_rel, len(rels))
    accum["ft"] += float(rels[:k_ft].sum()) / k_ft
    accum["st"] += float(rels[:k_st].sum()) / k_ft
    top_f_eff = min(top_f, len(rels))
    rel_top_f = int(rels[:top_f_eff].sum())
    p = rel_top_f / top_f_eff
    r = rel_top_f / num_rel
    accum["f1"] += (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    k_dcg = k_st
    dcg_val = dcg_at_k(rels.tolist(), k_dcg)
    idcg_val = ideal_dcg_at_k(num_rel, k_dcg)
    accum["ndcg"] += dcg_val / idcg_val if idcg_val > 0 else 0.0
    accum["ap"].append(average_precision_from_binary(rels.tolist()))
    rel_pos = [j + 1 for j, rv in enumerate(rels) if rv == 1 and (j + 1) <= k_dcg]
    accum["anmrr"].append(compute_anmrr(rel_pos, num_rel, k_dcg))


def finalize_accum(accum):
    n = accum["count"]
    if n == 0:
        return {k: float("nan") for k in (
            "num_queries", "NN_accuracy", "FT_mean", "ST_mean",
            "F1_mean", "nDCG@2R_mean", "mAP", "ANMRR_mean")}
    return {
        "num_queries": n,
        "NN_accuracy": 100.0 * accum["nn_correct"] / n,
        "FT_mean":      accum["ft"]   / n,
        "ST_mean":      accum["st"]   / n,
        "F1_mean":      accum["f1"]   / n,
        "nDCG@2R_mean": accum["ndcg"] / n,
        "mAP":          float(np.mean(accum["ap"]))    if accum["ap"]    else float("nan"),
        "ANMRR_mean":   float(np.mean(accum["anmrr"])) if accum["anmrr"] else float("nan"),
    }


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def empty_pointcloud_result():
    empty = np.zeros((0, 3), dtype=np.float32)
    return PointCloudResult(
        point_cloud=None,
        points=empty,
        colors=empty,
        num_points=0,
        bbox_min=np.zeros(3, dtype=np.float32),
        bbox_max=np.zeros(3, dtype=np.float32),
        bbox_size=np.zeros(3, dtype=np.float32),
    )


def _filter_dino_result_by_ids(result, id_set, clip_score_map=None):
    """Derive a CLIP-pruned DINO result from a full DINO result.

    Pure id-intersection — scores and order (sorted by dino_score desc)
    are preserved. Optionally backfills ``clip_score`` on the kept
    candidates from a ``{obj_id: clip_score}`` map, so the derived pruned
    result matches what an explicit CLIP-gated DINO run would have
    produced.
    """
    if result is None:
        return None
    import copy as _copy
    kept = []
    for c in result.candidates:
        if c.object_id not in id_set:
            continue
        if clip_score_map is not None and c.object_id in clip_score_map:
            c = _copy.copy(c)
            c.clip_score = float(clip_score_map[c.object_id])
        kept.append(c)
    out = _copy.copy(result)
    out.candidates = kept
    return out


def _filter_shape_result_by_ids(result, id_set):
    """Derive a CLIP-pruned ShapeMatchingResult from a full one.

    Pure id-intersection — shape_scores and order are preserved.
    """
    if result is None:
        return None
    import copy as _copy
    kept = [c for c in result.candidates if c.object_id in id_set]
    out = _copy.copy(result)
    out.candidates = kept
    return out


def query_has_depth(query_path):
    d = os.path.dirname(query_path)
    stem = os.path.splitext(os.path.basename(query_path))[0]
    sibling_depth_dir = os.path.join(os.path.dirname(d), "depth")
    if os.path.isdir(sibling_depth_dir):
        for ext in (".png", ".tif", ".tiff", ".npy"):
            if os.path.isfile(os.path.join(sibling_depth_dir, stem + ext)):
                return True
    for suffix in ("_depth.png", "_depth.tif", "_depth.tiff", "_depth.npy"):
        if os.path.isfile(os.path.join(d, stem + suffix)):
            return True
    if os.path.isfile(os.path.join(d, stem + ".npz")):
        return True
    return False


def build_pipeline(cfg, cad_mesh_items=None):
    """Initialise all pipeline components.

    Parameters
    ----------
    cfg : EvalConfig
    cad_mesh_items : list of (obj_id, mesh_path), optional
        When the default basename-based obj_id extraction is wrong (e.g.
        YCBV ``<name>/meshes/model.obj``), pass a pre-built list.

    Returns (pipeline_config, clip_retr, dino_rer, fusion, shape_m).
    """
    config = PipelineConfig(
        description_file=cfg.desc_file,
        reference_images_dir=cfg.ref_dir,
        cad_models_dir="",
        clip_top_k=cfg.clip_top_k,
        dino_top_k=cfg.dino_top_k,
        ulip2_top_k=cfg.ulip2_top_k,
        fusion_top_k=cfg.fusion_top_k,
        fusion_method=cfg.fusion_method,
        weight_clip=cfg.weight_clip,
        weight_dino=cfg.weight_dino,
        weight_ulip=cfg.weight_ulip,
        ulip_repo_path=cfg.ulip_repo_path,
        ulip2_checkpoint=cfg.ulip2_checkpoint,
        ulip2_mode=cfg.ulip2_mode,
        ulip2_use_partial_views=cfg.ulip2_use_partial_views,
    )

    print("[init] Loading CLIP descriptions...")
    clip_retr = CLIPRetriever(config)
    clip_retr.load_descriptions()

    print("[init] Loading DINOv2 reference images...")
    dino_rer = DINOReRanker(config)
    dino_rer.load_reference_images()

    fusion_mod = ScoreFusion(config)

    # --- ULIP CAD embeddings ---
    shape_m = None
    if cad_mesh_items is None and cfg.cad_mesh_glob:
        mesh_paths = sorted(_glob.glob(cfg.cad_mesh_glob))
        if mesh_paths:
            cad_mesh_items = [
                (os.path.splitext(os.path.basename(p))[0], p)
                for p in mesh_paths
            ]
        else:
            print(f"[init] WARNING: cad_mesh_glob matched 0 files: "
                  f"{cfg.cad_mesh_glob}")

    if cad_mesh_items:
        shape_m = ShapeMatcher(config)
        try:
            shape_m._load_model()
        except Exception as exc:
            print(f"[init] ULIP model load failed: {exc}")
            shape_m = None

        if shape_m is not None and cfg.ulip2_use_partial_views:
            # --- Partial-view path: encode per-view .npz embeddings ---
            partial_items = shape_m._collect_partial_items(cfg.ref_dir)
            if not partial_items:
                print(f"[init] WARNING: no partial PCs found in {cfg.ref_dir}. "
                      f"Falling back to full-mesh encoding.")
                cfg.ulip2_use_partial_views = False
                config.ulip2_use_partial_views = False

        if shape_m is not None and cfg.ulip2_use_partial_views:
            mesh_map = {oid: p for oid, p in cad_mesh_items}
            shape_m._partial_view_paths = dict(partial_items)
            cache_path = shape_m._get_partial_cache_path(
                cfg.ref_dir, partial_items)

            if shape_m._try_load_partial_cache(cache_path):
                for oid in shape_m._cad_embeddings:
                    if oid not in shape_m._cad_paths and oid in mesh_map:
                        shape_m._cad_paths[oid] = mesh_map[oid]
                print(f"[init] ULIP partial-view cache loaded "
                      f"({len(shape_m._cad_embeddings)} models).")
            else:
                print(f"[init] Encoding {len(partial_items)} objects "
                      f"(partial views)...")
                ok = 0
                for obj_id, view_files in tqdm(sorted(partial_items.items()),
                                               desc="ULIP partial",
                                               unit="obj"):
                    view_embs = []
                    for view_idx, npz_path in sorted(view_files):
                        try:
                            data = np.load(npz_path)
                            emb = shape_m.encode_pointcloud(
                                data["points"], colors=data.get("colors"))
                            view_embs.append(
                                emb.squeeze(0).detach().cpu())
                        except Exception as e:
                            tqdm.write(
                                f"[warn] {obj_id} view {view_idx}: {e}")
                    if view_embs:
                        shape_m._cad_embeddings[obj_id] = torch.stack(
                            view_embs, dim=0)
                        shape_m._cad_paths[obj_id] = mesh_map.get(
                            obj_id, "")
                        ok += 1
                    elif obj_id in mesh_map:
                        if shape_m._encode_and_cache(
                                obj_id, mesh_map[obj_id]):
                            ok += 1

                print(f"[init] ULIP partial: {ok}/{len(partial_items)} ok.")
                if ok == 0:
                    shape_m = None
                else:
                    shape_m._partial_mode = True
                    shape_m._save_partial_cache(cache_path)

        elif shape_m is not None:
            # --- Full-mesh path (legacy) ---
            print(f"[init] Encoding {len(cad_mesh_items)} ULIP CAD meshes...")
            all_paths = [p for _, p in cad_mesh_items]
            cad_dir = os.path.commonpath(all_paths)
            if not os.path.isdir(cad_dir):
                cad_dir = os.path.dirname(cad_dir)
            cache_path = shape_m._get_cache_path(cad_dir, cad_mesh_items)

            if shape_m._try_load_cache(cache_path):
                print(f"[init] ULIP CAD cache loaded "
                      f"({len(shape_m._cad_embeddings)} models).")
            else:
                ok = 0
                for obj_id, mesh_path in tqdm(cad_mesh_items,
                                              desc="ULIP CAD", unit="mesh"):
                    if shape_m._encode_and_cache(obj_id, mesh_path):
                        ok += 1
                print(f"[init] ULIP CAD: {ok}/{len(cad_mesh_items)} ok.")
                if ok == 0:
                    shape_m = None
                else:
                    shape_m._save_cache(cache_path)

    return config, clip_retr, dino_rer, fusion_mod, shape_m


# ---------------------------------------------------------------------------
# ULIP query cache
# ---------------------------------------------------------------------------

def load_ulip_query_cache(cache_path):
    if not cache_path or not os.path.isfile(cache_path):
        return None
    print(f"[ulip-cache] Loading {cache_path}...")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    print(f"[ulip-cache] {len(cache)} embeddings loaded.")
    return cache


def pre_encode_ulip_queries(img_paths, shape_m, batch_size=32):
    shape_m._load_image_encoder()
    shape_m.image_encoder.load()
    enc = shape_m.image_encoder
    preprocess = enc.preprocess
    model = enc.model
    device = enc.device

    print(f"[ulip-preenc] Encoding {len(img_paths)} images "
          f"(batch={batch_size})...")
    tensors, valid_paths = [], []
    for p in img_paths:
        try:
            tensors.append(preprocess(Image.open(p).convert("RGB")))
            valid_paths.append(p)
        except Exception as exc:
            tqdm.write(f"[ulip-preenc] skip {p}: {exc}")

    cache = {}
    model.eval()
    for i in tqdm(range(0, len(tensors), batch_size),
                  desc="ULIP pre-enc", unit="batch"):
        batch = torch.stack(tensors[i:i + batch_size]).to(device)
        with torch.no_grad():
            emb = model(batch)
            emb = F.normalize(emb, p=2, dim=-1)
        for j, p in enumerate(valid_paths[i:i + batch_size]):
            cache[p] = emb[j:j + 1].cpu()

    print(f"[ulip-preenc] Done: {len(cache)} embeddings.")
    return cache


# ---------------------------------------------------------------------------
# Per-query retrieval
# ---------------------------------------------------------------------------

def run_query(pipeline_cfg, clip_retr, dino_rer, fusion_mod, shape_m,
              roi, cfg, ulip_query_emb=None,
              dino_full_top_k=None, ulip_full_top_k=None):
    """Run one query through CLIP + (one full DINO) + (one full ULIP) + fusion.

    The CLIP-pruned DINO and CLIP-pruned ULIP variants are derived by
    id-intersecting the full rankings with the CLIP candidate set; no
    re-ranking happens. This is mathematically equivalent to running a
    separate CLIP-gated DINO/ULIP pass, because both stages compute
    per-object scores independent of the candidate pool (DINO aggregates
    views per object; ULIP does per-object cosine sim).

    ``dino_full_top_k`` and ``ulip_full_top_k`` control the depth of the
    full rankings. They should be ≥ the number of reference objects for
    the respective stage so that every CLIP candidate survives the full
    pass — the run_evaluation wrapper computes them from the loaded
    reference counts.
    """
    clip_res = clip_retr.retrieve(roi, top_k=cfg.clip_top_k)
    clip_candidate_ids = [c.object_id for c in clip_res.candidates]
    clip_id_set = set(clip_candidate_ids)
    clip_score_map = {c.object_id: float(c.score)
                      for c in clip_res.candidates}

    # Full DINO: single pass over the whole reference set.
    dino_k = dino_full_top_k or cfg.dino_top_k
    dino_res_full = dino_rer.rerank(roi, clip_result=None, top_k=dino_k)
    dino_res_clip_pruned = _filter_dino_result_by_ids(
        dino_res_full, clip_id_set, clip_score_map=clip_score_map,
    )

    # Full ULIP: single pass over the whole CAD set.
    ulip_fell_back = False
    shape_res_full = None
    shape_res_clip_pruned = None
    if shape_m is not None:
        ulip_k = ulip_full_top_k or cfg.ulip2_top_k
        prev_mode = pipeline_cfg.ulip2_mode
        pipeline_cfg.ulip2_mode = "cross"
        try:
            if ulip_query_emb is not None:
                cached = ulip_query_emb.to(shape_m.config.device)
                orig_encode = shape_m.encode_image
                shape_m.encode_image = lambda _img: cached
                try:
                    shape_res_full = shape_m.match(
                        empty_pointcloud_result(),
                        top_k=ulip_k,
                        candidate_ids=None,
                        query_image=roi,
                    )
                finally:
                    shape_m.encode_image = orig_encode
            else:
                shape_res_full = shape_m.match(
                    empty_pointcloud_result(),
                    top_k=ulip_k,
                    candidate_ids=None,
                    query_image=roi,
                )
            ulip_fell_back = True
        finally:
            pipeline_cfg.ulip2_mode = prev_mode

        shape_res_clip_pruned = _filter_shape_result_by_ids(
            shape_res_full, clip_id_set,
        )

    fusion_top_k = len(clip_candidate_ids) if clip_candidate_ids else cfg.fusion_top_k
    fused_du_clip_pruned = fusion_mod.fuse(
        None,
        dino_res_clip_pruned,
        shape_res_clip_pruned,
        top_k=fusion_top_k,
    )

    # Final decision must be a CLIP candidate: choose highest fused CLIP score.
    if clip_candidate_ids:
        filtered = [c for c in fused_du_clip_pruned.candidates
                    if c.object_id in clip_id_set]
        filtered.sort(key=lambda c: c.fused_score, reverse=True)
        fused_du_clip_pruned.candidates = filtered
        fused_du_clip_pruned.best_match = filtered[0] if filtered else None

    return {
        "clip_res": clip_res,
        "dino_res_full": dino_res_full,
        "dino_res_clip_pruned": dino_res_clip_pruned,
        "shape_res_full": shape_res_full,
        "shape_res_clip_pruned": shape_res_clip_pruned,
        "fused_du_clip_pruned": fused_du_clip_pruned,
        "ulip_fell_back": ulip_fell_back,
    }


# ---------------------------------------------------------------------------
# Ranking extractors  → list of (object_id, score)
# ---------------------------------------------------------------------------

def clip_ranking(clip_res):
    return [(c.object_id, float(c.score)) for c in clip_res.candidates]


def dino_ranking(dino_res):
    return [(c.object_id, float(c.dino_score)) for c in dino_res.candidates]


def ulip_ranking(shape_res):
    if shape_res is None:
        return []
    return [(c.object_id, float(c.shape_score)) for c in shape_res.candidates]


def fusion_ranking(fusion_res):
    return [(c.object_id, float(c.fused_score)) for c in fusion_res.candidates]


# ---------------------------------------------------------------------------
# Image crop helpers
# ---------------------------------------------------------------------------

def crop_by_bbox(image, bbox):
    """Crop PIL Image by BOP-style [x, y, w, h] bbox."""
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))


def crop_with_mask(image, mask):
    """Crop PIL Image using a binary mask (grey background)."""
    mask_array = np.array(mask) > 0
    if mask_array.sum() == 0:
        return None
    img_array = np.array(image)
    masked = np.full(img_array.shape, (205, 205, 205), dtype=np.uint8)
    masked[mask_array] = img_array[mask_array]
    coords = np.argwhere(mask_array)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return Image.fromarray(masked[y0:y1, x0:x1])


# ---------------------------------------------------------------------------
# JSON output (private helpers)
# ---------------------------------------------------------------------------


def _make_per_query_record(out, gt_label, category, fname, to_label_fn):
    """Build a per-query record with explicit full-set vs CLIP-pruned fields.

    ``pred`` and ``clip_pruned_dino_ulip_pred`` are the same value — the
    top-1 of the explicit clip-pruned DINO+ULIP fusion. There is no
    ambiguous ``fusion_pred`` / ``fusion_top5`` field; readers should pick
    between ``dino_candidates_full`` / ``dino_candidates_clip_pruned`` (and
    the matching ULIP pair) based on which pool they want to evaluate.
    """
    clip_ids = clip_ranking(out["clip_res"])
    dino_ids_full = dino_ranking(out["dino_res_full"])
    dino_ids_clip_pruned = dino_ranking(out["dino_res_clip_pruned"])
    ulip_ids_full = ulip_ranking(out["shape_res_full"])
    ulip_ids_clip_pruned = ulip_ranking(out["shape_res_clip_pruned"])
    du_clip = fusion_ranking(out["fused_du_clip_pruned"])

    top5_clip = [{"label": to_label_fn(oid), "clip_score": s}
                 for oid, s in clip_ids[:5]]
    top5_dino_full = [{"label": to_label_fn(oid), "dino_score": s}
                      for oid, s in dino_ids_full[:5]]
    top5_dino_clip_pruned = [{"label": to_label_fn(oid), "dino_score": s}
                             for oid, s in dino_ids_clip_pruned[:5]]
    top5_ulip_full = [{"label": to_label_fn(oid), "ulip_score": s}
                      for oid, s in ulip_ids_full[:5]]
    top5_ulip_clip_pruned = [{"label": to_label_fn(oid), "ulip_score": s}
                             for oid, s in ulip_ids_clip_pruned[:5]]

    top5_clip_pruned_du = []
    for c in out["fused_du_clip_pruned"].candidates[:5]:
        top5_clip_pruned_du.append({
            "label": to_label_fn(c.object_id),
            "fused_score": float(c.fused_score),
            "dino_score": float(c.dino_score),
            "ulip_score": float(c.ulip_score),
        })

    matched_files = [oid for oid, _ in dino_ids_full[:5]]

    pred_label = to_label_fn(du_clip[0][0]) if du_clip else None

    return {
        "category": category,
        "filename": fname,
        "gt": gt_label,
        "pred": pred_label,
        "clip_candidates": top5_clip,
        "dino_candidates_full": top5_dino_full,
        "dino_candidates_clip_pruned": top5_dino_clip_pruned,
        "ulip_candidates_full": top5_ulip_full,
        "ulip_candidates_clip_pruned": top5_ulip_clip_pruned,
        "matched_files": matched_files,
        "clip_pruned_dino_ulip_pred": pred_label,
        "clip_pruned_dino_ulip_top5": top5_clip_pruned_du,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(cfg, to_label_fn, query_factory, components,
                   ulip_cache=None):
    """Run the full OSCAR+ evaluation.

    Parameters
    ----------
    cfg : EvalConfig
    to_label_fn : callable
        Maps pipeline object_id to the label used for relevance checks.
    query_factory : callable(k) -> iterable of (roi, gt_label, img_path, category, fname)
    components : tuple
        ``(pipeline_config, clip_retr, dino_rer, fusion, shape_m)`` from
        :func:`build_pipeline`.
    ulip_cache : dict, optional
        ``{img_path: tensor(1, embed_dim)}`` pre-computed ULIP query
        embeddings.

    Returns
    -------
    dict mapping *k* to the summary dict for that topk run.
    """
    pipeline_cfg, clip_retr, dino_rer, fusion_mod, shape_m = components
    os.makedirs(cfg.result_folder, exist_ok=True)

    # --- Auto-expand full-ranking depth so CLIP candidates survive filtering ---
    ref_objects = (len(dino_rer._ref_embeddings)
                   if getattr(dino_rer, "_ref_embeddings", None) else 0)
    cad_objects = (len(shape_m._cad_embeddings)
                   if (shape_m is not None and shape_m._cad_embeddings)
                   else 0)
    dino_full_top_k = max(cfg.dino_top_k, ref_objects) if ref_objects else cfg.dino_top_k
    ulip_full_top_k = max(cfg.ulip2_top_k, cad_objects) if cad_objects else cfg.ulip2_top_k

    auto_dino = dino_full_top_k != cfg.dino_top_k
    auto_ulip = ulip_full_top_k != cfg.ulip2_top_k
    print(
        f"[eval] Full-ranking depth — DINO: {dino_full_top_k} "
        f"({'auto-expanded from ' + str(cfg.dino_top_k) if auto_dino else 'from cfg'}, "
        f"{ref_objects} ref objects); "
        f"ULIP: {ulip_full_top_k} "
        f"({'auto-expanded from ' + str(cfg.ulip2_top_k) if auto_ulip else 'from cfg'}, "
        f"{cad_objects} CAD objects). CLIP-pruned variants are derived "
        f"by id-intersecting these full rankings (no re-ranking)."
    )

    summaries = {}
    for k in cfg.topk:
        accums = {name: make_accum() for name in RANKING_KEYS}
        ulip_fallback_count = 0
        total_queries = 0
        per_query_records = []

        for roi, gt_label, img_path, category, fname in query_factory(k):
            try:
                ulip_emb = (ulip_cache.get(img_path)
                            if ulip_cache else None)
                out = run_query(
                    pipeline_cfg, clip_retr, dino_rer, fusion_mod,
                    shape_m, roi, cfg, ulip_query_emb=ulip_emb,
                    dino_full_top_k=dino_full_top_k,
                    ulip_full_top_k=ulip_full_top_k,
                )
            except Exception as exc:
                tqdm.write(f"[warn] query failed ({img_path}): {exc}")
                continue

            total_queries += 1
            if out["ulip_fell_back"]:
                ulip_fallback_count += 1

            # Incremental metrics
            c_ids  = clip_ranking(out["clip_res"])
            d_full_ids = dino_ranking(out["dino_res_full"])
            d_clip_ids = dino_ranking(out["dino_res_clip_pruned"])
            u_full_ids = ulip_ranking(out["shape_res_full"])
            u_clip_ids = ulip_ranking(out["shape_res_clip_pruned"])
            du_clip_ids = fusion_ranking(out["fused_du_clip_pruned"])

            update_accum(accums["clip_only"],        c_ids,  gt_label,
                         to_label_fn, cfg.TOP_F)
            update_accum(accums["dino_only_full"],   d_full_ids, gt_label,
                         to_label_fn, cfg.TOP_F)
            update_accum(accums["ulip_only_full"],   u_full_ids, gt_label,
                         to_label_fn, cfg.TOP_F)
            update_accum(accums["dino_only_clip_pruned"], d_clip_ids,
                         gt_label, to_label_fn, cfg.TOP_F)
            update_accum(accums["ulip_only_clip_pruned"], u_clip_ids,
                         gt_label, to_label_fn, cfg.TOP_F)
            update_accum(accums["clip_pruned_dino_ulip"], du_clip_ids,
                         gt_label, to_label_fn, cfg.TOP_F)

            per_query_records.append(
                _make_per_query_record(out, gt_label, category, fname,
                                      to_label_fn)
            )

        # --- Finalize metrics ---
        variants = {name: finalize_accum(accums[name])
                    for name in RANKING_KEYS}
        primary = variants["clip_pruned_dino_ulip"]

        summary = {
            "primary": "clip_pruned_dino_ulip",
            **primary,
            "variants": variants,
            "config": {
                "ref_dir": cfg.ref_dir,
                "desc_file": cfg.desc_file,
                "cad_mesh_glob": cfg.cad_mesh_glob,
                "result_folder": cfg.result_folder,
                "topk": cfg.topk, "TOP_F": cfg.TOP_F,
                "clip_top_k": cfg.clip_top_k,
                "dino_top_k": cfg.dino_top_k,
                "ulip2_top_k": cfg.ulip2_top_k,
                "dino_full_top_k_used": dino_full_top_k,
                "ulip_full_top_k_used": ulip_full_top_k,
                "fusion_top_k": cfg.fusion_top_k,
                "fusion_method": cfg.fusion_method,
                "weight_clip": cfg.weight_clip,
                "weight_dino": cfg.weight_dino,
                "weight_ulip": cfg.weight_ulip,
                "ulip_repo_path": cfg.ulip_repo_path,
                "ulip2_checkpoint": cfg.ulip2_checkpoint,
                "ulip2_mode": cfg.ulip2_mode,
                "ulip2_use_partial_views": cfg.ulip2_use_partial_views,
                "ulip_active": shape_m is not None,
            },
            "ulip_fallback_cross_count": ulip_fallback_count,
            "total_queries_seen": total_queries,
        }

        results_path = os.path.join(cfg.result_folder,
                                    f"results_topk_{k}.json")
        summary_path = os.path.join(cfg.result_folder,
                                    f"metrics_summary_topk_{k}.json")
        with open(results_path, "w") as f:
            json.dump(per_query_records, f, indent=2)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # --- Console summary ---
        print(f"\n=== OSCAR+ evaluation (top-k={k}, "
              f"primary=clip_pruned_dino_ulip) ===")
        for key in ("num_queries", "NN_accuracy", "FT_mean", "ST_mean",
                     "F1_mean", "nDCG@2R_mean", "mAP", "ANMRR_mean"):
            print(f"  {key}: {primary.get(key)}")
        print(f"  ulip_fallback_cross_count: {ulip_fallback_count}")
        print(f"  total_queries_seen: {total_queries}")
        print(f"\n  Per-variant NN_accuracy:")
        for name in RANKING_KEYS:
            print(f"    {name:<28s} "
                  f"{variants[name].get('NN_accuracy')}")
        print(f"\n  Results: {results_path}")
        print(f"  Summary: {summary_path}\n")

        summaries[k] = summary

    return summaries
