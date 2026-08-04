#!/usr/bin/env python3
"""
stage1_reproduce.py — minimal, single-run SHREC'18 Stage-1 retrieval.
=============================================================================

One run, one configuration. Edit the CONFIG block, run the script, get this
run's metrics + per-query results written to OUTPUT_DIR. It talks to the
pipeline modules directly (CLIP/DINOv2/ULIP-2 encoders via eval_common,
ScoreFusion, GeometryReRanker) — it is NOT a wrapper over the big ablation
orchestrator. Gallery embeddings and GeDi descriptors are generated on first
use and loaded from cache afterwards.

The full ablation grid, resumability, benchmarks and the official leaderboard
scorer live in `experiment1_shrec18_stage1.py`. This file is the readable
"do one thing" version for understanding and quick custom runs. See
`docs/STAGE1_IMPLEMENTATION.md` for the pseudo-code.

Run:
    docker compose up -d gedi          # only if GEOMETRY is set
    docker compose run --rm oscar bash -lc \
        "cd /app && python3 experiments/stage1_reproduce.py"
=============================================================================
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# ===========================================================================
# CONFIG — everything you would change lives here.
# ===========================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- data paths (chosen manually) ------------------------------------------
DATA_ROOT  = os.path.join(ROOT, "eval/datasets/shrec18/shrec18_full")   # cad/, rgbd/, results/
IMAGES_DIR = os.path.join(ROOT, "object_images/shrec18_v2")             # rendered views + partial .npz
DESC_FILE  = os.path.join(ROOT, "object_database/shrec18_v2/descriptions_attributes.json")
OUTPUT_DIR = os.path.join(ROOT, "object_retrieval/results_stage1_singlerun")  # metrics land here
CACHE_DIR  = os.path.join(OUTPUT_DIR, "_cache")   # gallery embeddings + GeDi descriptors

# --- the values we ablate (change these per run) ---------------------------
WEIGHTS      = (0.30, 0.40, 0.30)   # (w_text/CLIP, w_view/DINO, w_shape/ULIP); renormalise if a channel is off
VIEW_BUDGET  = 42                   # V: FPS-ordered reference views the DINO channel aggregates (8/16/32/42)
APPEARANCE   = "dino"               # "dino" | "siglip"
SHAPE_MODE   = "pc_rgb"             # "pc_rgb" (XYZ+RGB query PC) | "fullmesh" | "cross" (query image) | "off"
FUSION       = "weighted_sum"       # "weighted_sum" | "rank_fusion" (RRF, k=60)
SCOPE        = "full"               # "full" | "clip_topk" | "dino_topk" | "clip_threshold"
CLIP_PRUNE_K = 20                   # shortlist size for the *_topk scopes
CLIP_TAU     = 0.37                 # threshold for scope="clip_threshold" (top-K fallback if empty)
GEOMETRY     = None                 # None | "fitness" | "chamfer_ransac" | "chamfer_icp" | "both_borda"
GEOM_K       = 20                   # geometry re-ranking shortlist depth
SKIP_ICP     = True                 # skip ICP refinement in B2 (+0.0001 nDCG, ~38% cheaper)
LIMIT_QUERIES = None                # None = all; e.g. 20 for a smoke test

# view aggregation ("soft-k-max"): top-k soft-max pooling over per-view sims
DINO_TOPK, DINO_TAU = 5, 0.5
ULIP_TOPK, ULIP_TAU = 8, 0.5
SHAPE_AGG_VIEWS     = 16            # partial-view refs: pool over the first N FPS views

# ===========================================================================
# Wiring. Everything below is the experiment; only CONFIG above should change.
# ===========================================================================
for _p in (ROOT, os.path.join(ROOT, "object_retrieval"),
           os.path.join(ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Data readers reused from the canonical script (SHREC GT reconstruction and
# query preparation — reading data, not experiment logic).
from experiment1_shrec18_stage1 import (          # noqa: E402
    DEFAULTS, load_official_gt, prepare_queries, validate_inputs)


def stack_views(emb_map, object_ids):
    """{obj: [(emb, path)...] | tensor} -> (M, D) matrix + per-object (start,end)."""
    import torch
    chunks, ranges, pos = [], [], 0
    for oid in object_ids:
        v = emb_map.get(oid)
        if v is None:
            ranges.append((pos, pos)); continue
        t = (torch.stack([e for e, _ in v]) if isinstance(v, list)
             else (v if v.dim() == 2 else v.unsqueeze(0)))
        chunks.append(t.float().cpu())
        ranges.append((pos, pos + t.shape[0])); pos += t.shape[0]
    big = torch.cat(chunks) if chunks else torch.zeros(0, 1)
    return big, ranges


def aggregate(sims, ranges, prefix, top_k, tau):
    """Per-object top-k soft-max pooling of per-view similarities (CNOS/OPEN)."""
    import torch
    out = np.full(len(ranges), -np.inf, dtype=np.float32)
    for i, (s, e) in enumerate(ranges):
        cnt = e - s
        if prefix is not None:
            cnt = min(cnt, prefix)
        if cnt <= 0:
            continue
        row = sims[s:s + cnt]
        k = min(top_k, cnt)
        vals, _ = row.topk(k)
        w = torch.softmax(vals / tau, dim=0)
        out[i] = float((w * vals).sum())
    return out


def build_eval_config():
    """Translate CONFIG into the EvalConfig that eval_common.build_pipeline wants."""
    from eval_common import EvalConfig
    mode = {"pc_rgb": ("pc", True), "fullmesh": ("pc", False),
            "cross": ("cross", True), "off": ("cross", True)}[SHAPE_MODE]
    overrides = {"num_views": None}                      # encode all views; trim at scoring time
    if APPEARANCE == "siglip":
        overrides["appearance_encoder"] = "siglip"
    return EvalConfig(
        ref_dir=IMAGES_DIR, desc_file=DESC_FILE,
        cad_mesh_glob=("" if SHAPE_MODE == "off"
                       else os.path.join(DATA_ROOT, "cad", "*.obj")),
        result_folder=CACHE_DIR,
        clip_top_k=10**6, dino_top_k=10**6, ulip2_top_k=10**6, fusion_top_k=10**6,
        weight_clip=WEIGHTS[0], weight_dino=WEIGHTS[1], weight_ulip=WEIGHTS[2],
        ulip2_mode=mode[0], ulip2_use_partial_views=mode[1],
        pipeline_overrides=overrides)


def score_channels(q, roi, clip_r, dino_r, shape_m, dino_big, dino_ranges,
                   shape_big, shape_ranges, object_ids, use_shape):
    """The three per-query score vectors over the whole gallery."""
    import torch
    vecs = {}
    # S_text — CLIP crop-vs-description similarity.
    res = clip_r.retrieve(roi, top_k=10**6)
    smap = {c.object_id: c.score for c in res.candidates}
    vecs["clip"] = np.array([smap.get(o, -np.inf) for o in object_ids], np.float32)
    # S_view — DINO/SigLIP crop-vs-views, top-k soft-max over V views.
    with torch.no_grad():
        qe = dino_r.encode_image(roi)
        sims = (qe @ dino_big.T).squeeze(0).float().cpu()
    vecs["dino"] = aggregate(sims, dino_ranges, VIEW_BUDGET, DINO_TOPK, DINO_TAU)
    # S_shape — ULIP query (point cloud in pc-mode, image in cross-mode) vs CAD.
    if use_shape:
        if SHAPE_MODE == "cross":
            qs = shape_m.encode_image(roi)
        else:
            data = np.load(q["npz"])
            qs = shape_m.encode_pointcloud(data["points"], colors=data["colors"])
        with torch.no_grad():
            sims = (qs.float().to(dino_big.device) @ shape_big.T).squeeze(0).float().cpu()
        prefix = SHAPE_AGG_VIEWS if SHAPE_MODE in ("pc_rgb", "cross") else None
        vecs["shape"] = aggregate(sims, shape_ranges, prefix, ULIP_TOPK, ULIP_TAU)
    return vecs


def fuse(vecs, fusion_mod, object_ids, cad_dir):
    """Rank the gallery: optional scope prune, then production ScoreFusion."""
    from pipeline.step3_clip_retrieval import CLIPCandidate, CLIPRetrievalResult
    from pipeline.step4_dino_reranking import DINOCandidate, DINOReRankingResult
    from pipeline.step5_shape_matching import ShapeCandidate, ShapeMatchingResult
    n = len(object_ids)
    for ch, v in vecs.items():                       # clamp -inf so min-max stays defined
        if not np.isfinite(v).all():
            fin = v[np.isfinite(v)]
            vecs[ch] = np.nan_to_num(v, neginf=float(fin.min()) if fin.size else 0.0)

    # scope: candidate pool + tail (tail keeps its pruning-channel order)
    if SCOPE in ("clip_topk", "clip_threshold"):
        prune = vecs["clip"]
    elif SCOPE == "dino_topk":
        prune = vecs["dino"]
    else:
        prune = None
    if prune is not None:
        order = np.argsort(-prune, kind="stable")
        if SCOPE == "clip_threshold":
            keep = int((prune >= CLIP_TAU).sum()) or min(CLIP_PRUNE_K, n)
        else:
            keep = min(CLIP_PRUNE_K, n)
        pool, tail = order[:keep], order[keep:]
    else:
        pool, tail = np.arange(n), np.array([], int)

    active = [ch for ch, w in zip(("clip", "dino", "shape"), WEIGHTS)
              if ch in vecs and w > 0]
    if FUSION == "weighted_sum" and len(active) == 1:    # single-channel shortcut
        ch = active[0]
        ranked = pool[np.argsort(-vecs[ch][pool], kind="stable")]
        return list(ranked) + list(tail)

    def ordered(ch):
        return pool[np.argsort(-vecs[ch][pool], kind="stable")]
    clip_res = dino_res = shape_res = None
    if "clip" in active:
        clip_res = CLIPRetrievalResult([CLIPCandidate(object_ids[i], float(vecs["clip"][i]))
                                        for i in ordered("clip")], np.zeros(1, np.float32))
    if "dino" in active:
        dino_res = DINOReRankingResult([DINOCandidate(object_ids[i], float(vecs["dino"][i]), 0.0)
                                        for i in ordered("dino")], np.zeros(1, np.float32))
    if "shape" in active:
        shape_res = ShapeMatchingResult([ShapeCandidate(
            object_id=object_ids[i], shape_score=float(vecs["shape"][i]),
            cad_model_path=os.path.join(cad_dir, object_ids[i] + ".obj"))
            for i in ordered("shape")], np.zeros(1, np.float32))
    fused = fusion_mod.fuse(clip_res, dino_res, shape_res, method=FUSION, top_k=n)
    idx = {o: i for i, o in enumerate(object_ids)}
    ranked = [idx[c.object_id] for c in fused.candidates]
    seen = set(ranked)
    return ranked + [i for i in pool if i not in seen] + list(tail)


def geometry_rerank(reranker, q, ranking, object_ids, cad_dir):
    """Re-order the top-GEOM_K by a geometric signal (GeDi + RANSAC)."""
    import open3d as o3d
    from pipeline.step6_fusion import FusedCandidate
    top = ranking[:GEOM_K]
    cands = [FusedCandidate(object_id=object_ids[i], fused_score=0.0,
                            cad_model_path=os.path.join(cad_dir, object_ids[i] + ".obj"))
             for i in top]
    # query cloud: center + unit-normalise + estimate normals (as in B2)
    d = np.load(q["npz"]); pts = d["points"].astype(np.float64)
    pts -= pts.mean(0); r = np.linalg.norm(pts, axis=1).max()
    if r > 0:
        pts /= r
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    res = reranker.rerank(cands, pcd, signal=("chamfer_ransac"
          if GEOMETRY == "both_borda" else GEOMETRY), all_aligned=True, query_id=q["id"])
    gc = {c.object_id: c for c in res.candidates}

    def key(oid):
        c = gc.get(oid)
        if c is None or c.registration_failed:
            return (-np.inf, 0.0)
        if GEOMETRY == "fitness":
            return (c.ransac_fitness, 0.0)
        if GEOMETRY == "chamfer_icp":
            return (-c.d_icp, c.ransac_fitness)
        return (-c.d_ransac, c.ransac_fitness)       # chamfer_ransac
    top_ids = [object_ids[i] for i in top]
    if GEOMETRY == "both_borda":                     # mean rank of fitness & -d_ransac
        def ranks(vals):
            o = np.argsort(np.argsort(-np.asarray(vals), kind="stable"))
            return o.astype(float)
        fit = [gc[o].ransac_fitness if gc.get(o) and not gc[o].registration_failed
               else -np.inf for o in top_ids]
        dst = [-gc[o].d_ransac if gc.get(o) and not gc[o].registration_failed
               else -np.inf for o in top_ids]
        mean_rank = (ranks(fit) + ranks(dst)) / 2
        new = [top_ids[i] for i in np.argsort(mean_rank, kind="stable")]
    else:
        new = sorted(top_ids, key=key, reverse=True)
    idx = {o: i for i, o in enumerate(object_ids)}
    return [idx[o] for o in new] + ranking[GEOM_K:]


# ---- metrics (category-level; graded: 2=subcategory, 1=category) ----------

def compute_metrics(ranked_ids, q_label, cad_labels, freq):
    qc, qs = q_label
    rel = np.zeros(len(ranked_ids))
    for i, cid in enumerate(ranked_ids):
        lab = cad_labels.get(cid)
        if lab and lab[0] == qc:
            rel[i] = 2.0 if lab[1] == qs else 1.0
    cat = rel >= 1.0
    f = freq
    disc = 1.0 / np.log2(np.arange(f) + 2.0)
    dcg = float((rel[:f] * disc).sum())
    idcg = float((np.sort(rel)[::-1][:f] * disc).sum())
    hits = int(cat[:f].sum())
    hidx = np.flatnonzero(cat[:f])
    ap = float(((np.arange(hidx.size) + 1) / (hidx + 1)).sum() / f) if hidx.size else 0.0
    kk = min(GEOM_K, len(rel))
    dk = 1.0 / np.log2(np.arange(kk) + 2.0)
    dcgk = float((rel[:kk] * dk).sum()); idcgk = float((np.sort(rel)[::-1][:kk] * dk).sum())
    return {
        "nDCG": dcg / idcg if idcg else 0.0,
        "precision": hits / f, "recall": hits / f, "F1": hits / f,   # f = |category| = |relevant|
        "AP": ap, "NN": float(cat[0]),
        "nDCG@K": dcgk / idcgk if idcgk else 0.0,
        "NN_sub@1": float(rel[0] >= 2.0),
        "hit_sub@K": float((rel[:kk] >= 2.0).any()),
    }


# ===========================================================================
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    import torch  # noqa: F401 (loads libgomp before open3d)
    # ULIP samples the query point cloud stochastically; seed so a run is
    # reproducible. (The canonical grid instead caches each query's embedding.)
    torch.manual_seed(0); np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    from eval_common import build_pipeline
    from pipeline.config import PipelineConfig
    from pipeline.step6_fusion import ScoreFusion

    print(f"[stage1] SHAPE_MODE={SHAPE_MODE} APPEARANCE={APPEARANCE} V={VIEW_BUDGET} "
          f"FUSION={FUSION} SCOPE={SCOPE} GEOMETRY={GEOMETRY} K={GEOM_K}")

    # --- load encoders + gallery embeddings (generate on first use, else load) ---
    cfg = build_eval_config()
    pipe_cfg, clip_r, dino_r, _f, shape_m = build_pipeline(cfg)
    use_shape = SHAPE_MODE != "off" and shape_m is not None
    if SHAPE_MODE != "off" and shape_m is None:
        print("[stage1] WARNING: shape encoder failed to load — running without S_shape.")

    # --- data: gallery ids, official GT, queries -------------------------------
    paths = {"data_root": DATA_ROOT, "images_dir": IMAGES_DIR, "desc_file": DESC_FILE,
             "results_root": OUTPUT_DIR, "stage1_root": DEFAULTS["stage1_root"]}
    object_ids = validate_inputs(paths, allow_partial=False)
    gt = load_official_gt(DATA_ROOT, paths["stage1_root"])
    cad_labels, freqs = gt["cad"], gt["freqs"]
    index = prepare_queries(DATA_ROOT, paths["stage1_root"], gt)
    if LIMIT_QUERIES:
        index = index[:LIMIT_QUERIES]

    # --- pre-stack reference embeddings; fusion + geometry modules -------------
    dino_big, dino_ranges = stack_views(dino_r._ref_embeddings, object_ids)
    dino_big = dino_big.to(pipe_cfg.device)
    shape_big = shape_ranges = None
    if use_shape:
        shape_big, shape_ranges = stack_views(shape_m._cad_embeddings, object_ids)
        shape_big = shape_big.to(pipe_cfg.device)
    fusion_mod = ScoreFusion(PipelineConfig(
        weight_clip=WEIGHTS[0], weight_dino=WEIGHTS[1], weight_ulip=WEIGHTS[2],
        fusion_method=FUSION))
    reranker = None
    if GEOMETRY:
        from pipeline.step_b2_geometry_reranking import GeometryReRanker
        gcfg = PipelineConfig(geometry_reranking_top_k=GEOM_K,
                              geometry_reranking_signal=GEOMETRY,
                              gedi_cache_dir=os.path.join(CACHE_DIR, "gedi"))
        reranker = GeometryReRanker(gcfg)
        os.makedirs(gcfg.gedi_cache_dir, exist_ok=True)

    # --- the run --------------------------------------------------------------
    from PIL import Image
    cad_dir = os.path.join(DATA_ROOT, "cad")
    sums, per_query = {}, []
    for n, q in enumerate(index, 1):
        qc = tuple(q["category"])[0]
        if freqs.get(qc, 0) == 0:
            continue                                 # no official GT for this category
        roi = Image.open(q["png"]).convert("RGB")
        vecs = score_channels(q, roi, clip_r, dino_r, shape_m, dino_big, dino_ranges,
                              shape_big, shape_ranges, object_ids, use_shape)
        ranking = fuse(vecs, fusion_mod, object_ids, cad_dir)
        if reranker is not None:
            ranking = geometry_rerank(reranker, q, ranking, object_ids, cad_dir)
        ranked_ids = [object_ids[i] for i in ranking]
        m = compute_metrics(ranked_ids, tuple(q["category"]), cad_labels, freqs[qc])
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + v
        per_query.append({"id": q["id"], "category": list(q["category"]),
                          "top10": ranked_ids[:10], "nDCG": round(m["nDCG"], 4),
                          "AP": round(m["AP"], 4)})
        if n % 50 == 0 or n == len(index):
            print(f"  {n}/{len(index)}")

    nq = len(per_query)
    metrics = {k: v / nq for k, v in sums.items()}

    # --- save this run's metrics + per-query results to OUTPUT_DIR -------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = {k: v for k, v in globals().items()
              if k.isupper() and isinstance(v, (int, float, str, tuple, type(None)))}
    with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w") as fh:
        json.dump({"config": config, "num_queries": nq,
                   "gallery_size": len(object_ids), "metrics": metrics}, fh, indent=2)
    with open(os.path.join(OUTPUT_DIR, "results_per_query.json"), "w") as fh:
        json.dump(per_query, fh)
    print(f"\n[stage1] n={nq}  " + "  ".join(f"{k}={metrics[k]:.4f}" for k in
          ("nDCG", "precision", "AP", "NN", "nDCG@K", "NN_sub@1")))
    print(f"[stage1] wrote metrics_summary.json + results_per_query.json -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
