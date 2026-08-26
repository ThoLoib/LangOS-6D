#!/usr/bin/env python3
"""MI3DOR cross-mode fusion-weight sweep -> ternary heatmap CSV.

Stage-2 counterpart of the Stage-1 weight sweep. The MI3DOR driver has no
per-channel score cache, so we score every query ONCE (reusing eval_common's
exact run_query), cache the three per-channel score maps, then re-fuse over a
weight simplex — min-max per channel + weighted sum, the ScoreFusion recipe.
Metric reuses eval_common's update_accum/finalize_accum so FT/NN match the
driver exactly.

SELF-CHECK: the BASE point (0.3,0.4,0.3) must reproduce the reported full-fusion
FT (~0.682, partial/cross run). If it deviates >0.03 the fusion normalisation is
wrong and the script exits non-zero rather than emit a misleading heatmap.
"""
import os, sys, csv
import numpy as np

# The MI3DOR driver uses paths relative to object_retrieval/, so this must be
# run from there (cd object_retrieval && python3 mi3dor_weight_sweep.py) with
# sibling imports — mirroring how the driver itself is invoked.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_mi3dor_eval_oscarplus import (  # noqa: E402
    cfg, to_category_label, _get_categories,
    _collect_filtered_cad_mesh_items, _make_query_factory, _collect_query_paths,
)
from eval_common import (  # noqa: E402
    build_pipeline, run_query, make_accum, update_accum, finalize_accum,
    load_ulip_query_cache, pre_encode_ulip_queries,
)

BASE_W = (0.3, 0.4, 0.3)
BASE_FT_EXPECTED = 0.682
STEP = 0.05
OUT_CSV = os.environ.get("MI3DOR_SWEEP_CSV",
                         "results_mi3dor_wsweep/weight_sweep_mi3dor.csv")


def _simplex(step):
    n = int(round(1.0 / step))
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((round(i * step, 4), round(j * step, 4), round(k * step, 4)))
    return pts


def _minmax(d):
    if not d:
        return d
    vs = np.fromiter(d.values(), dtype=np.float64)
    lo, hi = float(vs.min()), float(vs.max())
    rng = (hi - lo) or 1e-9
    return {k: (v - lo) / rng for k, v in d.items()}


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    categories = _get_categories()
    cad_mesh_items = _collect_filtered_cad_mesh_items(categories)
    cfg.ulip2_use_partial_views = True          # cross-mode partial gallery (reported)
    cfg.result_folder = os.path.dirname(OUT_CSV)
    print(f"[sweep] building pipeline (cross-mode partial) over {len(categories)} categories ...")
    components = build_pipeline(cfg, cad_mesh_items=cad_mesh_items)
    pipeline_cfg, clip_retr, dino_rer, fusion_mod, shape_m = components

    ulip_cache = load_ulip_query_cache(cfg.ulip_query_cache_path)
    if ulip_cache is None:
        print("[sweep] encoding ULIP query images (once) ...")
        ulip_cache = pre_encode_ulip_queries(_collect_query_paths(categories), shape_m)

    # gallery |C| per category (FT normaliser), mirroring run_evaluation
    gallery_ids = set(getattr(dino_rer, "_ref_embeddings", {}) or {})
    if shape_m is not None and getattr(shape_m, "_cad_embeddings", None):
        gallery_ids |= set(shape_m._cad_embeddings)
    glc = {}
    for oid in gallery_ids:
        lab = to_category_label(oid)
        glc[lab] = glc.get(lab, 0) + 1
    ref_objects = len(getattr(dino_rer, "_ref_embeddings", {}) or {})
    cad_objects = len(shape_m._cad_embeddings) if (shape_m and shape_m._cad_embeddings) else 0
    dino_k = max(cfg.dino_top_k, ref_objects) if ref_objects else cfg.dino_top_k
    ulip_k = max(cfg.ulip2_top_k, cad_objects) if cad_objects else cfg.ulip2_top_k
    clip_rows = len(getattr(clip_retr, "_desc_labels", []) or [])
    clip_k = max(cfg.clip_top_k, clip_rows, ref_objects, 1_000_000 if clip_rows == 0 else 0)

    # ---- score every query ONCE, cache normalised per-channel maps ----
    print("[sweep] scoring queries once (run_query) + caching normalised channels ...")
    cache = []
    n = 0
    for roi, gt_label, img_path, category, fname in _make_query_factory(categories)(cfg.topk[0]):
        try:
            emb = ulip_cache.get(img_path) if ulip_cache else None
            out = run_query(pipeline_cfg, clip_retr, dino_rer, fusion_mod, shape_m,
                            roi, cfg, ulip_query_emb=emb,
                            dino_full_top_k=dino_k, ulip_full_top_k=ulip_k,
                            clip_full_top_k=clip_k)
        except Exception as exc:
            print(f"[sweep][warn] query failed ({img_path}): {exc}")
            continue
        cm = _minmax({c.object_id: float(c.score) for c in out["clip_res"].candidates})
        dm = _minmax({c.object_id: float(c.dino_score) for c in out["dino_res_full"].candidates})
        um = _minmax({c.object_id: float(c.shape_score) for c in out["shape_res_full"].candidates}) \
            if out["shape_res_full"] is not None else {}
        cache.append((cm, dm, um, gt_label, glc.get(gt_label, 0)))
        n += 1
        if n % 1000 == 0:
            print(f"[sweep]   cached {n} queries")
    print(f"[sweep] cached {len(cache)} queries; gallery {len(gallery_ids)} objs / {len(glc)} cats")

    # ---- sweep the weight simplex (cheap: normalisation already done) ----
    def eval_point(wc, wd, wu):
        acc = make_accum()
        for cm, dm, um, gt, nr in cache:
            objs = set(cm) | set(dm) | set(um)
            fused = [(o, wc * cm.get(o, 0.0) + wd * dm.get(o, 0.0) + wu * um.get(o, 0.0))
                     for o in objs]
            fused.sort(key=lambda x: x[1], reverse=True)
            update_accum(acc, fused, gt, to_category_label, cfg.TOP_F, nr)
        m = finalize_accum(acc)
        return float(m["FT_mean"]), float(m["NN_accuracy"])

    # self-check FIRST
    bft, bnn = eval_point(*BASE_W)
    print(f"[sweep] SELF-CHECK BASE {BASE_W}: FT={bft:.4f} NN={bnn:.4f} "
          f"(expected FT~{BASE_FT_EXPECTED})")
    if abs(bft - BASE_FT_EXPECTED) > 0.03:
        print(f"[sweep] ABORT: BASE FT {bft:.4f} deviates >0.03 from {BASE_FT_EXPECTED} "
              f"— fusion normalisation does not match ScoreFusion; not emitting heatmap.")
        sys.exit(4)

    rows = []
    best = (-1, None)
    for (wc, wd, wu) in _simplex(STEP):
        ft, nn = eval_point(wc, wd, wu)
        rows.append((wc, wd, wu, round(ft, 4), round(nn, 4)))
        if ft > best[0]:
            best = (ft, (wc, wd, wu))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["w_clip", "w_dino", "w_ulip", "FT", "NN"])
        w.writerows(rows)
    print(f"[sweep] wrote {len(rows)} points -> {OUT_CSV}")
    print(f"[sweep] BASE FT={bft:.4f} | optimum FT={best[0]:.4f} at "
          f"w={best[1]} (Δ={best[0]-bft:+.4f})")


if __name__ == "__main__":
    main()
