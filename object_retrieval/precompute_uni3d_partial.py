#!/usr/bin/env python3
"""
Precompute the **Uni3D** partial-view gallery caches for Stage-3.

The Stage-3 shape arm normally uses ULIP-2 partial-view embeddings
(``.ulip_partial_cache_*`` with a ULIP fingerprint). Swapping to Uni3D
(``--uni3d``) needs the SAME per-object partial-view embeddings but produced by
the Uni3D encoder. The partial cache path is encoder-keyed
(step5 ``_get_partial_cache_path`` adds ``encoder=uni3d``), so these live in
their own cache files and never collide with the ULIP-2 ones.

This driver calls ``build_pipeline`` once per dataset with the Uni3D override:
its partial-view branch (eval_common) encodes every object's per-view .npz with
``encode_pointcloud`` (which dispatches to Uni3D) and writes the cache. Datasets
whose Uni3D cache already exists are a fast cache-hit (no re-encode).

Run (inside the oscar container, from object_retrieval/):
    python3 precompute_uni3d_partial.py --datasets all
Needs the Uni3D checkpoint at /uni3d/modelzoo/uni3d-g/model.pt (mounted).
"""

import argparse
import time

from eval_common import build_pipeline
from stage3_gallery import (DATASET_LAYOUT, PROXY_DATASETS, TARGET_DATASETS,
                            UNI3D_OVERRIDES, _base_cfg, _mesh_items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all",
                    help="'all' (proxies+targets), 'proxies', or comma list")
    args = ap.parse_args()

    if args.datasets == "all":
        dsets = list(TARGET_DATASETS) + list(PROXY_DATASETS)
    elif args.datasets == "proxies":
        dsets = list(PROXY_DATASETS)
    else:
        dsets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for ds in dsets:
        if ds not in DATASET_LAYOUT:
            raise SystemExit(f"unknown dataset {ds!r}")
        print(f"\n{'='*60}\n[uni3d-precompute] {ds}\n{'='*60}", flush=True)
        t0 = time.time()
        cfg = _base_cfg(ds, extra_overrides=UNI3D_OVERRIDES)
        # build_pipeline's partial branch encodes + saves the Uni3D cache on
        # miss (cache-hit is near-instant, so this is safe to re-run).
        _config, _clip, _dino, _fusion, shape_m = build_pipeline(
            cfg, cad_mesh_items=_mesh_items(ds))
        n = len(shape_m._cad_embeddings) if shape_m else 0
        print(f"[uni3d-precompute] {ds}: {n} objects in "
              f"{time.time()-t0:.1f}s", flush=True)

    print("\n[uni3d-precompute] DONE for", dsets, flush=True)


if __name__ == "__main__":
    main()
