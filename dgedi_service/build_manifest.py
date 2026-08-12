#!/usr/bin/env python3
"""
Build the gallery manifest for dGeDi descriptor precompute.

Emits ``{namespaced_id: repo_relative_mesh_path}`` for every gallery object of
the requested datasets, WITHOUT constructing the retrieval pipeline (no
encoders / no GPU) — it enumerates the render dirs under
``object_images/<ds>/`` (which are exactly the gallery ids that
``stage3_gallery._absorb_dataset`` uses) and resolves each to its native-scale
pose mesh via ``stage3_gallery._pose_mesh_path``.

Paths are stored **relative to the OSCAR repo root** so the manifest is
container-portable (precompute prepends ``--repo-root``, default ``/oscar``).

Usage (host):
    python3 dgedi_service/build_manifest.py --datasets all \
        --out object_retrieval/.dgedi_gallery/manifest.json
"""

import argparse
import json
import os

# Import layout + resolver from the Stage-3 gallery module.
import sys
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, "..", "object_retrieval"))
from stage3_gallery import (DATASET_LAYOUT, PROXY_DATASETS, TARGET_DATASETS,  # noqa: E402
                            namespaced_id, _pose_mesh_path)

_REPO = os.path.abspath(os.path.join(_THIS, ".."))


def gallery_ids(ds: str):
    """Render-dir names under object_images/<ds> = the gallery obj ids."""
    ref = os.path.join(_REPO, "object_images", ds)
    if not os.path.isdir(ref):
        return []
    return sorted(d for d in os.listdir(ref)
                  if os.path.isdir(os.path.join(ref, d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all",
                    help="'all' (proxies+targets), 'proxies', or comma list")
    ap.add_argument("--out", default="object_retrieval/.dgedi_gallery/manifest.json")
    args = ap.parse_args()

    if args.datasets == "all":
        dsets = list(TARGET_DATASETS) + list(PROXY_DATASETS)
    elif args.datasets == "proxies":
        dsets = list(PROXY_DATASETS)
    else:
        dsets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    manifest, missing = {}, 0
    for ds in dsets:
        if ds not in DATASET_LAYOUT:
            raise SystemExit(f"unknown dataset {ds!r}")
        for oid in gallery_ids(ds):
            path, _units = _pose_mesh_path(ds, oid)   # absolute host path
            if not os.path.isfile(path):
                missing += 1
                continue
            rel = os.path.relpath(path, _REPO)
            manifest[namespaced_id(ds, oid)] = rel

    out = os.path.join(_REPO, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=0)
    print(f"[manifest] {len(manifest)} objects across {dsets} "
          f"({missing} meshes unresolved) -> {out}")


if __name__ == "__main__":
    main()
