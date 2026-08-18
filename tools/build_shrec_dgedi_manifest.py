#!/usr/bin/env python3
"""Build the dGeDi gallery manifest for the SHREC'18 Stage-1 gallery.

Emits ``{id: repo_relative_mesh_path}`` for every SHREC'18 gallery CAD, the
input format ``dgedi_service/precompute_gallery.py`` consumes. Unlike
``dgedi_service/build_manifest.py`` (BOP/proxy datasets, ids read from
``object_images/<ds>/`` render dirs), SHREC is not in ``DATASET_LAYOUT`` and its
render dirs are not local — so ids come straight from the CAD filenames, which
ARE the Stage-1 gallery ids (verified: all 3308 CAD stems == the old
``_cache/gedi_descriptors/cad/*.npz`` ids). Geometry needs no download.

Usage (host):
    python3 tools/build_shrec_dgedi_manifest.py \
        --out object_retrieval/.dgedi_gallery_shrec/manifest.json
"""
import argparse
import glob
import json
import os

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, ".."))
_CAD_REL = "eval/datasets/shrec18/shrec18_full/cad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cad-dir", default=os.path.join(_REPO, _CAD_REL),
                    help="SHREC'18 CAD directory (contains <id>.obj).")
    ap.add_argument("--out",
                    default=os.path.join(_REPO,
                                         "object_retrieval/.dgedi_gallery_shrec/manifest.json"))
    args = ap.parse_args()

    objs = sorted(glob.glob(os.path.join(args.cad_dir, "*.obj")))
    if not objs:
        raise SystemExit(f"no .obj under {args.cad_dir}")

    manifest = {}
    for p in objs:
        oid = os.path.splitext(os.path.basename(p))[0]
        rel = os.path.relpath(p, _REPO)          # repo-relative, container-portable
        manifest[oid] = rel

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=0, sort_keys=True)
    print(f"[shrec-manifest] {len(manifest)} objects -> {args.out}")


if __name__ == "__main__":
    main()
