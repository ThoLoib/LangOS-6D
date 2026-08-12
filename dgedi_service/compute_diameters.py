#!/usr/bin/env python3
"""
Compute each gallery object's real diameter (in METRES) for dGeDi co-scaling.

dGeDi (per the reference demo.py) co-scales query and candidate by ONE common
divisor so their local descriptors are at a consistent physical scale. We keep
the gallery features precomputed (each candidate self-normalized by its own
diameter) and, at query time, normalize the QUERY by each *candidate's*
diameter. That needs the candidate diameter in the SAME unit as the query
cloud, which is metres (the driver back-projects depth in metres).

diameter_m = diameter(native units) * (1.0 if dataset is metric-metres else
0.001 for millimetre datasets). Fast 2-pass farthest-point estimate on the mesh
vertices (~exact extent, matches the self-normalization scale closely enough).

Run (dgedi container):
    docker compose run --rm dgedi python3 /oscar/dgedi_service/compute_diameters.py \
        --manifest /oscar/object_retrieval/.dgedi_gallery/manifest.json \
        --out      /oscar/object_retrieval/.dgedi_gallery/diameters.json
"""

import argparse
import json
import os

import numpy as np
import open3d as o3d

# Native units per dataset (mirrors stage3_gallery.DATASET_LAYOUT units_m).
UNITS_M = {"gso": True, "housecat6d": True, "itodd": False,
           "ycbv": False, "tless": False, "lmo": False}


def diameter_2pass(pts: np.ndarray) -> float:
    """Fast diameter estimate: farthest from centroid, then farthest from that."""
    c = pts.mean(0)
    a = pts[np.argmax(np.linalg.norm(pts - c, axis=1))]
    b = pts[np.argmax(np.linalg.norm(pts - a, axis=1))]
    return float(np.linalg.norm(pts[np.argmax(np.linalg.norm(pts - b, axis=1))] - b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default="/oscar")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    diam, missing = {}, 0
    for i, (nsid, rel) in enumerate(sorted(manifest.items())):
        ds = nsid.split("/")[0]
        path = os.path.join(args.repo_root, rel)
        mesh = o3d.io.read_triangle_mesh(path)
        pts = np.asarray(mesh.vertices, dtype=np.float64)
        if pts.shape[0] < 2:
            pc = o3d.io.read_point_cloud(path)
            pts = np.asarray(pc.points, dtype=np.float64)
        if pts.shape[0] < 2:
            missing += 1
            continue
        d_native = diameter_2pass(pts)
        diam[nsid] = d_native * (1.0 if UNITS_M.get(ds, False) else 0.001)  # -> metres
        if (i + 1) % 200 == 0:
            print(f"[diam] {i+1}/{len(manifest)}", flush=True)

    json.dump(diam, open(args.out, "w"))
    print(f"[diam] wrote {len(diam)} diameters ({missing} skipped) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
