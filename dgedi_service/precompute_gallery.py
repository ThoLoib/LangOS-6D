#!/usr/bin/env python3
"""
Precompute dGeDi descriptors for every gallery object (runs in oscar-dgedi).

For each ``{id: mesh}`` in the manifest: load the mesh, uniformly sample a
point cloud (deterministic, seeded from the id — same reproducibility rule as
``step_b2._load_cad_pointcloud``), self-normalize + extract dGeDi features, and
save ``<out>/<id>.npz`` (arrays ``points`` MxD self-normalized, ``feats`` MxD).
The service (server.py) loads these at startup for /rerank.

Run (host):
    docker compose run --rm dgedi python3 \
        /oscar/dgedi_service/precompute_gallery.py \
        --manifest /oscar/object_retrieval/.dgedi_gallery/manifest.json \
        --out      /oscar/object_retrieval/.dgedi_gallery
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import open3d as o3d

# server.py provides the model loader + self-normalizing feature extractor;
# reuse them so precompute and serving are bit-identical.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server  # noqa: E402


def sample_cloud(mesh_path: str, n_points: int) -> np.ndarray:
    """Deterministic uniform surface sample -> (n,3). Empty on failure."""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        # Some 'meshes' may already be point clouds (.ply); try that.
        pcd = o3d.io.read_point_cloud(mesh_path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        return pts
    mesh.compute_vertex_normals()
    # o3d's uniform sampler draws from the GLOBAL RNG (no seed arg) — seed from
    # the id so the cloud is a pure function of the model, not of call order.
    o3d.utility.random.seed(
        int(hashlib.sha1(os.path.basename(mesh_path).encode()).hexdigest()[:8], 16)
        % (2 ** 31 - 1))
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    return np.asarray(pcd.points, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default="/oscar",
                    help="prefix for the manifest's repo-relative mesh paths")
    ap.add_argument("--config", default="/dgedi/config_dgedi.yaml")
    ap.add_argument("--mode", default="multi_scale")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-points", type=int, default=10000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    os.makedirs(args.out, exist_ok=True)

    print(f"[precompute] loading dGeDi ({args.mode}) ...", flush=True)
    server._STATE["model"] = server.load_model(args.config, args.mode, args.device)
    server._STATE["device"] = args.device

    done = skipped = failed = 0
    for i, (nsid, rel) in enumerate(sorted(manifest.items())):
        out_path = os.path.join(args.out, server.id_to_fname(nsid))
        if os.path.isfile(out_path) and not args.overwrite:
            skipped += 1
            continue
        mesh_path = os.path.join(args.repo_root, rel)
        try:
            pts = sample_cloud(mesh_path, args.n_points)
            if pts.shape[0] < 4:
                raise ValueError(f"cloud too small ({pts.shape[0]} pts)")
            pcd, feats = server.compute_feats(pts)     # self-normalized
            np.savez_compressed(
                out_path,
                points=np.asarray(pcd.points, dtype=np.float32),
                feats=feats.astype(np.float32))
            done += 1
        except Exception as exc:
            print(f"[precompute] FAIL {nsid} ({mesh_path}): {exc}", flush=True)
            failed += 1
        if (i + 1) % 50 == 0:
            print(f"[precompute] {i+1}/{len(manifest)} "
                  f"(done={done} skip={skipped} fail={failed})", flush=True)

    print(f"[precompute] DONE: {done} written, {skipped} skipped, "
          f"{failed} failed -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
