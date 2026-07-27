#!/usr/bin/env python3
"""Re-render the SHREC'18 query PNG crops in place using the current
prepare_queries renderer (solid surface), leaving the .npz point clouds
untouched.  Use after changing the query renderer.

  python3 tools/regen_query_pngs.py \
      --stage1 eval/datasets/shrec18/stage1 \
      --rgbd   eval/datasets/shrec18/shrec18_full/rgbd
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import torch  # noqa: F401  (libgomp must load before open3d)
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
import experiment1_shrec18_stage1 as E

ap = argparse.ArgumentParser()
ap.add_argument("--stage1", required=True)
ap.add_argument("--rgbd", required=True)
ap.add_argument("--size", type=int, default=448)
args = ap.parse_args()

idx_path = os.path.join(args.stage1, "gt", "queries_index.json")
with open(idx_path) as f:
    index = json.load(f)
print(f"[regen] {len(index)} queries from {idx_path}")

n_ok = n_splat = 0
for q in tqdm(index, desc="re-render PNG", unit="q"):
    qid = q["id"]
    ply = os.path.join(args.rgbd, f"{qid}.ply")
    png = q["png"]
    m = o3d.io.read_triangle_mesh(ply)
    pts = np.asarray(m.vertices, dtype=np.float32)
    if len(pts) == 0:
        continue
    if not m.has_vertex_normals():
        try: m.compute_vertex_normals()
        except Exception: pass
    nrm = np.asarray(m.vertex_normals, dtype=np.float32) if m.has_vertex_normals() else None
    n, u, v = E._view_basis(pts, nrm)
    img = None
    if m.has_triangles():
        img = E._render_query_surface(m, n, u, v, args.size)
    if img is not None:
        n_ok += 1
    else:
        cols = (np.asarray(m.vertex_colors, dtype=np.float32)
                if m.has_vertex_colors() else np.full((len(pts),3),0.5,np.float32))
        img = E._render_query_splat(pts, cols, n, u, v, args.size)
        n_splat += 1
    Image.fromarray(E._crop_to_content(img)).save(png)

print(f"[regen] done: {n_ok} surface renders, {n_splat} splat fallbacks")
