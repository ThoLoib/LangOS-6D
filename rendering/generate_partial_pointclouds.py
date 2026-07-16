#!/usr/bin/env python3
# =============================================================================
# rendering/generate_partial_pointclouds.py
# =============================================================================
#
# Offline preprocessing: generates partial point clouds from CAD meshes
# using front-face culling from the same 8 viewpoints used in rendering.py.
#
# Each view produces one .npz file with keys "points" (N,3) and "colors" (N,3),
# stored alongside existing rendered PNGs and camera matrices.
#
# No Blender needed — uses trimesh for surface sampling + visibility filtering.
#
# Usage:
#   python rendering/generate_partial_pointclouds.py \
#       --cad_dir object_database/ycbv_gso/ \
#       --images_dir object_images/ycbv_gso/ \
#       --num_points 10000
# =============================================================================

import argparse
import logging
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def normalize_mesh(mesh) -> None:
    """Normalize mesh in-place: center on bbox center, scale max dimension to 1.0.

    Replicates the normalization in rendering.py (normalize_and_center_objects
    with normalization_range=1.0).
    """
    bounds = mesh.bounds  # (2, 3): min, max
    center = (bounds[0] + bounds[1]) / 2.0
    max_dim = (bounds[1] - bounds[0]).max()
    if max_dim <= 0:
        return
    mesh.vertices -= center
    mesh.vertices /= max_dim


def load_camera_matrix(cam_matrix_path: str) -> np.ndarray:
    """Load a 3x4 RT matrix saved by rendering.py."""
    RT = np.load(cam_matrix_path)
    if RT.shape != (3, 4):
        raise ValueError(f"Expected (3,4) matrix, got {RT.shape} from {cam_matrix_path}")
    return RT


def sample_visible_surface(
    mesh, cam_pos: np.ndarray, num_points: int,
    oversample_factor: int = 5, with_colors: bool = True,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Sample points visible from a camera position using front-face culling.

    Samples points uniformly on the mesh surface, then keeps only those whose
    face normal points towards the camera (dot(normal, view_dir) > 0). This is
    a fast approximation of visibility that handles most cases well for convex
    and mildly concave objects.

    Args:
        mesh: trimesh.Trimesh object (normalized).
        cam_pos: (3,) camera position in world coordinates.
        num_points: desired number of output points.
        oversample_factor: sample this many more points than needed, then filter.
        with_colors: extract face colors if available.

    Returns:
        (points, colors) or None if too few visible points (<100).
    """
    import trimesh as _trimesh

    n_sample = num_points * oversample_factor
    points, face_indices = _trimesh.sample.sample_surface(mesh, n_sample)

    # Front-face visibility: keep points whose face normal points toward camera
    normals = mesh.face_normals[face_indices]
    view_dirs = cam_pos - points
    view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8
    dots = np.sum(normals * view_dirs, axis=1)
    visible = dots > 0

    vis_points = points[visible].astype(np.float32)
    vis_faces = face_indices[visible]

    if len(vis_points) < 100:
        return None

    # Extract colors
    colors = None
    if with_colors:
        try:
            face_colors = mesh.visual.face_colors[vis_faces][:, :3]
            colors = (face_colors / 255.0).astype(np.float32)
        except (AttributeError, IndexError):
            pass

    # Resample to exact target count (deterministic seed for reproducibility)
    content_hash = hash(vis_points.tobytes()) & 0xFFFFFFFF
    rng = np.random.RandomState(content_hash)
    n = len(vis_points)
    if n >= num_points:
        indices = rng.choice(n, num_points, replace=False)
    else:
        indices = rng.choice(n, num_points, replace=True)

    vis_points = vis_points[indices]
    if colors is not None:
        colors = colors[indices]

    return vis_points, colors


def _discover_view_indices(obj_images_dir: str, obj_id: str) -> list:
    """Discover all available view indices by finding CamMatrix .npy files.

    Supports any number of views (ablation O4: V in {8, 16, 32}).
    Returns sorted list of integer view indices.
    """
    import re
    indices = []
    pattern = re.compile(rf"^{re.escape(obj_id)}_view(\d+)_CamMatrix\.npy$")
    for fname in os.listdir(obj_images_dir):
        m = pattern.match(fname)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def process_object(obj_id: str, cad_dir: str, images_dir: str,
                   num_points: int, overwrite: bool = False,
                   mesh_path: Optional[str] = None) -> int:
    """Generate partial point clouds for all views of one object.

    Auto-discovers all available camera matrices (view0, view1, ..., viewN)
    so that the same script works for V=8, 16, or 32 rendered views
    (thesis ablation O4).

    Uses front-face culling to approximate visibility from each camera viewpoint.

    Args:
        mesh_path: Explicit path to the mesh file.  When provided, skips the
                   automatic mesh discovery in *cad_dir*.

    Returns:
        Number of views successfully processed.
    """
    import trimesh

    obj_images_dir = os.path.join(images_dir, obj_id)
    if not os.path.isdir(obj_images_dir):
        logger.warning("No images directory for %s at %s", obj_id, obj_images_dir)
        return 0

    # Find mesh file
    if mesh_path is None:
        obj_cad_dir = os.path.join(cad_dir, obj_id)
        mesh_path = _find_mesh(obj_cad_dir if os.path.isdir(obj_cad_dir) else cad_dir, obj_id)
    if not mesh_path:
        logger.warning("No mesh found for %s", obj_id)
        return 0

    # Load and normalize mesh
    mesh = trimesh.load(mesh_path, force="mesh")
    # Convert texture-based visuals to per-face colors for sampling
    if hasattr(mesh.visual, 'to_color'):
        try:
            mesh.visual = mesh.visual.to_color()
        except Exception:
            pass
    normalize_mesh(mesh)

    # Auto-discover all available views from camera matrices
    view_indices = _discover_view_indices(obj_images_dir, obj_id)
    if not view_indices:
        logger.warning("No camera matrices found for %s", obj_id)
        return 0

    count = 0
    for view_idx in view_indices:
        out_path = os.path.join(obj_images_dir, f"{obj_id}_view{view_idx}_partial.npz")
        if os.path.isfile(out_path) and not overwrite:
            count += 1
            continue

        cam_path = os.path.join(obj_images_dir, f"{obj_id}_view{view_idx}_CamMatrix.npy")
        RT = load_camera_matrix(cam_path)
        R = RT[:3, :3]
        t = RT[:3, 3]
        cam_pos = -R.T @ t  # Camera position in world coords

        result = sample_visible_surface(mesh, cam_pos, num_points)
        if result is None:
            logger.debug("Too few visible points for %s view %d, skipping", obj_id, view_idx)
            continue

        points, colors = result
        save_dict = {"points": points}
        if colors is not None:
            save_dict["colors"] = colors

        np.savez_compressed(out_path, **save_dict)
        count += 1

    return count


def _find_mesh(search_dir: str, obj_id: str) -> Optional[str]:
    """Find a mesh file for an object, mirroring step5's logic."""
    if not os.path.isdir(search_dir):
        # Maybe the mesh is a direct file in cad_dir
        for ext in (".obj", ".ply", ".glb", ".gltf"):
            candidate = search_dir + ext
            if os.path.isfile(candidate):
                return candidate
        return None

    preferred = ("textured_simple.obj", "model.obj", "mesh.obj")
    allowed = {".obj", ".ply", ".glb", ".gltf"}
    candidates = []

    for root, _, files in os.walk(search_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in allowed:
                candidates.append(os.path.join(root, fname))

    if not candidates:
        return None

    def sort_key(path):
        base = os.path.basename(path).lower()
        in_meshes = 0 if os.path.basename(os.path.dirname(path)).lower() == "meshes" else 1
        try:
            pref_idx = preferred.index(base)
        except ValueError:
            pref_idx = len(preferred)
        return (in_meshes, pref_idx, path)

    return sorted(candidates, key=sort_key)[0]


def _build_mesh_map_from_glob(pattern: str):
    """Build {obj_id: mesh_path} from a glob pattern.

    The obj_id is the file stem (basename without extension).
    Use this for datasets where the mesh filename is the object id
    (e.g. MI3DOR: ``model/test/airplane/airplane_test_0001.obj``).
    """
    import glob as _g
    paths = sorted(_g.glob(pattern))
    mapping = {}
    for p in paths:
        obj_id = os.path.splitext(os.path.basename(p))[0]
        mapping[obj_id] = p
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Generate partial point clouds from CAD meshes via front-face culling"
    )
    parser.add_argument("--cad_dir", default="",
                        help="Path to CAD models directory (e.g. object_database/ycbv_gso/)")
    parser.add_argument("--images_dir", required=True,
                        help="Path to rendered images directory (e.g. object_images/ycbv_gso/)")
    parser.add_argument("--mesh-glob", default="",
                        help="Glob pattern to find meshes directly. "
                             "obj_id = file stem.  Use when CAD layout doesn't "
                             "match <cad_dir>/<obj_id>/ convention.  "
                             "E.g.: 'object_database/MI3DOR/model/test/*/*.obj'")
    parser.add_argument("--num_points", type=int, default=10000,
                        help="Points per partial PC (default: 10000, matches ULIP-2)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing .npz files")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not os.path.isdir(args.images_dir):
        logger.error("Images directory not found: %s", args.images_dir)
        sys.exit(1)

    # Build obj_id -> mesh_path mapping
    mesh_map = {}
    if args.mesh_glob:
        mesh_map = _build_mesh_map_from_glob(args.mesh_glob)
        if not mesh_map:
            logger.error("--mesh-glob matched 0 files: %s", args.mesh_glob)
            sys.exit(1)
        logger.info("Mesh glob matched %d meshes", len(mesh_map))
    elif args.cad_dir:
        if not os.path.isdir(args.cad_dir):
            logger.error("CAD directory not found: %s", args.cad_dir)
            sys.exit(1)
    else:
        logger.error("Provide either --cad_dir or --mesh-glob")
        sys.exit(1)

    # Discover objects from the images directory
    obj_ids = sorted([
        d for d in os.listdir(args.images_dir)
        if os.path.isdir(os.path.join(args.images_dir, d))
    ])
    logger.info("Found %d objects in %s", len(obj_ids), args.images_dir)

    t0 = time.time()
    total_views = 0
    success_objects = 0
    missing_meshes = []

    for i, obj_id in enumerate(obj_ids):
        explicit_mesh = mesh_map.get(obj_id)
        if mesh_map and explicit_mesh is None:
            missing_meshes.append(obj_id)
            continue

        views = process_object(
            obj_id, args.cad_dir, args.images_dir,
            args.num_points, overwrite=args.overwrite,
            mesh_path=explicit_mesh,
        )
        if views > 0:
            success_objects += 1
            total_views += views
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info(
                "Progress: %d/%d objects (%.1fs elapsed, %d views generated)",
                i + 1, len(obj_ids), elapsed, total_views,
            )

    elapsed = time.time() - t0
    logger.info(
        "Done: %d objects, %d views in %.1fs",
        success_objects, total_views, elapsed,
    )
    if missing_meshes:
        logger.warning(
            "%d objects in images_dir have no matching mesh: %s",
            len(missing_meshes),
            missing_meshes[:10] if len(missing_meshes) > 10
            else missing_meshes,
        )


if __name__ == "__main__":
    main()
