#!/usr/bin/env python3
"""Export real GeDi matches and a thesis-ready correspondence figure."""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import math
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY_ID = "0007851779a24a969d153d32953c9a2d"
DEFAULT_CAD_ID = "7e984643df66189454e185afc91dc396"
DEFAULT_QUERY_DIR = ROOT / "eval/datasets/shrec18/stage1/queries"
DEFAULT_CAD_DIR = ROOT / "eval/datasets/shrec18/shrec18_full/cad"
DEFAULT_OUTPUT = ROOT / "object_retrieval/gedi_correspondence_figure"


def prepare_open3d_runtime() -> None:
    """Expose PyTorch's bundled OpenMP runtime to Open3D."""
    torch_lib = Path("/usr/local/lib/python3.11/dist-packages/torch/lib")
    candidates = sorted(torch_lib.glob("libgomp.so*"))
    if candidates:
        ctypes.CDLL(str(candidates[0]), mode=ctypes.RTLD_GLOBAL)


def normalize(points: np.ndarray) -> np.ndarray:
    """Match the independent unit-sphere normalization used in Stage 1."""
    points = np.asarray(points, dtype=np.float64)
    points = points - points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(points, axis=1).max(initial=0.0)
    if radius > 0:
        points = points / radius
    return np.ascontiguousarray(points, dtype=np.float32)


def load_pair(query_path: Path, cad_path: Path, seed: int):
    import open3d as o3d

    with np.load(query_path) as data:
        query = normalize(data["points"])
    if len(query) > 500_000:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(len(query), 500_000, replace=False))
        query = query[keep]

    o3d.utility.random.seed(seed)
    mesh = o3d.io.read_triangle_mesh(str(cad_path))
    if mesh.is_empty():
        raise RuntimeError(f"Could not load CAD mesh: {cad_path}")
    cad_cloud = mesh.sample_points_uniformly(number_of_points=10_000)
    cad = normalize(np.asarray(cad_cloud.points))
    return query, cad


def compute_descriptors(url: str, points: np.ndarray, keypoints: int, seed: int):
    import httpx

    encoded = base64.b64encode(points.astype(np.float32).tobytes()).decode("ascii")
    started = time.perf_counter()
    response = httpx.post(
        f"{url.rstrip('/')}/compute_descriptors",
        json={
            "points": encoded,
            "num_keypoints": min(keypoints, len(points)),
            "seed": seed,
        },
        timeout=httpx.Timeout(connect=10, read=240, write=30, pool=10),
    )
    response.raise_for_status()
    payload = response.json()
    indices = np.asarray(payload["keypoint_indices"], dtype=np.int64)
    descriptors = np.frombuffer(
        base64.b64decode(payload["descriptors"]), dtype=np.float32
    ).reshape(-1, int(payload["dim"]))
    if len(indices) != len(descriptors):
        raise RuntimeError("GeDi returned mismatched keypoint and descriptor counts")
    return {
        "indices": indices,
        "points": points[indices],
        "descriptors": descriptors,
        "server_time_s": float(payload["compute_time_s"]),
        "wall_time_s": time.perf_counter() - started,
    }


def cloud(points: np.ndarray):
    import open3d as o3d

    return o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    )


def features(descriptors: np.ndarray):
    import open3d as o3d

    result = o3d.pipelines.registration.Feature()
    result.data = np.ascontiguousarray(descriptors.T, dtype=np.float64)
    return result


def descriptor_matches(query_desc: np.ndarray, cad_desc: np.ndarray):
    """Return one-way descriptor matches and their mutual-match mask."""
    from scipy.spatial import cKDTree

    cad_tree = cKDTree(cad_desc)
    q_to_c_dist, q_to_c = cad_tree.query(query_desc, k=1, workers=-1)
    query_tree = cKDTree(query_desc)
    _, c_to_q = query_tree.query(cad_desc, k=1, workers=-1)
    query_index = np.arange(len(query_desc), dtype=np.int64)
    mutual = c_to_q[q_to_c] == query_index
    pairs = np.column_stack((query_index, q_to_c)).astype(np.int64)
    return pairs, q_to_c_dist.astype(np.float32), mutual


def run_ransac(query_result: dict, cad_result: dict, seed: int):
    import open3d as o3d

    threshold = 0.03
    o3d.utility.random.seed(seed)
    started = time.perf_counter()
    result = (
        o3d.pipelines.registration
        .registration_ransac_based_on_feature_matching(
            cloud(query_result["points"]),
            cloud(cad_result["points"]),
            features(query_result["descriptors"]),
            features(cad_result["descriptors"]),
            mutual_filter=True,
            max_correspondence_distance=threshold,
            estimation_method=(
                o3d.pipelines.registration
                .TransformationEstimationPointToPoint(False)
            ),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration
                .CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration
                .CorrespondenceCheckerBasedOnDistance(threshold),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                100_000, 0.999
            ),
        )
    )
    return result, time.perf_counter() - started, threshold


def transformed_distances(
    query_points: np.ndarray,
    cad_points: np.ndarray,
    pairs: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    source = query_points[pairs[:, 0]]
    homogeneous = np.column_stack((source, np.ones(len(source))))
    transformed = (transform @ homogeneous.T).T[:, :3]
    return np.linalg.norm(transformed - cad_points[pairs[:, 1]], axis=1)


def farthest_subset(points: np.ndarray, candidates: np.ndarray, count: int):
    """Choose a deterministic spatially spread subset of candidate indices."""
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) <= count:
        return candidates
    xyz = np.asarray(points)
    if len(xyz) != len(candidates):
        raise ValueError("points and candidates must have equal length")
    selected = [int(np.argmin(xyz[:, 0]))]
    min_dist = np.full(len(candidates), np.inf)
    for _ in range(1, count):
        last = xyz[selected[-1]]
        min_dist = np.minimum(min_dist, np.linalg.norm(xyz - last, axis=1))
        min_dist[selected] = -1
        selected.append(int(np.argmax(min_dist)))
    return candidates[np.asarray(selected)]


def mesh_reference_image(mesh_path: Path) -> np.ndarray:
    """Render a lightweight shaded reference directly from an OBJ mesh."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with mesh_path.open("r", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                indices = [
                    int(token.split("/")[0]) - 1 for token in line.split()[1:]
                ]
                for index in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[index], indices[index + 1]])

    xyz = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    centered = xyz - xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:2].T
    polygons = projected[triangles]

    normals = np.cross(
        centered[triangles[:, 1]] - centered[triangles[:, 0]],
        centered[triangles[:, 2]] - centered[triangles[:, 0]],
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.asarray([0.35, -0.45, 0.82])
    shade = 0.30 + 0.70 * np.abs(normals @ light)
    colors = plt.cm.Blues(0.08 + 0.34 * shade)

    figure, axis = plt.subplots(figsize=(5.2, 2.6), dpi=140)
    figure.patch.set_alpha(0)
    axis.set_facecolor("none")
    axis.add_collection(
        PolyCollection(
            polygons,
            facecolors=colors,
            edgecolors="#294d66",
            linewidths=0.34,
            antialiased=True,
        )
    )
    axis.scatter(
        projected[:, 0],
        projected[:, 1],
        s=0.13,
        c="#17384f",
        alpha=0.45,
        linewidths=0,
    )
    axis.autoscale_view()
    axis.set_aspect("equal")
    axis.axis("off")
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba()).copy()
    plt.close(figure)
    opaque_y, opaque_x = np.where(image[:, :, 3] > 8)
    if len(opaque_x):
        padding = 8
        x0 = max(0, int(opaque_x.min()) - padding)
        x1 = min(image.shape[1], int(opaque_x.max()) + padding + 1)
        y0 = max(0, int(opaque_y.min()) - padding)
        y1 = min(image.shape[0], int(opaque_y.max()) + padding + 1)
        image = image[y0:y1, x0:x1]
    return image


def load_shrec_query_ply(ply_path: Path):
    """Read the binary SHREC PLY fields used to build the Stage-1 crop."""
    with ply_path.open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Incomplete PLY header: {ply_path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        vertex_count = int(
            next(line.split()[2] for line in header_lines
                 if line.startswith("element vertex "))
        )
        face_count = int(
            next(line.split()[2] for line in header_lines
                 if line.startswith("element face "))
        )
        vertex_dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        records = np.fromfile(handle, dtype=vertex_dtype, count=vertex_count)
        faces: list[list[int]] = []
        for _ in range(face_count):
            count = int(np.fromfile(handle, dtype="u1", count=1)[0])
            indices = np.fromfile(handle, dtype="<i4", count=count).tolist()
            for index in range(1, count - 1):
                faces.append([indices[0], indices[index], indices[index + 1]])

    points = np.column_stack((records["x"], records["y"], records["z"]))
    normals = np.column_stack((records["nx"], records["ny"], records["nz"]))
    return points.astype(np.float64), normals.astype(np.float64), np.asarray(
        faces, dtype=np.int64
    )


def project_query_to_saved_crop(
    normalized_points: np.ndarray,
    normalized_keypoints: np.ndarray,
    ply_path: Path,
    image_shape: tuple[int, int],
):
    """Reproduce Stage-1's mean-normal projection and content crop exactly."""
    raw_points, normals, _ = load_shrec_query_ply(ply_path)
    view = normals.mean(axis=0)
    if np.linalg.norm(view) < 1e-6:
        centered = raw_points - raw_points.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        view = vt[-1]
    view /= np.linalg.norm(view)
    up = np.asarray([0.0, 0.0, 1.0])
    if abs(np.dot(up, view)) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0])
    horizontal = np.cross(up, view)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(view, horizontal)

    size = 448
    padding = size // 12
    draw = size - 2 * padding
    raw_x = raw_points @ horizontal
    raw_y = raw_points @ vertical
    span = max(np.ptp(raw_x), np.ptp(raw_y), 1e-9)

    def full_pixels(raw):
        x = raw @ horizontal
        y = raw @ vertical
        px = (x - raw_x.min()) / span * draw + padding
        py = size - 1 - ((y - raw_y.min()) / span * draw + padding)
        return np.column_stack((px, py))

    # _render_query_surface adds 200k interior samples but retains every raw
    # vertex. Interior samples cannot extend the projected extrema, so the
    # crop bounds are determined by these vertices plus the one-pixel splat.
    vertex_pixels = full_pixels(raw_points)
    splat_radius = 1
    crop_margin = 8
    crop_x0 = max(
        int(np.floor(vertex_pixels[:, 0].min())) - splat_radius - crop_margin,
        0,
    )
    crop_x1 = min(
        int(np.ceil(vertex_pixels[:, 0].max())) + splat_radius + crop_margin + 1,
        size,
    )
    crop_y0 = max(
        int(np.floor(vertex_pixels[:, 1].min())) - splat_radius - crop_margin,
        0,
    )
    crop_y1 = min(
        int(np.ceil(vertex_pixels[:, 1].max())) + splat_radius + crop_margin + 1,
        size,
    )

    raw_center = raw_points.mean(axis=0)
    raw_radius = np.linalg.norm(raw_points - raw_center, axis=1).max()
    query_raw = normalized_points * raw_radius + raw_center
    keypoint_raw = normalized_keypoints * raw_radius + raw_center
    query_pixels = full_pixels(query_raw) - np.asarray([crop_x0, crop_y0])
    keypoint_pixels = full_pixels(keypoint_raw) - np.asarray([crop_x0, crop_y0])

    expected_height = crop_y1 - crop_y0
    expected_width = crop_x1 - crop_x0
    actual_height, actual_width = image_shape
    query_pixels[:, 0] *= actual_width / max(expected_width, 1)
    query_pixels[:, 1] *= actual_height / max(expected_height, 1)
    keypoint_pixels[:, 0] *= actual_width / max(expected_width, 1)
    keypoint_pixels[:, 1] *= actual_height / max(expected_height, 1)
    return query_pixels, keypoint_pixels


def load_obj_mesh(mesh_path: Path):
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with mesh_path.open("r", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                indices = [
                    int(token.split("/")[0]) - 1 for token in line.split()[1:]
                ]
                for index in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[index], indices[index + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def render_figure(
    path_pdf: Path,
    path_png: Path,
    query_all: np.ndarray,
    cad_all: np.ndarray,
    query_keypoints: np.ndarray,
    cad_keypoints: np.ndarray,
    pairs: np.ndarray,
    inlier_mask: np.ndarray,
    tentative_count: int,
    inlier_count: int,
    fitness: float,
    seed: int,
    query_image_path: Path | None = None,
    cad_mesh_path: Path | None = None,
    query_ply_path: Path | None = None,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    def project_cad_panel(all_points, keypoints):
        center = all_points.mean(axis=0)
        _, _, vt = np.linalg.svd(all_points - center, full_matrices=False)
        basis = vt[:2].T
        all_projected = (all_points - center) @ basis
        keypoints_projected = (keypoints - center) @ basis
        panel_center = np.median(all_projected, axis=0)
        panel_span = np.ptp(
            all_projected - panel_center, axis=0
        ).max(initial=1.0)
        scale = max(panel_span, 1e-9)
        return (
            (all_projected - panel_center) / scale,
            (keypoints_projected - panel_center) / scale,
            basis,
        )

    c_all_2d, c_kp_2d, cad_basis = project_cad_panel(
        cad_all, cad_keypoints
    )
    c_all_2d[:, 0] += 0.72
    c_kp_2d[:, 0] += 0.72

    inlier_candidates = np.flatnonzero(inlier_mask)
    outlier_candidates = np.flatnonzero(~inlier_mask)
    show_inliers = farthest_subset(
        query_keypoints[pairs[inlier_candidates, 0]],
        inlier_candidates,
        28,
    )
    show_outliers = farthest_subset(
        query_keypoints[pairs[outlier_candidates, 0]],
        outlier_candidates,
        12,
    )

    fig, ax = plt.subplots(figsize=(8.1, 3.15), constrained_layout=True)
    if (
        query_image_path
        and query_image_path.is_file()
        and query_ply_path
        and query_ply_path.is_file()
    ):
        query_reference = plt.imread(query_image_path)
        query_reference = np.clip(
            (query_reference - 0.5) * 1.20 + 0.5, 0.0, 1.0
        )
        height, width = query_reference.shape[:2]
        query_extent = (-1.34, -0.10, -0.342, 0.342)
        q_pixels, q_kp_pixels = project_query_to_saved_crop(
            query_all,
            query_keypoints,
            query_ply_path,
            (height, width),
        )

        def pixels_to_panel(pixel_points):
            result = np.empty_like(pixel_points, dtype=np.float64)
            result[:, 0] = (
                query_extent[0]
                + pixel_points[:, 0] / max(width - 1, 1)
                * (query_extent[1] - query_extent[0])
            )
            result[:, 1] = (
                query_extent[3]
                - pixel_points[:, 1] / max(height - 1, 1)
                * (query_extent[3] - query_extent[2])
            )
            return result

        q_all_2d = pixels_to_panel(q_pixels)
        q_kp_2d = pixels_to_panel(q_kp_pixels)
        ax.imshow(
            query_reference,
            extent=query_extent,
            alpha=0.62,
            interpolation="bilinear",
            zorder=0,
        )
    else:
        centered = query_all - query_all.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:2].T
        q_all_2d = centered @ basis
        q_kp_2d = (
            query_keypoints - query_all.mean(axis=0)
        ) @ basis
        center = np.median(q_all_2d, axis=0)
        scale = max(np.ptp(q_all_2d - center, axis=0).max(), 1e-9)
        q_all_2d = (q_all_2d - center) / scale
        q_kp_2d = (q_kp_2d - center) / scale
        q_all_2d[:, 0] -= 0.72
        q_kp_2d[:, 0] -= 0.72

    if cad_mesh_path and cad_mesh_path.is_file():
        from matplotlib.collections import PolyCollection

        mesh_vertices, mesh_faces = load_obj_mesh(cad_mesh_path)
        mesh_projected = mesh_vertices @ cad_basis
        mesh_min = mesh_projected.min(axis=0)
        mesh_range = np.maximum(
            mesh_projected.max(axis=0) - mesh_min, 1e-9
        )
        cad_min = c_all_2d.min(axis=0)
        cad_range = np.maximum(c_all_2d.max(axis=0) - cad_min, 1e-9)
        mesh_projected = (
            (mesh_projected - mesh_min) / mesh_range * cad_range + cad_min
        )
        mesh_polygons = mesh_projected[mesh_faces]
        mesh_normals = np.cross(
            mesh_vertices[mesh_faces[:, 1]] - mesh_vertices[mesh_faces[:, 0]],
            mesh_vertices[mesh_faces[:, 2]] - mesh_vertices[mesh_faces[:, 0]],
        )
        mesh_normals /= np.maximum(
            np.linalg.norm(mesh_normals, axis=1, keepdims=True), 1e-12
        )
        shade = 0.30 + 0.70 * np.abs(
            mesh_normals @ np.asarray([0.35, -0.45, 0.82])
        )
        ax.add_collection(
            PolyCollection(
                mesh_polygons,
                facecolors=plt.cm.Blues(0.06 + 0.26 * shade),
                edgecolors="#315b73",
                linewidths=0.24,
                alpha=0.58,
                zorder=0,
            )
        )
    ax.scatter(
        q_all_2d[:, 0], q_all_2d[:, 1], s=1.45, c="#5f6872",
        alpha=0.42, linewidths=0, rasterized=True, zorder=1,
    )
    ax.scatter(
        c_all_2d[:, 0], c_all_2d[:, 1], s=1.35, c="#5f6872",
        alpha=0.38, linewidths=0, rasterized=True, zorder=1,
    )

    def segments(which):
        selected_pairs = pairs[which]
        return np.stack(
            (
                q_kp_2d[selected_pairs[:, 0]],
                c_kp_2d[selected_pairs[:, 1]],
            ),
            axis=1,
        )

    if len(show_outliers):
        ax.add_collection(
            LineCollection(
                segments(show_outliers), colors="#d55e5e", linewidths=0.6,
                alpha=0.48, zorder=2,
            )
        )
    if len(show_inliers):
        ax.add_collection(
            LineCollection(
                segments(show_inliers), colors="#168a5b", linewidths=0.85,
                alpha=0.78, zorder=3,
            )
        )

    for which, color in ((show_outliers, "#d55e5e"), (show_inliers, "#168a5b")):
        if len(which):
            selected_pairs = pairs[which]
            endpoints = np.vstack(
                (
                    q_kp_2d[selected_pairs[:, 0]],
                    c_kp_2d[selected_pairs[:, 1]],
                )
            )
            ax.scatter(
                endpoints[:, 0], endpoints[:, 1], s=8, c=color,
                edgecolors="white", linewidths=0.25, zorder=4,
            )

    ax.text(
        -0.72, 0.405, "Observed partial point cloud",
        ha="center", va="bottom", fontsize=10,
    )
    ax.text(
        0.72, 0.405, "CAD reference point cloud",
        ha="center", va="bottom", fontsize=10,
    )
    ax.text(
        0.0, -0.465, "Tentative local-descriptor correspondences",
        ha="center", va="top", fontsize=8.5,
    )
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.54, 0.50)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "shown_inliers": int(len(show_inliers)),
        "shown_outliers": int(len(show_outliers)),
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-id", default=DEFAULT_QUERY_ID)
    parser.add_argument("--cad-id", default=DEFAULT_CAD_ID)
    parser.add_argument("--query-dir", type=Path, default=DEFAULT_QUERY_DIR)
    parser.add_argument("--cad-dir", type=Path, default=DEFAULT_CAD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gedi-url", default="http://gedi:5060")
    parser.add_argument("--keypoints", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Rebuild PDF/PNG from the saved NPZ without rerunning GeDi.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_path = args.query_dir / f"{args.query_id}.npz"
    query_image_path = args.query_dir / f"{args.query_id}.png"
    cad_path = args.cad_dir / f"{args.cad_id}.obj"
    stem = "gedi_correspondences_keyboard"

    if args.render_only:
        data_path = args.output_dir / f"{stem}.npz"
        metadata = json.loads(
            (args.output_dir / f"{stem}.json").read_text()
        )
        with np.load(data_path) as data:
            tentative_pairs = data["tentative_pairs"]
            official = data["ransac_correspondences"]
            official_set = {tuple(pair) for pair in official.tolist()}
            official_in_tentative = np.asarray(
                [
                    tuple(pair) in official_set
                    for pair in tentative_pairs.tolist()
                ],
                dtype=bool,
            )
            figure_pairs = np.vstack(
                (official, tentative_pairs[~official_in_tentative])
            )
            figure_inlier_mask = np.zeros(len(figure_pairs), dtype=bool)
            figure_inlier_mask[:len(official)] = True
            render_figure(
                args.output_dir / f"{stem}.pdf",
                args.output_dir / f"{stem}.png",
                data["query_points"],
                data["cad_points"],
                data["query_keypoints"],
                data["cad_keypoints"],
                figure_pairs,
                figure_inlier_mask,
                len(tentative_pairs),
                len(official),
                float(metadata["ransac_fitness"]),
                int(metadata["seed"]),
                query_image_path,
                cad_path,
                ROOT / "eval/datasets/shrec18/shrec18_full/rgbd" / f"{args.query_id}.ply",
            )
        print(f"Re-rendered {args.output_dir / (stem + '.pdf')}")
        return

    prepare_open3d_runtime()
    import open3d as o3d

    query, cad = load_pair(query_path, cad_path, args.seed)

    print(f"Computing query descriptors for {len(query):,} points...", flush=True)
    query_result = compute_descriptors(
        args.gedi_url, query, args.keypoints, args.seed
    )
    print(f"Computing CAD descriptors for {len(cad):,} points...", flush=True)
    cad_result = compute_descriptors(
        args.gedi_url, cad, args.keypoints, args.seed + 1
    )

    pairs, descriptor_distances, mutual_match_mask = descriptor_matches(
        query_result["descriptors"], cad_result["descriptors"]
    )
    print(
        f"Running OSCAR RANSAC on {len(pairs):,} descriptor matches "
        f"({int(mutual_match_mask.sum()):,} mutual)...",
        flush=True,
    )
    ransac, ransac_time, threshold = run_ransac(
        query_result, cad_result, args.seed
    )
    pair_geometry_distances = transformed_distances(
        query_result["points"],
        cad_result["points"],
        pairs,
        np.asarray(ransac.transformation),
    )
    distance_inlier_mask = pair_geometry_distances <= threshold
    official_correspondences = np.asarray(
        ransac.correspondence_set, dtype=np.int64
    ).reshape(-1, 2)
    official_set = {tuple(pair) for pair in official_correspondences.tolist()}
    official_in_tentative = np.asarray(
        [tuple(pair) in official_set for pair in pairs.tolist()], dtype=bool
    )
    figure_pairs = np.vstack(
        (official_correspondences, pairs[~official_in_tentative])
    )
    figure_inlier_mask = np.zeros(len(figure_pairs), dtype=bool)
    figure_inlier_mask[:len(official_correspondences)] = True

    stem = "gedi_correspondences_keyboard"
    data_path = args.output_dir / f"{stem}.npz"
    np.savez_compressed(
        data_path,
        query_points=query,
        cad_points=cad,
        query_keypoint_indices=query_result["indices"],
        cad_keypoint_indices=cad_result["indices"],
        query_keypoints=query_result["points"],
        cad_keypoints=cad_result["points"],
        query_descriptors=query_result["descriptors"],
        cad_descriptors=cad_result["descriptors"],
        tentative_pairs=pairs,
        descriptor_distances=descriptor_distances,
        mutual_match_mask=mutual_match_mask,
        geometric_distances=pair_geometry_distances.astype(np.float32),
        geometric_inlier_mask=distance_inlier_mask,
        ransac_correspondences=official_correspondences,
        ransac_inlier_mask_on_tentative=official_in_tentative,
        transformation=np.asarray(ransac.transformation),
    )

    figure_info = render_figure(
        args.output_dir / f"{stem}.pdf",
        args.output_dir / f"{stem}.png",
        query,
        cad,
        query_result["points"],
        cad_result["points"],
        figure_pairs,
        figure_inlier_mask,
        len(pairs),
        len(official_correspondences),
        float(ransac.fitness),
        args.seed,
        args.query_dir / f"{args.query_id}.png",
        cad_path,
        ROOT / "eval/datasets/shrec18/shrec18_full/rgbd" / f"{args.query_id}.ply",
    )
    metadata = {
        "query_id": args.query_id,
        "cad_id": args.cad_id,
        "query_category": "keyboard",
        "cad_category": "keyboard",
        "cached_pair_selection_fitness": 0.811078140454995,
        "seed": args.seed,
        "descriptor_dimension": int(query_result["descriptors"].shape[1]),
        "query_points": int(len(query)),
        "cad_points": int(len(cad)),
        "query_keypoints": int(len(query_result["points"])),
        "cad_keypoints": int(len(cad_result["points"])),
        "tentative_descriptor_matches": int(len(pairs)),
        "mutual_descriptor_matches": int(mutual_match_mask.sum()),
        "geometric_inliers_at_threshold": int(distance_inlier_mask.sum()),
        "ransac_correspondences": int(len(official_correspondences)),
        "open3d_used_non_mutual_fallback": bool(mutual_match_mask.sum() < 100),
        "ransac_fitness": float(ransac.fitness),
        "ransac_inlier_rmse": (
            None if not math.isfinite(ransac.inlier_rmse)
            else float(ransac.inlier_rmse)
        ),
        "ransac_threshold": threshold,
        "ransac_iterations": 100_000,
        "ransac_confidence": 0.999,
        "ransac_time_s": ransac_time,
        "query_descriptor_server_time_s": query_result["server_time_s"],
        "cad_descriptor_server_time_s": cad_result["server_time_s"],
        **figure_info,
        "note": (
            "The cached fitness selected this pair. All descriptors, matches, "
            "inlier labels, and the transformation in this export come from "
            "the seeded rerun recorded here."
        ),
    }
    metadata_path = args.output_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Saved {data_path}")


if __name__ == "__main__":
    main()
