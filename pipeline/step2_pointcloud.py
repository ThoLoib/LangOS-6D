# =============================================================================
# pipeline/step2_pointcloud.py – Schritt 2: Punktwolke erzeugen
# =============================================================================
#
# Ziel:
#   Aus dem Tiefenbild (Depth Map), der Segmentierungsmaske (Schritt 1) und
#   den Kameraintrinsics eine partielle 3D-Punktwolke des Zielobjekts
#   erzeugen.
#
# Pipeline:
#   RGB-D + Kamera-Intrinsics → segmentierte Tiefe → Punktwolke des Objekts
#
# Tools:
#   • Open3D – 3D-Datenverarbeitung und Punktwolken
#     Ref: http://www.open3d.org/docs/release/
#     Paper: "Open3D: A Modern Library for 3D Data Processing"
#             (Zhou, Park & Koltun, 2018)
#
# Outputs:
#   - Partielle Punktwolke des segmentierten Objekts (Open3D PointCloud)
#   - Optional: Downsampled Version für effiziente Weiterverarbeitung
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenstruktur für Punktwolken-Ergebnisse
# ---------------------------------------------------------------------------

@dataclass
class PointCloudResult:
    """Ergebnis der Punktwolkenerzeugung (Schritt 2).

    Attributes:
        point_cloud: Open3D PointCloud-Objekt (mit Farben).
        points: Numpy-Array der 3D-Punkte (N, 3).
        colors: Numpy-Array der Farben (N, 3), normalisiert auf [0, 1].
        num_points: Anzahl der Punkte.
        bbox_min: Minimale Ecke der 3D-Bounding-Box.
        bbox_max: Maximale Ecke der 3D-Bounding-Box.
        bbox_size: Größe der 3D-Bounding-Box (Breite, Höhe, Tiefe).
    """
    point_cloud: object  # open3d.geometry.PointCloud (vermeidet Import-Pflicht)
    points: np.ndarray
    colors: np.ndarray
    num_points: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    bbox_size: np.ndarray


# ---------------------------------------------------------------------------
# Punktwolken-Generator
# ---------------------------------------------------------------------------

class PointCloudGenerator:
    """Erzeugt 3D-Punktwolken aus RGB-D-Daten und Segmentierungsmasken.

    Verwendet Open3D für die Rückprojektion von Tiefenpixeln in den
    3D-Raum unter Berücksichtigung der Kameraintrinsics.

    The caller is responsible for converting the depth image to float32
    meters before passing it to generate(). No internal heuristic is applied.

    Usage:
        >>> gen = PointCloudGenerator(config)
        >>> result = gen.generate(rgb, depth, mask)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._check_open3d()

    @staticmethod
    def _check_open3d():
        """Prüft ob Open3D installiert ist."""
        try:
            import open3d as o3d  # noqa: F401
        except ImportError:
            raise ImportError(
                "Open3D nicht installiert. Installieren mit:\n"
                "  pip install open3d\n"
                "Ref: http://www.open3d.org/docs/release/"
            )

    def _gate_depth(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove depth outliers within the mask using median-relative gating.

        Operates on the 2D depth image before backprojection.
        """
        valid = mask & (depth > 0)
        n_valid = valid.sum()
        if n_valid == 0:
            return depth

        valid_depths = depth[valid]
        median_z = np.median(valid_depths)
        tol = self.config.depth_gate_tolerance
        z_min = median_z * (1.0 - tol)
        z_max = median_z * (1.0 + tol)

        gated = depth.copy()
        out_of_range = mask & ((depth < z_min) | (depth > z_max))
        gated[out_of_range] = 0.0

        n_after = (mask & (gated > 0)).sum()
        n_removed = n_valid - n_after
        logger.info(
            "  Depth gating: median=%.4fm, window=[%.4f, %.4f]m, "
            "%d → %d valid pixels (removed %d, %.1f%%)",
            median_z, z_min, z_max, n_valid, n_after, n_removed,
            100.0 * n_removed / max(n_valid, 1),
        )
        return gated

    def generate(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        mask: np.ndarray,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        depth_trunc: Optional[float] = None,
    ) -> Optional[PointCloudResult]:
        """Erzeugt eine Punktwolke aus RGB-D-Daten für den maskierten Bereich.

        Die Tiefenpixel werden mithilfe des Pinhole-Kameramodells in den
        3D-Raum rückprojiziert:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth[v, u]

        Args:
            rgb_image: RGB-Bild als numpy-Array (H, W, 3), uint8.
            depth_image: Tiefenbild als numpy-Array (H, W), float32, in Metern.
                         The caller must convert to meters before calling.
            mask: Binäre Segmentierungsmaske (H, W), bool.
            fx, fy: Fokuslängen (überschreiben Config-Werte).
            cx, cy: Hauptpunkt (überschreiben Config-Werte).
            depth_trunc: Max. Tiefe in Metern (Punkte darüber werden ignoriert).

        Returns:
            PointCloudResult oder None bei Fehler.
        """
        import open3d as o3d

        # Parameter mit Config-Defaults auffüllen
        fx = fx or self.config.camera_fx
        fy = fy or self.config.camera_fy
        cx = cx or self.config.camera_cx
        cy = cy or self.config.camera_cy
        depth_trunc = depth_trunc or self.config.depth_trunc

        # --- Tiefenbild vorbereiten (already in meters, no heuristic) ---
        depth = depth_image.astype(np.float32)

        # --- Maske anwenden ---
        mask_bool = np.asarray(mask, dtype=bool)
        n_mask_pixels = mask_bool.sum()
        n_valid_depth = (mask_bool & (depth > 0) & (depth < depth_trunc)).sum()
        logger.info(
            "  Mask: %d pixels, %d with valid depth (before gating)",
            n_mask_pixels, n_valid_depth,
        )

        # --- Depth gating (2D, before backprojection) ---
        if self.config.depth_gate_enabled:
            depth = self._gate_depth(depth, mask_bool)

        # --- Maske anwenden: nur Objektpixel behalten ---
        segmented_depth = np.where(mask_bool, depth, 0.0)

        # --- Rückprojektion ---
        points, colors = self._backproject_manual(
            rgb_image, segmented_depth, fx, fy, cx, cy, depth_trunc
        )
        logger.info("  Backprojected: %d raw 3D points", len(points))

        if len(points) == 0:
            logger.warning("Keine gültigen Tiefenpunkte im maskierten Bereich.")
            return None

        # --- Open3D PointCloud erstellen ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # --- Optional: Voxel-Downsampling ---
        if self.config.voxel_size > 0:
            n_before = len(pcd.points)
            pcd = pcd.voxel_down_sample(voxel_size=self.config.voxel_size)
            logger.info(
                "  Downsampling: %d → %d points (voxel: %.3fm)",
                n_before, len(pcd.points), self.config.voxel_size,
            )

        # --- Statistical Outlier Removal (configurable) ---
        if self.config.sor_nb_neighbors > 0:
            n_before = len(pcd.points)
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.config.sor_nb_neighbors,
                std_ratio=self.config.sor_std_ratio,
            )
            logger.info(
                "  SOR: %d → %d points (removed %d)",
                n_before, len(pcd.points), n_before - len(pcd.points),
            )

        # --- Radius Outlier Removal (optional) ---
        if self.config.ror_enabled:
            n_before = len(pcd.points)
            pcd, _ = pcd.remove_radius_outlier(
                nb_points=self.config.ror_nb_points,
                radius=self.config.ror_radius,
            )
            logger.info(
                "  ROR: %d → %d points (removed %d)",
                n_before, len(pcd.points), n_before - len(pcd.points),
            )

        # --- Bounding Box berechnen ---
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors)

        if len(pts) == 0:
            logger.warning("No points remaining after filtering.")
            return None

        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        bbox_size = bbox_max - bbox_min

        logger.info(
            "  Final: %d points, BBox=[%.4f, %.4f, %.4f]m",
            len(pts), bbox_size[0], bbox_size[1], bbox_size[2],
        )

        return PointCloudResult(
            point_cloud=pcd,
            points=pts,
            colors=cols,
            num_points=len(pts),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            bbox_size=bbox_size,
        )

    @staticmethod
    def _backproject_manual(
        rgb: np.ndarray,
        depth: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_trunc: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rückprojektion von Tiefenpixeln in 3D via Pinhole-Modell.

        Vektorisierte Implementierung für hohe Performance.

        Pinhole-Modell:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth[v, u]

        Args:
            rgb: (H, W, 3), uint8.
            depth: (H, W), float32, in Metern, 0 = ungültig.
            fx, fy, cx, cy: Kameraparameter.
            depth_trunc: Max. Tiefe.

        Returns:
            (points, colors): Arrays der Form (N, 3).
        """
        h, w = depth.shape

        # Gültige Pixel: Tiefe > 0 und < max
        valid = (depth > 0) & (depth < depth_trunc)
        vs, us = np.where(valid)
        zs = depth[valid]

        # Rückprojektion
        xs = (us.astype(np.float32) - cx) * zs / fx
        ys = (vs.astype(np.float32) - cy) * zs / fy

        points = np.stack([xs, ys, zs], axis=-1)  # (N, 3)

        # Farben normalisieren
        colors = rgb[vs, us].astype(np.float32) / 255.0  # (N, 3)

        return points, colors

    def save_pointcloud(self, result: PointCloudResult, path: str) -> None:
        """Speichert eine Punktwolke als PLY-Datei.

        Args:
            result: PointCloudResult aus generate().
            path: Zielpfad (z.B. "output/object.ply").
        """
        import open3d as o3d
        o3d.io.write_point_cloud(path, result.point_cloud)
        logger.info(f"Punktwolke gespeichert: {path}")

    @staticmethod
    def load_pointcloud(path: str) -> 'PointCloudResult':
        """Lädt eine Punktwolke aus einer PLY-Datei.

        Args:
            path: Pfad zur PLY-Datei.

        Returns:
            PointCloudResult.
        """
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(pts)
        bbox_min = pts.min(axis=0) if len(pts) > 0 else np.zeros(3)
        bbox_max = pts.max(axis=0) if len(pts) > 0 else np.zeros(3)

        return PointCloudResult(
            point_cloud=pcd,
            points=pts,
            colors=cols,
            num_points=len(pts),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            bbox_size=bbox_max - bbox_min,
        )
