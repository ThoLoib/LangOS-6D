# =============================================================================
# pipeline/step7_scale_estimation.py – Schritt 7: Ausrichtung + Skalenbestimmung
# =============================================================================
#
# Ziel:
#   1. Coarse Alignment: CAD-Modell ueber RANSAC + ICP an die beobachtete
#      Punktwolke ausrichten -> korrekte Orientierung.
#   2. Partial-Aware Scale: Skalierungsfaktor nur aus den gut sichtbaren
#      Achsen ableiten (partielle Punktwolke -> nicht alle Dimensionen
#      sind vollstaendig beobachtet).
#
# Problem bei partiellen Punktwolken:
#   Von einer einzelnen Kameraansicht sehen wir nur die Vorderseite.
#   Die Tiefe (Achse von Kamera weg) ist systematisch unterschaetzt.
#   Z.B. bei einer zylindrischen Dose: Breite und Hoehe sind fast
#   vollstaendig sichtbar, aber die Tiefe nur ~50%.
#
#   Loesung: Nach Alignment die 2 am besten sichtbaren Achsen (groesste
#   Beobachtungs-/CAD-Ratio) fuer die Skalierung verwenden.
#
# Inputs:
#   - Punktwolke des Objekts (Schritt 2)
#   - Ausgewaehltes CAD-Modell (Schritt 6)
#
# Outputs:
#   - Skalierungsfaktor (float)
#   - Coarse-Alignment-Transformation (4x4)
#   - Skaliertes CAD-Modell
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .step2_pointcloud import PointCloudResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenstruktur
# ---------------------------------------------------------------------------

@dataclass
class ScaleEstimationResult:
    """Ergebnis der Skalenbestimmung (Schritt 7)."""
    scale_factor: float
    scale_per_axis: np.ndarray
    observed_size: np.ndarray
    cad_size: np.ndarray
    method: str
    confidence: float = 1.0
    coarse_alignment: Optional[np.ndarray] = None
    visible_axes: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Skalenschaetzer mit Alignment
# ---------------------------------------------------------------------------

class ScaleEstimator:
    """Bestimmt Orientierung und Skalierungsfaktor zwischen Beobachtung und CAD.

    Zweistufiger Ansatz:

    1. Coarse Alignment (RANSAC + ICP): Richtet das CAD-Modell an der
       beobachteten Punktwolke aus, sodass die Achsen korrespondieren.

    2. Partial-Aware Scale: Nach Alignment wird die Ausdehnung der
       alignierten Punktwolken entlang jeder Achse verglichen.
       Die Achse mit der schlechtesten Sichtbarkeit wird heruntergewichtet.
       Die Skala wird aus den 2 besten Achsen abgeleitet.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def estimate(
        self,
        observed_pc: PointCloudResult,
        cad_model_path: str,
        method: str = "align_then_scale",
    ) -> ScaleEstimationResult:
        """Schaetzt Ausrichtung + Skalierungsfaktor.

        Args:
            observed_pc: Punktwolke des beobachteten Objekts (Schritt 2).
            cad_model_path: Pfad zum CAD-Modell (OBJ, PLY, GLB, ...).
            method: "align_then_scale" (default) | "max_extent" (Fallback).

        Returns:
            ScaleEstimationResult mit Alignment und Skalierungsfaktor.
        """
        import open3d as o3d

        # --- CAD-Modell als Punktwolke laden ---
        cad_pcd = self._load_cad_pointcloud(cad_model_path)
        if cad_pcd is None:
            logger.warning("CAD-Modell konnte nicht geladen werden -> scale=1.0")
            return self._fallback_result(observed_pc, method)

        cad_bbox_size = self._pcd_bbox_size(cad_pcd)
        obs_bbox_size = observed_pc.bbox_size

        if method == "max_extent" or cad_bbox_size is None:
            return self._max_extent_scale(obs_bbox_size, cad_bbox_size or np.ones(3))

        # --- Schritt A: Coarse Alignment (RANSAC + ICP) ---
        logger.info("Coarse Alignment (RANSAC + ICP)...")
        source = observed_pc.point_cloud  # beobachtete Wolke
        target = cad_pcd                  # CAD-Wolke

        alignment = self._coarse_align(source, target)
        if alignment is None:
            logger.warning("Alignment fehlgeschlagen -> Fallback max_extent")
            return self._max_extent_scale(obs_bbox_size, cad_bbox_size)

        T = alignment.transformation  # 4x4
        fitness = alignment.fitness
        logger.info("  RANSAC+ICP: fitness=%.4f, inlier_rmse=%.6f m",
                     fitness, alignment.inlier_rmse)

        # --- Schritt B: Aligned BBoxes vergleichen ---
        # Beobachtete Wolke in CAD-Koordinaten transformieren
        source_aligned = o3d.geometry.PointCloud(source)
        source_aligned.transform(T)

        obs_aligned_size = self._pcd_bbox_size(source_aligned)
        cad_size = self._pcd_bbox_size(target)

        if obs_aligned_size is None or cad_size is None:
            return self._max_extent_scale(obs_bbox_size, cad_bbox_size)

        # --- Schritt C: Partial-Aware Scale ---
        #  Die Achse mit dem niedrigsten obs/cad-Ratio ist die, die am
        #  schlechtesten beobachtet wurde (typischerweise Tiefe).
        #  Wir nutzen die 2 besten Achsen fuer die Skala.
        safe_cad = np.where(cad_size > 1e-8, cad_size, 1.0)
        ratios = obs_aligned_size / safe_cad  # (3,)

        # Sortiere: Die 2 Achsen mit hoechstem Ratio sind am besten sichtbar
        sorted_idx = np.argsort(ratios)[::-1]   # absteigend
        best_2 = sorted_idx[:2]
        scale_factor = float(np.mean(ratios[best_2]))

        # Konfidenz aus Fitness + Ratio-Konsistenz
        ratio_spread = abs(ratios[best_2[0]] - ratios[best_2[1]])
        confidence = float(min(fitness, max(0.0, 1.0 - ratio_spread)))

        logger.info(
            "Partial-Aware Scale: factor=%.4f (aus Achsen %s), "
            "ratios=[%.3f, %.3f, %.3f], conf=%.2f",
            scale_factor, best_2.tolist(),
            ratios[0], ratios[1], ratios[2], confidence,
        )

        return ScaleEstimationResult(
            scale_factor=scale_factor,
            scale_per_axis=ratios,
            observed_size=obs_aligned_size,
            cad_size=cad_size,
            method="align_then_scale",
            confidence=confidence,
            coarse_alignment=np.array(T),
            visible_axes=best_2,
        )

    # -----------------------------------------------------------------------
    # Coarse Alignment: RANSAC + ICP
    # -----------------------------------------------------------------------

    def _coarse_align(self, source, target):
        """RANSAC global registration + ICP refinement.

        Args:
            source: Beobachtete Punktwolke (Open3D PointCloud).
            target: CAD-Punktwolke (Open3D PointCloud).

        Returns:
            ICP-RegistrationResult oder None bei Fehler.
        """
        import open3d as o3d

        voxel_size = self.config.voxel_size or 0.005

        # Downsampling + Normals
        src_down = source.voxel_down_sample(voxel_size)
        tgt_down = target.voxel_down_sample(voxel_size)

        for pcd in (src_down, tgt_down):
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2, max_nn=30
                )
            )

        # FPFH Features
        src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            src_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
        )
        tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            tgt_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
        )

        # RANSAC
        try:
            ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_down, tgt_down,
                src_fpfh, tgt_fpfh,
                mutual_filter=True,
                max_correspondence_distance=voxel_size * 1.5,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
            )
        except Exception as e:
            logger.warning("RANSAC fehlgeschlagen: %s", e)
            return None

        # ICP Refinement (Point-to-Plane)
        for pcd in (source, target):
            if not pcd.has_normals():
                pcd.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
                )

        try:
            icp = o3d.pipelines.registration.registration_icp(
                source, target,
                max_correspondence_distance=voxel_size * 3,
                init=ransac.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.config.icp_max_iterations
                ),
            )
            return icp
        except Exception as e:
            logger.warning("ICP fehlgeschlagen: %s", e)
            return ransac

    # -----------------------------------------------------------------------
    # Hilfsfunktionen
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_cad_pointcloud(cad_path: str, n_points: int = 10000):
        """Laedt ein CAD-Modell und sampelt eine Punktwolke."""
        import open3d as o3d
        try:
            mesh = o3d.io.read_triangle_mesh(cad_path)
            if mesh.is_empty():
                return None
            mesh.compute_vertex_normals()
            pcd = mesh.sample_points_uniformly(number_of_points=n_points)
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
            return pcd
        except Exception as e:
            logger.warning("CAD laden fehlgeschlagen (%s): %s", cad_path, e)
            return None

    @staticmethod
    def _pcd_bbox_size(pcd) -> Optional[np.ndarray]:
        """Axis-Aligned Bounding Box Groesse einer PointCloud."""
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return None
        return pts.max(axis=0) - pts.min(axis=0)

    def _fallback_result(self, observed_pc: PointCloudResult,
                          method: str) -> ScaleEstimationResult:
        return ScaleEstimationResult(
            scale_factor=1.0,
            scale_per_axis=np.ones(3),
            observed_size=observed_pc.bbox_size,
            cad_size=np.zeros(3),
            method=method,
            confidence=0.0,
        )

    def _max_extent_scale(
        self, observed: np.ndarray, cad: np.ndarray
    ) -> ScaleEstimationResult:
        """Fallback: Skalierung basierend auf maximaler Ausdehnung.

        Rotationsinvariant - keine Achszuordnung noetig.
        """
        obs_max = observed.max()
        cad_max = cad.max()

        if cad_max == 0:
            return ScaleEstimationResult(
                scale_factor=1.0, scale_per_axis=np.ones(3),
                observed_size=observed, cad_size=cad,
                method="max_extent", confidence=0.0,
            )

        scale = obs_max / cad_max

        obs_ratio = observed / (obs_max + 1e-8)
        cad_ratio = cad / (cad_max + 1e-8)
        aspect_diff = np.abs(np.sort(obs_ratio) - np.sort(cad_ratio)).mean()
        confidence = max(0.0, 1.0 - aspect_diff)

        return ScaleEstimationResult(
            scale_factor=float(scale),
            scale_per_axis=np.full(3, scale),
            observed_size=observed,
            cad_size=cad,
            method="max_extent",
            confidence=float(confidence),
        )

    @staticmethod
    def apply_scale(cad_model_path: str, scale_factor: float, output_path: str) -> str:
        """Skaliert ein CAD-Modell und speichert es."""
        try:
            import trimesh
            mesh = trimesh.load(cad_model_path, force="mesh")
            mesh.apply_scale(scale_factor)
            mesh.export(output_path)
            logger.info("Skaliertes Modell gespeichert: %s (x%.4f)", output_path, scale_factor)
            return output_path
        except ImportError:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(cad_model_path)
            mesh.scale(scale_factor, center=mesh.get_center())
            o3d.io.write_triangle_mesh(output_path, mesh)
            logger.info("Skaliertes Modell gespeichert: %s (x%.4f)", output_path, scale_factor)
            return output_path