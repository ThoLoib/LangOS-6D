# =============================================================================
# pipeline/step7_scale_estimation.py – Thesis Step C (part 1): Coarse Alignment
#                                       + Scale Estimation
# =============================================================================
#
# Thesis reference: Section 3.4, Step C
#
# Two-stage process:
#
#   1. Coarse Alignment (RANSAC + ICP)
#      Registers the CAD model to the observed partial point cloud.
#      When Sub-step B2 provides a RANSAC transformation, that is used
#      directly as ICP initialisation — avoiding redundant descriptor
#      computation.  This reuse pattern follows FreeZe (Caraffa et al.,
#      ECCV 2025) which chains coarse registration into ICP refinement.
#
#      Descriptors: GeDi (Poiesi & Boscaini, 2022) preferred, with FPFH
#      (Rusu et al., ICRA 2009) as fallback when the GeDi service is
#      unavailable.
#
#      ICP: Point-to-Plane with correspondence distance 3×voxel_size
#      (thesis Sec. 3.4).
#
#   2. Partial-Aware Scale Estimation
#      From a single depth view only the front surface is observed; the
#      depth axis is systematically underestimated.  After alignment the
#      per-axis observed/CAD ratios are computed; the 2 axes with the
#      highest ratio (= best visibility) determine the scale factor.
#
# Inputs:
#   - Observed partial point cloud (Step 2)
#   - Selected CAD model (Step B1/B2)
#   - Optional: RANSAC transform from Sub-step B2
#
# Outputs:
#   - Scale factor (float)
#   - Coarse alignment transformation (4×4)
#   - Scaled CAD model
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
        init_transform: Optional[np.ndarray] = None,
    ) -> ScaleEstimationResult:
        """Schaetzt Ausrichtung + Skalierungsfaktor.

        Args:
            observed_pc: Punktwolke des beobachteten Objekts (Schritt 2).
            cad_model_path: Pfad zum CAD-Modell (OBJ, PLY, GLB, ...).
            method: "align_then_scale" (default) | "max_extent" (Fallback).
            init_transform: Optional 4x4 initial transform from Sub-step B2
                           RANSAC.  When provided, skips descriptor computation
                           and uses this as ICP initialisation.

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

        alignment = self._coarse_align(source, target, init_transform=init_transform)
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

        # --- Fallback: ICP-Alignment unreliable -> use sorted-bbox estimate ---
        # When confidence is very low the ICP alignment was degenerate (common
        # for heavily truncated partial views).  The sorted-bbox scale is
        # rotation-invariant and gives a better estimate in that case.
        # The ICP transformation T is still returned for coarse alignment.
        _min_conf = getattr(self.config, "scale_icp_min_confidence", 0.15)
        if confidence < _min_conf:
            fast_scale, fast_conf = self.estimate_fast(observed_pc, cad_model_path)
            logger.warning(
                "ICP scale confidence %.2f < %.2f — overriding with sorted-bbox "
                "scale=%.4f (conf=%.2f); keeping ICP transform for alignment.",
                confidence, _min_conf, fast_scale, fast_conf,
            )
            scale_factor = fast_scale
            confidence = fast_conf

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

    def estimate_fast(
        self,
        observed_pc: PointCloudResult,
        cad_model_path: str,
    ) -> Tuple[float, float]:
        """Fast, deterministic scale estimate for scale gate screening.

        Compares sorted bounding box dimensions without any ICP or RANSAC.
        Sorting the dimensions makes the comparison rotation-invariant and
        partial-view robust: the two largest dimensions are most likely to
        be fully visible in a single-depth-view partial point cloud.

        This method is intentionally cheap — it is called once per fused
        candidate during the scale gate pass. The full RANSAC+ICP estimate()
        is still used in Step 7 to produce the coarse alignment for Step 8.

        Args:
            observed_pc: Observed partial point cloud (Step 2).
            cad_model_path: Path to the CAD mesh.

        Returns:
            (scale_factor, confidence)
            confidence in [0, 1] — based on how consistently the two
            largest dimensions agree on the same scale ratio.
        """
        import open3d as o3d

        obs_size = observed_pc.bbox_size
        if obs_size is None or float(obs_size.max()) < 1e-6:
            return 1.0, 0.0

        try:
            mesh = o3d.io.read_triangle_mesh(cad_model_path)
            if mesh.is_empty():
                return 1.0, 0.0
            cad_size = np.asarray(
                mesh.get_axis_aligned_bounding_box().get_extent()
            )
        except Exception as exc:
            logger.warning("estimate_fast: CAD load failed (%s): %s",
                           cad_model_path, exc)
            return 1.0, 0.0

        if float(cad_size.max()) < 1e-6:
            return 1.0, 0.0

        # Sort dimensions largest-first; partial-view robust — largest
        # two dims are most reliably observed from a single depth view.
        obs_sorted = np.sort(obs_size)[::-1]
        cad_sorted = np.sort(cad_size)[::-1]

        safe_cad = np.where(cad_sorted[:2] > 1e-6, cad_sorted[:2], 1.0)
        ratios = obs_sorted[:2] / safe_cad

        scale_factor = float(np.median(ratios))
        ratio_spread = float(abs(ratios[0] - ratios[1]))
        confidence = max(0.0, 1.0 - ratio_spread / (scale_factor + 1e-8))

        logger.debug(
            "estimate_fast: obs=%s cad=%s ratios=%s → scale=%.4f conf=%.2f",
            np.round(obs_sorted[:2], 4), np.round(cad_sorted[:2], 4),
            np.round(ratios, 4), scale_factor, confidence,
        )

        return scale_factor, confidence

    # -----------------------------------------------------------------------
    # Coarse Alignment: RANSAC + ICP
    # -----------------------------------------------------------------------

    def _coarse_align(self, source, target, init_transform=None):
        """RANSAC global registration + ICP refinement (thesis Step C).

        Uses GeDi descriptors (Poiesi & Boscaini, 2022) when available,
        falling back to FPFH (Rusu et al., ICRA 2009) otherwise.

        If *init_transform* is provided (e.g. from Sub-step B2 RANSAC),
        the descriptor computation is skipped and ICP is initialised
        directly — following the FreeZe (Caraffa et al., 2025) pattern
        of chaining coarse registration into ICP refinement.

        ICP correspondence distance: 3×voxel_size (thesis Sec. 3.4).

        Args:
            source: Observed point cloud (Open3D PointCloud).
            target: CAD point cloud (Open3D PointCloud).
            init_transform: Optional 4x4 initial transformation (from B2).

        Returns:
            ICP RegistrationResult or None on failure.
        """
        import open3d as o3d

        voxel_size = self.config.voxel_size or 0.005

        # If we already have a B2 RANSAC transform, skip descriptor computation
        if init_transform is not None:
            logger.info("  Using B2 RANSAC transform as ICP initialisation.")
            ransac_T = init_transform
        else:
            # Try GeDi descriptors first, fall back to FPFH
            ransac_T = self._ransac_with_descriptors(source, target, voxel_size)
            if ransac_T is None:
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
                init=ransac_T,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.config.icp_max_iterations
                ),
            )
            return icp
        except Exception as e:
            logger.warning("ICP fehlgeschlagen: %s", e)
            # Return a minimal result with the RANSAC transform
            return None

    def _ransac_with_descriptors(self, source, target, voxel_size: float):
        """Run RANSAC using GeDi (preferred) or FPFH (fallback) descriptors.

        Returns:
            4x4 transformation matrix, or None on failure.
        """
        import open3d as o3d

        # --- Try GeDi ---
        try:
            from .gedi_descriptors import GeDiDescriptorModule
            gedi_mod = GeDiDescriptorModule(self.config)
            if gedi_mod.available:
                logger.info("  Coarse alignment: using GeDi descriptors.")
                src_gedi = gedi_mod.compute(source)
                tgt_gedi = gedi_mod.compute(target)

                if len(src_gedi.descriptors_np) > 0 and len(tgt_gedi.descriptors_np) > 0:
                    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                        src_gedi.keypoints, tgt_gedi.keypoints,
                        src_gedi.features, tgt_gedi.features,
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
                    return np.array(ransac.transformation)
        except Exception as exc:
            logger.info("  GeDi RANSAC failed (%s), falling back to FPFH.", exc)

        # --- Fallback: FPFH ---
        logger.info("  Coarse alignment: using FPFH descriptors (fallback).")
        src_down = source.voxel_down_sample(voxel_size)
        tgt_down = target.voxel_down_sample(voxel_size)

        for pcd in (src_down, tgt_down):
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2, max_nn=30
                )
            )

        src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            src_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
        )
        tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            tgt_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
        )

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
            return np.array(ransac.transformation)
        except Exception as e:
            logger.warning("FPFH RANSAC fehlgeschlagen: %s", e)
            return None

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