# =============================================================================
# pipeline/step_b2_geometry_reranking.py -- Thesis Sub-step B2: Geometry
# =============================================================================
#
# Thesis reference: Section 3.3 (Sub-step B2)
#
# After the multi-signal fusion in Step B1, the top-k fused candidates are
# re-ranked using geometric signals computed between the observed partial
# point cloud and each candidate's CAD partial view.
#
# Two geometry signals (thesis Sec. 3.3, Eq. S_geo):
#
#   1. GeDi correspondence score (S_GeDi)
#      GeDi local geometric descriptors (Poiesi & Boscaini, IEEE T-PAMI
#      2022) are matched via RANSAC (Fischler & Bolles, 1981).  The inlier
#      count serves as a geometric compatibility score — higher = better
#      structural match.  This follows ROCA (Gumeli et al., CVPR 2022)
#      which uses geometric fit as a re-ranking signal.
#
#   2. Trimmed one-sided Chamfer score (S_chamfer)
#      Nearest-neighbour distances from observed to CAD, top 10% trimmed
#      to handle partial overlap.  Inspired by U-RED (Di et al., 2023)
#      which uses a trimmed Chamfer variant for partial-shape comparison.
#      Lower = better geometric fit.
#
# The RANSAC transformation of the best candidate is forwarded to Step C
# as ICP initialisation, avoiding redundant descriptor computation.
# This reuse pattern follows FreeZe (Caraffa et al., ECCV 2025) which
# chains coarse registration into ICP refinement.
#
# Inputs:
#   - Fused candidate shortlist (Step B1 / Step 6)
#   - Observed partial point cloud (Step 2)
#
# Outputs:
#   - Re-ranked candidate list with geometry scores
#   - RANSAC transformation for the best candidate (reused in Step C)
# =============================================================================

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np

from .config import PipelineConfig
from .step6_fusion import FusedCandidate
from .utils import trimmed_chamfer_distance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GeometryCandidate:
    """Candidate after geometry re-ranking (Sub-step B2).

    Attributes:
        object_id: CAD model identifier.
        fused_score: Original fused score from Step B1.
        gedi_score: GeDi RANSAC inlier count (higher = better).
        chamfer_score: Trimmed one-sided Chamfer distance (lower = better).
        geometry_score: Combined geometry score used for re-ranking.
        ransac_transformation: 4x4 rigid transformation from RANSAC (if computed).
        ransac_fitness: RANSAC fitness (fraction of inliers).
        cad_model_path: Path to CAD model.
        best_view_path: Best reference image path (carried from fusion).
        clip_score: From fusion.
        dino_score: From fusion.
        ulip_score: From fusion.
    """
    object_id: str
    fused_score: float = 0.0
    gedi_score: float = 0.0
    chamfer_score: float = float("inf")
    geometry_score: float = 0.0
    ransac_transformation: Optional[np.ndarray] = None
    ransac_fitness: float = 0.0
    cad_model_path: str = ""
    best_view_path: str = ""
    clip_score: float = 0.0
    dino_score: float = 0.0
    ulip_score: float = 0.0


@dataclass
class GeometryReRankingResult:
    """Result of Sub-step B2 geometry re-ranking.

    Attributes:
        candidates: Re-ranked candidate list (best first).
        signal: Which geometry signal was used ("gedi" | "chamfer" | "both").
        best_candidate: Top-ranked candidate after re-ranking.
        best_transformation: RANSAC transformation for the best candidate
                            (reusable in Step C to skip redundant registration).
    """
    candidates: List[GeometryCandidate]
    signal: str
    best_candidate: Optional[GeometryCandidate] = None
    best_transformation: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Geometry Re-ranking Module
# ---------------------------------------------------------------------------

class GeometryReRanker:
    """Re-ranks fused candidates using geometric signals (GeDi + Chamfer).

    Thesis Sub-step B2: operates on the small fused shortlist so that
    per-candidate cost remains tractable.

    Usage:
        >>> reranker = GeometryReRanker(config)
        >>> result = reranker.rerank(fused_candidates, observed_pcd)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._gedi_module = None

    def _get_gedi(self):
        """Lazy-load GeDi descriptor module."""
        if self._gedi_module is None:
            from .gedi_descriptors import GeDiDescriptorModule
            self._gedi_module = GeDiDescriptorModule(self.config)
        return self._gedi_module

    def rerank(
        self,
        fused_candidates: List[FusedCandidate],
        observed_pcd,
        signal: Optional[str] = None,
    ) -> GeometryReRankingResult:
        """Re-rank fused candidates using geometric signals.

        Args:
            fused_candidates: Shortlist from Step B1 fusion.
            observed_pcd: Open3D PointCloud of the observed partial object.
            signal: Override for config.geometry_reranking_signal.

        Returns:
            GeometryReRankingResult with re-ranked candidates.
        """
        import open3d as o3d

        signal = signal or self.config.geometry_reranking_signal
        top_k = self.config.geometry_reranking_top_k
        candidates = fused_candidates[:top_k]

        if not candidates:
            return GeometryReRankingResult(candidates=[], signal=signal)

        logger.info(
            "Sub-step B2: geometry re-ranking %d candidates (signal=%s)...",
            len(candidates), signal,
        )

        voxel_size = self.config.voxel_size or 0.005

        # Compute GeDi descriptors for observed cloud (once)
        obs_gedi = None
        if signal in ("gedi", "both"):
            gedi_mod = self._get_gedi()
            if gedi_mod.available:
                obs_gedi = gedi_mod.compute(observed_pcd)
                logger.info("  GeDi: %d query keypoints computed.",
                            len(np.asarray(obs_gedi.keypoints.points)))
            else:
                logger.warning("  GeDi not available, falling back to chamfer-only.")
                if signal == "gedi":
                    signal = "chamfer"

        # Downsample observed cloud for Chamfer (once)
        obs_down = None
        if signal in ("chamfer", "both"):
            obs_down = observed_pcd.voxel_down_sample(voxel_size)

        # Score each candidate
        geo_candidates = []
        for fc in candidates:
            gc = GeometryCandidate(
                object_id=fc.object_id,
                fused_score=fc.fused_score,
                cad_model_path=fc.cad_model_path,
                best_view_path=getattr(fc, "best_view_path", ""),
                clip_score=fc.clip_score,
                dino_score=fc.dino_score,
                ulip_score=getattr(fc, "ulip_score", 0.0),
            )

            # Load CAD point cloud
            cad_pcd = self._load_cad_pointcloud(fc.cad_model_path)
            if cad_pcd is None:
                logger.warning("  %s: CAD load failed, skipping geometry.",
                               fc.object_id)
                geo_candidates.append(gc)
                continue

            # GeDi RANSAC
            if signal in ("gedi", "both") and obs_gedi is not None:
                gedi_score, ransac_result = self._gedi_ransac(
                    obs_gedi, cad_pcd, voxel_size,
                )
                gc.gedi_score = gedi_score
                if ransac_result is not None:
                    gc.ransac_transformation = np.array(ransac_result.transformation)
                    gc.ransac_fitness = ransac_result.fitness

            # Trimmed Chamfer
            if signal in ("chamfer", "both") and obs_down is not None:
                cad_down = cad_pcd.voxel_down_sample(voxel_size)
                gc.chamfer_score = trimmed_chamfer_distance(
                    np.asarray(obs_down.points),
                    np.asarray(cad_down.points),
                    trim_ratio=self.config.chamfer_trim_ratio,
                )

            # Combined geometry score
            gc.geometry_score = self._compute_geometry_score(gc, signal)

            logger.info(
                "  %s: gedi=%.0f chamfer=%.6f geo_score=%.4f",
                fc.object_id[:30], gc.gedi_score, gc.chamfer_score,
                gc.geometry_score,
            )
            geo_candidates.append(gc)

        # Sort by geometry score (higher = better)
        geo_candidates.sort(key=lambda c: c.geometry_score, reverse=True)

        best = geo_candidates[0] if geo_candidates else None
        if best:
            logger.info(
                "Sub-step B2 winner: %s (geo_score=%.4f, gedi=%.0f, chamfer=%.6f)",
                best.object_id, best.geometry_score,
                best.gedi_score, best.chamfer_score,
            )

        return GeometryReRankingResult(
            candidates=geo_candidates,
            signal=signal,
            best_candidate=best,
            best_transformation=best.ransac_transformation if best else None,
        )

    def _gedi_ransac(
        self,
        obs_gedi,
        cad_pcd,
        voxel_size: float,
    ):
        """Run GeDi descriptor matching + RANSAC on a CAD candidate.

        GeDi (Poiesi & Boscaini, 2022) descriptors are matched via
        RANSAC (Fischler & Bolles, 1981) with mutual filtering.
        The inlier count serves as S_GeDi — a higher count indicates
        better geometric compatibility (Yang et al., Teaser 2020
        uses a similar inlier-based fitness measure).

        Args:
            obs_gedi: GeDiResult for the observed cloud.
            cad_pcd: Open3D PointCloud of the CAD model.
            voxel_size: Voxel size for correspondence distance.

        Returns:
            (inlier_count, ransac_result) or (0, None) on failure.
        """
        import open3d as o3d

        gedi_mod = self._get_gedi()
        cad_gedi = gedi_mod.compute(cad_pcd)

        if len(cad_gedi.descriptors_np) == 0:
            return 0.0, None

        try:
            ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                obs_gedi.keypoints,
                cad_gedi.keypoints,
                obs_gedi.features,
                cad_gedi.features,
                mutual_filter=True,
                max_correspondence_distance=voxel_size * 1.5,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        voxel_size * 1.5
                    ),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    100000, 0.999
                ),
            )
            inlier_count = len(ransac.correspondence_set)
            return float(inlier_count), ransac
        except Exception as exc:
            logger.warning("GeDi RANSAC failed: %s", exc)
            return 0.0, None

    @staticmethod
    def _compute_geometry_score(gc: GeometryCandidate, signal: str) -> float:
        """Compute a single geometry score for ranking.

        For GeDi: score = inlier count (higher = better).
        For Chamfer: score = -chamfer (negated so higher = better).
        For both: normalize and average (GeDi dominates per thesis).
        """
        if signal == "gedi":
            return gc.gedi_score
        elif signal == "chamfer":
            return -gc.chamfer_score  # negate: lower chamfer = higher score
        else:  # "both"
            # Combine: GeDi as primary, Chamfer as tiebreaker
            # Chamfer is negated and scaled to be in similar range as GeDi
            return gc.gedi_score + (-gc.chamfer_score * 1000.0)

    @staticmethod
    def _load_cad_pointcloud(cad_path: str, n_points: int = 10000):
        """Load a CAD model and sample a point cloud."""
        import open3d as o3d

        if not cad_path or not os.path.isfile(cad_path):
            # Try common mesh extensions
            for ext in (".obj", ".ply", ".glb", ".stl"):
                alt = cad_path + ext if cad_path else ""
                if os.path.isfile(alt):
                    cad_path = alt
                    break
            else:
                return None

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
        except Exception as exc:
            logger.warning("CAD load failed (%s): %s", cad_path, exc)
            return None
