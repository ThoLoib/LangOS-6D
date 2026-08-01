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
#   2. Trimmed one-sided surface distance (S_surface = -D_trim)
#      Nearest-neighbour distances from observed to CAD, top 10% trimmed
#      to handle partial overlap.  Inspired by U-RED (Di et al., 2023)
#      which uses a trimmed Chamfer variant for partial-shape comparison.
#      Lower = better geometric fit.
#
#      IMPORTANT: the observation is rigidly transformed into the CAD frame
#      BEFORE the distance is evaluated (thesis Eq.
#      eq:methods_trimmed_surface_distance defines D_trim(P, C_j | T_j)).
#      A candidate whose registration fails receives an invalid score rather
#      than an unaligned distance.
#
# Signals (thesis ablation E2, subsec:eval_baselines):
#
#   "none"              no geometry re-ranking (handled by the caller)
#   "fitness"           GeDi -> RANSAC; rank by RANSAC fitness, no distance
#   "chamfer_unaligned" D_trim WITHOUT alignment — diagnostic control only,
#                       not a proposed method.  It exists to show that the
#                       gain comes from evaluating distance after alignment.
#   "chamfer_ransac"    apply T_RANSAC, then D_trim
#   "chamfer_icp"       refine T_RANSAC with point-to-plane ICP, then D_trim
#
# Legacy aliases kept for back-compat with existing configs//callers:
#   "gedi" -> "fitness", "chamfer" -> "chamfer_unaligned",
#   "both" -> "chamfer_ransac"
#
# The transformation of the best candidate is forwarded to Step C as ICP
# initialisation, avoiding redundant descriptor computation.  This reuse
# pattern follows FreeZe (Caraffa et al., ECCV 2025) which chains coarse
# registration into ICP refinement.
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
import hashlib
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np

from .config import PipelineConfig
from .step6_fusion import FusedCandidate
from .utils import trimmed_chamfer_distance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry signals (thesis ablation E2)
# ---------------------------------------------------------------------------
# Signals that evaluate D_trim after applying a rigid transform.
_ALIGNED_SIGNALS = ("chamfer_ransac", "chamfer_icp")
# Signals that evaluate a surface distance at all (aligned or not).
_DISTANCE_SIGNALS = _ALIGNED_SIGNALS + ("chamfer_unaligned",)
VALID_SIGNALS = ("fitness",) + _DISTANCE_SIGNALS

# Back-compat: older configs/callers use these names.  "both" maps to the
# aligned distance because that is what it was *intended* to be; its previous
# behaviour (GeDi score plus an UNALIGNED distance, combined with an
# arbitrary 1000x scale factor) did not correspond to any thesis config.
_SIGNAL_ALIASES = {
    "gedi": "fitness",
    "chamfer": "chamfer_unaligned",
    "both": "chamfer_ransac",
}


def _normalize_signal(signal: str) -> str:
    """Map legacy signal names onto the current vocabulary."""
    resolved = _SIGNAL_ALIASES.get(signal, signal)
    if resolved not in VALID_SIGNALS:
        raise ValueError(
            f"Unknown geometry signal {signal!r}; expected one of "
            f"{VALID_SIGNALS} (or a legacy alias {tuple(_SIGNAL_ALIASES)})."
        )
    if resolved != signal:
        logger.info("Geometry signal %r -> %r (legacy alias).", signal, resolved)
    return resolved


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
        chamfer_score: Trimmed one-sided surface distance D_trim
                       (lower = better).  For the aligned signals this is
                       evaluated AFTER applying `transformation`, per thesis
                       Eq. eq:methods_trimmed_surface_distance.
        geometry_score: Combined geometry score used for re-ranking.
        ransac_transformation: 4x4 rigid transformation from RANSAC (if computed).
        ransac_fitness: RANSAC fitness (fraction of inliers).  Preferred over
                        the raw inlier count for comparisons, since it is less
                        sensitive to the number of sampled keypoints
                        (thesis Sec. 3.3).
        icp_transformation: RANSAC transform refined by point-to-plane ICP.
        icp_fitness: ICP fitness (overlap fraction).
        icp_inlier_rmse: ICP inlier RMSE (E2 geometry diagnostic).
        transformation: The transform actually applied before measuring
                        D_trim — T_ICP when refinement succeeded, else T_RANSAC.
        registration_failed: True when no usable rigid alignment was found.
                             Such candidates receive an invalid geometry score
                             and rank last; they are never compared using raw,
                             unaligned distances (thesis Sec. 3.3).
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
    # Populated only under rerank(all_aligned=True): D_trim under the RANSAC
    # transform and under the ICP-refined transform, both measured from the
    # SAME registration.  See the `all_aligned` docstring for why.
    d_ransac: float = float("inf")
    d_icp: float = float("inf")
    geometry_score: float = 0.0
    ransac_transformation: Optional[np.ndarray] = None
    ransac_fitness: float = 0.0
    icp_transformation: Optional[np.ndarray] = None
    icp_fitness: float = 0.0
    icp_inlier_rmse: float = 0.0
    transformation: Optional[np.ndarray] = None
    registration_failed: bool = False
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
        signal: Which geometry signal was used (see VALID_SIGNALS).
        best_candidate: Top-ranked candidate after re-ranking.
        best_transformation: Rigid transformation for the best candidate —
                            ICP-refined when available, else the RANSAC
                            estimate (reusable in Step C to skip redundant
                            registration).
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
        all_aligned: bool = False,
        query_id: Optional[str] = None,
    ) -> GeometryReRankingResult:
        """Re-rank fused candidates using geometric signals.

        Args:
            fused_candidates: Shortlist from Step B1 fusion.
            observed_pcd: Open3D PointCloud of the observed partial object.
            signal: Override for config.geometry_reranking_signal.
            query_id: Cache key for the query cloud's descriptors.  Only used
                when `config.gedi_cache_dir` is set; without it the query
                descriptors are recomputed on every call, which matters
                because a batch evaluation re-enters `rerank` per signal.
            all_aligned: Also populate `d_ransac` and `d_icp` on every
                candidate, so that the three aligned signals (fitness,
                chamfer_ransac, chamfer_icp) can be read off ONE registration.

                GeDi + RANSAC costs ~35 s/query and dominates everything else
                (the trimmed distance alone, measured as the chamfer_unaligned
                ablation, is ~0.5 s/query; ICP is local refinement and far
                cheaper than feature RANSAC).  Scoring the three signals in
                separate passes therefore pays that 35 s three times to obtain
                three cheap readouts of the same alignment — ~60 h instead of
                ~26 h over the 2,101 SHREC'18 queries.

                Note this makes fitness and d_ransac come from the SAME RANSAC
                run rather than from independent stochastic ones.  That is a
                deliberate change: they are then two readouts of one alignment,
                which is what the E2 grid is meant to compare.

        Returns:
            GeometryReRankingResult with re-ranked candidates.
        """
        import open3d as o3d

        signal = _normalize_signal(signal or self.config.geometry_reranking_signal)
        top_k = self.config.geometry_reranking_top_k
        candidates = fused_candidates[:top_k]

        if not candidates:
            return GeometryReRankingResult(candidates=[], signal=signal)

        logger.info(
            "Sub-step B2: geometry re-ranking %d candidates (signal=%s)...",
            len(candidates), signal,
        )

        voxel_size = self.config.voxel_size or 0.005
        needs_registration = (signal in _ALIGNED_SIGNALS or signal == "fitness"
                              or all_aligned)
        needs_distance = signal in _DISTANCE_SIGNALS or all_aligned

        # Compute GeDi descriptors for observed cloud (once)
        obs_gedi = None
        if needs_registration:
            gedi_mod = self._get_gedi()
            if gedi_mod.available:
                obs_gedi = self._cached_gedi(gedi_mod, observed_pcd,
                                             "query", query_id)
                logger.info("  GeDi: %d query keypoints computed.",
                            len(np.asarray(obs_gedi.keypoints.points)))
            else:
                # Without GeDi there is no alignment, and an unaligned
                # distance is explicitly NOT an acceptable substitute for an
                # aligned one (thesis Sec. 3.3).  Degrade only to the
                # diagnostic signal, and say so loudly.
                logger.warning(
                    "  GeDi unavailable — cannot align.  Falling back to the "
                    "UNALIGNED diagnostic signal; these numbers are not "
                    "comparable to the aligned variants.")
                signal = "chamfer_unaligned"
                needs_registration = False
                needs_distance = True
                # No alignment means no aligned distances to harvest.
                all_aligned = False

        # Downsample observed cloud for the surface distance (once)
        obs_down = None
        if needs_distance:
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

            # --- Rigid registration: GeDi correspondences -> RANSAC -------
            if needs_registration and obs_gedi is not None:
                gedi_score, ransac_result = self._gedi_ransac(
                    obs_gedi, cad_pcd, voxel_size, cad_id=fc.object_id,
                )
                gc.gedi_score = gedi_score
                if ransac_result is not None and gedi_score > 0:
                    gc.ransac_transformation = np.array(ransac_result.transformation)
                    gc.ransac_fitness = float(ransac_result.fitness)
                    gc.transformation = gc.ransac_transformation
                else:
                    gc.registration_failed = True

                # Optional ICP refinement (thesis: T_j = T_ICP if it
                # succeeds, else T_RANSAC).
                #
                # skip_icp exists because ICP is the single most expensive
                # per-pair component after descriptors (~1.1 s/pair of a
                # ~2.9 s/pair budget) and it was measured to contribute
                # Delta 0.0001 nDCG at K=5.  At a deeper shortlist that is
                # tens of hours to re-confirm a null, so the final-depth run
                # may skip it and cite the shallow result instead.  The
                # `signal == "chamfer_icp"` path deliberately ignores the
                # flag: that arm IS the ICP measurement, so silently
                # returning d_icp = d_ransac would fabricate its result.
                skip_icp = (getattr(self.config, "geometry_skip_icp", False)
                            and signal != "chamfer_icp")
                if ((signal == "chamfer_icp" or all_aligned)
                        and not skip_icp
                        and not gc.registration_failed):
                    icp = self._icp_refine(
                        observed_pcd, cad_pcd, gc.ransac_transformation,
                    )
                    if icp is not None:
                        gc.icp_transformation = np.array(icp.transformation)
                        gc.icp_fitness = float(icp.fitness)
                        gc.icp_inlier_rmse = float(icp.inlier_rmse)
                        # T_j for the ICP signal only.  Under all_aligned with
                        # a non-ICP requested signal, the reported transform
                        # must stay T_RANSAC — harvesting d_icp on the side
                        # must not change what `signal` means.
                        if signal == "chamfer_icp":
                            gc.transformation = gc.icp_transformation

            # --- Trimmed one-sided surface distance -----------------------
            if needs_distance and obs_down is not None:
                if gc.registration_failed:
                    # Never fall back to an unaligned distance for a failed
                    # registration — it would be silently incomparable.
                    gc.chamfer_score = float("inf")
                else:
                    cad_down = cad_pcd.voxel_down_sample(voxel_size)

                    def _d_trim(transform) -> float:
                        """D_trim with the OBSERVATION carried into the CAD
                        frame by `transform` (None = unaligned diagnostic).
                        This is the T_j of
                        Eq. eq:methods_trimmed_surface_distance and matches
                        the one-sided obs->CAD direction."""
                        src = obs_down
                        if transform is not None:
                            src = o3d.geometry.PointCloud(obs_down)
                            src.transform(transform)
                        return trimmed_chamfer_distance(
                            np.asarray(src.points),
                            np.asarray(cad_down.points),
                            trim_ratio=self.config.chamfer_trim_ratio,
                        )

                    if all_aligned:
                        # Both aligned distances off the one registration.
                        # ICP falling back to T_RANSAC mirrors the thesis rule
                        # T_j = T_ICP if it succeeds, else T_RANSAC.
                        gc.d_ransac = _d_trim(gc.ransac_transformation)
                        gc.d_icp = (_d_trim(gc.icp_transformation)
                                    if gc.icp_transformation is not None
                                    else gc.d_ransac)

                    if signal == "chamfer_ransac":
                        gc.chamfer_score = (gc.d_ransac if all_aligned
                                            else _d_trim(gc.ransac_transformation))
                    elif signal == "chamfer_icp":
                        gc.chamfer_score = (gc.d_icp if all_aligned
                                            else _d_trim(gc.transformation))
                    elif signal == "chamfer_unaligned":
                        gc.chamfer_score = _d_trim(None)
                    # signal == "fitness": no distance is part of that signal,
                    # so chamfer_score stays inf even under all_aligned.

            # Combined geometry score
            gc.geometry_score = self._compute_geometry_score(gc, signal)

            logger.info(
                "  %s: fitness=%.4f D_trim=%.6f geo_score=%.4f%s",
                fc.object_id[:30], gc.ransac_fitness, gc.chamfer_score,
                gc.geometry_score,
                " [registration FAILED]" if gc.registration_failed else "",
            )
            geo_candidates.append(gc)

        # Rank: valid registrations first, then by the signal's score
        # (higher = better), tie-broken by RANSAC fitness (thesis Sec. 3.3 —
        # avoids combining an inlier statistic and a distance with an
        # arbitrary scale factor).
        geo_candidates.sort(
            key=lambda c: (not c.registration_failed, c.geometry_score,
                           c.ransac_fitness),
            reverse=True,
        )

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
            best_transformation=(
                (best.transformation if best.transformation is not None
                 else best.ransac_transformation)
                if best else None
            ),
        )

    def _cached_gedi(self, gedi_mod, pcd, kind: str, key: Optional[str]):
        """GeDi descriptors for one cloud, via the on-disk cache when enabled.

        Falls back to the uncached path when ``config.gedi_cache_dir`` is unset
        or no cache key is available, so existing callers are unaffected.
        Validation is by input-cloud fingerprint (see
        ``GeDiDescriptorModule.cloud_fingerprint``), not by stored settings.
        """
        cache_dir = getattr(self.config, "gedi_cache_dir", None)
        if not cache_dir or not key:
            return gedi_mod.compute(pcd)
        path = os.path.join(cache_dir, kind, f"{key}.npz")
        return gedi_mod.compute_and_cache(
            pcd, path, provenance={"kind": kind, "key": key,
                                   "voxel_size": self.config.voxel_size})

    def _gedi_ransac(
        self,
        obs_gedi,
        cad_pcd,
        voxel_size: float,
        cad_id: Optional[str] = None,
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
        # Descriptor extraction is per CLOUD (~3.4 s) while this RANSAC fit is
        # per PAIR (~0.43 s).  Recomputing the CAD descriptors on every pair
        # therefore costs ~8x the fit itself and scales with the shortlist
        # depth K, which is what made a deeper K unaffordable.  With a cache
        # dir configured the same CAD is extracted once for the whole grid.
        cad_gedi = self._cached_gedi(gedi_mod, cad_pcd, "cad", cad_id)

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

    def _icp_refine(self, observed_pcd, cad_pcd, init_transform):
        """Refine a RANSAC transform with point-to-plane ICP.

        Returns the Open3D registration result, or None if ICP could not be
        run (e.g. normals unavailable) — in which case the caller keeps
        T_RANSAC, per the thesis' T_j definition.
        """
        import open3d as o3d

        try:
            target = cad_pcd
            if not target.has_normals():
                target.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=(self.config.voxel_size or 0.005) * 4,
                        max_nn=30,
                    )
                )
            return o3d.pipelines.registration.registration_icp(
                observed_pcd,
                target,
                self.config.icp_threshold,
                init_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.config.icp_max_iterations
                ),
            )
        except Exception as exc:
            logger.warning("ICP refinement failed: %s", exc)
            return None

    @staticmethod
    def _compute_geometry_score(gc: GeometryCandidate, signal: str) -> float:
        """Single geometry score for ranking (higher = better).

        Failed registrations get -inf so they always sort last, rather than
        being compared via a raw unaligned distance (thesis Sec. 3.3).
        """
        if gc.registration_failed:
            return float("-inf")
        if signal == "fitness":
            # RANSAC fitness, not the raw inlier count: fitness is less
            # sensitive to the number of sampled keypoints (thesis Sec. 3.3).
            return gc.ransac_fitness
        # All distance signals rank by increasing D_trim, i.e. by -D_trim
        # descending.  No cross-signal scale factor is applied; the RANSAC
        # fitness tie-break lives in the sort key.
        if np.isinf(gc.chamfer_score):
            return float("-inf")
        return -gc.chamfer_score

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
            # Deterministic sampling: sample_points_uniformly() draws from
            # Open3D's GLOBAL RNG and takes no seed argument (0.19), so
            # without this the same CAD yields a different cloud on every
            # call — irreproducible geometry scores, and a descriptor cache
            # that can never hit.  Seed from the CAD path so the cloud is a
            # pure function of the model, not of call order.
            # Masked into the non-negative int32 range — Open3D's seed() is
            # bound to a 32-bit signed int and rejects larger values.
            o3d.utility.random.seed(
                int(hashlib.sha1(os.path.basename(cad_path).encode()
                                 ).hexdigest()[:8], 16) % (2 ** 31 - 1))
            pcd = mesh.sample_points_uniformly(number_of_points=n_points)
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
            return pcd
        except Exception as exc:
            logger.warning("CAD load failed (%s): %s", cad_path, exc)
            return None
