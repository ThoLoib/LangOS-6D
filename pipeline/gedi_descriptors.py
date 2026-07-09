# =============================================================================
# pipeline/gedi_descriptors.py -- GeDi Local Geometric Descriptors
# =============================================================================
#
# HTTP bridge to the GeDi descriptor service (Poiesi & Boscaini, IEEE
# T-PAMI 2022, "Learning General and Distinctive 3D Local Deep Descriptors
# for Point Cloud Registration").
#
# Used in:
#   • Sub-step B2 (geometry re-ranking) — RANSAC inlier count as S_GeDi
#   • Step C (coarse alignment) — RANSAC transform as ICP initialisation
#
# GeDi is a PointNet++-based (Qi et al., NeurIPS 2017) local descriptor
# trained on 3DMatch (Zeng et al., CVPR 2017).  It produces per-keypoint
# 32-dim descriptors suitable for feature-matching RANSAC (Fischler &
# Bolles, 1981).
#
# GeDi runs in a separate Docker container (Dockerfile.gedi) to isolate
# its CUDA 11.x / PyTorch 2.0.1 stack from OSCAR's CUDA 12.2 environment
# — the same container-isolation pattern used for FoundationPose (Wen et
# al., 2024).
#
# Fallback: when the GeDi service is unavailable, callers fall back to
# FPFH descriptors (Rusu et al., ICRA 2009).
#
# Repository: https://github.com/fabiopoiesi/gedi
#
# Usage:
#   >>> gedi_module = GeDiDescriptorModule(config)
#   >>> result = gedi_module.compute(point_cloud)
#   >>> # result.features is an o3d.pipelines.registration.Feature (dim x N)
# =============================================================================

import base64
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import PipelineConfig

logger = logging.getLogger(__name__)

# Default GeDi service URL (docker-compose service name)
_DEFAULT_GEDI_URL = "http://gedi:5060"


@dataclass
class GeDiResult:
    """Result of GeDi descriptor computation.

    Attributes:
        keypoints: Open3D PointCloud of sampled keypoints.
        features: Open3D Feature (dim x N_keypoints) for RANSAC.
        descriptors_np: Raw descriptor array (N_keypoints, dim).
    """
    keypoints: object      # o3d.geometry.PointCloud
    features: object       # o3d.pipelines.registration.Feature
    descriptors_np: np.ndarray


class GeDiDescriptorModule:
    """Computes GeDi local geometric descriptors on Open3D point clouds.

    Calls the GeDi Docker service via HTTP. Descriptors are returned in
    Open3D Feature format for direct use with RANSAC registration.

    Supports disk caching for CAD partial views to avoid repeated HTTP
    calls during B2 re-ranking.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._available = None  # None = not checked, True/False after check
        self._gedi_url = getattr(config, "gedi_url", _DEFAULT_GEDI_URL)

    @property
    def available(self) -> bool:
        """Check whether the GeDi service is reachable."""
        if self._available is not None:
            return self._available

        try:
            import httpx
            resp = httpx.get(
                f"{self._gedi_url}/health",
                timeout=5.0,
            )
            if resp.status_code == 200:
                logger.info("GeDi service available at %s", self._gedi_url)
                self._available = True
            else:
                logger.warning(
                    "GeDi service returned %d at %s",
                    resp.status_code, self._gedi_url,
                )
                self._available = False
        except Exception as exc:
            logger.warning(
                "GeDi service not reachable at %s: %s. "
                "Start it with: docker compose up -d gedi",
                self._gedi_url, exc,
            )
            self._available = False

        return self._available

    def compute(
        self,
        point_cloud,
        num_keypoints: Optional[int] = None,
    ) -> GeDiResult:
        """Compute GeDi descriptors on an Open3D point cloud via HTTP.

        Args:
            point_cloud: Open3D PointCloud (must have points).
            num_keypoints: Number of keypoints to sample (default: config value).

        Returns:
            GeDiResult with keypoints (PointCloud), features (Feature), and
            raw descriptors (ndarray).
        """
        import httpx
        import open3d as o3d

        pts_np = np.asarray(point_cloud.points).astype(np.float32)

        if len(pts_np) < 100:
            logger.warning("Point cloud too small (%d pts) for GeDi.", len(pts_np))
            return self._empty_result()

        n_kp = num_keypoints or self.config.gedi_num_keypoints

        # Encode point cloud as base64
        pts_b64 = base64.b64encode(pts_np.tobytes()).decode("ascii")

        # Call GeDi service
        try:
            resp = httpx.post(
                f"{self._gedi_url}/compute_descriptors",
                json={
                    "points": pts_b64,
                    "num_keypoints": n_kp,
                },
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as exc:
            logger.warning("GeDi HTTP call failed: %s", exc)
            return self._empty_result()

        if result.get("num_keypoints", 0) == 0:
            logger.warning("GeDi returned 0 descriptors.")
            return self._empty_result()

        # Decode response
        kp_indices = np.array(result["keypoint_indices"], dtype=int)
        desc_bytes = base64.b64decode(result["descriptors"])
        dim = result["dim"]
        descriptors = np.frombuffer(desc_bytes, dtype=np.float32).reshape(-1, dim)

        # Pack into Open3D format
        kp_pcd = o3d.geometry.PointCloud()
        kp_pcd.points = o3d.utility.Vector3dVector(pts_np[kp_indices])

        features = o3d.pipelines.registration.Feature()
        features.data = descriptors.T  # (dim, N_keypoints)

        logger.debug(
            "GeDi: computed %d descriptors (dim=%d) on %d-point cloud.",
            len(kp_indices), dim, len(pts_np),
        )

        return GeDiResult(
            keypoints=kp_pcd,
            features=features,
            descriptors_np=descriptors,
        )

    def compute_and_cache(
        self,
        point_cloud,
        cache_path: str,
        num_keypoints: Optional[int] = None,
        force: bool = False,
    ) -> GeDiResult:
        """Compute descriptors with disk caching.

        Args:
            point_cloud: Open3D PointCloud.
            cache_path: Path to save/load cached descriptors (.npz).
            num_keypoints: Number of keypoints.
            force: Recompute even if cache exists.

        Returns:
            GeDiResult.
        """
        import open3d as o3d

        if not force and os.path.isfile(cache_path):
            try:
                data = np.load(cache_path)
                kp_pcd = o3d.geometry.PointCloud()
                kp_pcd.points = o3d.utility.Vector3dVector(data["keypoints"])
                features = o3d.pipelines.registration.Feature()
                features.data = data["descriptors"].T
                logger.debug("GeDi cache hit: %s", cache_path)
                return GeDiResult(
                    keypoints=kp_pcd,
                    features=features,
                    descriptors_np=data["descriptors"],
                )
            except Exception as exc:
                logger.warning("GeDi cache load failed (%s): %s", cache_path, exc)

        result = self.compute(point_cloud, num_keypoints)

        # Save to cache
        if result.descriptors_np.size > 0:
            try:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    keypoints=np.asarray(result.keypoints.points),
                    descriptors=result.descriptors_np,
                )
                logger.debug("GeDi cache saved: %s", cache_path)
            except Exception as exc:
                logger.warning("GeDi cache save failed: %s", exc)

        return result

    @staticmethod
    def _empty_result() -> GeDiResult:
        """Return an empty result for degenerate cases."""
        import open3d as o3d
        return GeDiResult(
            keypoints=o3d.geometry.PointCloud(),
            features=o3d.pipelines.registration.Feature(),
            descriptors_np=np.empty((0, 32)),
        )
