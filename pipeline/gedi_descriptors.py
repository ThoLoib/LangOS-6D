# =============================================================================
# pipeline/gedi_descriptors.py -- GeDi Local Geometric Descriptors
# =============================================================================
#
# Wraps the GeDi descriptor (Poiesi & Boscaini, IEEE T-PAMI 2022) for use
# in OSCAR+ Sub-step B2 (geometry re-ranking) and Step C (coarse alignment).
#
# GeDi computes rotation- and scale-invariant local 3D descriptors via a
# learned local reference frame (LRF) + PointNet++ backbone.  The descriptors
# are packed into Open3D's Feature format so they can be used directly with
# Open3D's RANSAC-based global registration -- the same interface as FPFH.
#
# Repository: https://github.com/fabiopoiesi/gedi
# Paper: "Learning General and Distinctive 3D Local Deep Descriptors
#         for Point Cloud Registration"
#
# Dependencies (inside Docker container):
#   - GeDi repo cloned + pointnet2_ops_lib compiled for CUDA 12.2
#   - torchgeometry (or kornia as replacement)
#   - open3d.ml.torch (for radius_search)
#
# Usage:
#   >>> gedi_module = GeDiDescriptorModule(config)
#   >>> keypoints, features = gedi_module.compute(point_cloud)
#   >>> # features is an o3d.pipelines.registration.Feature (dim x N)
# =============================================================================

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import PipelineConfig

logger = logging.getLogger(__name__)


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

    Lazy-loads the GeDi model on first use.  Descriptors are returned in
    Open3D Feature format for direct use with RANSAC registration.

    The module also supports caching descriptors for CAD partial views
    to avoid recomputation during B2 re-ranking.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._gedi = None
        self._available = None  # None = not checked, True/False after check

    @property
    def available(self) -> bool:
        """Check whether GeDi can be loaded (repo present, deps installed)."""
        if self._available is not None:
            return self._available

        repo_path = self.config.gedi_repo_path
        checkpoint = self.config.gedi_checkpoint

        if not repo_path or not os.path.isdir(repo_path):
            logger.warning(
                "GeDi repo not found at '%s'. Set config.gedi_repo_path.",
                repo_path,
            )
            self._available = False
            return False

        if not checkpoint or not os.path.isfile(checkpoint):
            logger.warning(
                "GeDi checkpoint not found at '%s'. Set config.gedi_checkpoint.",
                checkpoint,
            )
            self._available = False
            return False

        # Try importing critical deps
        try:
            import torch  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def _load_model(self):
        """Load the GeDi network and LRF module."""
        if self._gedi is not None:
            return

        if not self.available:
            raise RuntimeError(
                "GeDi is not available. Check gedi_repo_path and gedi_checkpoint."
            )

        repo_path = self.config.gedi_repo_path
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        # Import GeDi class from the repo
        from gedi import GeDi  # type: ignore

        gedi_config = {
            "dim": self.config.gedi_dim,
            "samples_per_batch": self.config.gedi_samples_per_batch,
            "samples_per_patch_lrf": self.config.gedi_samples_per_patch_lrf,
            "samples_per_patch_out": self.config.gedi_samples_per_patch_out,
            "r_lrf": self.config.gedi_r_lrf,
            "fchkpt_gedi_net": self.config.gedi_checkpoint,
        }

        logger.info("Loading GeDi model (dim=%d, r_lrf=%.2f)...",
                     gedi_config["dim"], gedi_config["r_lrf"])
        self._gedi = GeDi(config=gedi_config)
        logger.info("GeDi model loaded successfully.")

    def compute(
        self,
        point_cloud,
        num_keypoints: Optional[int] = None,
    ) -> GeDiResult:
        """Compute GeDi descriptors on an Open3D point cloud.

        Args:
            point_cloud: Open3D PointCloud (must have points).
            num_keypoints: Number of keypoints to sample (default: config value).

        Returns:
            GeDiResult with keypoints (PointCloud), features (Feature), and
            raw descriptors (ndarray).
        """
        import open3d as o3d
        import torch

        self._load_model()

        n_kp = num_keypoints or self.config.gedi_num_keypoints
        pts_np = np.asarray(point_cloud.points)

        if len(pts_np) < 100:
            logger.warning("Point cloud too small (%d pts) for GeDi.", len(pts_np))
            return self._empty_result()

        # Sample keypoints randomly
        n_kp = min(n_kp, len(pts_np))
        kp_indices = np.random.choice(len(pts_np), n_kp, replace=False)
        kp_pts = torch.tensor(pts_np[kp_indices]).float()
        pcd_tensor = torch.tensor(pts_np).float()

        # Compute descriptors via GeDi
        descriptors = self._gedi.compute(pts=kp_pts, pcd=pcd_tensor)

        # Pack into Open3D format
        kp_pcd = o3d.geometry.PointCloud()
        kp_pcd.points = o3d.utility.Vector3dVector(pts_np[kp_indices])

        features = o3d.pipelines.registration.Feature()
        features.data = descriptors.T  # (dim, N_keypoints)

        logger.debug(
            "GeDi: computed %d descriptors (dim=%d) on %d-point cloud.",
            n_kp, descriptors.shape[1], len(pts_np),
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
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
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
