# =============================================================================
# pipeline/step8_pose_estimation.py – Thesis Step C (part 2): 6D Pose
# =============================================================================
#
# Thesis reference: Section 3.4, Step C — Pose Estimation
#
# Estimates the 6D pose (3D rotation + 3D translation) of the detected
# object in camera coordinates, using the scaled CAD model from Step 7.
#
# Two backends:
#
#   • FoundationPose (Wen et al., CVPR 2024)
#     Model-based 6D pose estimation via render-and-compare with neural
#     object field.  Runs in a separate Docker container, called via HTTP
#     (same isolation pattern as GeDi).
#     Ref: https://github.com/NVlabs/FoundationPose
#
#   • ICP (Besl & McKay, 1992) — classical fallback
#     FPFH-based RANSAC (Rusu et al., 2009) for coarse alignment, then
#     Point-to-Plane ICP refinement.  Correspondence distance: 3×voxel_size
#     (thesis Sec. 3.4).  When a coarse alignment from Step 7 is available,
#     RANSAC is skipped and ICP starts from that transform.
#
# Inputs:
#   - Scaled CAD model (Step 7)
#   - RGB image (original)
#   - Depth image (required for FoundationPose, optional for ICP)
#   - Segmentation mask (Step 1)
#   - Camera intrinsics
#
# Outputs:
#   - 4×4 transformation matrix [R|t] (camera ← object)
#   - Confidence estimate
# =============================================================================

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .config import PipelineConfig
from .step2_pointcloud import PointCloudResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenstruktur für Pose-Ergebnisse
# ---------------------------------------------------------------------------

@dataclass
class PoseEstimationResult:
    """Ergebnis der 6D Pose Estimation (Schritt 8).

    Die Pose beschreibt die Transformation vom Objekt-Koordinatensystem
    ins Kamera-Koordinatensystem: p_cam = R @ p_obj + t

    Attributes:
        pose_matrix: 4×4 homogene Transformationsmatrix [R|t; 0 0 0 1].
        rotation: 3×3 Rotationsmatrix R.
        translation: 3D-Translationsvektor t (in Metern).
        confidence: Schätzung der Pose-Qualität (0–1).
        method: Verwendete Methode.
        cad_model_path: Pfad zum verwendeten CAD-Modell.
        scale_factor: Angewandter Skalierungsfaktor.
    """
    pose_matrix: np.ndarray      # (4, 4)
    rotation: np.ndarray         # (3, 3)
    translation: np.ndarray      # (3,)
    confidence: float
    method: str
    cad_model_path: str = ""
    scale_factor: float = 1.0


# ---------------------------------------------------------------------------
# Pose Estimation Modul
# ---------------------------------------------------------------------------

class PoseEstimator:
    """Schätzt die 6D-Pose eines Objekts relativ zur Kamera.

    Unterstützt mehrere Backends:
    1. foundationpose: NVIDIA FoundationPose (via HTTP to separate container)
    2. icp: Klassische Point-to-Point ICP-Registrierung

    Die ICP-Methode ist als Fallback implementiert und benötigt keine
    externen Modelle.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device

    def estimate(
        self,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        mask: np.ndarray,
        cad_model_path: str,
        scale_factor: float = 1.0,
        observed_pc: Optional[PointCloudResult] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        method: Optional[str] = None,
        initial_pose: Optional[np.ndarray] = None,
    ) -> PoseEstimationResult:
        """Schätzt die 6D-Pose des Objekts."""
        method = method or self.config.pose_method

        logger.info(f"Pose Estimation mit Methode: {method}")

        if method == "foundationpose":
            return self._estimate_foundationpose(
                rgb_image, depth_image, mask, cad_model_path,
                scale_factor, fx, fy, cx, cy, initial_pose,
                observed_pc=observed_pc,
            )
        elif method == "icp":
            return self._estimate_icp(
                observed_pc, cad_model_path, scale_factor, initial_pose,
            )
        else:
            raise ValueError(f"Unbekannte Pose-Methode: {method}")

    # -----------------------------------------------------------------------
    # Methode 1: FoundationPose (via HTTP)
    # -----------------------------------------------------------------------

    def _estimate_foundationpose(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        mask: np.ndarray,
        cad_path: str,
        scale: float,
        fx: Optional[float],
        fy: Optional[float],
        cx: Optional[float],
        cy: Optional[float],
        initial_pose: Optional[np.ndarray] = None,
        observed_pc: Optional[PointCloudResult] = None,
    ) -> PoseEstimationResult:
        """6D Pose via FoundationPose HTTP service.

        Calls the FoundationPose container over the Docker network.
        Falls back to ICP on any error.
        """
        try:
            logger.info("Calling FoundationPose service...")
            if depth is None:
                raise RuntimeError("FoundationPose benoetigt ein Tiefenbild, erhielt aber None.")

            K = self._camera_matrix(fx, fy, cx, cy)
            url = self.config.foundationpose_url.rstrip("/") + "/estimate_pose"

            from .foundationpose_bridge import call_foundationpose

            pose_matrix, fp_conf = call_foundationpose(
                url=url,
                rgb=rgb,
                depth=depth,
                mask=mask,
                K=K,
                cad_path=cad_path,
                scale=scale,
                refine_iter=int(self.config.foundationpose_est_refine_iter),
                debug=int(self.config.foundationpose_debug),
                debug_dir=os.path.join(self.config.output_dir, "foundationpose_debug"),
            )

            return PoseEstimationResult(
                pose_matrix=pose_matrix,
                rotation=np.array(pose_matrix[:3, :3]),
                translation=np.array(pose_matrix[:3, 3]),
                confidence=float(fp_conf),
                method="foundationpose",
                cad_model_path=cad_path,
                scale_factor=scale,
            )

        except Exception as e:
            logger.warning("FoundationPose nicht verfuegbar oder fehlgeschlagen: %s", e)
            logger.info("Fallback auf ICP...")

            if observed_pc is None and depth is not None:
                from .step2_pointcloud import PointCloudGenerator

                pc_gen = PointCloudGenerator(self.config)
                observed_pc = pc_gen.generate(rgb, depth, mask, fx, fy, cx, cy)

            if observed_pc is not None:
                return self._estimate_icp(observed_pc, cad_path, scale, initial_pose)

            return self._identity_pose(cad_path, scale, "foundationpose_fallback")

    def _camera_matrix(
        self,
        fx: Optional[float],
        fy: Optional[float],
        cx: Optional[float],
        cy: Optional[float],
    ) -> np.ndarray:
        """Builds camera intrinsic matrix from args or config defaults."""
        k_fx = float(fx if fx is not None else self.config.camera_fx)
        k_fy = float(fy if fy is not None else self.config.camera_fy)
        k_cx = float(cx if cx is not None else self.config.camera_cx)
        k_cy = float(cy if cy is not None else self.config.camera_cy)

        return np.array(
            [[k_fx, 0.0, k_cx], [0.0, k_fy, k_cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    # -----------------------------------------------------------------------
    # Methode 2: ICP (Iterative Closest Point)
    # -----------------------------------------------------------------------

    def _estimate_icp(
        self,
        observed_pc: Optional[PointCloudResult],
        cad_path: str,
        scale: float,
        initial_pose: Optional[np.ndarray] = None,
    ) -> PoseEstimationResult:
        """6D Pose via ICP registration (Besl & McKay, 1992).

        Pipeline:
        1. CAD model → sampled point cloud (scaled).
        2. FPFH RANSAC (Rusu et al., 2009) for coarse alignment —
           skipped when initial_pose is provided from Step 7.
        3. Point-to-Plane ICP refinement with correspondence distance
           3×voxel_size (thesis Sec. 3.4).
        """
        if observed_pc is None:
            logger.warning("Keine Punktwolke für ICP verfügbar.")
            return self._identity_pose(cad_path, scale, "icp_error")

        try:
            import open3d as o3d
        except ImportError:
            raise ImportError(
                "Open3D für ICP benötigt. Installieren mit: pip install open3d"
            )

        # --- CAD-Modell als Punktwolke laden ---
        cad_mesh = o3d.io.read_triangle_mesh(cad_path)
        if cad_mesh.is_empty():
            logger.warning(f"CAD-Mesh leer: {cad_path}")
            return self._identity_pose(cad_path, scale, "icp_error")

        # Skalieren
        cad_mesh.scale(scale, center=cad_mesh.get_center())
        cad_pcd = cad_mesh.sample_points_uniformly(number_of_points=10000)
        cad_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )

        # --- Beobachtete Punktwolke vorbereiten ---
        source = observed_pc.point_cloud
        if not source.has_normals():
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.01, max_nn=30
                )
            )

        # --- FPFH Features für RANSAC ---
        logger.info("Berechne FPFH Features für RANSAC...")
        voxel_size = self.config.voxel_size or 0.005

        source_down = source.voxel_down_sample(voxel_size)
        target_down = cad_pcd.voxel_down_sample(voxel_size)

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )

        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
        )

        # --- Initiale Transformation ---
        if initial_pose is not None:
            logger.info("Verwende Coarse-Alignment aus Schritt 7 als Startpose.")
            init_transform = initial_pose
        else:
            # --- RANSAC Global Registration ---
            logger.info("RANSAC Global Registration...")
            ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down,
                source_fpfh, target_fpfh,
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
            init_transform = ransac_result.transformation

        # --- ICP Refinement ---
        logger.info("ICP Point-to-Plane Verfeinerung...")
        icp_result = o3d.pipelines.registration.registration_icp(
            source, cad_pcd,
            max_correspondence_distance=voxel_size * 3,  # thesis: 3×voxel_size
            init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.config.icp_max_iterations
            ),
        )

        pose = icp_result.transformation  # (4, 4)
        fitness = icp_result.fitness
        rmse = icp_result.inlier_rmse

        logger.info(
            f"ICP Ergebnis: Fitness={fitness:.4f}, RMSE={rmse:.6f}m"
        )

        return PoseEstimationResult(
            pose_matrix=np.array(pose),
            rotation=np.array(pose[:3, :3]),
            translation=np.array(pose[:3, 3]),
            confidence=float(fitness),
            method="icp",
            cad_model_path=cad_path,
            scale_factor=scale,
        )

    @staticmethod
    def _identity_pose(
        cad_path: str, scale: float, method: str
    ) -> PoseEstimationResult:
        """Gibt eine Identitätspose zurück (kein Ergebnis)."""
        return PoseEstimationResult(
            pose_matrix=np.eye(4),
            rotation=np.eye(3),
            translation=np.zeros(3),
            confidence=0.0,
            method=method,
            cad_model_path=cad_path,
            scale_factor=scale,
        )
