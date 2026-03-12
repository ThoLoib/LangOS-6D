# =============================================================================
# pipeline/step8_pose_estimation.py – Schritt 8: 6D Pose Estimation
# =============================================================================
#
# Ziel:
#   Die 6D-Pose (3D-Rotation + 3D-Translation) des erkannten Objekts
#   in der Kamera-Koordinatenebene schätzen, unter Verwendung des
#   skalierten CAD-Modells.
#
# Optionen:
#   • FoundationPose – NVIDIA, State-of-the-Art für model-based Pose
#     Ref: https://github.com/NVlabs/FoundationPose
#     Paper: "FoundationPose: Unified 6D Object Pose Estimation and
#             Tracking of Novel Objects" (Wen et al., 2024)
#
#   • MegaPose – Model-based Pose Estimation mit CNOS
#     Ref: https://github.com/megapose6d/megapose6d
#     Paper: "MegaPose: 6D Pose Estimation of Novel Objects via
#             Render & Compare" (Labbe et al., 2022)
#
#   • ICP + RANSAC – Klassische geometrische Registrierung
#     Ref: Open3D ICP Tutorial
#          http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
#     Paper: "A Method for Registration of 3-D Shapes" (Besl & McKay, 1992)
#
# Inputs:
#   - Skaliertes CAD-Modell (Schritt 7)
#   - RGB-Bild (original)
#   - Tiefenbild (optional, für ICP)
#   - Segmentierungsmaske (Schritt 1)
#   - Kameraintrinsics
#
# Outputs:
#   - 4×4 Transformationsmatrix [R|t] (Kamera ← Objekt)
#   - Konfidenzschätzung
# =============================================================================

import logging
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
    1. foundationpose: NVIDIA FoundationPose (State-of-the-Art)
    2. megapose: MegaPose (Render & Compare)
    3. icp: Klassische Point-to-Point ICP-Registrierung

    Die ICP-Methode ist als Fallback implementiert und benötigt keine
    externen Modelle. FoundationPose und MegaPose erfordern separate
    Installation.

    Usage:
        >>> estimator = PoseEstimator(config)
        >>> result = estimator.estimate(
        ...     rgb_image, depth_image, mask,
        ...     cad_model_path="cad/object.obj",
        ...     scale_factor=0.85
        ... )
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
        """Schätzt die 6D-Pose des Objekts.

        Args:
            rgb_image: RGB-Bild (H, W, 3), uint8.
            depth_image: Tiefenbild (H, W), float32 in Metern (für ICP).
            mask: Segmentierungsmaske (H, W), bool.
            cad_model_path: Pfad zum (skalierten) CAD-Modell.
            scale_factor: Angewandter Skalierungsfaktor.
            observed_pc: Vorberechnete Punktwolke (für ICP, spart Neuberechnung).
            fx, fy, cx, cy: Kameraintrinsics (überschreiben Config).
            method: Backend ("foundationpose" | "megapose" | "icp").

        Returns:
            PoseEstimationResult mit Transformation und Konfidenz.
        """
        method = method or self.config.pose_method

        logger.info(f"Pose Estimation mit Methode: {method}")

        if method == "foundationpose":
            return self._estimate_foundationpose(
                rgb_image, depth_image, mask, cad_model_path,
                scale_factor, fx, fy, cx, cy, initial_pose,
            )
        elif method == "megapose":
            return self._estimate_megapose(
                rgb_image, mask, cad_model_path, scale_factor,
                fx, fy, cx, cy,
            )
        elif method == "icp":
            return self._estimate_icp(
                observed_pc, cad_model_path, scale_factor, initial_pose,
            )
        else:
            raise ValueError(f"Unbekannte Pose-Methode: {method}")

    # -----------------------------------------------------------------------
    # Methode 1: FoundationPose (NVIDIA)
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
    ) -> PoseEstimationResult:
        """6D Pose via FoundationPose.

        FoundationPose nutzt einen neuronalen Render-and-Compare-Ansatz:
        1. Initiale Pose-Hypothesen generieren.
        2. CAD-Modell rendern und mit dem Eingabebild vergleichen.
        3. Iterativ verfeinern.

        Ref: https://github.com/NVlabs/FoundationPose
        Paper: Wen et al., "FoundationPose: Unified 6D Object Pose
               Estimation and Tracking of Novel Objects", CVPR 2024.

        HINWEIS: Erfordert separate Installation von FoundationPose.
        """
        try:
            # FoundationPose Integration
            # Das folgende ist ein Interface-Template – die tatsächliche
            # Integration hängt von der FoundationPose-Installation ab.
            logger.info("Versuche FoundationPose zu verwenden...")

            # Beispiel-Interface (muss an die tatsächliche API angepasst werden):
            # from foundationpose import FoundationPoseEstimator
            # estimator = FoundationPoseEstimator(cad_path, scale)
            # pose = estimator.estimate(rgb, depth, mask, K)

            raise NotImplementedError(
                "FoundationPose-Integration noch nicht implementiert.\n"
                "Bitte das FoundationPose-Repository klonen und konfigurieren:\n"
                "  https://github.com/NVlabs/FoundationPose\n"
                "Alternativ: method='icp' als Fallback verwenden."
            )

        except (ImportError, NotImplementedError) as e:
            logger.warning(f"FoundationPose nicht verfügbar: {e}")
            logger.info("Fallback auf ICP...")
            # ICP als Fallback wenn Tiefe verfügbar
            if depth is not None:
                from .step2_pointcloud import PointCloudGenerator
                pc_gen = PointCloudGenerator(self.config)
                observed_pc = pc_gen.generate(
                    rgb, depth, mask, fx, fy, cx, cy
                )
                if observed_pc:
                    return self._estimate_icp(observed_pc, cad_path, scale, initial_pose)

            return self._identity_pose(cad_path, scale, "foundationpose_fallback")

    # -----------------------------------------------------------------------
    # Methode 2: MegaPose
    # -----------------------------------------------------------------------

    def _estimate_megapose(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        cad_path: str,
        scale: float,
        fx: Optional[float],
        fy: Optional[float],
        cx: Optional[float],
        cy: Optional[float],
    ) -> PoseEstimationResult:
        """6D Pose via MegaPose.

        MegaPose ist ein Render-and-Compare-Ansatz, der:
        1. CAD-Modell aus vielen Blickwinkeln rendert.
        2. Features mit dem Eingabebild vergleicht.
        3. Die beste Pose iterativ verfeinert.

        Ref: https://github.com/megapose6d/megapose6d
        Paper: Labbe et al., "MegaPose: 6D Pose Estimation of Novel
               Objects via Render & Compare", CoRL 2022.

        HINWEIS: Erfordert separate Installation von MegaPose.
        """
        try:
            logger.info("Versuche MegaPose zu verwenden...")

            # Beispiel-Interface (anpassbar):
            # from megapose.inference import MegaPoseInference
            # inference = MegaPoseInference(cad_path)
            # pose = inference.run(rgb, mask, K)

            raise NotImplementedError(
                "MegaPose-Integration noch nicht implementiert.\n"
                "Bitte das MegaPose-Repository klonen und konfigurieren:\n"
                "  https://github.com/megapose6d/megapose6d\n"
                "Alternativ: method='icp' als Fallback verwenden."
            )

        except (ImportError, NotImplementedError) as e:
            logger.warning(f"MegaPose nicht verfügbar: {e}")
            return self._identity_pose(cad_path, scale, "megapose_fallback")

    # -----------------------------------------------------------------------
    # Methode 3: ICP (Iterative Closest Point)
    # -----------------------------------------------------------------------

    def _estimate_icp(
        self,
        observed_pc: Optional[PointCloudResult],
        cad_path: str,
        scale: float,
        initial_pose: Optional[np.ndarray] = None,
    ) -> PoseEstimationResult:
        """6D Pose via ICP-Registrierung.

        ICP (Iterative Closest Point) registriert die beobachtete
        Punktwolke mit dem CAD-Modell durch iterative Minimierung
        der Punkt-zu-Punkt-Distanz.

        Pipeline:
        1. CAD-Modell laden und Punktwolke sampeln.
        2. RANSAC für grobe initiale Ausrichtung.
        3. Point-to-Plane ICP für feine Verfeinerung.

        Ref: Open3D ICP Registration Tutorial
             http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
        Paper: Besl & McKay, "A Method for Registration of 3-D Shapes",
               IEEE TPAMI 1992.

        Args:
            observed_pc: Punktwolke des beobachteten Objekts.
            cad_path: Pfad zum CAD-Modell.
            scale: Skalierungsfaktor.

        Returns:
            PoseEstimationResult.
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
            max_correspondence_distance=self.config.icp_threshold,
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
