# =============================================================================
# pipeline/step7_scale_estimation.py – Schritt 7: Skalenbestimmung
# =============================================================================
#
# Ziel:
#   Die korrekte Skala des CAD-Modells relativ zur beobachteten Szene
#   bestimmen, damit das Modell physikalisch korrekt platziert werden kann.
#
# Methode:
#   Vergleich der 3D-Bounding-Boxen:
#     scale = size(point_cloud_bbox) / size(cad_model_bbox)
#
#   Dabei wird die Größe entlang jeder Achse oder als maximale Ausdehnung
#   verglichen. Da die beobachtete Punktwolke partiell ist (nur eine Seite
#   sichtbar), wird ein robuster Schätzer verwendet.
#
# Tools:
#   • Open3D – Bounding-Box-Berechnung
#     Ref: http://www.open3d.org/docs/release/
#
#   • Trimesh – Für CAD-Modell-Analyse
#     Ref: https://trimsh.org/
#
# Inputs:
#   - Punktwolke des Objekts (Schritt 2)
#   - Ausgewähltes CAD-Modell (Schritt 6)
#
# Outputs:
#   - Skalierungsfaktor (float)
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
# Datenstruktur für Skalierungsergebnisse
# ---------------------------------------------------------------------------

@dataclass
class ScaleEstimationResult:
    """Ergebnis der Skalenbestimmung (Schritt 7).

    Attributes:
        scale_factor: Verhältnis observed / cad (einheitenlos).
        scale_per_axis: Skalierung pro Achse (x, y, z).
        observed_size: Größe der beobachteten Bounding Box (m).
        cad_size: Größe der CAD-Bounding-Box (Modelleinheiten).
        method: Verwendete Methode.
        confidence: Schätzung der Zuverlässigkeit (0–1).
    """
    scale_factor: float
    scale_per_axis: np.ndarray
    observed_size: np.ndarray
    cad_size: np.ndarray
    method: str
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Skalenschätzer
# ---------------------------------------------------------------------------

class ScaleEstimator:
    """Bestimmt den Skalierungsfaktor zwischen beobachtetem Objekt und CAD-Modell.

    Da die beobachtete Punktwolke partiell ist (nur sichtbare Oberfläche),
    werden mehrere Strategien unterstützt:

    1. max_extent: Verhältnis der maximalen Ausdehnung beider BBs.
       Robuster bei partiellen Ansichten.

    2. per_axis: Skalierung pro Achse (nur sinnvoll bei guter Ausrichtung).

    3. median_axis: Median der achsweisen Skalierungen.
       Kompromiss zwischen Robustheit und Genauigkeit.

    Usage:
        >>> estimator = ScaleEstimator(config)
        >>> result = estimator.estimate(point_cloud, "path/to/cad.obj")
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def estimate(
        self,
        observed_pc: PointCloudResult,
        cad_model_path: str,
        method: str = "max_extent",
    ) -> ScaleEstimationResult:
        """Schätzt den Skalierungsfaktor.

        Args:
            observed_pc: Punktwolke des beobachteten Objekts (Schritt 2).
            cad_model_path: Pfad zum CAD-Modell (OBJ, PLY, GLB, ...).
            method: "max_extent" | "per_axis" | "median_axis".

        Returns:
            ScaleEstimationResult mit dem Skalierungsfaktor.
        """
        # --- CAD-Modell Bounding Box ---
        cad_bbox_size = self._get_cad_bbox_size(cad_model_path)

        if cad_bbox_size is None or np.all(cad_bbox_size == 0):
            logger.warning("CAD-Modell-BBox konnte nicht bestimmt werden.")
            return ScaleEstimationResult(
                scale_factor=1.0,
                scale_per_axis=np.ones(3),
                observed_size=observed_pc.bbox_size,
                cad_size=np.zeros(3),
                method=method,
                confidence=0.0,
            )

        observed_size = observed_pc.bbox_size  # (3,)

        # --- Skalierung berechnen ---
        if method == "max_extent":
            result = self._max_extent_scale(observed_size, cad_bbox_size)
        elif method == "per_axis":
            result = self._per_axis_scale(observed_size, cad_bbox_size)
        elif method == "median_axis":
            result = self._median_axis_scale(observed_size, cad_bbox_size)
        else:
            raise ValueError(f"Unbekannte Methode: {method}")

        result.observed_size = observed_size
        result.cad_size = cad_bbox_size
        result.method = method

        logger.info(
            f"Skalenbestimmung ({method}): Faktor={result.scale_factor:.4f}, "
            f"Observed={observed_size}, CAD={cad_bbox_size}"
        )

        return result

    @staticmethod
    def _get_cad_bbox_size(cad_path: str) -> Optional[np.ndarray]:
        """Berechnet die Bounding-Box-Größe eines CAD-Modells.

        Versucht zunächst Trimesh, dann Open3D als Fallback.
        """
        try:
            import trimesh
            mesh = trimesh.load(cad_path, force="mesh")
            bbox_size = mesh.bounding_box.extents  # (3,)
            return np.array(bbox_size, dtype=np.float64)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Trimesh-Fehler bei {cad_path}: {e}")

        try:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(cad_path)
            if mesh.is_empty():
                return None
            bbox = mesh.get_axis_aligned_bounding_box()
            size = np.array(bbox.get_max_bound()) - np.array(bbox.get_min_bound())
            return size
        except Exception as e:
            logger.warning(f"Fehler beim Laden des CAD-Modells {cad_path}: {e}")
            return None

    def _max_extent_scale(
        self, observed: np.ndarray, cad: np.ndarray
    ) -> ScaleEstimationResult:
        """Skalierung basierend auf maximaler Ausdehnung.

        Robusteste Methode für partielle Punktwolken:
        Die größte Dimension der beobachteten Wolke wird mit der größten
        Dimension des CAD-Modells verglichen.

        HINWEIS: Bei partiellen Ansichten ist die beobachtete Größe kleiner
        als die tatsächliche, daher ist die berechnete Skala eine Unterschätzung.
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

        # Konfidenz: Höher wenn das Aspektverhältnis ähnlich ist
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
    def _per_axis_scale(
        observed: np.ndarray, cad: np.ndarray
    ) -> ScaleEstimationResult:
        """Achsweise Skalierung.

        Nur sinnvoll wenn das CAD-Modell und die Beobachtung gleich
        ausgerichtet sind (z.B. nach ICP oder bekannter Pose).
        """
        safe_cad = np.where(cad > 1e-8, cad, 1.0)
        scale_per_axis = observed / safe_cad
        scale_factor = float(scale_per_axis.mean())

        return ScaleEstimationResult(
            scale_factor=scale_factor,
            scale_per_axis=scale_per_axis,
            observed_size=observed,
            cad_size=cad,
            method="per_axis",
            confidence=0.7,  # Niedrigere Konfidenz wegen Ausrichtungsannahme
        )

    @staticmethod
    def _median_axis_scale(
        observed: np.ndarray, cad: np.ndarray
    ) -> ScaleEstimationResult:
        """Median der achsweisen Skalierungen.

        Robuster als per_axis, da Ausreißer (z.B. durch partielle Sicht)
        weniger Einfluss haben.
        """
        safe_cad = np.where(cad > 1e-8, cad, 1.0)
        scale_per_axis = observed / safe_cad
        scale_factor = float(np.median(scale_per_axis))

        return ScaleEstimationResult(
            scale_factor=scale_factor,
            scale_per_axis=scale_per_axis,
            observed_size=observed,
            cad_size=cad,
            method="median_axis",
            confidence=0.8,
        )

    @staticmethod
    def apply_scale(cad_model_path: str, scale_factor: float, output_path: str) -> str:
        """Skaliert ein CAD-Modell und speichert es.

        Args:
            cad_model_path: Pfad zum Original-CAD-Modell.
            scale_factor: Skala (z.B. 0.5 = halbieren).
            output_path: Pfad für das skalierte Modell.

        Returns:
            Pfad zum skalierten Modell.
        """
        try:
            import trimesh
            mesh = trimesh.load(cad_model_path, force="mesh")
            mesh.apply_scale(scale_factor)
            mesh.export(output_path)
            logger.info(f"Skaliertes Modell gespeichert: {output_path} (×{scale_factor:.4f})")
            return output_path
        except ImportError:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(cad_model_path)
            mesh.scale(scale_factor, center=mesh.get_center())
            o3d.io.write_triangle_mesh(output_path, mesh)
            logger.info(f"Skaliertes Modell gespeichert: {output_path} (×{scale_factor:.4f})")
            return output_path
