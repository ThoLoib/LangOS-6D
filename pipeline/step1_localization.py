# =============================================================================
# pipeline/step1_localization.py – Schritt 1: Objekt lokalisieren
# =============================================================================
#
# Ziel:
#   Aus einem RGB-Bild und einem Sprachprompt (z.B. "greife nach der
#   Mayonnaisetube") das Zielobjekt lokalisieren und eine präzise
#   Segmentierungsmaske erzeugen.
#
# Pipeline:
#   prompt → GroundingDINO (Bounding Box) → SAM2 (Segmentierungsmaske)
#
# Modelle:
#   • GroundingDINO – Open-Set Object Detection mit Sprachprompts
#     Ref: https://github.com/IDEA-Research/GroundingDINO
#     Paper: "Grounding DINO: Marrying DINO with Grounded Pre-Training
#             for Open-Set Object Detection" (Liu et al., 2023)
#
#   • SAM2 – Segment Anything Model 2 (Meta)
#     Ref: https://github.com/facebookresearch/segment-anything-2
#     Paper: "SAM 2: Segment Anything in Images and Videos" (Ravi et al., 2024)
#
#   • LangSAM – Kombination von GroundingDINO + SAM in einer API
#     Ref: https://github.com/luca-medeiros/lang-segment-anything
#
# Outputs:
#   - RGB-Bild (original)
#   - Segmentierungsmaske (binär, H×W)
#   - Bounding Box [x_min, y_min, x_max, y_max]
#   - ROI-Ausschnitt (zugeschnittenes Bild des Objekts)
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import torch
from PIL import Image

from .config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenstruktur für Lokalisierungsergebnisse
# ---------------------------------------------------------------------------

@dataclass
class LocalizationResult:
    """Ergebnis der Objektlokalisierung (Schritt 1).

    Attributes:
        rgb_image: Originales RGB-Bild (PIL).
        mask: Binäre Segmentierungsmaske (H, W), bool.
        bbox: Bounding Box [x_min, y_min, x_max, y_max] in Pixeln.
        roi_image: Zugeschnittenes Bild des erkannten Objekts.
        confidence: Konfidenz der Detektion.
        prompt: Verwendeter Sprachprompt.
    """
    rgb_image: Image.Image
    mask: np.ndarray
    bbox: List[float]
    roi_image: Image.Image
    confidence: float
    prompt: str


# ---------------------------------------------------------------------------
# Lokalisierungsmodul
# ---------------------------------------------------------------------------

class ObjectLocalizer:
    """Lokalisiert Objekte in RGB-Bildern mit GroundingDINO + SAM.

    Nutzt LangSAM als komfortablen Wrapper, der beide Modelle kombiniert.

    Usage:
        >>> localizer = ObjectLocalizer(config)
        >>> result = localizer.localize(image, "mayonnaise tube")
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self.model = None  # Lazy-Loading

    def _load_model(self):
        """Lädt das LangSAM-Modell (GroundingDINO + SAM) bei Erstverwendung.

        LangSAM verbindet GroundingDINO zur Texterkennung mit SAM zur
        präzisen Segmentierung in einer einzigen API.
        Ref: https://github.com/luca-medeiros/lang-segment-anything
        """
        if self.model is not None:
            return

        logger.info("Lade LangSAM-Modell (GroundingDINO + SAM)...")
        try:
            from lang_sam import LangSAM
            self.model = LangSAM()
            logger.info("LangSAM erfolgreich geladen.")
        except ImportError:
            raise ImportError(
                "LangSAM nicht installiert. Installieren mit:\n"
                "  pip install lang-sam\n"
                "Ref: https://github.com/luca-medeiros/lang-segment-anything"
            )

    def localize(
        self,
        rgb_image: Image.Image,
        prompt: str,
        top_k: int = 1,
    ) -> Optional[LocalizationResult]:
        """Lokalisiert ein Objekt im Bild anhand eines Sprachprompts.

        Args:
            rgb_image: RGB-Eingabebild (PIL).
            prompt: Natürlichsprachige Beschreibung des Zielobjekts,
                    z.B. "mayonnaise tube" oder "Mayonnaisetube".
            top_k: Anzahl der gewünschten Detektionen (Standard: 1, beste).

        Returns:
            LocalizationResult mit Maske, BBox und ROI-Ausschnitt,
            oder None falls kein Objekt gefunden wurde.
        """
        self._load_model()

        # --- GroundingDINO → Bounding Box + SAM → Maske ---
        masks, boxes, phrases, logits = self.model.predict(
            rgb_image.convert("RGB"), prompt
        )

        if len(masks) == 0:
            logger.warning(f"Kein Objekt für Prompt '{prompt}' gefunden.")
            return None

        # Beste Detektion auswählen (höchste Konfidenz)
        best_idx = logits.argmax().item()
        mask_tensor = masks[best_idx]  # (H, W) Tensor
        mask_np = mask_tensor.squeeze().cpu().numpy().astype(bool)
        bbox = boxes[best_idx].cpu().numpy().tolist()  # [x1, y1, x2, y2]
        confidence = logits[best_idx].item()

        logger.info(
            f"Objekt gefunden: '{phrases[best_idx]}' "
            f"(Konfidenz: {confidence:.3f}, BBox: {bbox})"
        )

        # --- ROI-Ausschnitt erzeugen ---
        roi_image = self._extract_roi(rgb_image, mask_np, bbox)

        return LocalizationResult(
            rgb_image=rgb_image,
            mask=mask_np,
            bbox=bbox,
            roi_image=roi_image,
            confidence=confidence,
            prompt=prompt,
        )

    def localize_all(
        self,
        rgb_image: Image.Image,
        prompt: str,
    ) -> List[LocalizationResult]:
        """Gibt alle gefundenen Objekte zurück (nicht nur das beste).

        Nützlich wenn mehrere Instanzen desselben Objekttyps im Bild sind.

        Args:
            rgb_image: RGB-Eingabebild.
            prompt: Sprachprompt.

        Returns:
            Liste von LocalizationResult-Objekten, sortiert nach Konfidenz.
        """
        self._load_model()

        masks, boxes, phrases, logits = self.model.predict(
            rgb_image.convert("RGB"), prompt
        )

        if len(masks) == 0:
            logger.warning(f"Keine Objekte für Prompt '{prompt}' gefunden.")
            return []

        results = []
        # Sortiere nach Konfidenz (absteigend)
        sorted_indices = logits.argsort(descending=True)

        for idx in sorted_indices:
            idx = idx.item()
            mask_np = masks[idx].squeeze().cpu().numpy().astype(bool)
            bbox = boxes[idx].cpu().numpy().tolist()
            confidence = logits[idx].item()

            roi_image = self._extract_roi(rgb_image, mask_np, bbox)

            results.append(LocalizationResult(
                rgb_image=rgb_image,
                mask=mask_np,
                bbox=bbox,
                roi_image=roi_image,
                confidence=confidence,
                prompt=prompt,
            ))

        return results

    @staticmethod
    def _extract_roi(
        image: Image.Image,
        mask: np.ndarray,
        bbox: List[float],
        background_color: Tuple[int, int, int] = (205, 205, 205),
    ) -> Image.Image:
        """Erzeugt einen ROI-Ausschnitt mit neutralem Hintergrund.

        Adaptiert aus OSCAR – object_retrieval/i2i_seg_clip.py:crop_with_mask()

        Args:
            image: Originalbild (PIL).
            mask: Binäre Maske (bool, H×W).
            bbox: [x1, y1, x2, y2] Bounding Box.
            background_color: Hintergrundfarbe.

        Returns:
            Zugeschnittenes PIL-Bild.
        """
        img_array = np.array(image)
        canvas = np.full_like(img_array, background_color, dtype=np.uint8)
        canvas[mask] = img_array[mask]

        # Bounding Box aus Maske für sauberen Zuschnitt
        coords = np.argwhere(mask)
        if len(coords) == 0:
            # Fallback auf BBox
            x1, y1, x2, y2 = [int(c) for c in bbox]
            return Image.fromarray(canvas[y1:y2, x1:x2])

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return Image.fromarray(canvas[y0:y1, x0:x1])
