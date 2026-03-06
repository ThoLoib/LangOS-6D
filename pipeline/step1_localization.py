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
#   Beide Modelle werden direkt via HuggingFace transformers geladen
#   Beide Modelle werden direkt via HuggingFace transformers geladen
#   (kein LangSAM-Wrapper erforderlich).
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

    Nutzt HuggingFace transformers direkt (kein LangSAM nötig).
    GroundingDINO liefert Bounding Boxes, SAM segmentiert präzise.

    Ref:
        GroundingDINO: https://huggingface.co/IDEA-Research/grounding-dino-base
        SAM: https://huggingface.co/facebook/sam-vit-large

    Usage:
        >>> localizer = ObjectLocalizer(config)
        >>> result = localizer.localize(image, "mayonnaise tube")
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self._gdino_model = None
        self._gdino_processor = None
        self._sam_model = None
        self._sam_processor = None

    def _load_model(self):
        """Lädt GroundingDINO + SAM via HuggingFace transformers."""
        if self._gdino_model is not None:
            return

        from transformers import (
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
            SamModel,
            SamProcessor,
        )

        gdino_id = self.config.grounding_dino_model
        sam_id = self.config.sam_model

        logger.info("Lade GroundingDINO (%s)...", gdino_id)
        self._gdino_processor = AutoProcessor.from_pretrained(gdino_id)
        self._gdino_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id)
            .to(self.device)
            .eval()
        )

        logger.info("Lade SAM (%s)...", sam_id)
        self._sam_processor = SamProcessor.from_pretrained(sam_id)
        self._sam_model = SamModel.from_pretrained(sam_id).to(self.device).eval()

        logger.info("GroundingDINO + SAM erfolgreich geladen.")

    def _detect(self, rgb_image: Image.Image, prompt: str):
        """Führt GroundingDINO-Detektion aus.

        Returns:
            Dict mit 'scores' (Tensor), 'labels' (list[str]), 'boxes' (Tensor).
        """
        # GroundingDINO erwartet Prompt mit abschließendem Punkt
        text = prompt.strip()
        if not text.endswith("."):
            text += "."

        inputs = self._gdino_processor(
            images=rgb_image, text=text, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._gdino_model(**inputs)

        results = self._gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.config.detection_confidence,
            text_threshold=self.config.detection_confidence,
            target_sizes=[(rgb_image.height, rgb_image.width)],
        )
        return results[0]  # single image

    def _segment(self, rgb_image: Image.Image, bbox: List[float]) -> np.ndarray:
        """Erzeugt eine SAM-Maske aus einem Bounding-Box-Prompt.

        Args:
            rgb_image: RGB-Bild.
            bbox: [x1, y1, x2, y2] in Pixelkoordinaten.

        Returns:
            Binäre Maske (H, W), bool.
        """
        input_boxes = [[[bbox[0], bbox[1], bbox[2], bbox[3]]]]
        inputs = self._sam_processor(
            rgb_image, input_boxes=input_boxes, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._sam_model(**inputs)

        masks = self._sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        # masks[0] shape: (1, 3, H, W) – 3 Masken pro Box, beste wählen
        iou_scores = outputs.iou_scores.cpu()  # (1, 1, 3)
        best_mask_idx = iou_scores[0, 0].argmax().item()
        return masks[0][0, best_mask_idx].numpy().astype(bool)

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

        det = self._detect(rgb_image, prompt)

        if len(det["scores"]) == 0:
            logger.warning("Kein Objekt für Prompt '%s' gefunden.", prompt)
            return None

        # Beste Detektion auswählen (höchste Konfidenz)
        best_idx = det["scores"].argmax().item()
        bbox = det["boxes"][best_idx].cpu().tolist()  # [x1, y1, x2, y2]
        confidence = det["scores"][best_idx].item()
        label = det["labels"][best_idx]

        logger.info(
            "Objekt gefunden: '%s' (Konfidenz: %.3f, BBox: %s)",
            label, confidence, bbox,
        )

        # SAM-Segmentierung mit BBox-Prompt
        mask_np = self._segment(rgb_image, bbox)

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

        det = self._detect(rgb_image, prompt)

        if len(det["scores"]) == 0:
            logger.warning("Keine Objekte für Prompt '%s' gefunden.", prompt)
            return []

        results = []
        sorted_indices = det["scores"].argsort(descending=True)

        for idx in sorted_indices:
            idx = idx.item()
            bbox = det["boxes"][idx].cpu().tolist()
            confidence = det["scores"][idx].item()

            mask_np = self._segment(rgb_image, bbox)
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
