# =============================================================================
# pipeline/utils.py – Gemeinsame Hilfsfunktionen für alle Pipeline-Schritte
# =============================================================================

import os
import json
import logging
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datei-/Ordner-Hilfsfunktionen
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    """Erstellt ein Verzeichnis falls es nicht existiert.

    Args:
        path: Verzeichnispfad.

    Returns:
        Den selben Pfad (für Method-Chaining).
    """
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Bild-Hilfsfunktionen
# ---------------------------------------------------------------------------

def crop_with_mask(
    image: Image.Image,
    mask: np.ndarray,
    background_color: Tuple[int, int, int] = (205, 205, 205),
) -> Optional[Image.Image]:
    """Extrahiert den segmentierten Bereich aus einem Bild.

    Das Objekt wird auf einen einfarbigen Hintergrund gesetzt und
    auf die minimale Bounding Box zugeschnitten.

    Ref: Adaptiert aus OSCAR – object_retrieval/i2i_seg_clip.py

    Args:
        image: RGB-Bild (PIL).
        mask: Binäre Maske (H, W), True/1 = Vordergrund.
        background_color: Hintergrundfarbe für maskierte Bereiche.

    Returns:
        Zugeschnittenes PIL-Bild oder None, falls die Maske leer ist.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.sum() == 0:
        logger.warning("Leere Maske – kein Objekt gefunden.")
        return None

    img_array = np.array(image)
    canvas = np.full_like(img_array, background_color, dtype=np.uint8)
    canvas[mask_bool] = img_array[mask_bool]

    # Bounding Box des Vordergrunds bestimmen
    coords = np.argwhere(mask_bool)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = canvas[y0:y1, x0:x1]
    return Image.fromarray(cropped)


def load_depth_image(depth_path: str, depth_scale: float = 1000.0) -> np.ndarray:
    """Lädt ein Tiefenbild und konvertiert es in Meter.

    Args:
        depth_path: Pfad zum Tiefenbild (16-bit PNG üblich bei BOP-Datensätzen).
        depth_scale: Divisor zur Umrechnung in Meter (z.B. 1000 für mm → m).

    Returns:
        Tiefenbild als float32-Array in Metern (H, W).
    """
    depth_raw = np.array(Image.open(depth_path))
    return depth_raw.astype(np.float32) / depth_scale


def load_camera_intrinsics(json_path: str, image_id: int = 0) -> Dict:
    """Lädt Kamera-Intrinsics aus einer BOP-kompatiblen JSON-Datei.

    BOP-Format: scene_camera.json → {image_id: {"cam_K": [9 floats], "depth_scale": float}}

    Args:
        json_path: Pfad zu scene_camera.json.
        image_id: Bild-ID (Key im JSON).

    Returns:
        Dict mit 'fx', 'fy', 'cx', 'cy', 'depth_scale'.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Exakten Key suchen; Fallback auf ersten verfügbaren Key (Intrinsics sind
    # in BOP-Szenen meist für alle Frames identisch)
    key = str(image_id)
    if key not in data:
        key = next(iter(data))
    entry = data[key]
    K = entry["cam_K"]  # 3x3 Matrix als flache Liste [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    return {
        "fx": K[0],
        "fy": K[4],
        "cx": K[2],
        "cy": K[5],
        "depth_scale": entry.get("depth_scale", 1.0),
    }


def load_object_descriptions(desc_file: str) -> Tuple[List[str], List[str]]:
    """Lädt Objektbeschreibungen aus einer OSCAR-kompatiblen JSON-Datei.

    Format: {object_id: {"image_descriptions": {"view_name": "text", ...}}}

    Args:
        desc_file: Pfad zur JSON-Datei.

    Returns:
        (texts, labels) – Liste aller Beschreibungstexte und zugehörige Label-IDs.
    """
    with open(desc_file, "r") as f:
        descriptions = json.load(f)

    texts: List[str] = []
    labels: List[str] = []
    for obj_id, entry in descriptions.items():
        for _, text in entry.get("image_descriptions", {}).items():
            texts.append(text)
            labels.append(obj_id)
    return texts, labels


def ensure_dir(path: str) -> str:
    """Erstellt einen Ordner, falls er nicht existiert."""
    os.makedirs(path, exist_ok=True)
    return path
