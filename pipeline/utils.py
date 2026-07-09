# =============================================================================
# pipeline/utils.py – Gemeinsame Hilfsfunktionen für alle Pipeline-Schritte
# =============================================================================

import os
import json
import logging
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict

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


# ---------------------------------------------------------------------------
# Geometrische Distanzmetriken
# ---------------------------------------------------------------------------

def trimmed_chamfer_distance(
    source: np.ndarray,
    target: np.ndarray,
    trim_ratio: float = 0.1,
) -> float:
    """Trimmed one-sided Chamfer distance (source → target).

    For each point in *source*, find the nearest neighbour in *target*.
    Discard the top *trim_ratio* fraction of distances (the largest ones)
    and return the mean of the remaining distances.  This is robust to
    partial overlap — the trimmed tail absorbs query regions that have
    no corresponding CAD surface (e.g. back faces, occluded areas).

    Thesis reference: Sec. 3.3 (Sub-step B2), Equation for S_chamfer.

    Args:
        source: (N, 3) query point cloud (observed partial PC).
        target: (M, 3) reference point cloud (CAD partial view or full mesh).
        trim_ratio: Fraction of largest distances to discard (default 0.1 = 10 %).

    Returns:
        Mean of the trimmed nearest-neighbour distances (lower = better fit).
        Returns ``float('inf')`` if either cloud is empty.
    """
    if len(source) == 0 or len(target) == 0:
        return float("inf")

    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    dists, _ = tree.query(source, k=1)

    # Trim the largest `trim_ratio` fraction
    n_keep = max(1, int(len(dists) * (1.0 - trim_ratio)))
    dists_sorted = np.sort(dists)[:n_keep]
    return float(dists_sorted.mean())


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
