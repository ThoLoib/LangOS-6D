# =============================================================================
# pipeline/visualization.py – Visualisierung der Pipeline-Zwischenergebnisse
# =============================================================================
#
# Speichert nach jedem Schritt ein Bild in den Output-Ordner:
#
#   step1_mask.png          – RGB-Bild mit Maskenoverlay + BBox
#   step1_roi.png           – Zugeschnittener ROI-Ausschnitt
#   step2_pointcloud.png    – 3D-Punktwolke (3 Ansichten)
#   step2_depth_masked.png  – Segmentiertes Tiefenbild
#   step3_clip_top5.png     – Top-5 CLIP-Kandidaten mit Scores
#   step4_dino_top5.png     – Top-5 DINOv2-Kandidaten mit Scores
#   step5_shape_top5.png    – Top-5 Shape-Kandidaten mit Scores
#   step6_fusion_result.png – Finales Ergebnis nach Fusion
#   summary.png             – Alle Schritte auf einen Blick
#
# =============================================================================

import logging
import os
from typing import Optional, List, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def _get_font(size: int = 14):
    """Versucht eine halbwegs lesbare Schrift zu laden."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _add_text(draw: ImageDraw.Draw, text: str, xy, color="white", bg="black", font=None):
    """Text mit Hintergrund-Box für Lesbarkeit."""
    font = font or _get_font(14)
    bbox = draw.textbbox(xy, text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=bg)
    draw.text(xy, text, fill=color, font=font)


# ---------------------------------------------------------------------------
# Schritt 1: Lokalisierung
# ---------------------------------------------------------------------------

def viz_step1_mask(rgb_image: Image.Image, mask: np.ndarray, bbox: list,
                   confidence: float, prompt: str, output_dir: str) -> str:
    """Speichert das RGB-Bild mit halbtransparentem Masken-Overlay und BBox.

    Returns:
        Pfad zum gespeicherten Bild.
    """
    img = rgb_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    mask_rgba[mask, :] = [0, 255, 0, 100]  # Grün, halbtransparent
    overlay = Image.fromarray(mask_rgba, "RGBA")
    img = Image.alpha_composite(img, overlay).convert("RGB")

    draw = ImageDraw.Draw(img)
    # BBox zeichnen
    x1, y1, x2, y2 = [int(c) for c in bbox]
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
    font = _get_font(16)
    _add_text(draw, f"{prompt}  ({confidence:.2f})", (x1, max(0, y1 - 22)),
              color="lime", bg=(0, 0, 0), font=font)

    path = os.path.join(output_dir, "step1_mask.png")
    img.save(path)
    logger.info(f"  [VIZ] Maske + BBox → {path}")
    return path


def viz_step1_roi(roi_image: Image.Image, output_dir: str) -> str:
    """Speichert den ROI-Ausschnitt."""
    path = os.path.join(output_dir, "step1_roi.png")
    roi_image.save(path)
    logger.info(f"  [VIZ] ROI-Ausschnitt → {path}")
    return path


# ---------------------------------------------------------------------------
# Schritt 2: Punktwolke
# ---------------------------------------------------------------------------

def viz_step2_depth_masked(depth_image: np.ndarray, mask: np.ndarray,
                           output_dir: str) -> str:
    """Speichert das maskierte Tiefenbild als Falschfarben-Bild."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masked_depth = np.where(mask, depth_image, 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(depth_image, cmap="turbo")
    axes[0].set_title("Depth (vollständig)")
    axes[0].axis("off")

    axes[1].imshow(masked_depth, cmap="turbo")
    axes[1].set_title("Depth (maskiert)")
    axes[1].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "step2_depth_masked.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [VIZ] Maskiertes Tiefenbild → {path}")
    return path


def viz_step2_pointcloud(points: np.ndarray, colors: np.ndarray,
                         output_dir: str) -> str:
    """Speichert die Punktwolke als 3-Ansichten-Plot (Front, Top, Side)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 5))

    views = [
        ("Frontal (XY)", 0, 1, "X", "Y"),
        ("Draufsicht (XZ)", 0, 2, "X", "Z"),
        ("Seite (YZ)", 1, 2, "Y", "Z"),
    ]

    # Farben auf [0,1] sicherstellen
    c = colors if colors.max() <= 1.0 else colors / 255.0

    for i, (title, ax_x, ax_y, xlabel, ylabel) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.scatter(points[:, ax_x], points[:, ax_y], c=c, s=0.3, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.invert_yaxis()

    plt.suptitle(f"Punktwolke ({len(points)} Punkte)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "step2_pointcloud.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [VIZ] Punktwolke (3 Ansichten) → {path}")
    return path


# ---------------------------------------------------------------------------
# Schritt 3–5: Retrieval-Kandidaten
# ---------------------------------------------------------------------------

def _load_reference_thumbnail(object_id: str, ref_images_dir: str,
                              size: int = 128) -> Optional[Image.Image]:
    """Versucht, ein Referenzbild für ein Objekt zu laden."""
    obj_dir = os.path.join(ref_images_dir, object_id)
    if not os.path.isdir(obj_dir):
        return None

    # Erstes Bild im Ordner nehmen
    for fname in sorted(os.listdir(obj_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(os.path.join(obj_dir, fname)).convert("RGB")
            img.thumbnail((size, size))
            return img
    return None


def _viz_candidates(candidates: list, score_attr: str, title: str,
                    ref_images_dir: str, output_dir: str, filename: str,
                    query_image: Optional[Image.Image] = None) -> str:
    """Generische Visualisierung von Top-K Retrieval-Kandidaten.

    Args:
        candidates: Liste von Candidate-Objekten mit .object_id und score-Attribut.
        score_attr: Name des Score-Attributs (z.B. "score", "dino_score", "shape_score").
        title: Titel für das Bild.
        ref_images_dir: Verzeichnis mit Referenzbildern.
        output_dir: Ausgabeverzeichnis.
        filename: Dateiname.
        query_image: Optionales Query-Bild (wird links angezeigt).

    Returns:
        Pfad zum gespeicherten Bild.
    """
    top_k = min(5, len(candidates))
    thumb_size = 160
    padding = 10
    font = _get_font(13)
    font_title = _get_font(18)

    # Breite: Query (optional) + Top-K Thumbnails
    cols = (1 if query_image else 0) + top_k
    width = cols * (thumb_size + padding) + padding
    height = thumb_size + 80  # Platz für Titel + Score-Text

    canvas = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # Titel
    draw.text((padding, 5), title, fill="white", font=font_title)

    y_off = 35
    x = padding

    # Query-Bild (links)
    if query_image:
        q = query_image.copy()
        q.thumbnail((thumb_size, thumb_size))
        canvas.paste(q, (x, y_off))
        draw.text((x, y_off + thumb_size + 2), "Query", fill="cyan", font=font)
        x += thumb_size + padding
        # Trennlinie
        draw.line([(x - 5, y_off), (x - 5, y_off + thumb_size)], fill="gray", width=2)

    # Kandidaten
    for i in range(top_k):
        c = candidates[i]
        score = getattr(c, score_attr, 0.0)
        obj_id = c.object_id

        thumb = _load_reference_thumbnail(obj_id, ref_images_dir, thumb_size)
        if thumb:
            canvas.paste(thumb, (x, y_off))
        else:
            # Platzhalter
            draw.rectangle([x, y_off, x + thumb_size, y_off + thumb_size],
                           fill=(60, 60, 60), outline="gray")
            draw.text((x + 10, y_off + 60), "N/A", fill="gray", font=font)

        # Rank + Score
        rank_color = ["gold", "silver", "#cd7f32", "white", "white"][i]
        label = f"#{i+1} {score:.3f}"
        draw.text((x, y_off + thumb_size + 2), label, fill=rank_color, font=font)

        # Object-ID (gekürzt auf 20 Zeichen)
        short_id = obj_id[:22] + "…" if len(obj_id) > 22 else obj_id
        draw.text((x, y_off + thumb_size + 18), short_id, fill="gray", font=_get_font(10))

        x += thumb_size + padding

    path = os.path.join(output_dir, filename)
    canvas.save(path)
    logger.info(f"  [VIZ] {title} → {path}")
    return path


def viz_step3_clip(clip_result, ref_images_dir: str, output_dir: str,
                   query_image: Optional[Image.Image] = None) -> str:
    """Top-5 CLIP-Kandidaten visualisieren."""
    return _viz_candidates(
        clip_result.candidates, "score", "Schritt 3: CLIP Retrieval",
        ref_images_dir, output_dir, "step3_clip_top5.png", query_image
    )


def viz_step4_dino(dino_result, ref_images_dir: str, output_dir: str,
                   query_image: Optional[Image.Image] = None) -> str:
    """Top-5 DINOv2-Kandidaten visualisieren."""
    return _viz_candidates(
        dino_result.candidates, "dino_score", "Schritt 4: DINOv2 Re-Ranking",
        ref_images_dir, output_dir, "step4_dino_top5.png", query_image
    )


def viz_step5_shape(shape_result, ref_images_dir: str, output_dir: str) -> str:
    """Top-5 Shape-Kandidaten visualisieren."""
    return _viz_candidates(
        shape_result.candidates, "shape_score", "Schritt 5: ULIP-2 Shape Matching",
        ref_images_dir, output_dir, "step5_shape_top5.png"
    )


def viz_step6_fusion(fusion_result, ref_images_dir: str, output_dir: str,
                     query_image: Optional[Image.Image] = None) -> str:
    """Fusionsergebnis visualisieren."""
    candidates = fusion_result.ranked_candidates if hasattr(fusion_result, 'ranked_candidates') else []
    if fusion_result.best_match and (not candidates or fusion_result.best_match not in candidates):
        candidates = [fusion_result.best_match] + list(candidates)
    return _viz_candidates(
        candidates, "fused_score", "Schritt 6: Fusion (Endergebnis)",
        ref_images_dir, output_dir, "step6_fusion_result.png", query_image
    )


# ---------------------------------------------------------------------------
# Summary: Alle Schritte auf einen Blick
# ---------------------------------------------------------------------------

def viz_summary(output_dir: str, prompt: str, timing: dict) -> str:
    """Erstellt ein Zusammenfassungsbild aus allen gespeicherten Schritt-Bildern.

    Lädt die einzelnen Step-Bilder und arrangiert sie in einem Grid.

    Returns:
        Pfad zum Summary-Bild.
    """
    step_files = [
        ("step1_mask.png", "1: Lokalisierung"),
        ("step1_roi.png", "1: ROI"),
        ("step2_depth_masked.png", "2: Tiefenbild"),
        ("step2_pointcloud.png", "2: Punktwolke"),
        ("step3_clip_top5.png", "3: CLIP"),
        ("step4_dino_top5.png", "4: DINOv2"),
        ("step5_shape_top5.png", "5: Shape"),
        ("step6_fusion_result.png", "6: Fusion"),
    ]

    images = []
    labels = []
    for fname, label in step_files:
        img_path = os.path.join(output_dir, fname)
        if os.path.exists(img_path):
            images.append(Image.open(img_path).convert("RGB"))
            labels.append(label)

    if not images:
        logger.warning("[VIZ] Keine Schritt-Bilder für Summary gefunden.")
        return ""

    # Grid: 2 Spalten
    cols = 2
    rows = (len(images) + cols - 1) // cols
    cell_w = 640
    header_h = 50
    label_h = 25

    # Bilder skalieren
    thumbs = []
    for img in images:
        ratio = cell_w / img.width
        new_h = int(img.height * ratio)
        thumbs.append(img.resize((cell_w, new_h), Image.LANCZOS))

    max_h = max(t.height for t in thumbs) if thumbs else 200
    cell_h = max_h + label_h

    width = cols * cell_w
    height = header_h + rows * cell_h + 40  # +40 für Timing

    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font_title = _get_font(20)
    font_label = _get_font(14)
    font_timing = _get_font(12)

    # Header
    draw.text((10, 10), f"OSCAR+ Pipeline – \"{prompt}\"", fill="white", font=font_title)

    for i, (thumb, label) in enumerate(zip(thumbs, labels)):
        row = i // cols
        col = i % cols
        x = col * cell_w
        y = header_h + row * cell_h

        draw.text((x + 5, y + 2), label, fill="cyan", font=font_label)
        canvas.paste(thumb, (x, y + label_h))

    # Timing am unteren Rand
    y_timing = header_h + rows * cell_h + 5
    total = timing.get("total", 0)
    parts = [f"{k}: {v:.1f}s" for k, v in timing.items() if k != "total"]
    timing_text = f"Gesamt: {total:.1f}s | " + " | ".join(parts)
    draw.text((10, y_timing), timing_text, fill="gray", font=font_timing)

    path = os.path.join(output_dir, "summary.png")
    canvas.save(path)
    logger.info(f"  [VIZ] Summary → {path}")
    return path
