#!/usr/bin/env python3
# =============================================================================
# pipeline/debug_steps.py – Etappe-für-Etappe Debug der OSCAR+ Pipeline
# =============================================================================
#
# Führt die Pipeline schrittweise aus und speichert nach jedem Schritt
# ein detailliertes Diagnosebild. Ideal um Fehler früh zu erkennen.
#
# Verwendung:
#   python -m pipeline.debug_steps \
#       --rgb  eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
#       --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
#       --prompt "mustard bottle" \
#       --descriptions object_database/ycbv_gso/descriptions_attributes.json \
#       --reference_images object_images/ycbv_gso/ \
#       --cad_models object_database/ycbv_gso/ \
#       --camera eval/datasets/ycbv_gso/test/000048/scene_camera.json \
#       --until_step 4        # Stop nach Schritt 4
#
# Ausgabe je Schritt:
#   debug_01_localization.png  – Prompt → Objekt (Maske, ROI, Text)
#   debug_02_pointcloud.png    – Tiefenbild + Punktwolke 2D/3D
#   debug_03_clip.png          – ROI vs Top-5 CLIP-Kandidaten + Scores
#   debug_04_dino.png          – ROI vs bestes DINO-Match + Score-Tabelle
#   debug_05_ulip.png          – Punktwolke 3D + Top-3 ULIP-Matches
#   debug_06_fusion.png        – Score-Tabelle (CLIP/DINO/ULIP/Fusion) + Vergleich
#   debug_07_scale_pose.png    – Modellüberlagerung auf Szene + Scale/Pose Info
# =============================================================================

import argparse
import logging
import os
from typing import Optional, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("debug")


# =============================================================================
# PIL-Hilfsfunktionen
# =============================================================================

def _font(size: int = 14) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


def _text(draw: ImageDraw.Draw, text: str, xy, fg="white",
          bg=(0, 0, 0), size: int = 14):
    """Text mit dunkler Hintergrundbox."""
    font = _font(size)
    bb = draw.textbbox(xy, text, font=font)
    draw.rectangle([bb[0]-2, bb[1]-2, bb[2]+2, bb[3]+2], fill=bg)
    draw.text(xy, text, fill=fg, font=font)


def _hstack(images: list, pad: int = 8,
            bg=(18, 18, 18)) -> Image.Image:
    """Horizontal stapeln (gleiche Höhe)."""
    h = max(img.height for img in images)
    w = sum(img.width for img in images) + pad * (len(images) + 1)
    out = Image.new("RGB", (w, h + 2 * pad), bg)
    x = pad
    for img in images:
        out.paste(img, (x, pad + (h - img.height) // 2))
        x += img.width + pad
    return out


def _vstack(images: list, pad: int = 8, bg=(18, 18, 18)) -> Image.Image:
    """Vertikal stapeln (gleiche Breite)."""
    w = max(img.width for img in images)
    h = sum(img.height for img in images) + pad * (len(images) + 1)
    out = Image.new("RGB", (w + 2 * pad, h), bg)
    y = pad
    for img in images:
        out.paste(img, (pad, y))
        y += img.height + pad
    return out


def _banner(text: str, width: int, height: int = 44,
            bg=(25, 25, 50), fg_style: str = "title") -> Image.Image:
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    draw.text((10, 8), text, fill="white", font=_font(20))
    return img


def _label_top(img: Image.Image, text: str, fg="cyan", size=14) -> Image.Image:
    """Label-Zeile über ein Bild legen."""
    lh = size + 8
    out = Image.new("RGB", (img.width, img.height + lh), (18, 18, 18))
    out.paste(img, (0, lh))
    ImageDraw.Draw(out).text((4, 3), text, fill=fg, font=_font(size))
    return out


def _load_thumb(obj_id: str, ref_dir: str, size: int = 160) -> Optional[Image.Image]:
    """Lädt erstes Referenzbild für ein Objekt."""
    obj_dir = os.path.join(ref_dir, obj_id)
    if not os.path.isdir(obj_dir):
        return None
    for fname in sorted(os.listdir(obj_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(os.path.join(obj_dir, fname)).convert("RGB")
                img.thumbnail((size, size), Image.LANCZOS)
                return img
            except Exception:
                continue
    return None


def _placeholder(size: int, label: str = "N/A") -> Image.Image:
    img = Image.new("RGB", (size, size), (50, 50, 50))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size - 1, size - 1], outline=(90, 90, 90))
    draw.text((4, size // 2 - 8), label[:10], fill=(130, 130, 130), font=_font(12))
    return img


RANK_COLORS = [
    (255, 215, 0),    # gold
    (192, 192, 192),  # silver
    (205, 127, 50),   # bronze
    (200, 200, 200),  # white-ish
    (180, 180, 180),
    (160, 160, 160),
    (140, 140, 140),
    (120, 120, 120),
]


# =============================================================================
# Debug-Bild 1: Prompt-Analyse + Lokalisierung
# =============================================================================

def save_debug_step1(rgb_image: Image.Image, mask: np.ndarray, bbox: list,
                     roi_image: Image.Image, original_prompt: str,
                     extracted_name: str, confidence: float,
                     output_dir: str) -> str:
    """3 Panels: Szene+Maske | ROI | Textanalyse"""
    tw = 360  # Breite je Panel

    # --- Panel A: Szene mit Maskenoverlay + BBox ---
    scene = rgb_image.copy().convert("RGB")
    # Grüner Masken-Overlay
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask] = [0, 220, 80, 110]
    scene_rgba = scene.convert("RGBA")
    scene_rgba.alpha_composite(Image.fromarray(overlay, "RGBA"))
    scene = scene_rgba.convert("RGB")
    draw_s = ImageDraw.Draw(scene)
    x1, y1, x2, y2 = [int(c) for c in bbox]
    draw_s.rectangle([x1, y1, x2, y2], outline=(0, 255, 100), width=3)
    _text(draw_s, f"{extracted_name}  conf={confidence:.2f}",
          (x1, max(0, y1 - 24)), fg=(0, 255, 100), bg=(0, 0, 0), size=15)
    ratio = tw / scene.width
    pa = scene.resize((tw, int(scene.height * ratio)), Image.LANCZOS)
    pa = _label_top(pa, "A — Szene + Segmentierungsmaske + BBox", fg="cyan")

    # --- Panel B: Segmentierter ROI-Ausschnitt ---
    roi_r = roi_image.copy()
    roi_r.thumbnail((tw, tw), Image.LANCZOS)
    pb = Image.new("RGB", (tw, pa.height), (22, 22, 22))
    pb.paste(roi_r, ((tw - roi_r.width) // 2, (pa.height - roi_r.height - 20) // 2 + 20))
    pb = _label_top(pb, "B — Segmentierter Objektausschnitt (ROI)", fg="cyan")
    pb = pb.crop((0, 0, pb.width, pa.height))

    # --- Panel C: Text-Info ---
    pc = Image.new("RGB", (tw, pa.height), (14, 14, 18))
    draw_t = ImageDraw.Draw(pc)
    draw_t.text((10, 14), "SCHRITT 1 — LOKALISIERUNG", fill=(0, 200, 255), font=_font(19))

    y = 60
    draw_t.text((10, y), "Eingabe-Prompt:", fill=(170, 170, 170), font=_font(13))
    y += 20
    # Zeilenumbruch bei langen Prompts
    words = original_prompt.split()
    line, lines = "", []
    for w in words:
        if len(line + " " + w) > 36:
            lines.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    for l in lines:
        draw_t.text((14, y), l, fill=(255, 255, 255), font=_font(14))
        y += 20
    y += 10

    draw_t.text((10, y), "Extrahierter Objektname (→ GroundingDINO):",
                fill=(170, 170, 170), font=_font(13))
    y += 20
    draw_t.text((14, y), f'"{extracted_name}"', fill=(255, 220, 0), font=_font(17))
    y += 36

    draw_t.text((10, y), "Detektions-Ergebnis:", fill=(170, 170, 170), font=_font(13))
    y += 20
    conf_col = (60, 230, 100) if confidence > 0.55 else (255, 165, 0) if confidence > 0.35 else (255, 60, 60)
    draw_t.text((14, y), f"Konfidenz: {confidence:.3f}", fill=conf_col, font=_font(16))
    y += 22
    draw_t.text((14, y),
                f"BBox: [{x1}, {y1}] → [{x2}, {y2}] ({x2-x1}×{y2-y1}px)",
                fill=(190, 190, 190), font=_font(13))
    y += 22
    draw_t.text((14, y),
                f"Bildgröße: {rgb_image.width} × {rgb_image.height} px",
                fill=(150, 150, 150), font=_font(13))
    y += 36

    qual = ("✓ Gute Detektion" if confidence > 0.6
            else "⚠  Detektion unsicher" if confidence > 0.35
            else "✗  Schwache Detektion")
    draw_t.text((10, y), qual, fill=conf_col, font=_font(16))

    final = _hstack([pa, pb, pc], pad=6)
    final = _vstack([_banner("SCHRITT 1: Prompt-Analyse + Objektlokalisierung",
                             final.width), final], pad=0)
    path = os.path.join(output_dir, "debug_01_localization.png")
    final.save(path)
    logger.info("  [DEBUG 1] → %s", path)
    return path


# =============================================================================
# Debug-Bild 2: Tiefenbild + Punktwolke
# =============================================================================

def save_debug_step2(depth_image: np.ndarray, mask: np.ndarray,
                     points: np.ndarray, colors: np.ndarray,
                     num_points: int, bbox_size,
                     output_dir: str) -> str:
    """2×2 Raster: depth-raw | depth-masked | PC frontal | PC top"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#141414")

    c_norm = np.clip(colors / 255.0 if colors.max() > 1.0 else colors, 0, 1)
    step = max(1, len(points) // 3000)

    # Depth raw
    ax = axes[0, 0]
    ax.set_facecolor("#141414")
    im = ax.imshow(depth_image, cmap="turbo")
    plt.colorbar(im, ax=ax, label="mm", fraction=0.046)
    ax.set_title("Tiefenbild (vollständig)", color="white", fontsize=12)
    ax.axis("off")

    # Depth masked
    ax = axes[0, 1]
    ax.set_facecolor("#141414")
    im = ax.imshow(np.where(mask, depth_image, 0), cmap="turbo")
    plt.colorbar(im, ax=ax, label="mm", fraction=0.046)
    ax.set_title("Tiefenbild maskiert (Objekt)", color="white", fontsize=12)
    ax.axis("off")

    # PC frontal (XY)
    ax = axes[1, 0]
    ax.set_facecolor("#0d0d0d")
    ax.scatter(points[::step, 0], points[::step, 1], c=c_norm[::step], s=0.8, alpha=0.85)
    ax.set_title(f"Punktwolke — Frontal (X/Y) | {num_points} Punkte",
                 color="white", fontsize=12)
    ax.set_xlabel("X (m)", color="gray"); ax.set_ylabel("Y (m)", color="gray")
    ax.invert_yaxis()
    ax.tick_params(colors="gray"); [s.set_color("gray") for s in ax.spines.values()]

    # PC top (XZ)
    ax = axes[1, 1]
    ax.set_facecolor("#0d0d0d")
    ax.scatter(points[::step, 0], -points[::step, 2], c=c_norm[::step], s=0.8, alpha=0.85)
    ax.set_title("Punktwolke — Draufsicht (X / -Z)", color="white", fontsize=12)
    ax.set_xlabel("X (m)", color="gray"); ax.set_ylabel("-Z (m)", color="gray")
    ax.tick_params(colors="gray"); [s.set_color("gray") for s in ax.spines.values()]

    sz = (f"BBox: {bbox_size[0]:.3f} × {bbox_size[1]:.3f} × {bbox_size[2]:.3f} m"
          if hasattr(bbox_size, "__len__") else f"BBox: {bbox_size}")
    fig.suptitle(f"SCHRITT 2: Punktwolkenerzeugung — {sz}",
                 color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "debug_02_pointcloud.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#141414")
    plt.close(fig)
    logger.info("  [DEBUG 2] → %s", path)
    return path


# =============================================================================
# Debug-Bild 3: CLIP Retrieval
# =============================================================================

def save_debug_step3(roi_image: Image.Image, candidates: list,
                     ref_dir: str, output_dir: str) -> str:
    """Links: Query ROI  |  Rechts: Top-5 CLIP-Kandidaten mit Score + Beschreibung"""
    top_n = min(5, len(candidates))
    thumb = 170
    pad = 8

    # ROI Panel
    roi = roi_image.copy()
    roi.thumbnail((280, 320), Image.LANCZOS)
    roi_panel = Image.new("RGB", (280, thumb * top_n + pad * (top_n + 1) + 36), (16, 16, 16))
    roi_d = ImageDraw.Draw(roi_panel)
    roi_d.text((8, 8), "Query-Bild (ROI)", fill=(0, 200, 255), font=_font(15))
    roi_panel.paste(roi, ((280 - roi.width) // 2, 36))

    # Kandidaten Panel
    cw = thumb + 380 + pad * 2
    ch = thumb * top_n + pad * (top_n + 1) + 36
    cand = Image.new("RGB", (cw, ch), (20, 20, 20))
    cd = ImageDraw.Draw(cand)
    cd.text((8, 8), "CLIP Kandidaten (semantische Ähnlichkeit der Beschreibungen)",
            fill=(255, 180, 60), font=_font(14))

    for i, c in enumerate(candidates[:top_n]):
        y = 36 + i * (thumb + pad)
        t = _load_thumb(c.object_id, ref_dir, thumb) or _placeholder(thumb, c.object_id[:8])
        cand.paste(t, (pad, y))

        tx = thumb + pad * 2
        rc = RANK_COLORS[i]
        cd.text((tx, y + 4), f"#{i+1}", fill=rc, font=_font(19))
        cd.text((tx + 34, y + 6), f"Score: {c.score:.4f}", fill=rc, font=_font(15))
        cd.text((tx, y + 28), c.object_id[:45], fill="white", font=_font(13))

        # Beschreibung
        desc = getattr(c, "description", "")
        if desc:
            short = desc[:85] + ("…" if len(desc) > 85 else "")
            cd.text((tx, y + 46), f'"{short}"', fill=(160, 160, 160), font=_font(11))

        # Score-Bar
        blen = max(2, min(int(c.score * 220), 220))
        cd.rectangle([tx, y + thumb - 18, tx + 220, y + thumb - 8], fill=(45, 45, 45))
        cd.rectangle([tx, y + thumb - 18, tx + blen, y + thumb - 8], fill=rc)
        cd.text((tx + 224, y + thumb - 17), f"{c.score:.3f}", fill=rc, font=_font(11))

    final = _hstack([roi_panel, cand], pad=pad)
    final = _vstack([_banner("SCHRITT 3: CLIP Retrieval — Semantische Kandidatensuche",
                             final.width, bg=(25, 20, 10)), final], pad=0)
    path = os.path.join(output_dir, "debug_03_clip.png")
    final.save(path)
    logger.info("  [DEBUG 3] → %s", path)
    return path


# =============================================================================
# Debug-Bild 4: DINOv2 Re-Ranking
# =============================================================================

def save_debug_step4(roi_image: Image.Image, candidates: list,
                     ref_dir: str, output_dir: str) -> str:
    """
    Zeile 1: Direkte Gegenüberstellung Query ↔ bestes DINO-Match (best_view_path)
    Zeile 2: Score-Tabelle aller Kandidaten (DINO + CLIP Score nebeneinander)
    """
    top_n = min(5, len(candidates))
    thumb2 = 220
    pad = 8
    w_total = 1000

    # --- Zeile 1: Direktvergleich ROI vs. Best View ---
    roi = roi_image.copy()
    roi.thumbnail((thumb2, thumb2), Image.LANCZOS)
    p_roi = Image.new("RGB", (thumb2 + 20, thumb2 + 36), (14, 14, 14))
    p_roi.paste(roi, (10 + (thumb2 - roi.width) // 2, 30))
    ImageDraw.Draw(p_roi).text((6, 6), "Query (ROI)", fill=(0, 200, 255), font=_font(14))

    p_best = Image.new("RGB", (thumb2 + 20, thumb2 + 36), (14, 14, 14))
    d_best = ImageDraw.Draw(p_best)
    if candidates:
        best = candidates[0]
        bv_path = getattr(best, "best_view_path", "")
        if bv_path and os.path.exists(bv_path):
            bv = Image.open(bv_path).convert("RGB")
            bv.thumbnail((thumb2, thumb2), Image.LANCZOS)
            p_best.paste(bv, (10 + (thumb2 - bv.width) // 2, 30))
        else:
            t = _load_thumb(best.object_id, ref_dir, thumb2)
            if t:
                p_best.paste(t, (10 + (thumb2 - t.width) // 2, 30))
        d_best.text((6, 6),
                    f"Top-1: {best.object_id[:30]}  DINO={best.dino_score:.4f}",
                    fill=(255, 215, 0), font=_font(13))

    arrow = Image.new("RGB", (60, thumb2 + 36), (18, 18, 18))
    ImageDraw.Draw(arrow).text((10, thumb2 // 2 - 10), "→", fill="white", font=_font(34))

    row1 = _hstack([p_roi, arrow, p_best], pad=6)

    # --- Zeile 2: Ranking-Tabelle ---
    rh = 60
    tw = w_total
    table = Image.new("RGB", (tw, rh * top_n + 36), (18, 18, 18))
    td = ImageDraw.Draw(table)
    td.text((10, 8), "DINOv2 Ranking — alle Kandidaten",
            fill=(100, 200, 255), font=_font(15))
    td.text((tw - 250, 10), "██ DINO-Score  ██ CLIP-Score",
            fill=(160, 160, 160), font=_font(12))

    for i, c in enumerate(candidates[:top_n]):
        y = 36 + i * rh
        bg = (28, 28, 28) if i % 2 == 0 else (22, 22, 22)
        td.rectangle([0, y, tw, y + rh], fill=bg)

        tn = _load_thumb(c.object_id, ref_dir, rh - 6) or _placeholder(rh - 6)
        table.paste(tn, (4, y + 3))

        rc = RANK_COLORS[i]
        tx_x = rh + 10
        td.text((tx_x, y + 6), f"#{i+1}", fill=rc, font=_font(16))
        td.text((tx_x + 32, y + 6), c.object_id[:40], fill="white", font=_font(13))

        # DINO bar
        db = max(2, min(int(c.dino_score * 280), 280))
        td.rectangle([tx_x + 30, y + 32, tx_x + 310, y + 44], fill=(40, 40, 40))
        td.rectangle([tx_x + 30, y + 32, tx_x + 30 + db, y + 44], fill=(100, 180, 255))
        td.text((tx_x + 315, y + 30), f"DINO {c.dino_score:.4f}",
                fill=(100, 180, 255), font=_font(12))

        # CLIP bar
        cb = max(2, min(int(c.clip_score * 280), 280))
        td.rectangle([tx_x + 420, y + 32, tx_x + 700, y + 44], fill=(40, 40, 40))
        td.rectangle([tx_x + 420, y + 32, tx_x + 420 + cb, y + 44], fill=(255, 180, 60))
        td.text((tx_x + 705, y + 30), f"CLIP {c.clip_score:.4f}",
                fill=(255, 180, 60), font=_font(12))

    final = _vstack([row1, table], pad=8)
    # Breite angleichen
    if row1.width != table.width:
        bg_full = Image.new("RGB", (max(row1.width, table.width),
                                    row1.height + table.height + 16), (18, 18, 18))
        bg_full.paste(row1, (0, 0))
        bg_full.paste(table, (0, row1.height + 8))
        final = bg_full

    final = _vstack([_banner("SCHRITT 4: DINOv2 Re-Ranking — Bildbasiertes Verfeinern",
                             final.width, bg=(10, 25, 15)), final], pad=0)
    path = os.path.join(output_dir, "debug_04_dino.png")
    final.save(path)
    logger.info("  [DEBUG 4] → %s", path)
    return path


# =============================================================================
# Debug-Bild 5: ULIP-2 Shape Matching
# =============================================================================

def save_debug_step5(points: np.ndarray, colors: np.ndarray,
                     candidates: list, ref_dir: str, output_dir: str) -> str:
    """
    Links: 3D-Punktwolke des beobachteten Objekts (Matplotlib 3D)
    Rechts: Top-3 ULIP-2 Matches mit Thumbnail + Score
    """
    top_n = min(3, len(candidates))
    c_norm = np.clip(colors / 255.0 if colors.max() > 1.0 else colors, 0, 1)
    step = max(1, len(points) // 2500)

    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_facecolor("#141414")

    # 3D Scatter
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.set_facecolor("#141414")
    ax3d.scatter(points[::step, 0], points[::step, 1], points[::step, 2],
                 c=c_norm[::step], s=1.5, alpha=0.85, depthshade=True)
    ax3d.set_title(f"Beobachtete Punktwolke\n({len(points)} Punkte)",
                   color="white", fontsize=12, pad=4)
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.pane.fill = False
        axis.label.set_color("gray")
        axis._axinfo["tick"]["color"] = "gray"
    ax3d.set_xlabel("X", color="gray", fontsize=9)
    ax3d.set_ylabel("Y", color="gray", fontsize=9)
    ax3d.set_zlabel("Z", color="gray", fontsize=9)

    # Tabellen-Panel
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor("#141414")
    ax2.axis("off")
    ax2.set_title("ULIP-2 Top-3 Shape Matches", color="white", fontsize=13)

    rank_mpl = ["gold", "silver", "#cd7f32"]
    for i, c in enumerate(candidates[:top_n]):
        col = rank_mpl[i] if i < 3 else "white"
        yp = 0.92 - i * 0.30
        ax2.text(0.02, yp,
                 f"#{i+1}  {c.object_id}",
                 transform=ax2.transAxes, color=col,
                 fontsize=12, fontweight="bold", va="top")
        ax2.text(0.02, yp - 0.07,
                 f"Shape Score: {c.shape_score:.4f}",
                 transform=ax2.transAxes, color="white",
                 fontsize=11, va="top")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    tmp = os.path.join(output_dir, "_tmp_s5.png")
    fig.savefig(tmp, dpi=115, bbox_inches="tight", facecolor="#141414")
    plt.close(fig)

    # Referenz-Thumbnails rechts anhängen (PIL)
    mpl_img = Image.open(tmp).convert("RGB")
    os.remove(tmp)

    th_size = mpl_img.height // max(top_n, 1) - 12
    side_w = th_size + 30
    side = Image.new("RGB", (side_w, mpl_img.height), (16, 16, 16))
    sd = ImageDraw.Draw(side)
    sd.text((4, 4), "Referenzbilder", fill=(180, 180, 180), font=_font(13))
    for i, c in enumerate(candidates[:top_n]):
        y = 24 + i * (th_size + 10)
        t = _load_thumb(c.object_id, ref_dir, th_size) or _placeholder(th_size, "N/A")
        side.paste(t, ((side_w - t.width) // 2, y))
        sd.text((4, y + t.height + 1), f"#{i+1}", fill=RANK_COLORS[i], font=_font(12))

    final = _hstack([mpl_img, side], pad=4)
    final = _vstack([_banner("SCHRITT 5: ULIP-2 Shape Matching — 3D Geometrie-Vergleich",
                             final.width, bg=(15, 15, 30)), final], pad=0)
    path = os.path.join(output_dir, "debug_05_ulip.png")
    final.save(path)
    logger.info("  [DEBUG 5] → %s", path)
    return path


# =============================================================================
# Debug-Bild 6: Fusion — Score-Tableau
# =============================================================================

def save_debug_step6(candidates: list, ref_dir: str,
                     roi_image: Image.Image, output_dir: str) -> str:
    """
    Obere Hälfte: Score-Tabelle (CLIP | DINO | ULIP | Fused) mit Balken
    Untere Hälfte: ROI vs. Gewinner-Modell Gegenüberstellung
    """
    top_n = min(8, len(candidates))
    if top_n == 0:
        return ""

    rh = 66     # row height
    th = 50     # thumbnail size
    bar_max = 200
    col_id = 280
    col_bar = bar_max + 80

    canvas_w = th + 14 + col_id + col_bar * 4
    canvas_h = 50 + top_n * rh + 14

    canvas = Image.new("RGB", (canvas_w, canvas_h), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)

    # Spalten-Header
    draw.text((10, 10), "# ", fill="white", font=_font(14))
    draw.text((th + 20, 10), "Objekt-ID", fill="white", font=_font(14))
    cx_h = th + 14 + col_id
    for label, col in [("CLIP", (255, 180, 60)), ("DINO", (100, 200, 255)),
                        ("ULIP", (180, 255, 100)), ("FUSION", (255, 80, 80))]:
        draw.text((cx_h + 8, 10), label, fill=col, font=_font(14))
        cx_h += col_bar

    draw.line([(0, 32), (canvas_w, 32)], fill=(60, 60, 60), width=1)

    score_attrs = [
        ("clip_score",  (255, 180, 60)),
        ("dino_score",  (100, 200, 255)),
        ("ulip_score",  (180, 255, 100)),
        ("fused_score", (255, 80, 80)),
    ]

    for i, c in enumerate(candidates[:top_n]):
        y = 36 + i * rh
        row_bg = (30, 30, 30) if i % 2 == 0 else (22, 22, 22)
        draw.rectangle([0, y, canvas_w, y + rh - 1], fill=row_bg)

        # Highlight winner
        if i == 0:
            draw.rectangle([0, y, canvas_w, y + rh - 1], outline=(255, 215, 0), width=2)

        t = _load_thumb(c.object_id, ref_dir, th) or _placeholder(th)
        canvas.paste(t, (4, y + (rh - th) // 2))

        rc = RANK_COLORS[i]
        draw.text((th + 8, y + 8), f"#{i+1}", fill=rc, font=_font(16))
        draw.text((th + 34, y + 8), c.object_id[:36], fill="white", font=_font(13))

        cx = th + 14 + col_id
        for attr, bar_col in score_attrs:
            val = getattr(c, attr, 0.0)
            blen = max(2, min(int(val * bar_max), bar_max))
            draw.rectangle([cx + 4, y + 18, cx + 4 + bar_max, y + 32], fill=(40, 40, 40))
            draw.rectangle([cx + 4, y + 18, cx + 4 + blen, y + 32], fill=bar_col)
            draw.text((cx + 4, y + 36), f"{val:.4f}", fill=bar_col, font=_font(12))
            cx += col_bar

    # Gegenüberstellung
    cmp_h = 220
    compare = Image.new("RGB", (canvas_w, cmp_h), (12, 12, 18))
    cd = ImageDraw.Draw(compare)
    cd.text((10, 8), "Gegenüberstellung: Query (ROI)  ↔  Bestes Modell",
            fill=(255, 215, 0), font=_font(16))
    cd.line([(0, 34), (canvas_w, 34)], fill=(50, 50, 60))

    qr = roi_image.copy(); qr.thumbnail((180, 180), Image.LANCZOS)
    compare.paste(qr, (16, 42))
    cd.text((16, 42 + qr.height + 4), "Query", fill=(0, 200, 255), font=_font(13))

    cd.text((210, 100), "→", fill="gray", font=_font(36))

    best = candidates[0]
    bt = _load_thumb(best.object_id, ref_dir, 180) or _placeholder(180)
    compare.paste(bt, (256, 42))
    cd.text((256, 42 + bt.height + 4),
            f"{best.object_id[:32]}  Fusion={best.fused_score:.4f}",
            fill=(255, 215, 0), font=_font(13))

    # Score badges
    bx = 460
    for label, attr, col in [("CLIP", "clip_score", (255, 180, 60)),
                              ("DINO", "dino_score", (100, 200, 255)),
                              ("ULIP", "ulip_score", (180, 255, 100))]:
        val = getattr(best, attr, 0.0)
        cd.text((bx, 50), label, fill=col, font=_font(13))
        cd.text((bx, 68), f"{val:.4f}", fill=col, font=_font(16))
        bx += 100

    full = _vstack([canvas, compare], pad=4)
    full = _vstack([_banner("SCHRITT 6: Score-Fusion — CLIP · DINO · ULIP → Finales Ranking",
                            full.width, bg=(30, 10, 10)), full], pad=0)
    path = os.path.join(output_dir, "debug_06_fusion.png")
    full.save(path)
    logger.info("  [DEBUG 6] → %s", path)
    return path


# =============================================================================
# Debug-Bild 7+8: Scale + Pose / Modellüberlagerung
# =============================================================================

def save_debug_step7_8(rgb_image: Image.Image, bbox: list,
                        scale_factor: float, best_object_id: str,
                        ref_dir: str, pose_info: dict,
                        obs_size: Optional[np.ndarray],
                        cad_size: Optional[np.ndarray],
                        output_dir: str) -> str:
    """
    Links: Originalszene mit Modell-Thumbnail (50% Alpha) in BBox eingeblendet
    Mitte: Bestes Modell-Referenzbild
    Rechts: Scale + Pose Infos als Text
    """
    pad = 8
    scene_w = 500

    # --- Panel A: Szene + Überlagerung ---
    scene = rgb_image.copy().convert("RGB")
    sc = min(1.0, scene_w / scene.width)
    scene_s = scene.resize((int(scene.width * sc), int(scene.height * sc)), Image.LANCZOS)
    x1, y1, x2, y2 = [max(0, int(c * sc)) for c in bbox]
    bw, bh = max(x2 - x1, 10), max(y2 - y1, 10)

    # Alpha-blend best model thumbnail into BBox
    best_t = _load_thumb(best_object_id, ref_dir, max(bw, bh))
    if best_t:
        region = scene_s.crop((x1, y1, x1 + bw, y1 + bh))
        best_r = best_t.resize((bw, bh), Image.LANCZOS)
        blended = Image.blend(region, best_r, alpha=0.5)
        scene_s.paste(blended, (x1, y1))

    draw_s = ImageDraw.Draw(scene_s)
    draw_s.rectangle([x1, y1, x1 + bw, y1 + bh], outline=(255, 215, 0), width=3)
    _text(draw_s, f"{best_object_id[:28]}  ×{scale_factor:.3f}",
          (x1, max(0, y1 - 22)), fg=(255, 215, 0), bg=(0, 0, 0), size=13)

    pa = _label_top(scene_s, "A — Szene + Modellüberlagerung (50% Alpha)")

    # --- Panel B: Modell-Referenzbild ---
    ref_img = _load_thumb(best_object_id, ref_dir, 300) or _placeholder(300, best_object_id[:8])
    pb = _label_top(ref_img, f"B — Referenzbild: {best_object_id[:30]}", fg=(255, 215, 0))

    # --- Panel C: Infos ---
    pc_h = max(pa.height, pb.height, 350)
    pc = Image.new("RGB", (340, pc_h), (12, 12, 18))
    td = ImageDraw.Draw(pc)
    td.text((10, 14), "SCHRITT 7 + 8", fill=(0, 200, 255), font=_font(20))
    td.text((10, 42), "Skalenbestimmung + Pose", fill=(160, 160, 200), font=_font(15))
    td.line([(10, 65), (330, 65)], fill=(50, 50, 80), width=1)

    y = 78
    td.text((10, y), "Skalierungsfaktor:", fill=(170, 170, 170), font=_font(14))
    y += 22
    sc_col = (60, 230, 100) if 0.5 < scale_factor < 3.0 else (255, 100, 50)
    td.text((16, y), f"× {scale_factor:.4f}", fill=sc_col, font=_font(20))
    y += 34

    if obs_size is not None:
        td.text((10, y), "Beob. Objekt-BBox (m):", fill=(150, 150, 150), font=_font(13))
        y += 18
        td.text((16, y), f"  {obs_size[0]:.3f} × {obs_size[1]:.3f} × {obs_size[2]:.3f}",
                fill="white", font=_font(13))
        y += 20
    if cad_size is not None:
        td.text((10, y), "CAD-Modell-BBox:", fill=(150, 150, 150), font=_font(13))
        y += 18
        td.text((16, y), f"  {cad_size[0]:.3f} × {cad_size[1]:.3f} × {cad_size[2]:.3f}",
                fill="white", font=_font(13))
        y += 22

    td.line([(10, y), (330, y)], fill=(50, 50, 80), width=1)
    y += 10
    td.text((10, y), "Pose-Ergebnis:", fill=(170, 170, 170), font=_font(14))
    y += 22
    if pose_info:
        for k, v in pose_info.items():
            td.text((16, y), f"{k}:", fill=(160, 160, 160), font=_font(13))
            y += 18
            td.text((20, y), str(v), fill="white", font=_font(13))
            y += 18
    else:
        td.text((16, y), "(Pose nicht berechnet)", fill=(100, 100, 100), font=_font(13))

    h_max = max(pa.height, pb.height, pc.height)
    # Pad panels to equal height
    def pad_h(img, h):
        if img.height >= h:
            return img
        out = Image.new("RGB", (img.width, h), (18, 18, 18))
        out.paste(img, (0, 0))
        return out

    final = _hstack([pad_h(pa, h_max), pad_h(pb, h_max), pad_h(pc, h_max)], pad=pad)
    final = _vstack([_banner("SCHRITT 7+8: Skalenbestimmung & Modellüberlagerung",
                             final.width, bg=(30, 10, 10)), final], pad=0)
    path = os.path.join(output_dir, "debug_07_scale_pose.png")
    final.save(path)
    logger.info("  [DEBUG 7+8] → %s", path)
    return path


# =============================================================================
# Haupt-Debug-Loop
# =============================================================================

def run_debug(args) -> None:
    from .config import PipelineConfig
    from .step1_localization import ObjectLocalizer
    from .step2_pointcloud import PointCloudGenerator
    from .step3_clip_retrieval import CLIPRetriever
    from .step4_dino_reranking import DINOReRanker
    from .step5_shape_matching import ShapeMatcher
    from .step6_fusion import ScoreFusion
    from .step7_scale_estimation import ScaleEstimator
    from .step8_pose_estimation import PoseEstimator
    from .run_pipeline import OSCARPlusPipeline
    from .utils import load_camera_intrinsics, ensure_dir

    out = ensure_dir(args.output)
    logger.info("=" * 60)
    logger.info("OSCAR+ Debug — bis Schritt %d  →  %s", args.until_step, out)
    logger.info("=" * 60)

    config = PipelineConfig(
        description_file=args.descriptions,
        reference_images_dir=args.reference_images,
        cad_models_dir=args.cad_models,
        output_dir=args.output,
        ulip_repo_path=args.ulip_repo,
        ulip2_checkpoint=args.ulip_checkpoint,
    )

    rgb_image = Image.open(args.rgb).convert("RGB")
    depth_raw = np.array(Image.open(args.depth))
    depth_m = (depth_raw.astype(np.float32) / config.depth_scale
               if depth_raw.max() > 100 else depth_raw.astype(np.float32))

    cam = {}
    if args.camera:
        img_id = int(os.path.splitext(os.path.basename(args.rgb))[0])
        cam = load_camera_intrinsics(args.camera, image_id=img_id)

    results = {}

    # ── SCHRITT 1 ──────────────────────────────────────────────────────────
    logger.info("\n─── Schritt 1: Lokalisierung ───")
    extracted = OSCARPlusPipeline._extract_object_name_heuristic(args.prompt)
    # Versuche Ollama (stumm)
    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434")
        resp = client.chat(
            model=config.ollama_model,
            messages=[
                {"role": "system",
                 "content": "Reply with ONLY the object name from the grasping instruction. No extra words."},
                {"role": "user", "content": args.prompt},
            ],
            options={"temperature": 0, "num_predict": 20},
        )
        extracted_llm = resp["message"]["content"].strip().lower()
        if extracted_llm:
            extracted = extracted_llm
            logger.info("  Ollama: '%s'", extracted)
    except Exception as e:
        logger.info("  Ollama nicht verfügbar (%s) → Heuristik: '%s'", e, extracted)

    logger.info("  Prompt:     '%s'", args.prompt)
    logger.info("  Extrahiert: '%s'", extracted)

    localizer = ObjectLocalizer(config)
    loc = localizer.localize(rgb_image, extracted)
    if loc is None:
        logger.error("Objekt nicht gefunden – Abbruch.")
        return
    results["localization"] = loc
    logger.info("  Konfidenz=%.3f  BBox=%s", loc.confidence, loc.bbox)
    save_debug_step1(rgb_image, loc.mask, loc.bbox, loc.roi_image,
                     args.prompt, extracted, loc.confidence, out)

    if args.until_step < 2:
        _done(out); return

    # ── SCHRITT 2 ──────────────────────────────────────────────────────────
    logger.info("\n─── Schritt 2: Punktwolke ───")
    pc_gen = PointCloudGenerator(config)
    pc = pc_gen.generate(np.array(rgb_image), depth_m, loc.mask,
                         fx=cam.get("fx"), fy=cam.get("fy"),
                         cx=cam.get("cx"), cy=cam.get("cy"))
    results["point_cloud"] = pc
    if pc:
        logger.info("  %d Punkte  BBox=%s", pc.num_points, pc.bbox_size)
        save_debug_step2(depth_m, loc.mask, pc.points, pc.colors,
                         pc.num_points, pc.bbox_size, out)
    else:
        logger.warning("  Keine Punktwolke erzeugt!")

    if args.until_step < 3:
        _done(out); return

    # ── SCHRITT 3 ──────────────────────────────────────────────────────────
    logger.info("\n─── Schritt 3: CLIP Retrieval ───")
    clip = CLIPRetriever(config)
    clip.load_descriptions()
    clip_res = clip.retrieve(loc.roi_image)
    results["clip_retrieval"] = clip_res
    logger.info("  %d Kandidaten", len(clip_res.candidates))
    for i, c in enumerate(clip_res.candidates[:5]):
        logger.info("    #%d  %s  score=%.4f", i+1, c.object_id, c.score)
    save_debug_step3(loc.roi_image, clip_res.candidates, args.reference_images, out)

    if args.until_step < 4:
        _done(out); return

    # ── SCHRITT 4 ──────────────────────────────────────────────────────────
    logger.info("\n─── Schritt 4: DINOv2 Re-Ranking ───")
    dino = DINOReRanker(config)
    dino.load_reference_images()
    dino_res = dino.rerank(loc.roi_image, clip_res)
    results["dino_reranking"] = dino_res
    logger.info("  %d Kandidaten", len(dino_res.candidates))
    for i, c in enumerate(dino_res.candidates[:5]):
        logger.info("    #%d  %s  dino=%.4f clip=%.4f",
                    i+1, c.object_id, c.dino_score, c.clip_score)
    save_debug_step4(loc.roi_image, dino_res.candidates, args.reference_images, out)

    if args.until_step < 5:
        _done(out); return

    # ── SCHRITT 5 ──────────────────────────────────────────────────────────
    shape_res = None
    if not args.ulip_checkpoint:
        logger.warning("\n  --ulip_checkpoint fehlt → Schritt 5 übersprungen.")
    elif pc:
        logger.info("\n─── Schritt 5: ULIP-2 Shape Matching ───")
        shape = ShapeMatcher(config)
        shape.load_cad_models()
        shape_res = shape.match(pc)
        results["shape_matching"] = shape_res
        logger.info("  %d Matches", len(shape_res.candidates))
        for i, c in enumerate(shape_res.candidates[:3]):
            logger.info("    #%d  %s  shape=%.4f", i+1, c.object_id, c.shape_score)
        save_debug_step5(pc.points, pc.colors,
                         shape_res.candidates, args.reference_images, out)

    if args.until_step < 6:
        _done(out); return

    # ── SCHRITT 6 ──────────────────────────────────────────────────────────
    logger.info("\n─── Schritt 6: Score-Fusion ───")
    fusion = ScoreFusion(config)
    fusion_res = fusion.fuse(
        clip_result=clip_res,
        dino_result=dino_res,
        shape_result=shape_res,
    )
    results["fusion"] = fusion_res
    if fusion_res.best_match:
        b = fusion_res.best_match
        logger.info("  Gewinner: %s  fused=%.4f", b.object_id, b.fused_score)
    save_debug_step6(fusion_res.candidates, args.reference_images,
                     loc.roi_image, out)

    if args.until_step < 7:
        _done(out); return

    # ── SCHRITT 7+8 ────────────────────────────────────────────────────────
    if not fusion_res.best_match:
        logger.warning("Kein Gewinner → Schritte 7+8 übersprungen.")
        _done(out); return

    best = fusion_res.best_match
    scale_factor = 1.0
    obs_size = cad_size = None

    if pc and best.cad_model_path:
        logger.info("\n─── Schritt 7: Skalenbestimmung ───")
        se = ScaleEstimator(config)
        sr = se.estimate(pc, best.cad_model_path)
        scale_factor = sr.scale_factor
        obs_size = sr.observed_size
        cad_size = sr.cad_size
        logger.info("  scale=%.4f  conf=%.2f", scale_factor, sr.confidence)

    pose_info = {}
    if args.until_step >= 8 and best.cad_model_path:
        logger.info("\n─── Schritt 8: Pose Estimation ───")
        pe = PoseEstimator(config)
        pr = pe.estimate(
            rgb_image=np.array(rgb_image),
            depth_image=depth_m,
            mask=loc.mask,
            cad_model_path=best.cad_model_path,
            scale_factor=scale_factor,
            observed_pc=pc,
            fx=cam.get("fx"), fy=cam.get("fy"),
            cx=cam.get("cx"), cy=cam.get("cy"),
        )
        pose_info = {
            "Methode": pr.method,
            "Konfidenz": f"{pr.confidence:.4f}",
            "t [m]": np.round(pr.translation, 3).tolist()
                     if hasattr(pr, "translation") else "N/A",
        }
        logger.info("  method=%s  conf=%.4f", pr.method, pr.confidence)

    save_debug_step7_8(rgb_image, loc.bbox, scale_factor, best.object_id,
                       args.reference_images, pose_info,
                       obs_size, cad_size, out)
    _done(out)


def _done(out: str) -> None:
    logger.info("\n✓ Debug fertig. Gespeicherte Bilder:")
    for f in sorted(os.listdir(out)):
        if f.startswith("debug_"):
            logger.info("    %s/%s", out, f)


# =============================================================================
# CLI
# =============================================================================

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OSCAR+ Schrittweiser Debug",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Nur mit Defaults laufen lassen (YCBV-GSO, Szene 000048, Schritt 1-6):
  python -m pipeline.debug_steps

  # Anderen Prompt testen:
  python -m pipeline.debug_steps --prompt "banana"

  # Nur Lokalisierung + Punktwolke (keine CLIP/DINO Modelle nötig):
  python -m pipeline.debug_steps --until_step 2

  # Komplett mit ULIP-2 (Schritt 5):
  python -m pipeline.debug_steps --until_step 6 \\
      --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt

  # Anderer Datensatz:
  python -m pipeline.debug_steps \\
      --rgb  eval/datasets/housecat6d/test/000001/rgb/000001.png \\
      --depth eval/datasets/housecat6d/test/000001/depth/000001.png \\
      --prompt "keyboard" \\
      --descriptions object_database/housecat6d/descriptions_attributes.json \\
      --reference_images object_images/housecat6d/ \\
      --cad_models object_database/housecat6d/ \\
      --camera eval/datasets/housecat6d/test/000001/scene_camera.json
        """,
    )
    # ── Datei-Defaults: YCBV-GSO Szene 000048, Bild 000001 ────────────────
    _RGB   = "eval/datasets/ycbv_gso/test/000048/rgb/000001.png"
    _DEPTH = "eval/datasets/ycbv_gso/test/000048/depth/000001.png"
    _CAM   = "eval/datasets/ycbv_gso/test/000048/scene_camera.json"
    _DESC  = "object_database/ycbv_gso/descriptions_attributes.json"
    _REFS  = "object_images/ycbv_gso/"
    _CADS  = "object_database/ycbv_gso/"
    # ─────────────────────────────────────────────────────────────────────────

    p.add_argument("--rgb",     default=_RGB,
                   help=f"RGB-Bild (default: {_RGB})")
    p.add_argument("--depth",   default=_DEPTH,
                   help=f"Tiefenbild (default: {_DEPTH})")
    p.add_argument("--prompt",  default="mustard bottle",
                   help='Suchprompt (default: "mustard bottle")')
    p.add_argument("--descriptions", default=_DESC,
                   help=f"descriptions_attributes.json (default: {_DESC})")
    p.add_argument("--reference_images", default=_REFS,
                   help=f"Referenzbild-Verzeichnis (default: {_REFS})")
    p.add_argument("--cad_models", default=_CADS,
                   help=f"CAD-Modell-Verzeichnis (default: {_CADS})")
    p.add_argument("--camera",  default=_CAM,
                   help=f"scene_camera.json (default: {_CAM})")
    p.add_argument("--output",  default="debug_output")
    p.add_argument("--until_step", type=int, default=6,
                   help="Bis welchem Schritt ausführen: 1-8 (default: 6)")
    p.add_argument("--ulip_repo",       default="/ulip",
                   help="ULIP-Repo-Pfad (default: /ulip)")
    p.add_argument("--ulip_checkpoint", default="",
                   help="ULIP-2 Checkpoint .pt (leer = Schritt 5 überspringen)")
    return p.parse_args()


if __name__ == "__main__":
    run_debug(_parse())
