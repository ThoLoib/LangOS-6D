# =============================================================================
# pipeline/step5_shape_matching.py – Schritt 5: Shape Matching (ULIP-2)
# =============================================================================
#
# Ziel:
#   Die partielle Punktwolke des segmentierten Objekts (Schritt 2) mit den
#   vollständigen CAD-Modell-Punktwolken vergleichen, um die geometrisch
#   ähnlichsten Modelle zu identifizieren.
#
# Pipeline:
#   ULIP-2(segmented point cloud) vs. ULIP-2(CAD model point clouds)
#   → Cosine Similarity → Top-K Shape Matches
#
# Modell:
#   • ULIP-2 – Unified Language-Image-Point Cloud Pre-training
#     Ref: https://github.com/salesforce/ULIP
#     HuggingFace: https://huggingface.co/datasets/SFXX/ulip
#     Paper: "ULIP-2: Towards Scalable Multimodal Pre-training for 3D
#             Understanding" (Xue et al., CVPR 2024)
#
#   ULIP-2 lernt einen gemeinsamen Embedding-Raum für Bilder, Text und
#   Punktwolken. Hier nutzen wir den Point-Cloud-Encoder (PointBERT),
#   um geometrische Ähnlichkeit zwischen der beobachteten Szene und
#   den CAD-Modellen zu berechnen.
#
#   Architektur (Colored PointBERT, 10k Punkte):
#     PointTransformer_Colored(xyzrgb) → 768-dim feat → pc_projection → 1280-dim
#     Das finale Embedding liegt im selben Raum wie OpenCLIP ViT-bigG-14.
#
# Effizientes Laden:
#   Wir laden NUR den point_encoder + pc_projection aus dem Checkpoint,
#   NICHT das gesamte OpenCLIP ViT-bigG-14 Modell (~5 GB). Dadurch bleibt
#   der Speicherbedarf bei ~400 MB statt ~5.5 GB.
#
# Inputs:
#   - Partielle Punktwolke (Schritt 2) mit optionalen RGB-Farben
#   - CAD-Modell-Punktwolken (vorberechnet oder on-the-fly)
#
# Outputs:
#   - Top-K Shape Matches mit Similarity Scores
#
# Voraussetzungen:
#   - ULIP-Repo geklont: git clone https://github.com/salesforce/ULIP.git
#   - Checkpoint heruntergeladen (z.B. ulip2_pointbert_10k.pt)
#   - knn_cuda installiert (für PointBERT Grouping):
#     pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
# =============================================================================

import logging
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PipelineConfig
from .step2_pointcloud import PointCloudResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenstruktur für Shape-Matching-Ergebnisse
# ---------------------------------------------------------------------------

@dataclass
class ShapeCandidate:
    """Einzelner Shape-Match-Kandidat.

    Attributes:
        object_id: Identifikator des CAD-Modells.
        shape_score: ULIP-2 Cosine-Similarity.
        cad_model_path: Pfad zum CAD-Modell.
    """
    object_id: str
    shape_score: float
    cad_model_path: str = ""


@dataclass
class ShapeMatchingResult:
    """Ergebnis des Shape Matchings (Schritt 5).

    Attributes:
        candidates: Liste der Top-K Shape Matches, sortiert nach Score.
        query_embedding: ULIP-2-Embedding der beobachteten Punktwolke.
    """
    candidates: List[ShapeCandidate]
    query_embedding: np.ndarray


# ---------------------------------------------------------------------------
# Hilfsfunktionen für Punktwolkenverarbeitung
# ---------------------------------------------------------------------------

def normalize_pointcloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    num_points: int = 10000,
) -> np.ndarray:
    """Normalisiert eine Punktwolke für ULIP-2.

    Schritte:
    1. Zentriere auf den Schwerpunkt (zero-mean) – nur XYZ.
    2. Skaliere auf Einheitskugel (max. Distanz = 1) – nur XYZ.
    3. Sample/Upsample auf fixe Punktanzahl.
    4. Optional: RGB-Farben anhängen → (N, 6).

    ULIP-2 PointBERT Colored erwartet 10.000 Punkte mit 6 Kanälen (xyzrgb).

    Args:
        points: Punktwolke (N, 3) – XYZ-Koordinaten.
        colors: Optionale Farben (N, 3) – RGB in [0, 1].
                Falls None und farbiger Encoder gewünscht, werden Nullen
                als Platzhalter verwendet.
        num_points: Gewünschte Anzahl Punkte.

    Returns:
        Normalisierte Punktwolke als (num_points, 3) oder (num_points, 6).
    """
    if len(points) == 0:
        dim = 6 if colors is not None else 3
        return np.zeros((num_points, dim), dtype=np.float32)

    # --- Zentrierung ---
    centroid = points.mean(axis=0)
    points_centered = points - centroid

    # --- Skalierung auf Einheitskugel ---
    max_dist = np.linalg.norm(points_centered, axis=1).max()
    if max_dist > 0:
        points_centered = points_centered / max_dist

    # --- Resampling auf fixe Punktanzahl ---
    n = len(points_centered)
    if n >= num_points:
        indices = np.random.choice(n, num_points, replace=False)
    else:
        indices = np.random.choice(n, num_points, replace=True)

    result_xyz = points_centered[indices].astype(np.float32)

    if colors is not None:
        # RGB-Farben normalisieren auf [0, 1] falls nötig
        if colors.max() > 1.0:
            colors = colors / 255.0
        result_rgb = colors[indices].astype(np.float32)
        return np.concatenate([result_xyz, result_rgb], axis=1)  # (N, 6)

    return result_xyz  # (N, 3)


def sample_pointcloud_from_mesh(
    mesh_path: str,
    num_points: int = 10000,
    with_colors: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Sampelt eine Punktwolke von der Oberfläche eines 3D-Meshes.

    Unterstützt OBJ, PLY, GLB und andere von Open3D/Trimesh
    unterstützte Formate.

    Args:
        mesh_path: Pfad zum 3D-Modell.
        num_points: Anzahl der zu sampleden Punkte.
        with_colors: Vertex-Farben extrahieren (falls vorhanden).

    Returns:
        Tuple (points, colors):
            points: (num_points, 3) numpy-Array.
            colors: (num_points, 3) numpy-Array in [0,1] oder None.
    """
    colors = None
    try:
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
        points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
        points = points.astype(np.float32)

        if with_colors and hasattr(mesh.visual, 'face_colors'):
            face_colors = mesh.visual.face_colors[face_indices][:, :3]
            colors = (face_colors / 255.0).astype(np.float32)

        return points, colors

    except ImportError:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not mesh.has_triangles():
            logger.warning(f"Mesh {mesh_path} hat keine Dreiecke.")
            return np.zeros((num_points, 3), dtype=np.float32), None
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        points = np.asarray(pcd.points, dtype=np.float32)

        if with_colors and pcd.has_colors():
            raw = np.asarray(pcd.colors)
            colors = np.clip(raw, 0.0, 1.0).astype(np.float32)

        return points, colors


# ---------------------------------------------------------------------------
# Leichtgewichtiger ULIP-2 Point-Cloud-Encoder Wrapper
# ---------------------------------------------------------------------------

class ULIP2PointEncoder(nn.Module):
    """Leichtgewichtiger Wrapper für den ULIP-2 Point-Cloud-Encoder.

    Lädt NUR den PointBERT/PointNeXt Encoder + pc_projection aus dem
    ULIP-2 Checkpoint, ohne das gesamte OpenCLIP ViT-bigG-14 Modell.

    Speicherbedarf: ~400 MB statt ~5.5 GB.

    Ref: "ULIP-2: Towards Scalable Multimodal Pre-training for 3D
          Understanding" (Xue et al., CVPR 2024)
    """

    def __init__(self, point_encoder: nn.Module, pc_projection: nn.Parameter):
        super().__init__()
        self.point_encoder = point_encoder
        self.pc_projection = pc_projection

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        """Encodiert eine Punktwolke in ein ULIP-2-Embedding.

        Args:
            pc: Punktwolke (B, N, C) mit C=3 (xyz) oder C=6 (xyzrgb).

        Returns:
            Normalisiertes Embedding (B, embed_dim).
        """
        pc_feat = self.point_encoder(pc)             # (B, 768)
        pc_embed = pc_feat @ self.pc_projection      # (B, embed_dim)
        pc_embed = F.normalize(pc_embed, p=2, dim=1)
        return pc_embed


# ---------------------------------------------------------------------------
# ULIP-2 Image Encoder (OpenCLIP ViT-bigG-14) für Cross-Modal Retrieval
# ---------------------------------------------------------------------------

class ULIP2ImageEncoder:
    """OpenCLIP ViT-bigG-14 Image Encoder für ULIP-2 Cross-Modal Retrieval.

    ULIP-2 friert den Image-Encoder während des Trainings ein, deshalb
    sind vanilla OpenCLIP ViT-bigG-14 Gewichte (laion2b_s39b_b160k) identisch
    mit dem ULIP-2 Image-Branch — kein separater Download nötig.

    Cross-Modal: image_embedding und pc_embedding liegen im selben 1280-dim
    Raum → Cosine-Similarity direkt vergleichbar.

    Ref: "ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding"
         Xue et al., CVPR 2024
    """

    def __init__(self, device: str = "cpu", checkpoint_path: Optional[str] = None):
        self.device = device
        self._checkpoint_path = checkpoint_path
        self.model: Optional[nn.Module] = None
        self.preprocess = None

    def load(self) -> None:
        """Lädt OpenCLIP ViT-bigG-14 (lazy init)."""
        if self.model is not None:
            return

        try:
            import open_clip  # type: ignore
        except ImportError:
            raise ImportError(
                "open_clip nicht installiert.\n"
                "  pip install open-clip-torch"
            )

        logger.info(
            "Lade OpenCLIP ViT-bigG-14 Image-Encoder "
            "(für ULIP-2 Cross-Modal Retrieval)..."
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-bigG-14", pretrained="laion2b_s39b_b160k"
        )

        # Optional: visuelle Gewichte aus ULIP-2 Checkpoint laden.
        # Da ULIP-2 den Image-Encoder einfriert, sind vanilla Gewichte
        # identisch — dieser Block ist nur zur Vollständigkeit.
        if self._checkpoint_path and os.path.isfile(self._checkpoint_path):
            try:
                ckpt = torch.load(
                    self._checkpoint_path, map_location="cpu", weights_only=False
                )
                sd = ckpt.get("state_dict", ckpt)
                sd = OrderedDict({k.replace("module.", ""): v for k, v in sd.items()})
                visual_sd = {k[len("visual."):]: v for k, v in sd.items()
                             if k.startswith("visual.")}
                if visual_sd:
                    res = model.visual.load_state_dict(visual_sd, strict=False)
                    logger.info(
                        f"Visuelle Gewichte aus Checkpoint geladen "
                        f"(fehlend={len(res.missing_keys)}, "
                        f"unerwartet={len(res.unexpected_keys)})"
                    )
                else:
                    logger.info(
                        "Keine visual.* Keys im Checkpoint — "
                        "verwende vanilla OpenCLIP (erwartet, ULIP-2 friert Image-Encoder ein)."
                    )
            except Exception as exc:
                logger.warning(
                    f"Konnte visuelle Gewichte nicht laden: {exc}. "
                    "Verwende vanilla OpenCLIP."
                )

        self.model = model.visual.to(self.device)
        self.model.eval()
        self.preprocess = preprocess

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"OpenCLIP ViT-bigG-14 Image-Encoder geladen: "
            f"{n_params / 1e6:.1f}M Parameter, Device: {self.device}"
        )

    def encode(self, image) -> torch.Tensor:
        """Encodiert ein PIL-Bild oder numpy-Array als ULIP-2-kompatibles Embedding.

        Args:
            image: PIL.Image oder numpy-Array (H, W, 3) uint8.

        Returns:
            L2-normalisierter Tensor (1, 1280).
        """
        self.load()
        from PIL import Image as PILImage  # type: ignore

        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image.astype(np.uint8))
        elif not isinstance(image, PILImage.Image):
            raise TypeError(
                f"Erwartet PIL.Image oder numpy.ndarray, got {type(image)}"
            )

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)  # (1, 1280)
            embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding


# ---------------------------------------------------------------------------
# ULIP-2 Shape Matching Modul
# ---------------------------------------------------------------------------

class ShapeMatcher:
    """Shape Matching via ULIP-2 Point-Cloud-Embeddings.

    Vergleicht die partielle Punktwolke des beobachteten Objekts mit
    den vollständigen CAD-Modell-Punktwolken im ULIP-2 Embedding-Raum.

    ULIP-2 bildet Punktwolken in denselben Embedding-Raum wie OpenCLIP
    Bilder und Texte ab (1280-dim für ViT-bigG-14). Dies ermöglicht
    auch cross-modale Vergleiche.

    Modell-Architektur:
        PointTransformer_Colored (PointBERT mit 6D input):
        - Input: (B, 10000, 6) xyzrgb
        - Backbone: 18-layer Transformer, 384-dim, 6 heads
        - Grouping: 512 groups × 32 points (FPS + kNN)
        - Output: 768-dim → pc_projection → 1280-dim

    Usage:
        >>> config = PipelineConfig(
        ...     ulip_repo_path="/path/to/ULIP",
        ...     ulip2_checkpoint="/path/to/checkpoint.pt",
        ... )
        >>> matcher = ShapeMatcher(config)
        >>> matcher.load_cad_models("object_database/ycbv_gso/")
        >>> result = matcher.match(point_cloud_result, top_k=5)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self.model: Optional[ULIP2PointEncoder] = None
        self.image_encoder: Optional[ULIP2ImageEncoder] = None

        # Gecachte CAD-Modell-Embeddings
        self._cad_embeddings: Dict[str, torch.Tensor] = {}
        self._cad_paths: Dict[str, str] = {}

    def _load_model(self):
        """Lädt den ULIP-2 Point-Cloud-Encoder.

        Ablauf:
        1. ULIP-Repo in sys.path hinzufügen
        2. PointTransformer_Colored aus ULIP-Repo importieren
        3. Modell-Architektur aus YAML-Config erstellen
        4. Nur point_encoder + pc_projection Gewichte aus Checkpoint laden
        5. In ULIP2PointEncoder Wrapper packen

        Benötigt:
        - ULIP-Repo unter config.ulip_repo_path
        - Checkpoint unter config.ulip2_checkpoint
        - knn_cuda installiert
        """
        if self.model is not None:
            return

        checkpoint_path = self.config.ulip2_checkpoint
        ulip_repo_path = self.config.ulip_repo_path

        # --- Validierung ---
        if not ulip_repo_path or not os.path.isdir(ulip_repo_path):
            raise FileNotFoundError(
                f"ULIP-Repo nicht gefunden: '{ulip_repo_path}'\n"
                "Setze ulip_repo_path in der Config auf den Pfad zum geklonten ULIP-Repo.\n"
                "  git clone https://github.com/salesforce/ULIP.git"
            )

        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"ULIP-2 Checkpoint nicht gefunden: '{checkpoint_path}'\n"
                "Download von HuggingFace:\n"
                "  https://huggingface.co/datasets/SFXX/ulip/tree/main/ULIP-2/pretrained_models\n"
                "Empfohlen: ULIP-2-PointBERT-10k-xyzrgb (402 MB)"
            )

        # --- ULIP-Repo in Python-Path einfügen ---
        if ulip_repo_path not in sys.path:
            sys.path.insert(0, ulip_repo_path)
            logger.info(f"ULIP-Repo zu sys.path hinzugefügt: {ulip_repo_path}")

        logger.info("Lade ULIP-2 Point-Cloud-Encoder (nur PointBERT + Projection)...")

        # --- 1. Modell-Architektur erstellen ---
        point_encoder, pc_feat_dims = self._create_point_encoder(ulip_repo_path)

        # --- 2. Checkpoint laden und Gewichte extrahieren ---
        embed_dim = self.config.ulip2_embed_dim  # 1280 für ViT-bigG-14
        pc_projection = nn.Parameter(torch.empty(pc_feat_dims, embed_dim))

        logger.info(f"Lade Checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Checkpoint-Format: {'state_dict': {...}, 'args': ..., 'epoch': ...}
        if "state_dict" in ckpt:
            state_dict = OrderedDict()
            for k, v in ckpt["state_dict"].items():
                # Entferne 'module.' Prefix (von DistributedDataParallel)
                state_dict[k.replace("module.", "")] = v
        else:
            state_dict = ckpt

        # --- 3. point_encoder Gewichte extrahieren ---
        pe_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("point_encoder."):
                pe_state_dict[k[len("point_encoder."):]] = v

        if not pe_state_dict:
            raise RuntimeError(
                "Keine point_encoder.* Keys im Checkpoint gefunden.\n"
                f"Verfügbare Top-Level Keys: {list(state_dict.keys())[:20]}"
            )

        result = point_encoder.load_state_dict(pe_state_dict, strict=False)
        if result.missing_keys:
            logger.warning(f"Fehlende Keys im point_encoder: {result.missing_keys}")
        if result.unexpected_keys:
            logger.warning(f"Unerwartete Keys im point_encoder: {result.unexpected_keys}")

        # --- 4. pc_projection Gewichte laden ---
        if "pc_projection" in state_dict:
            pc_projection.data.copy_(state_dict["pc_projection"])
            logger.info(
                f"pc_projection geladen: {pc_projection.shape} "
                f"({pc_feat_dims} → {embed_dim})"
            )
        else:
            logger.warning(
                "pc_projection nicht im Checkpoint gefunden. "
                "Verwende zufällige Initialisierung."
            )
            nn.init.normal_(pc_projection, std=embed_dim ** -0.5)

        # --- 5. Wrapper erstellen ---
        self.model = ULIP2PointEncoder(point_encoder, pc_projection)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Parameter zählen
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"ULIP-2 Point-Cloud-Encoder geladen: "
            f"{n_params / 1e6:.1f}M Parameter, "
            f"Embedding-Dim: {embed_dim}, "
            f"Device: {self.device}"
        )

    def _create_point_encoder(
        self, ulip_repo_path: str
    ) -> Tuple[nn.Module, int]:
        """Erstellt den Point-Cloud-Encoder basierend auf config.ulip2_backbone.

        Unterstützte Backbones:
        - "pointbert_colored": PointTransformer_Colored (6D input, ULIP-2 Standard)
        - "pointbert": PointTransformer (3D input)
        - "pointnext": PointNEXT

        Args:
            ulip_repo_path: Pfad zum geklonten ULIP-Repo.

        Returns:
            Tuple (point_encoder, pc_feat_dims):
                point_encoder: Das PyTorch-Modul.
                pc_feat_dims: Output-Feature-Dimension des Encoders.
        """
        from easydict import EasyDict
        from utils.config import cfg_from_yaml_file  # type: ignore  # ULIP utils

        backbone = self.config.ulip2_backbone

        if backbone == "pointbert_colored":
            from models.pointbert.point_encoder import PointTransformer_Colored  # type: ignore

            yaml_path = os.path.join(
                ulip_repo_path,
                "models", "pointbert",
                "ULIP_2_PointBERT_10k_colored_pointclouds.yaml",
            )
            if not os.path.isfile(yaml_path):
                raise FileNotFoundError(
                    f"PointBERT YAML-Config nicht gefunden: {yaml_path}"
                )
            config = cfg_from_yaml_file(yaml_path)

            # evaluate_3d=True verhindert, dass PointTransformer versucht,
            # den PointBERT-Pretrained-Checkpoint zu laden (der nicht nötig
            # ist, da wir den vollen ULIP-2 Checkpoint laden)
            dummy_args = EasyDict({"evaluate_3d": True})
            point_encoder = PointTransformer_Colored(config.model, args=dummy_args)
            pc_feat_dims = 768  # concat_f: cls_token (384) + max_pool (384)

        elif backbone == "pointbert":
            from models.pointbert.point_encoder import PointTransformer  # type: ignore

            yaml_path = os.path.join(
                ulip_repo_path,
                "models", "pointbert",
                "PointTransformer_8192point.yaml",
            )
            config = cfg_from_yaml_file(yaml_path)
            dummy_args = EasyDict({"evaluate_3d": True})
            point_encoder = PointTransformer(config.model, args=dummy_args)
            pc_feat_dims = 768

        elif backbone == "pointnext":
            from models.pointnext.pointnext import PointNEXT  # type: ignore

            point_encoder = PointNEXT()
            pc_feat_dims = 256

        else:
            raise ValueError(
                f"Unbekannter ULIP-2 Backbone: '{backbone}'. "
                "Erlaubt: 'pointbert_colored', 'pointbert', 'pointnext'"
            )

        logger.info(
            f"Point-Encoder erstellt: {backbone}, "
            f"Output-Dim: {pc_feat_dims}"
        )
        return point_encoder, pc_feat_dims

    def encode_pointcloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Encodiert eine Punktwolke in ein ULIP-2-Embedding.

        Args:
            points: Punktwolke (N, 3) – XYZ-Koordinaten.
            colors: Optionale Farben (N, 3) – RGB in [0, 1].
                    Für 'pointbert_colored': wird verwendet oder mit 0 aufgefüllt.
                    Für andere Backbones: wird ignoriert.

        Returns:
            Normalisierter Tensor (1, embed_dim).
        """
        self._load_model()

        use_colors = (
            self.config.ulip2_use_colors
            and self.config.ulip2_backbone == "pointbert_colored"
        )

        if use_colors:
            # xyzrgb → (N, 6)
            if colors is None:
                # Kein RGB vorhanden → Nullen als Platzhalter
                colors = np.zeros_like(points)
            pts_norm = normalize_pointcloud(
                points,
                colors=colors,
                num_points=self.config.ulip2_num_points,
            )  # (N, 6)
        else:
            # xyz only → (N, 3)
            pts_norm = normalize_pointcloud(
                points,
                colors=None,
                num_points=self.config.ulip2_num_points,
            )  # (N, 3)

        # (1, N, C) Tensor
        pts_tensor = torch.from_numpy(pts_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(pts_tensor)  # (1, embed_dim)

        return embedding

    def load_cad_models(
        self,
        cad_dir: Optional[str] = None,
        allowed_extensions: Tuple[str, ...] = (".obj", ".ply", ".glb", ".gltf"),
    ) -> None:
        """Lädt und encodiert CAD-Modelle als Punktwolken.

        Erwartete Ordnerstruktur:
            cad_dir/
                object_label_1.obj
                object_label_2.ply
                ...
            oder:
            cad_dir/
                object_label_1/
                    model.obj
                ...

        Args:
            cad_dir: Pfad zum CAD-Modell-Ordner.
                     Falls None, wird config.cad_models_dir verwendet.
            allowed_extensions: Erlaubte Dateierweiterungen.
        """
        self._load_model()

        cad_dir = cad_dir or self.config.cad_models_dir
        if not cad_dir:
            raise ValueError("Kein cad_models_dir konfiguriert.")

        logger.info(f"Lade CAD-Modelle aus: {cad_dir}")
        count = 0

        for entry in sorted(os.listdir(cad_dir)):
            entry_path = os.path.join(cad_dir, entry)

            # Fall 1: Direktes Mesh-File
            if os.path.isfile(entry_path):
                ext = os.path.splitext(entry)[1].lower()
                if ext not in allowed_extensions:
                    continue
                obj_id = os.path.splitext(entry)[0]
                if self._encode_and_cache(obj_id, entry_path):
                    count += 1
                continue

            # Fall 2: Unterordner mit Mesh-File
            if os.path.isdir(entry_path):
                mesh_file = self._find_mesh_in_dir(entry_path, allowed_extensions)
                if mesh_file:
                    if self._encode_and_cache(entry, mesh_file):
                        count += 1

        logger.info(f"CAD-Modell-Embeddings berechnet: {count} Modelle.")

    def _find_mesh_in_dir(
        self, dir_path: str, extensions: Tuple[str, ...]
    ) -> Optional[str]:
        """Findet rekursiv eine passende Mesh-Datei in einem Objektordner.

        Bevorzugt gängige Dateinamen im `meshes/`-Unterordner (GSO/YCBV-GSO),
        fällt sonst auf die erste gefundene Mesh-Datei zurück.
        """
        preferred_names = ("textured_simple.obj", "model.obj", "mesh.obj")
        candidates: List[str] = []

        for root, _, files in os.walk(dir_path):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in extensions:
                    candidates.append(os.path.join(root, fname))

        if not candidates:
            return None

        def sort_key(path: str) -> Tuple[int, int, str]:
            base = os.path.basename(path).lower()
            in_meshes = 0 if os.path.basename(os.path.dirname(path)).lower() == "meshes" else 1
            try:
                pref_idx = preferred_names.index(base)
            except ValueError:
                pref_idx = len(preferred_names)
            return (in_meshes, pref_idx, path)

        return sorted(candidates, key=sort_key)[0]

    def _encode_and_cache(self, obj_id: str, mesh_path: str) -> bool:
        """Sampelt Punkte von einem Mesh und berechnet das ULIP-2-Embedding."""
        try:
            use_colors = (
                self.config.ulip2_use_colors
                and self.config.ulip2_backbone == "pointbert_colored"
            )
            points, colors = sample_pointcloud_from_mesh(
                mesh_path,
                self.config.ulip2_num_points,
                with_colors=use_colors,
            )
            embedding = self.encode_pointcloud(points, colors=colors)
            self._cad_embeddings[obj_id] = embedding.squeeze(0)
            self._cad_paths[obj_id] = mesh_path
            return True
        except Exception as e:
            logger.warning(f"Fehler bei CAD-Modell {obj_id}: {e}")
            return False

    def _load_image_encoder(self) -> None:
        """Lädt den ULIP2ImageEncoder (lazy init)."""
        if self.image_encoder is None:
            self.image_encoder = ULIP2ImageEncoder(
                device=self.device,
                checkpoint_path=self.config.ulip2_checkpoint or None,
            )
        self.image_encoder.load()

    def encode_image(self, image) -> torch.Tensor:
        """Encodiert ein Query-Bild in das ULIP-2 Embedding (cross-modal).

        Args:
            image: PIL.Image oder numpy-Array (H, W, 3) uint8.

        Returns:
            L2-normalisierter Tensor (1, embed_dim).
        """
        self._load_image_encoder()
        return self.image_encoder.encode(image)

    def match(
        self,
        observed_pc: PointCloudResult,
        top_k: Optional[int] = None,
        candidate_ids: Optional[List[str]] = None,
        query_image=None,
    ) -> ShapeMatchingResult:
        """Findet die geometrisch ähnlichsten CAD-Modelle.

        Der Retrieval-Modus wird über config.ulip2_mode gesteuert:
          - "pc"    : Beobachtete Punktwolke → PC-Encoder → mit CAD-PC verglichen.
          - "cross" : Cropped Query-Bild → OpenCLIP Image-Encoder → mit CAD-PC verglichen.
                      Nutzt den gemeinsamen ULIP-2 Embedding-Raum cross-modal.
          - "both"  : Gewichteter Mittelwert aus PC- und Image-Embedding
                      (ulip2_image_weight aus Config).

        Args:
            observed_pc: Punktwolke des beobachteten Objekts (Schritt 2).
            top_k: Anzahl der Ergebnisse (überschreibt Config).
            candidate_ids: Optional – nur diese Objekte vergleichen.
            query_image: PIL.Image oder numpy-Array (H, W, 3) – Query-Bild für
                         cross-modal Retrieval. Wird ignoriert wenn mode="pc".

        Returns:
            ShapeMatchingResult mit sortierten Kandidaten.
        """
        if not self._cad_embeddings:
            raise RuntimeError(
                "CAD-Modelle nicht geladen. Rufe load_cad_models() auf."
            )

        top_k = top_k or self.config.ulip2_top_k
        mode = getattr(self.config, "ulip2_mode", "pc")

        # --- Query Embedding aufbauen ---
        query_emb: Optional[torch.Tensor] = None

        # PC-Embedding (immer nötig bei mode="pc" oder "both")
        pc_emb: Optional[torch.Tensor] = None
        if mode in ("pc", "both"):
            colors = getattr(observed_pc, "colors", None)
            pc_emb = self.encode_pointcloud(
                observed_pc.points, colors=colors
            )  # (1, embed_dim)

        # Image-Embedding (nötig bei mode="cross" oder "both")
        img_emb: Optional[torch.Tensor] = None
        if mode in ("cross", "both"):
            if query_image is None:
                if mode == "cross":
                    logger.warning(
                        "ulip2_mode='cross' aber kein query_image übergeben — "
                        "Fallback auf PC-Modus."
                    )
                    mode = "pc"
                    pc_emb = self.encode_pointcloud(
                        observed_pc.points,
                        colors=getattr(observed_pc, "colors", None),
                    )
                # mode="both" ohne Bild → nur PC nutzen
            else:
                img_emb = self.encode_image(query_image)  # (1, embed_dim)

        # Finales Query-Embedding zusammensetzen
        if mode == "pc" or (mode == "both" and img_emb is None):
            query_emb = pc_emb
            logger.info("ULIP-2: Retrieval-Modus = PC→PC (shape matching)")
        elif mode == "cross":
            query_emb = img_emb
            logger.info("ULIP-2: Retrieval-Modus = Image→PC (cross-modal)")
        else:  # "both" mit beiden Embeddings
            w_img = getattr(self.config, "ulip2_image_weight", 0.5)
            w_pc = 1.0 - w_img
            query_emb = F.normalize(
                w_img * img_emb + w_pc * pc_emb, p=2, dim=-1
            )
            logger.info(
                f"ULIP-2: Retrieval-Modus = both "
                f"(image_weight={w_img:.2f}, pc_weight={w_pc:.2f})"
            )

        # --- Kandidaten zusammenstellen ---
        if candidate_ids:
            obj_ids = [oid for oid in candidate_ids if oid in self._cad_embeddings]
        else:
            obj_ids = list(self._cad_embeddings.keys())

        if not obj_ids:
            logger.warning("Keine passenden CAD-Modell-Embeddings gefunden.")
            return ShapeMatchingResult(
                candidates=[], query_embedding=query_emb.cpu().numpy()
            )

        # --- Cosine Similarity ---
        cad_embs = torch.stack(
            [self._cad_embeddings[oid] for oid in obj_ids]
        ).to(self.device)  # (K, embed_dim)

        sims = (query_emb @ cad_embs.T).squeeze(0)  # (K,)

        # --- NaN-Scores filtern ---
        nan_mask = torch.isnan(sims)
        if nan_mask.any():
            logger.warning(
                "%d von %d CAD-Embeddings haben NaN-Similarity (werden ignoriert).",
                nan_mask.sum().item(), len(sims),
            )
            sims = torch.where(nan_mask, torch.tensor(-1.0, device=sims.device), sims)

        # --- Top-K ---
        k = min(top_k, len(obj_ids))
        topk_scores, topk_indices = sims.topk(k)

        candidates = []
        for score, idx in zip(topk_scores.tolist(), topk_indices.tolist()):
            obj_id = obj_ids[idx]
            candidates.append(ShapeCandidate(
                object_id=obj_id,
                shape_score=score,
                cad_model_path=self._cad_paths.get(obj_id, ""),
            ))

        logger.info(
            f"Shape Matching: {len(candidates)} Kandidaten "
            f"(Top: {candidates[0].object_id}, "
            f"Score={candidates[0].shape_score:.4f})"
        )

        return ShapeMatchingResult(
            candidates=candidates,
            query_embedding=query_emb.cpu().numpy(),
        )
