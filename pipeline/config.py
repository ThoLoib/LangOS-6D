# =============================================================================
# pipeline/config.py – Zentrale Konfiguration für die gesamte Pipeline
# =============================================================================
#
# Alle Hyperparameter, Pfade und Modellbezeichnungen an einem Ort.
# Kann per CLI oder YAML überschrieben werden.
# =============================================================================

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class PipelineConfig:
    """Zentrale Konfigurationsklasse für die OSCAR+ Pipeline.

    Attribute sind nach Pipeline-Schritten gruppiert.
    """

    # -------------------------------------------------------------------------
    # Allgemein
    # -------------------------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # -------------------------------------------------------------------------
    # Schritt 1 – Objektlokalisierung (GroundingDINO + SAM)
    # -------------------------------------------------------------------------
    # GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
    # SAM2: https://github.com/facebookresearch/segment-anything-2
    # LangSAM (Wrapper): https://github.com/luca-medeiros/lang-segment-anything
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base"
    sam_model: str = "facebook/sam-vit-large"
    detection_confidence: float = 0.3   # Mindest-Konfidenz für Bounding Boxes
    segmentation_threshold: float = 0.5  # SAM-Masken-Schwelle

    # -------------------------------------------------------------------------
    # Schritt 2 – Punktwolkenerzeugung
    # -------------------------------------------------------------------------
    # Open3D: http://www.open3d.org/docs/release/
    # Standardmäßige Kameraintrinsics (überschreibbar pro Datensatz)
    camera_fx: float = 591.0    # Fokuslänge x (Pixel)
    camera_fy: float = 591.0    # Fokuslänge y (Pixel)
    camera_cx: float = 320.0    # Hauptpunkt x
    camera_cy: float = 240.0    # Hauptpunkt y
    depth_scale: float = 10000.0  # Konversion: depth_raw / depth_scale = Meter
    depth_trunc: float = 10.0     # Maximale Tiefe in Metern
    voxel_size: float = 0.002    # Voxel-Downsampling-Größe (Meter)

    # -------------------------------------------------------------------------
    # Schritt 3 – Semantische Kandidatensuche (CLIP)
    # -------------------------------------------------------------------------
    # CLIP: https://github.com/openai/CLIP
    clip_model_name: str = "ViT-B/32"
    clip_top_k: int = 20         # Anzahl der CLIP-Kandidaten
    clip_threshold: float = 0.25  # Alternativ: Schwellen-basierte Filterung

    # Pfad zu den Objektbeschreibungen (JSON, erzeugt via description_generator)
    description_file: str = ""    # z.B. "object_database/ycbv_gso/descriptions_attributes.json"

    # -------------------------------------------------------------------------
    # Schritt 4 – Bildbasiertes Re-Ranking (DINOv2)
    # -------------------------------------------------------------------------
    # DINOv2: https://github.com/facebookresearch/dinov2
    dino_model_name: str = "facebook/dinov2-base"
    dino_top_k: int = 5          # Anzahl der DINOv2-Kandidaten nach Re-Ranking

    # Pfad zu vorgerenderten Referenzbildern
    reference_images_dir: str = ""  # z.B. "object_images/ycbv_gso/"

    # -------------------------------------------------------------------------
    # Schritt 5 – Shape Matching (ULIP-2)
    # -------------------------------------------------------------------------
    # ULIP-2: https://github.com/salesforce/ULIP
    # HuggingFace: https://huggingface.co/datasets/SFXX/ulip
    ulip_repo_path: str = ""       # Pfad zum geklonten ULIP-Repo (für Model-Imports)
    ulip2_checkpoint: str = ""     # Pfad zum ULIP-2-Modell-Checkpoint (.pt)
    ulip2_backbone: str = "pointbert_colored"  # "pointbert_colored" | "pointbert" | "pointnext"
    ulip2_top_k: int = 5          # Anzahl der Shape-Kandidaten
    ulip2_num_points: int = 10000  # Punktanzahl für normalisierte Point Clouds
    ulip2_use_colors: bool = True  # xyzrgb (True) oder nur xyz (False)
    ulip2_embed_dim: int = 1280    # Embedding-Dimension (1280 für ViT-bigG-14, 512 für ViT-B)

    # Pfad zu den CAD-Modellen (OBJ/PLY/GLB)
    cad_models_dir: str = ""       # z.B. "object_database/ycbv_gso/"

    # -------------------------------------------------------------------------
    # Schritt 6 – Fusion / Konsens
    # -------------------------------------------------------------------------
    fusion_method: str = "weighted_sum"  # "weighted_sum" | "intersection" | "rank_fusion"
    weight_clip: float = 0.3
    weight_dino: float = 0.4
    weight_ulip: float = 0.3
    fusion_top_k: int = 1               # Finale Anzahl Kandidaten

    # -------------------------------------------------------------------------
    # Schritt 7 – Skalenbestimmung
    # -------------------------------------------------------------------------
    # Keine spezifischen Hyperparameter – nutzt die Punktwolke aus Schritt 2
    # und die Bounding Box des ausgewählten CAD-Modells.

    # -------------------------------------------------------------------------
    # Schritt 8 – Pose Estimation
    # -------------------------------------------------------------------------
    # MegaPose: https://github.com/megapose6d/megapose6d
    # FoundationPose: https://github.com/NVlabs/FoundationPose
    pose_method: str = "icp"  # "foundationpose" | "megapose" | "icp"
    icp_max_iterations: int = 50
    icp_threshold: float = 0.02          # Konvergenz-Schwelle (Meter)

    # -------------------------------------------------------------------------
    # Prompt-Parsing (Ollama LLM)
    # -------------------------------------------------------------------------
    # Ollama: https://ollama.com/
    # Installiert im Docker-Container; Python-Client in requirements.txt
    ollama_host: str = "http://localhost:11434"  # Ollama läuft im Container (start.sh)
    ollama_model: str = "gemma3:4b"       # Modell für Prompt-Parsing (im Dockerfile gepullt)
    ollama_timeout: float = 30.0                  # Sekunden bis Timeout (Fallback auf Heuristik)

    # -------------------------------------------------------------------------
    # Ein-/Ausgabepfade
    # -------------------------------------------------------------------------
    output_dir: str = "pipeline_output"  # Ordner für alle Pipeline-Ergebnisse

    def to_dict(self) -> dict:
        """Konvertiert die Konfiguration in ein Dictionary (z.B. für JSON-Export)."""
        return {k: str(v) if isinstance(v, torch.device) else v
                for k, v in self.__dict__.items()}
