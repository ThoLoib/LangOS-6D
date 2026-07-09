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
    # Schritt 1 – Objektlokalisierung (GroundingDINO + SAM2.1)
    # -------------------------------------------------------------------------
    # GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
    # SAM2.1: https://github.com/facebookresearch/sam2
    #   HF: https://huggingface.co/facebook/sam2.1-hiera-large
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base"
    sam_model: str = "facebook/sam2.1-hiera-large"  # SAM2.1 (Ravi et al., 2024)
    detection_confidence: float = 0.3   # Mindest-Konfidenz für Bounding Boxes

    # Mask post-processing (thesis Step A: largest connected component + dilation)
    mask_largest_cc: bool = True         # Retain only largest connected component
    mask_dilation_kernel: int = 5        # Dilation kernel size (pixels)
    mask_dilation_iterations: int = 1    # Number of dilation iterations (0 = disabled)

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
    depth_trunc: float = 2.0      # Maximale Tiefe in Metern (2m für Tabletop-Szenen)
    voxel_size: float = 0.002    # Voxel-Downsampling-Größe (Meter)

    # Depth Gating (2D-Vorfilter auf maskierter Tiefe, vor Rückprojektion)
    depth_gate_enabled: bool = True
    depth_gate_tolerance: float = 0.3   # ±30% um den Median

    # Statistical Outlier Removal (3D-Nachfilter)
    sor_nb_neighbors: int = 10
    sor_std_ratio: float = 1.0

    # Radius Outlier Removal (3D-Nachfilter, optional)
    ror_enabled: bool = False
    ror_nb_points: int = 10       # Min. Nachbarn innerhalb des Radius
    ror_radius: float = 0.01     # Suchradius in Metern

    # -------------------------------------------------------------------------
    # Schritt 3 – Semantische Kandidatensuche (CLIP)
    # -------------------------------------------------------------------------
    # CLIP: https://github.com/openai/CLIP
    clip_model_name: str = "ViT-B/32"
    clip_top_k: int = 20         # Anzahl der CLIP-Kandidaten
    clip_threshold: float = 0.25  # Alternativ: Schwellen-basierte Filterung - noch nicht implementiert

    # Pfad zu den Objektbeschreibungen (JSON, erzeugt via description_generator)
    description_file: str = ""    # z.B. "object_database/ycbv_gso/descriptions_attributes.json"

    # -------------------------------------------------------------------------
    # Schritt 4 – Bildbasiertes Re-Ranking (DINOv2)
    # -------------------------------------------------------------------------
    # DINOv2: https://github.com/facebookresearch/dinov2
    dino_model_name: str = "facebook/dinov2-base"
    dino_pooling: str = "cls"    # "cls" = CLS token (CNOS/thesis default), "mean" = average pooling (legacy)
    dino_top_k: int = 5          # Anzahl der DINOv2-Kandidaten nach Re-Ranking

    # Multi-view aggregation for DINOv2 re-ranking (inspired by OPEN, Chu et al. 2024)
    # "max" = hard best-view (legacy), "mean" = average all views,
    # "softmax" = softmax-weighted over all views, "topk_softmax" = softmax over top-k views
    dino_view_aggregation: str = "topk_softmax"
    dino_view_topk: int = 5       # Number of top views for topk_softmax (CNOS default, thesis Table 4.1)
    dino_view_temperature: float = 0.5  # Softmax temperature (lower = sharper)

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

    # Modus für ULIP-2 Query-Embedding:
    #   "pc"    – Beobachtete Punktwolke → ULIP-PC-Encoder → mit CAD-PC-Embeddings vergleichen (wie bisher)
    #   "cross" – Cropped Query-Bild → OpenCLIP ViT-bigG-14 (ULIP Image-Branch) → mit CAD-PC-Embeddings
    #             vergleichen (cross-modal; nutzt den gemeinsamen Embedding-Raum von ULIP-2 voll aus).
    #             ULIP-2 friert den Image-Encoder während Training ein → vanilla OpenCLIP Gewichte sind korrekt.
    #   "both"  – Gewichteter Durchschnitt aus "pc" und "cross" Embeddings.
    ulip2_mode: str = "cross"         # "pc" | "cross" | "both"
    ulip2_image_weight: float = 0.5   # Gewicht für Image-Embedding in Modus "both" (PC-Gewicht = 1 - image_weight)
    ulip2_use_partial_views: bool = False  # True = precomputed partial PCs per view; False = full mesh (legacy)

    # Multi-view aggregation for ULIP-2 partial views (inspired by OPEN, Chu et al. 2024)
    # Same modes as dino_view_aggregation. Only applies when ulip2_use_partial_views=True.
    ulip_view_aggregation: str = "topk_softmax"
    ulip_view_topk: int = 8
    ulip_view_temperature: float = 0.5

    # Rotation sensitivity evaluation for ULIP Top-K candidates
    ulip2_rotation_eval: bool = False
    ulip2_rotation_eval_top_k: int = 5
    ulip2_rotation_eval_method: str = "icp"  # initially only "icp"
    ulip2_rotation_eval_weight: float = 0.0  # 0.0 = debug-only, >0 = optional rerank contribution

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
    # Sub-step B2 – Geometry Re-ranking (GeDi + trimmed Chamfer)
    # -------------------------------------------------------------------------
    # GeDi: https://github.com/fabiopoiesi/gedi
    # Paper: "Learning General and Distinctive 3D Local Deep Descriptors" (Poiesi & Boscaini, 2022)
    geometry_reranking_enabled: bool = True
    geometry_reranking_signal: str = "gedi"  # "gedi" | "chamfer" | "both"
    geometry_reranking_top_k: int = 5        # Shortlist size from fusion for B2 re-ranking

    # GeDi descriptor settings
    gedi_url: str = "http://gedi:5060"          # GeDi service URL (docker-compose service name)
    gedi_repo_path: str = "/gedi"             # Path to cloned fabiopoiesi/gedi repo (inside GeDi container)
    gedi_checkpoint: str = "/gedi/data/chkpts/3dmatch/chkpt.tar"  # GeDi pretrained checkpoint
    gedi_dim: int = 32                       # Descriptor output dimension
    gedi_r_lrf: float = 0.5                  # Local reference frame radius
    gedi_samples_per_batch: int = 500        # Batch size for GPU descriptor computation
    gedi_samples_per_patch_lrf: int = 4000   # Points for LRF computation
    gedi_samples_per_patch_out: int = 512    # Points sampled for PointNet++
    gedi_num_keypoints: int = 5000           # Number of keypoints to sample per cloud

    # Trimmed Chamfer settings (B2)
    chamfer_trim_ratio: float = 0.1          # Discard top 10% of distances

    # -------------------------------------------------------------------------
    # Candidate scale gate (after fusion, before pose estimation)
    # -------------------------------------------------------------------------
    # Tries fused candidates in rank order; accepts the first whose estimated
    # scale factor falls within [scale_gate_min, scale_gate_max].
    scale_gate_enabled: bool = False
    scale_gate_min: float = 0.8
    scale_gate_max: float = 1.2
    scale_gate_min_confidence: float = 0.0
    scale_gate_max_candidates: int = 5
    scale_gate_reject_policy: str = "fallback_best"  # "fallback_best" | "fail"

    # -------------------------------------------------------------------------
    # Schritt 7 – Skalenbestimmung
    # -------------------------------------------------------------------------
    # When the ICP-based scale confidence falls below this threshold the result
    # is overridden with the rotation-invariant sorted-bbox estimate (same
    # method used by the scale gate).  The ICP transformation is still kept
    # for coarse alignment in Step 8.
    scale_icp_min_confidence: float = 0.15

    # -------------------------------------------------------------------------
    # Schritt 8 – Pose Estimation
    # -------------------------------------------------------------------------
    # MegaPose: https://github.com/megapose6d/megapose6d
    # FoundationPose: https://github.com/NVlabs/FoundationPose
    pose_method: str = "icp"  # "foundationpose" | "megapose" | "icp"
    icp_max_iterations: int = 50
    icp_threshold: float = 0.02          # Konvergenz-Schwelle (Meter)
    foundationpose_url: str = "http://foundationpose:5050"  # FoundationPose service URL (docker-compose service name)
    foundationpose_est_refine_iter: int = 5             # Register-Refinement-Iterationen
    foundationpose_debug: int = 0                       # 0=keine GUI/Debug-Ausgaben

    # -------------------------------------------------------------------------
    # Prompt-Parsing (Ollama LLM)
    # -------------------------------------------------------------------------
    # Ollama: https://ollama.com/
    # Installiert im Docker-Container; Python-Client in requirements.txt
    ollama_host: str = "http://localhost:11434"  # Ollama läuft im Container (start.sh)
    ollama_model: str = "gemma3:4b"       # Modell für Prompt-Parsing (im Dockerfile gepullt)

    # -------------------------------------------------------------------------
    # Debug-Visualisierung
    # -------------------------------------------------------------------------
    gt_bbox_center_compensation: bool = False  # Compensate GT wireframe for mesh bbox-center offset

    # -------------------------------------------------------------------------
    # Ein-/Ausgabepfade
    # -------------------------------------------------------------------------
    output_dir: str = "pipeline_output"  # Ordner für alle Pipeline-Ergebnisse

    def to_dict(self) -> dict:
        """Konvertiert die Konfiguration in ein Dictionary (z.B. für JSON-Export)."""
        return {k: str(v) if isinstance(v, torch.device) else v
                for k, v in self.__dict__.items()}
