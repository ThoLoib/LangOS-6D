# =============================================================================
# pipeline/step4_dino_reranking.py – Schritt 4: Bildbasiertes Re-Ranking
# =============================================================================
#
# Ziel:
#   Die CLIP-Kandidaten (Schritt 3) werden anhand visueller Ähnlichkeit
#   neu gerankt: ROI-Bild vs. vorgerenderte Ansichten der CAD-Modelle.
#
# Pipeline:
#   Für jeden Kandidaten:
#     Renderings des CAD-Modells → DINOv2 Features
#     ROI-Bild → DINOv2 Features
#     → Cosine Similarity → Re-Ranking
#
# Modell:
#   • DINOv2 – Self-supervised Vision Transformer (Meta)
#     Ref: https://github.com/facebookresearch/dinov2
#     Paper: "DINOv2: Learning Robust Visual Features without Supervision"
#             (Oquab et al., 2023)
#
# Adaptiert aus: OSCAR – object_retrieval/retrieval_combi_clip.py
#                 (encode_image_dino, load_ref_dino_embeddings)
#
# Outputs:
#   - Verfeinerte Top-K Kandidaten (z.B. K=5)
# =============================================================================

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import PipelineConfig
from .step3_clip_retrieval import CLIPRetrievalResult, CLIPCandidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenstruktur für DINOv2-Re-Ranking-Ergebnisse
# ---------------------------------------------------------------------------

@dataclass
class DINOCandidate:
    """Einzelner DINOv2-Kandidat nach Re-Ranking.

    Attributes:
        object_id: Identifikator des CAD-Modells.
        dino_score: DINOv2 Cosine-Similarity.
        clip_score: Ursprünglicher CLIP-Score (für spätere Fusion).
        best_view_path: Pfad zum ähnlichsten Rendering.
    """
    object_id: str
    dino_score: float
    clip_score: float
    best_view_path: str = ""


@dataclass
class DINOReRankingResult:
    """Ergebnis des DINOv2-basierten Re-Rankings (Schritt 4).

    Attributes:
        candidates: Liste der Top-K verfeinerten Kandidaten.
        query_embedding: DINOv2-Embedding des ROI-Bildes.
    """
    candidates: List[DINOCandidate]
    query_embedding: np.ndarray


# ---------------------------------------------------------------------------
# DINOv2 Re-Ranking Modul
# ---------------------------------------------------------------------------

class DINOReRanker:
    """Re-Rankt CLIP-Kandidaten anhand visueller DINOv2-Ähnlichkeit.

    Vergleicht das ROI-Bild mit vorgerenderten Ansichten der CAD-Modelle
    (8+ Ansichten pro Modell, erzeugt via rendering/rendering.py).

    Dies ist das Kernprinzip von OSCAR:
    1. CLIP filtert semantisch unpassende Modelle heraus.
    2. DINOv2 vergleicht die tatsächliche visuelle Erscheinung.

    Ref: OSCAR – retrieval_combi_clip.py (Stage 2: DINO seg_crop → ref imgs)

    Usage:
        >>> reranker = DINOReRanker(config)
        >>> reranker.load_reference_images("object_images/ycbv_gso/")
        >>> result = reranker.rerank(roi_image, clip_result, top_k=5)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self.processor = None
        self.model = None

        # Gecachte Referenz-Embeddings: {object_id: [(embedding, path), ...]}
        self._ref_embeddings: Dict[str, List[Tuple[torch.Tensor, str]]] = {}
        # Flache Liste für schnellen Batch-Vergleich
        self._all_ref_embs: Optional[torch.Tensor] = None
        self._all_ref_keys: List[Tuple[str, str]] = []  # (object_id, path)

    def _load_model(self):
        """Lädt DINOv2 bei Erstverwendung.

        Ref: https://github.com/facebookresearch/dinov2
        """
        if self.model is not None:
            return

        logger.info(f"Lade DINOv2-Modell: {self.config.dino_model_name}...")
        try:
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(
                self.config.dino_model_name
            )
            self.model = AutoModel.from_pretrained(
                self.config.dino_model_name
            ).to(self.device)
            self.model.eval()
            logger.info("DINOv2 erfolgreich geladen.")
        except ImportError:
            raise ImportError(
                "transformers nicht installiert. Installieren mit:\n"
                "  pip install transformers\n"
                "Ref: https://huggingface.co/facebook/dinov2-base"
            )

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encodiert ein Bild in ein DINOv2-Embedding.

        Verwendet Average Pooling über die Patch-Tokens des ViT.

        Args:
            image: PIL.Image (RGB).

        Returns:
            Normalisierter Tensor (1, D).
        """
        self._load_model()
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # Average Pooling über alle Patch-Tokens
            features = outputs.last_hidden_state.mean(dim=1)
            features = F.normalize(features, p=2, dim=1)
        return features

    def load_reference_images(self, ref_dir: Optional[str] = None) -> None:
        """Lädt und encodiert vorgerenderte Referenzbilder aller CAD-Modelle.

        Erwartet die OSCAR-Ordnerstruktur:
            ref_dir/
                object_label_1/
                    view_001.png
                    view_002.png
                    ...
                object_label_2/
                    ...

        Adaptiert aus: OSCAR – retrieval_combi_clip.py:load_ref_dino_embeddings()

        Args:
            ref_dir: Pfad zum Referenzbilder-Ordner.
                     Falls None, wird config.reference_images_dir verwendet.
        """
        self._load_model()

        ref_dir = ref_dir or self.config.reference_images_dir
        if not ref_dir:
            raise ValueError("Kein reference_images_dir konfiguriert.")

        logger.info(f"Lade Referenzbilder aus: {ref_dir}")
        embs_list = []
        keys_list = []

        for label in sorted(os.listdir(ref_dir)):
            label_dir = os.path.join(ref_dir, label)
            if not os.path.isdir(label_dir):
                continue

            object_embs = []
            for fname in os.listdir(label_dir):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                img_path = os.path.join(label_dir, fname)
                try:
                    img = Image.open(img_path).convert("RGB")
                except (OSError, IOError) as e:
                    logger.warning(f"Fehler beim Laden von {img_path}: {e}")
                    continue

                emb = self.encode_image(img).squeeze(0)  # (D,)
                object_embs.append((emb, img_path))
                embs_list.append(emb)
                keys_list.append((label, img_path))

            if object_embs:
                self._ref_embeddings[label] = object_embs

        if embs_list:
            self._all_ref_embs = torch.stack(embs_list).to(self.device)
            self._all_ref_keys = keys_list

        logger.info(
            f"Referenz-Embeddings geladen: {len(self._ref_embeddings)} Objekte, "
            f"{len(embs_list)} Ansichten total."
        )

    def rerank(
        self,
        roi_image: Image.Image,
        clip_result: CLIPRetrievalResult,
        top_k: Optional[int] = None,
    ) -> DINOReRankingResult:
        """Re-Rankt CLIP-Kandidaten anhand visueller DINOv2-Ähnlichkeit.

        Nur die von CLIP vorselektierten Kandidaten werden verglichen,
        was die Berechnung erheblich beschleunigt.

        Args:
            roi_image: ROI-Bild des segmentierten Objekts (Schritt 1).
            clip_result: Ergebnis der CLIP-Suche (Schritt 3).
            top_k: Anzahl der finalen Kandidaten (überschreibt Config).

        Returns:
            DINOReRankingResult mit verfeinerten Kandidaten.
        """
        if not self._ref_embeddings:
            raise RuntimeError(
                "Referenzbilder nicht geladen. Rufe load_reference_images() auf."
            )

        top_k = top_k or self.config.dino_top_k

        # --- ROI DINOv2 Embedding ---
        query_emb = self.encode_image(roi_image)  # (1, D)

        # --- Nur Referenzbilder der CLIP-Kandidaten vergleichen ---
        clip_candidates = clip_result.candidates
        clip_score_map = {c.object_id: c.score for c in clip_candidates}

        candidate_embs = []
        candidate_keys = []
        for candidate in clip_candidates:
            obj_id = candidate.object_id
            if obj_id not in self._ref_embeddings:
                logger.debug(f"Keine Referenzbilder für {obj_id}, überspringe.")
                continue
            for emb, path in self._ref_embeddings[obj_id]:
                candidate_embs.append(emb)
                candidate_keys.append((obj_id, path))

        if not candidate_embs:
            logger.warning("Keine Referenzbilder für die CLIP-Kandidaten gefunden.")
            # Fallback: CLIP-Reihenfolge beibehalten
            return DINOReRankingResult(
                candidates=[
                    DINOCandidate(
                        object_id=c.object_id,
                        dino_score=0.0,
                        clip_score=c.score,
                    )
                    for c in clip_candidates[:top_k]
                ],
                query_embedding=query_emb.cpu().numpy(),
            )

        # --- Cosine Similarity berechnen ---
        cand_tensor = torch.stack(candidate_embs).to(self.device)  # (K, D)
        sims = (query_emb @ cand_tensor.T).squeeze(0)  # (K,)

        # --- Pro Objekt den besten DINOv2-Score finden ---
        best_per_object: Dict[str, Tuple[float, str]] = {}
        for idx, (obj_id, path) in enumerate(candidate_keys):
            score = sims[idx].item()
            if obj_id not in best_per_object or score > best_per_object[obj_id][0]:
                best_per_object[obj_id] = (score, path)

        # --- Sortieren nach DINOv2-Score ---
        sorted_objects = sorted(
            best_per_object.items(), key=lambda x: x[1][0], reverse=True
        )

        candidates = []
        for obj_id, (dino_score, best_path) in sorted_objects[:top_k]:
            candidates.append(DINOCandidate(
                object_id=obj_id,
                dino_score=dino_score,
                clip_score=clip_score_map.get(obj_id, 0.0),
                best_view_path=best_path,
            ))

        logger.info(
            f"DINOv2 Re-Ranking: {len(candidates)} Kandidaten "
            f"(Top: {candidates[0].object_id}, DINO={candidates[0].dino_score:.4f})"
        )

        return DINOReRankingResult(
            candidates=candidates,
            query_embedding=query_emb.cpu().numpy(),
        )
