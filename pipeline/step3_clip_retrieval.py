# =============================================================================
# pipeline/step3_clip_retrieval.py – Schritt 3: Semantische Kandidatensuche
# =============================================================================
#
# Ziel:
#   Top-K CAD-Modell-Kandidaten finden, indem das ROI-Bild (Schritt 1)
#   mit vorgenerierten Text-Beschreibungen der CAD-Modelle verglichen wird.
#
# Pipeline:
#   CLIP(ROI image embedding) vs. CLIP(text embeddings of CAD descriptions)
#   → Top-K Kandidaten (z.B. K=20)
#
# Modell:
#   • CLIP (Contrastive Language–Image Pre-training)
#     Ref: https://github.com/openai/CLIP
#     Paper: "Learning Transferable Visual Models From Natural Language
#             Supervision" (Radford et al., 2021)
#
# Adaptiert aus: OSCAR – object_retrieval/retrieval_combi_clip.py
#
# Outputs:
#   - Liste von (object_id, similarity_score) Tupeln, sortiert nach Score
# =============================================================================

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F

from .config import PipelineConfig
from .utils import load_object_descriptions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenstruktur für CLIP-Retrieval-Ergebnisse
# ---------------------------------------------------------------------------

@dataclass
class CLIPCandidate:
    """Einzelner CLIP-Kandidat.

    Attributes:
        object_id: Identifikator des CAD-Modells / Objekts.
        score: Cosine-Similarity zwischen Query und Beschreibung.
        description: Die gematchte Textbeschreibung.
    """
    object_id: str
    score: float
    description: str = ""


@dataclass
class CLIPRetrievalResult:
    """Ergebnis der CLIP-basierten Kandidatensuche (Schritt 3).

    Attributes:
        candidates: Liste der Top-K Kandidaten, sortiert nach Score.
        query_embedding: CLIP-Embedding des ROI-Bildes (für spätere Fusion).
        all_scores: Vollständiger Score-Vektor gegen alle Beschreibungen.
    """
    candidates: List[CLIPCandidate]
    query_embedding: np.ndarray
    all_scores: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# CLIP Retrieval Modul
# ---------------------------------------------------------------------------

class CLIPRetriever:
    """Semantische Objektsuche via CLIP Image-Text-Matching.

    Vergleicht das ROI-Bild mit vorgenerierten Textbeschreibungen der
    CAD-Modelle aus der Objektdatenbank (OSCAR-Prinzip).

    Der Retrieval-Prozess:
    1. ROI-Bild → CLIP Image Encoder → Image Embedding
    2. CAD-Beschreibungen → CLIP Text Encoder → Text Embeddings (vorab berechnet)
    3. Cosine-Similarity → Top-K Kandidaten

    Ref: OSCAR Pipeline – retrieval_combi_clip.py (encode_image_clip, encode_texts_clip)

    Usage:
        >>> retriever = CLIPRetriever(config)
        >>> retriever.load_descriptions("object_database/ycbv_gso/descriptions.json")
        >>> result = retriever.retrieve(roi_image, top_k=20)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self.model = None
        self.preprocess = None

        # Vorab berechnete Embeddings der Beschreibungen
        self._desc_embeddings: Optional[torch.Tensor] = None
        self._desc_texts: List[str] = []
        self._desc_labels: List[str] = []

    def _load_model(self):
        """Lädt das CLIP-Modell bei Erstverwendung.

        Ref: https://github.com/openai/CLIP
        """
        if self.model is not None:
            return

        logger.info(f"Lade CLIP-Modell: {self.config.clip_model_name}...")
        try:
            import clip
            self.model, self.preprocess = clip.load(
                self.config.clip_model_name, device=self.device
            )
            logger.info("CLIP erfolgreich geladen.")
        except ImportError:
            raise ImportError(
                "CLIP nicht installiert. Installieren mit:\n"
                "  pip install git+https://github.com/openai/CLIP.git\n"
                "Ref: https://github.com/openai/CLIP"
            )

    def load_descriptions(
        self,
        desc_file: Optional[str] = None,
        id_to_label: Optional[Dict[str, str]] = None,
    ) -> None:
        """Lädt und encodiert alle CAD-Objektbeschreibungen.

        Die Beschreibungen werden einmalig durch den CLIP Text Encoder
        geschickt und gecacht.

        Args:
            desc_file: Pfad zur Beschreibungs-JSON (OSCAR-Format).
                       Falls None, wird config.description_file verwendet.
            id_to_label: Optionales Mapping von Objekt-IDs auf Labels.
        """
        self._load_model()
        import clip

        desc_file = desc_file or self.config.description_file
        if not desc_file:
            raise ValueError("Kein description_file konfiguriert.")

        logger.info(f"Lade Beschreibungen aus: {desc_file}")
        self._desc_texts, self._desc_labels = load_object_descriptions(desc_file)

        # Optional: IDs zu menschenlesbaren Labels umwandeln
        if id_to_label:
            self._desc_labels = [
                id_to_label.get(lbl, lbl) for lbl in self._desc_labels
            ]

        logger.info(f"Encodiere {len(self._desc_texts)} Beschreibungen mit CLIP...")
        self._desc_embeddings = self._encode_texts_batch(self._desc_texts)
        logger.info("Beschreibungs-Embeddings berechnet.")

    def _encode_texts_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> torch.Tensor:
        """Encodiert eine Liste von Texten in CLIP-Embeddings.

        Args:
            texts: Liste von Beschreibungstexten.
            batch_size: Batch-Größe für die Encodierung.

        Returns:
            Normalisierter Tensor (N, D) auf dem konfigurierten Device.
        """
        import clip

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_text(tokens)
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb)

        return torch.cat(all_embeddings, dim=0)

    def encode_image(self, image) -> torch.Tensor:
        """Encodiert ein Bild in ein CLIP-Embedding.

        Args:
            image: PIL.Image (RGB).

        Returns:
            Normalisierter Tensor (1, D).
        """
        self._load_model()
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
        return F.normalize(emb, p=2, dim=1)

    def retrieve(
        self,
        roi_image,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        text_query: Optional[str] = None,
        text_query_weight: float = 0.0,
    ) -> CLIPRetrievalResult:
        """Findet die Top-K semantisch ähnlichsten CAD-Modelle.

        Args:
            roi_image:          PIL.Image des segmentierten Objekts (Schritt 1).
            top_k:              Anzahl der Kandidaten (überschreibt Config).
            threshold:          Minimale Similarity (optional, alternative zu top_k).
            text_query:         Optionaler Text-Query (z.B. "yellow mustard bottle").
                                Falls angegeben, wird Image- und Text-Ähnlichkeit gemischt:
                                ``score = (1-w)·img_sim + w·text_sim``
            text_query_weight:  Gewicht des Text-Querys (default: 0.0).

        Returns:
            CLIPRetrievalResult mit sortierten Kandidaten.
        """
        if self._desc_embeddings is None:
            raise RuntimeError(
                "Beschreibungen nicht geladen. Rufe load_descriptions() auf."
            )

        top_k = top_k or self.config.clip_top_k

        # --- Image Embedding ---
        query_emb = self.encode_image(roi_image)  # (1, D)

        # --- Cosine Similarity: Bild vs. Beschreibungen ---
        img_sims = (query_emb @ self._desc_embeddings.T).squeeze(0)  # (M,)

        # --- Optional: Text-Query mischen ---
        if text_query:
            import clip
            tokens = clip.tokenize([text_query], truncate=True).to(self.device)
            with torch.no_grad():
                txt_emb = self.model.encode_text(tokens)
            txt_emb = F.normalize(txt_emb, p=2, dim=1)  # (1, D)
            txt_sims = (txt_emb @ self._desc_embeddings.T).squeeze(0)  # (M,)
            sims = (1.0 - text_query_weight) * img_sims + text_query_weight * txt_sims
            logger.debug(
                "CLIP text_query=%r (w=%.2f) gemischt.", text_query, text_query_weight
            )
        else:
            sims = img_sims

        # --- Top-K oder Threshold-basierte Filterung ---
        if threshold is not None:
            keep_mask = sims >= threshold
            if keep_mask.sum() == 0:
                logger.warning(
                    f"Kein Kandidat über Threshold {threshold}. "
                    f"Fallback auf Top-{top_k}."
                )
                keep_indices = sims.topk(top_k).indices
            else:
                keep_indices = keep_mask.nonzero(as_tuple=True)[0]
                # Sortiere nach Score
                keep_scores = sims[keep_indices]
                sorted_order = keep_scores.argsort(descending=True)
                keep_indices = keep_indices[sorted_order][:top_k]
        else:
            keep_indices = sims.topk(min(top_k, len(sims))).indices

        # --- Kandidaten aufbauen ---
        candidates = []
        seen_objects = set()  # Deduplizierung auf Objekt-Ebene
        for idx in keep_indices.tolist():
            obj_id = self._desc_labels[idx]
            score = sims[idx].item()
            desc = self._desc_texts[idx]

            # Pro Objekt den besten Score behalten
            if obj_id not in seen_objects:
                candidates.append(CLIPCandidate(
                    object_id=obj_id,
                    score=score,
                    description=desc,
                ))
                seen_objects.add(obj_id)

        logger.info(
            f"CLIP Retrieval: {len(candidates)} Kandidaten "
            f"(Top-Score: {candidates[0].score:.4f} – {candidates[0].object_id})"
        )

        return CLIPRetrievalResult(
            candidates=candidates,
            query_embedding=query_emb.cpu().numpy(),
            all_scores=sims.cpu().numpy(),
        )

    def get_candidate_labels(self, result: CLIPRetrievalResult) -> List[str]:
        """Extrahiert die Objekt-IDs aus einem CLIP-Retrieval-Ergebnis."""
        return [c.object_id for c in result.candidates]
