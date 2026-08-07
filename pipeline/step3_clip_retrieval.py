# =============================================================================
# pipeline/step3_clip_retrieval.py – Thesis Step B1: Semantic Channel S_text
# =============================================================================
#
# Computes the text-based semantic score S_text (thesis Sec. 3.3, Step B1).
#
# CLIP (Radford et al., 2021) performs image–text alignment via contrastive
# pretraining. OSCAR (Pulli et al., 2025) establishes CLIP as competitive
# for caption-based CAD retrieval in the training-free setting.
#
# The text channel uses offline-generated natural-language descriptions of
# each CAD model. At query time, the ROI image embedding is compared against
# all description embeddings via cosine similarity.
#
# In the thesis default (full-database scoring), all candidates are scored.
# The OSCAR-style cascade (CLIP top-k → DINOv2/ULIP) is retained as
# ablation O2 (Pulli et al., 2025).
#
# Model:
#   • CLIP ViT-B/32 (Radford et al., 2021)
#     Ref: https://github.com/openai/CLIP
#
# Adapted from: OSCAR – object_retrieval/retrieval_combi_clip.py
#
# Outputs:
#   - Liste von (object_id, similarity_score) Tupeln, sortiert nach Score
# =============================================================================

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F

from .config import PipelineConfig

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

    # Foreign-view descriptions: 3-digit zero-padded view names (``_002.png``)
    # are stray renders, not one of the 42 real views (``_0.png``..``_41.png``).
    # The image loader quarantines them (see retrieval_mi3dor_eval_oscarplus.
    # _quarantine_foreign_views, SAME regex); the descriptions JSON was built
    # before that and still carries them, giving ~1817 MI3DOR objects a 43rd
    # (foreign) description that biases CLIP scoring. Drop them so CLIP scores
    # exactly the 42 real views per object, consistent with DINO/ULIP.
    _FOREIGN_VIEW_RE = re.compile(r"_[0-9]{3,}\.png$")

    @staticmethod
    def _load_object_descriptions(desc_file: str) -> Tuple[List[str], List[str]]:
        """Lädt Objektbeschreibungen aus einer OSCAR-kompatiblen JSON-Datei.

        Format: {object_id: {"image_descriptions": {"view_name": "text", ...}}}

        Foreign-view descriptions (3-digit ``_NNN.png`` view keys) are skipped
        so CLIP scores only the 42 real views, matching the image quarantine.

        Args:
            desc_file: Pfad zur JSON-Datei.

        Returns:
            (texts, labels) – Liste aller Beschreibungstexte und zugehörige Label-IDs.
        """
        with open(desc_file, "r") as f:
            descriptions = json.load(f)

        texts: List[str] = []
        labels: List[str] = []
        n_foreign = 0
        for obj_id, entry in descriptions.items():
            for view_name, text in entry.get("image_descriptions", {}).items():
                if CLIPRetriever._FOREIGN_VIEW_RE.search(str(view_name)):
                    n_foreign += 1
                    continue
                texts.append(text)
                labels.append(obj_id)
        if n_foreign:
            logger.info(
                "CLIP descriptions: skipped %d foreign-view (_NNN.png) entries; "
                "keeping only the 42 real views per object.", n_foreign)
        return texts, labels

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

        desc_file = desc_file or self.config.description_file
        if not desc_file:
            raise ValueError("Kein description_file konfiguriert.")

        logger.info(f"Lade Beschreibungen aus: {desc_file}")
        self._desc_texts, self._desc_labels = self._load_object_descriptions(desc_file)

        # Optional: IDs zu menschenlesbaren Labels umwandeln
        if id_to_label:
            self._desc_labels = [
                id_to_label.get(lbl, lbl) for lbl in self._desc_labels
            ]

        cache_path = self._cache_path(desc_file)
        if self._try_load_cache(cache_path):
            return

        logger.info(f"Encodiere {len(self._desc_texts)} Beschreibungen mit CLIP...")
        self._desc_embeddings = self._encode_texts_batch(self._desc_texts)
        logger.info("Beschreibungs-Embeddings berechnet.")
        self._save_cache(cache_path)

    def _cache_path(self, desc_file: str) -> str:
        """Cache-Pfad für die Text-Embeddings, neben der Beschreibungsdatei.

        Fingerprint = CLIP-Modellname + Beschreibungstexte (Inhalt, nicht
        Pfad/mtime) → cross-machine-stabil, wie die DINO/ULIP-Caches.
        Labels sind absichtlich NICHT Teil des Fingerprints: ein anderes
        id_to_label-Mapping ändert nicht die encodierten Texte, der Cache
        bleibt also gültig.
        """
        model_tag = self.config.clip_model_name.replace("/", "_")
        raw = f"v1:{len(self._desc_texts)}\n" + "\n".join(self._desc_texts)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        cache_dir = os.path.dirname(os.path.abspath(desc_file))
        return os.path.join(cache_dir, f".clip_text_cache_{model_tag}_{digest}.pt")

    def _try_load_cache(self, cache_path: str) -> bool:
        """Lädt gecachte Text-Embeddings, falls vorhanden."""
        if not os.path.isfile(cache_path):
            return False
        try:
            data = torch.load(cache_path, map_location=self.device, weights_only=True)
            self._desc_embeddings = data["embeddings"].to(self.device)
            logger.info("CLIP-Text-Cache geladen: %s", os.path.basename(cache_path))
            return True
        except Exception as e:
            logger.warning(
                "CLIP-Text-Cache konnte nicht geladen werden (%s), encodiere neu.", e
            )
            return False

    def _save_cache(self, cache_path: str) -> None:
        """Speichert die Text-Embeddings auf Platte."""
        try:
            torch.save({"embeddings": self._desc_embeddings.cpu()}, cache_path)
            logger.info("CLIP-Text-Embeddings gespeichert: %s", cache_path)
        except OSError as e:
            logger.warning("Konnte CLIP-Text-Cache nicht speichern: %s", e)

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
        # keep_indices is score-descending in both branches, so the first row
        # seen for each object is its best (max) view — object score = max over
        # its description rows. Gather all kept scores to CPU in ONE transfer to
        # avoid a per-row .item() GPU sync (161k syncs/query on MI3DOR ≈ +1s);
        # the result is bit-identical, just ~1s/query faster.
        keep_indices_list = keep_indices.tolist()
        keep_scores_list = sims[keep_indices].cpu().tolist()
        candidates = []
        seen_objects = set()  # Deduplizierung auf Objekt-Ebene
        for pos, idx in enumerate(keep_indices_list):
            obj_id = self._desc_labels[idx]
            # Pro Objekt den besten Score behalten (erste = höchste)
            if obj_id not in seen_objects:
                candidates.append(CLIPCandidate(
                    object_id=obj_id,
                    score=keep_scores_list[pos],
                    description=self._desc_texts[idx],
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
