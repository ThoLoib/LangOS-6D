# =============================================================================
# pipeline/step4_dino_reranking.py – Schritt 4: Bildbasiertes Re-Ranking
# =============================================================================
#
# Ziel:
#   Die CLIP-Kandidaten (Schritt 3) werden anhand visueller Aehnlichkeit
#   neu gerankt: ROI-Bild vs. vorgerenderte Ansichten der CAD-Modelle.
#
# Pipeline:
#   Fuer jeden Kandidaten:
#     Renderings des CAD-Modells -> DINOv2 Features
#     ROI-Bild -> DINOv2 Features
#     -> Cosine Similarity -> Re-Ranking
#
# Modell:
#   DINOv2 – Self-supervised Vision Transformer (Meta)
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

import hashlib
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
# Datenstruktur fuer DINOv2-Re-Ranking-Ergebnisse
# ---------------------------------------------------------------------------

@dataclass
class DINOCandidate:
    """Einzelner DINOv2-Kandidat nach Re-Ranking.

    Attributes:
        object_id: Identifikator des CAD-Modells.
        dino_score: DINOv2 Cosine-Similarity.
        clip_score: Urspruenglicher CLIP-Score (fuer spaetere Fusion).
        best_view_path: Pfad zum aehnlichsten Rendering.
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
    """Re-Rankt CLIP-Kandidaten anhand visueller DINOv2-Aehnlichkeit.

    Vergleicht das ROI-Bild mit vorgerenderten Ansichten der CAD-Modelle
    (8+ Ansichten pro Modell, erzeugt via rendering/rendering.py).

    Dies ist das Kernprinzip von OSCAR:
    1. CLIP filtert semantisch unpassende Modelle heraus.
    2. DINOv2 vergleicht die tatsaechliche visuelle Erscheinung.

    Ref: OSCAR – retrieval_combi_clip.py (Stage 2: DINO seg_crop -> ref imgs)

    Usage:
        >>> reranker = DINOReRanker(config)
        >>> reranker.load_reference_images("object_images/ycbv_gso/")
        >>> result = reranker.rerank(roi_image, clip_result, top_k=5)
    """

    CACHE_VERSION = 1  # Bump when embedding logic changes
    BATCH_SIZE = 32    # Images per DINOv2 forward pass

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self.processor = None
        self.model = None

        # Gecachte Referenz-Embeddings: {object_id: [(embedding, path), ...]}
        self._ref_embeddings: Dict[str, List[Tuple[torch.Tensor, str]]] = {}
        # Flache Liste fuer schnellen Batch-Vergleich
        self._all_ref_embs: Optional[torch.Tensor] = None
        self._all_ref_keys: List[Tuple[str, str]] = []  # (object_id, path)

    def _load_model(self):
        """Laedt DINOv2 bei Erstverwendung.

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

        Verwendet Average Pooling ueber die Patch-Tokens des ViT.

        Args:
            image: PIL.Image (RGB).

        Returns:
            Normalisierter Tensor (1, D).
        """
        self._load_model()
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # Average Pooling ueber alle Patch-Tokens
            features = outputs.last_hidden_state.mean(dim=1)
            features = F.normalize(features, p=2, dim=1)
        return features

    def _encode_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Encodiert einen Batch von Bildern in DINOv2-Embeddings.

        Args:
            images: Liste von PIL.Image (RGB).

        Returns:
            Normalisierter Tensor (N, D).
        """
        self._load_model()
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
            features = F.normalize(features, p=2, dim=1)
        return features

    # -------------------------------------------------------------------
    # Cache-Logik
    # -------------------------------------------------------------------

    @staticmethod
    def _dir_fingerprint(ref_dir: str) -> str:
        """Erzeugt einen Fingerprint aus Dateianzahl + neuester mtime.

        Aendert sich, wenn Bilder hinzugefuegt/entfernt/aktualisiert werden.
        """
        total_files = 0
        latest_mtime = 0.0
        for root, _dirs, files in os.walk(ref_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    total_files += 1
                    mt = os.path.getmtime(os.path.join(root, f))
                    if mt > latest_mtime:
                        latest_mtime = mt
        raw = f"v{DINOReRanker.CACHE_VERSION}:{total_files}:{latest_mtime:.6f}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_path(self, ref_dir: str) -> str:
        """Pfad zur Cache-Datei im ref_dir."""
        fp = self._dir_fingerprint(ref_dir)
        model_tag = self.config.dino_model_name.replace("/", "_")
        return os.path.join(ref_dir, f".dino_cache_{model_tag}_{fp}.pt")

    def _try_load_cache(self, ref_dir: str) -> bool:
        """Versucht, gecachte Embeddings zu laden.

        Returns:
            True wenn Cache geladen, False sonst.
        """
        cache_file = self._cache_path(ref_dir)
        if not os.path.isfile(cache_file):
            return False
        try:
            data = torch.load(cache_file, map_location=self.device, weights_only=True)
            self._all_ref_embs = data["embeddings"].to(self.device)
            self._all_ref_keys = data["keys"]
            # Rebuild per-object dict
            self._ref_embeddings.clear()
            for i, (obj_id, path) in enumerate(self._all_ref_keys):
                emb = self._all_ref_embs[i]
                self._ref_embeddings.setdefault(obj_id, []).append((emb, path))
            logger.info(
                "DINOv2-Embeddings aus Cache geladen: %d Objekte, %d Ansichten (%s)",
                len(self._ref_embeddings), len(self._all_ref_keys),
                os.path.basename(cache_file),
            )
            return True
        except Exception as e:
            logger.warning("Cache-Ladeversuch fehlgeschlagen: %s", e)
            return False

    def _save_cache(self, ref_dir: str) -> None:
        """Speichert berechnete Embeddings als .pt Cache."""
        cache_file = self._cache_path(ref_dir)
        try:
            torch.save(
                {
                    "embeddings": self._all_ref_embs.cpu(),
                    "keys": self._all_ref_keys,
                },
                cache_file,
            )
            logger.info("DINOv2-Embeddings gespeichert: %s", cache_file)
        except Exception as e:
            logger.warning("Cache-Speichern fehlgeschlagen: %s", e)

    # -------------------------------------------------------------------
    # Referenzbilder laden (mit Batch + Cache)
    # -------------------------------------------------------------------

    def load_reference_images(self, ref_dir: Optional[str] = None) -> None:
        """Laedt und encodiert vorgerenderte Referenzbilder aller CAD-Modelle.

        Erwartet die OSCAR-Ordnerstruktur:
            ref_dir/
                object_label_1/
                    view_001.png
                    view_002.png
                    ...
                object_label_2/
                    ...

        Optimierungen gegenueber dem Original:
        - Batched DINOv2 forward passes (BATCH_SIZE Bilder gleichzeitig)
        - Disk-Cache: Embeddings werden als .pt gespeichert und bei
          erneutem Aufruf sofort geladen (Fingerprint-basiert).

        Adaptiert aus: OSCAR – retrieval_combi_clip.py:load_ref_dino_embeddings()

        Args:
            ref_dir: Pfad zum Referenzbilder-Ordner.
                     Falls None, wird config.reference_images_dir verwendet.
        """
        self._load_model()

        ref_dir = ref_dir or self.config.reference_images_dir
        if not ref_dir:
            raise ValueError("Kein reference_images_dir konfiguriert.")

        # --- Schnellpfad: Cache laden ---
        if self._try_load_cache(ref_dir):
            return

        # --- Kein Cache -> Bilder batchweise encodieren ---
        logger.info(f"Lade Referenzbilder aus: {ref_dir} (kein Cache, berechne Embeddings...)")

        # Schritt 1: Alle Bildpfade sammeln
        all_paths: List[str] = []
        all_labels: List[str] = []
        for label in sorted(os.listdir(ref_dir)):
            label_dir = os.path.join(ref_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                all_paths.append(os.path.join(label_dir, fname))
                all_labels.append(label)

        if not all_paths:
            logger.warning("Keine Referenzbilder gefunden in %s", ref_dir)
            return

        total = len(all_paths)
        logger.info("  %d Referenzbilder gefunden, encodiere in Batches von %d...",
                     total, self.BATCH_SIZE)

        # Schritt 2: Batch-Encoding
        embs_list: List[torch.Tensor] = []
        keys_list: List[Tuple[str, str]] = []

        batch_imgs: List[Image.Image] = []
        batch_keys: List[Tuple[str, str]] = []

        for i, (img_path, label) in enumerate(zip(all_paths, all_labels)):
            try:
                img = Image.open(img_path).convert("RGB")
            except (OSError, IOError) as e:
                logger.warning(f"Fehler beim Laden von {img_path}: {e}")
                continue

            batch_imgs.append(img)
            batch_keys.append((label, img_path))

            if len(batch_imgs) >= self.BATCH_SIZE:
                batch_emb = self._encode_batch(batch_imgs)  # (B, D)
                embs_list.append(batch_emb.cpu())
                keys_list.extend(batch_keys)
                n_done = len(keys_list)
                if n_done % (self.BATCH_SIZE * 10) == 0 or n_done == total:
                    logger.info("  ... %d / %d encodiert (%.0f%%)",
                                n_done, total, 100.0 * n_done / total)
                batch_imgs.clear()
                batch_keys.clear()

        # Letzter partieller Batch
        if batch_imgs:
            batch_emb = self._encode_batch(batch_imgs)
            embs_list.append(batch_emb.cpu())
            keys_list.extend(batch_keys)

        if not embs_list:
            logger.warning("Keine Embeddings berechnet.")
            return

        # Schritt 3: Zusammenfuegen
        all_embs = torch.cat(embs_list, dim=0)  # (N, D)
        self._all_ref_embs = all_embs.to(self.device)
        self._all_ref_keys = keys_list

        # Per-Object Dict aufbauen
        self._ref_embeddings.clear()
        for i, (obj_id, path) in enumerate(keys_list):
            emb = self._all_ref_embs[i]
            self._ref_embeddings.setdefault(obj_id, []).append((emb, path))

        logger.info(
            "Referenz-Embeddings berechnet: %d Objekte, %d Ansichten total.",
            len(self._ref_embeddings), len(keys_list),
        )

        # Schritt 4: Cache speichern
        self._save_cache(ref_dir)

    def rerank(
        self,
        roi_image: Image.Image,
        clip_result: CLIPRetrievalResult,
        top_k: Optional[int] = None,
    ) -> DINOReRankingResult:
        """Re-Rankt CLIP-Kandidaten anhand visueller DINOv2-Aehnlichkeit.

        Nur die von CLIP vorselektierten Kandidaten werden verglichen,
        was die Berechnung erheblich beschleunigt.

        Args:
            roi_image: ROI-Bild des segmentierten Objekts (Schritt 1).
            clip_result: Ergebnis der CLIP-Suche (Schritt 3).
            top_k: Anzahl der finalen Kandidaten (ueberschreibt Config).

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
                logger.debug(f"Keine Referenzbilder fuer {obj_id}, ueberspringe.")
                continue
            for emb, path in self._ref_embeddings[obj_id]:
                candidate_embs.append(emb)
                candidate_keys.append((obj_id, path))

        if not candidate_embs:
            logger.warning("Keine Referenzbilder fuer die CLIP-Kandidaten gefunden.")
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