# =============================================================================
# pipeline/step6_fusion.py – Schritt 6: Score-Fusion / Konsens
# =============================================================================
#
# Ziel:
#   Die Ergebnisse aus CLIP (Schritt 3), DINOv2 (Schritt 4) und ULIP-2
#   (Schritt 5) zu einem finalen Ranking kombinieren.
#
# Methoden:
#   1. Gewichtete Summe:
#      score = w_clip * CLIP + w_dino * DINO + w_ulip * ULIP
#
#   2. Intersection / Konsens:
#      Erstes Modell, das in beiden Top-K vorkommt.
#
#   3. Reciprocal Rank Fusion (RRF):
#      Standardmethode aus dem Information Retrieval.
#      Ref: "Reciprocal Rank Fusion outperforms Condorcet and Individual
#            Rank Learning Methods" (Cormack et al., 2009)
#
# Inputs:
#   - CLIPRetrievalResult (Schritt 3)
#   - DINOReRankingResult (Schritt 4)
#   - ShapeMatchingResult (Schritt 5)
#
# Outputs:
#   - Finale Liste von Kandidaten mit fusionierten Scores
# =============================================================================

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np

from .config import PipelineConfig
from .step3_clip_retrieval import CLIPRetrievalResult
from .step4_dino_reranking import DINOReRankingResult
from .step5_shape_matching import ShapeMatchingResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenstruktur für Fusions-Ergebnisse
# ---------------------------------------------------------------------------

@dataclass
class FusedCandidate:
    """Kandidat nach Score-Fusion.

    Attributes:
        object_id: Identifikator des CAD-Modells.
        fused_score: Kombinierter Score.
        clip_score: Beitrag von CLIP (normalisiert).
        dino_score: Beitrag von DINOv2 (normalisiert).
        ulip_score: Beitrag von ULIP-2 (normalisiert).
        cad_model_path: Pfad zum CAD-Modell.
    """
    object_id: str
    fused_score: float
    clip_score: float = 0.0
    dino_score: float = 0.0
    ulip_score: float = 0.0
    cad_model_path: str = ""
    best_view_path: str = ""  # bestes Referenzbild (aus DINOv2)


@dataclass
class FusionResult:
    """Ergebnis der Score-Fusion (Schritt 6).

    Attributes:
        candidates: Finale sortierte Kandidatenliste.
        method: Verwendete Fusionsmethode.
        best_match: Der beste Kandidat (# 1).
    """
    candidates: List[FusedCandidate]
    method: str
    best_match: Optional[FusedCandidate] = None


# ---------------------------------------------------------------------------
# Fusion Modul
# ---------------------------------------------------------------------------

class ScoreFusion:
    """Kombiniert Scores aus verschiedenen Retrieval-Modalitäten.

    Unterstützt drei Fusionsmethoden:
    1. weighted_sum: Gewichtete Linearkombination der Scores.
    2. intersection: Erste Übereinstimmung in den Top-K-Listen.
    3. rank_fusion: Reciprocal Rank Fusion (RRF).

    Usage:
        >>> fusion = ScoreFusion(config)
        >>> result = fusion.fuse(clip_result, dino_result, shape_result)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def fuse(
        self,
        clip_result: Optional[CLIPRetrievalResult] = None,
        dino_result: Optional[DINOReRankingResult] = None,
        shape_result: Optional[ShapeMatchingResult] = None,
        method: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> FusionResult:
        """Führt die Score-Fusion durch.

        Mindestens eines der Ergebnisse muss vorhanden sein.
        Fehlende Modalitäten werden mit Score 0 behandelt.

        Args:
            clip_result: CLIP-Ergebnis (Schritt 3).
            dino_result: DINOv2-Ergebnis (Schritt 4).
            shape_result: ULIP-2-Ergebnis (Schritt 5).
            method: Fusionsmethode (überschreibt Config).
            top_k: Anzahl finaler Kandidaten (überschreibt Config).

        Returns:
            FusionResult mit dem finalen Ranking.
        """
        method = method or self.config.fusion_method
        top_k = top_k or self.config.fusion_top_k

        logger.info(f"Score-Fusion mit Methode: {method}")

        if method == "weighted_sum":
            return self._weighted_sum(clip_result, dino_result, shape_result, top_k)
        elif method == "intersection":
            return self._intersection(clip_result, dino_result, shape_result, top_k)
        elif method == "rank_fusion":
            return self._reciprocal_rank_fusion(
                clip_result, dino_result, shape_result, top_k
            )
        elif method == "majority_voting":
            return self._majority_voting(clip_result, dino_result, shape_result, top_k)
        else:
            raise ValueError(f"Unbekannte Fusionsmethode: {method}")

    # -----------------------------------------------------------------------
    # Methode 1: Gewichtete Summe
    # -----------------------------------------------------------------------

    def _weighted_sum(
        self,
        clip_result: Optional[CLIPRetrievalResult],
        dino_result: Optional[DINOReRankingResult],
        shape_result: Optional[ShapeMatchingResult],
        top_k: int,
    ) -> FusionResult:
        """Gewichtete Linearkombination der normalisierten Scores.

        score(obj) = w_clip * clip_score(obj)
                   + w_dino * dino_score(obj)
                   + w_ulip * ulip_score(obj)

        Scores werden pro Modalität auf [0, 1] normalisiert (Min-Max).
        """
        w_clip = self.config.weight_clip
        w_dino = self.config.weight_dino
        w_ulip = self.config.weight_ulip

        # --- Alle Scores sammeln ---
        scores: Dict[str, Dict[str, float]] = {}  # obj_id → {clip, dino, ulip}
        paths: Dict[str, str] = {}
        view_paths: Dict[str, str] = {}  # obj_id → bestes Referenzbild

        if clip_result:
            for c in clip_result.candidates:
                entry = scores.setdefault(c.object_id, {"clip": 0, "dino": 0, "ulip": 0})
                entry["clip"] = max(entry["clip"], c.score)

        if dino_result:
            for c in dino_result.candidates:
                entry = scores.setdefault(c.object_id, {"clip": 0, "dino": 0, "ulip": 0})
                entry["dino"] = max(entry["dino"], c.dino_score)
                entry["clip"] = max(entry.get("clip", 0), c.clip_score)
                if c.best_view_path:
                    view_paths[c.object_id] = c.best_view_path

        if shape_result:
            for c in shape_result.candidates:
                entry = scores.setdefault(c.object_id, {"clip": 0, "dino": 0, "ulip": 0})
                s = c.shape_score
                if not (isinstance(s, float) and np.isnan(s)):
                    entry["ulip"] = max(entry["ulip"], s)
                if c.cad_model_path:
                    paths[c.object_id] = c.cad_model_path  # echter OBJ-Pfad

        if not scores:
            logger.warning("Keine Kandidaten für die Fusion verfügbar.")
            return FusionResult(candidates=[], method="weighted_sum")

        # --- Normalisierung (Min-Max pro Modalität) ---
        all_clip = [s["clip"] for s in scores.values()]
        all_dino = [s["dino"] for s in scores.values()]
        all_ulip = [s["ulip"] for s in scores.values()]

        def _minmax(values):
            clean = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
            if not clean:
                return [0.0] * len(values)
            vmin, vmax = min(clean), max(clean)
            rng = vmax - vmin
            return [
                (v - vmin) / rng if rng > 0 and not (isinstance(v, float) and np.isnan(v))
                else 0.0
                for v in values
            ]

        norm_clip = _minmax(all_clip)
        norm_dino = _minmax(all_dino)
        norm_ulip = _minmax(all_ulip)

        # --- Fusion ---
        candidates = []
        for i, (obj_id, _) in enumerate(scores.items()):
            fused = (
                w_clip * norm_clip[i]
                + w_dino * norm_dino[i]
                + w_ulip * norm_ulip[i]
            )
            candidates.append(FusedCandidate(
                object_id=obj_id,
                fused_score=fused,
                clip_score=norm_clip[i],
                dino_score=norm_dino[i],
                ulip_score=norm_ulip[i],
                cad_model_path=paths.get(obj_id, ""),
                best_view_path=view_paths.get(obj_id, ""),
            ))

        # Sortieren nach fusioniertem Score (absteigend)
        candidates.sort(key=lambda x: x.fused_score, reverse=True)
        candidates = candidates[:top_k]

        logger.info(
            f"Weighted Sum Fusion: Top-{len(candidates)} "
            f"(Bester: {candidates[0].object_id}, Score={candidates[0].fused_score:.4f})"
        )

        return FusionResult(
            candidates=candidates,
            method="weighted_sum",
            best_match=candidates[0] if candidates else None,
        )

    # -----------------------------------------------------------------------
    # Methode 2: Intersection / Konsens
    # -----------------------------------------------------------------------

    def _intersection(
        self,
        clip_result: Optional[CLIPRetrievalResult],
        dino_result: Optional[DINOReRankingResult],
        shape_result: Optional[ShapeMatchingResult],
        top_k: int,
    ) -> FusionResult:
        """Findet Objekte, die in mehreren Top-K-Listen vorkommen.

        Priorität: Objekte die in allen drei Listen sind > in zweien > in einer.
        Innerhalb jeder Gruppe wird nach der Summe der Ränge sortiert.
        """
        # Rang-Listen aufbauen (1-basiert)
        rank_lists: Dict[str, Dict[str, int]] = {}

        if clip_result:
            for rank, c in enumerate(clip_result.candidates, 1):
                rank_lists.setdefault(c.object_id, {})["clip"] = rank

        if dino_result:
            for rank, c in enumerate(dino_result.candidates, 1):
                rank_lists.setdefault(c.object_id, {})["dino"] = rank

        if shape_result:
            for rank, c in enumerate(shape_result.candidates, 1):
                rank_lists.setdefault(c.object_id, {})["ulip"] = rank

        # Score = Anzahl der Listen, in denen es vorkommt (primär)
        #       + inverse Rangsumme (sekundär)
        scored = []
        for obj_id, ranks in rank_lists.items():
            num_lists = len(ranks)
            rank_sum = sum(ranks.values())
            # Hohe Verbreitung (num_lists) ist wichtiger als niedrige Rangsumme
            intersection_score = num_lists * 1000.0 - rank_sum
            scored.append((obj_id, intersection_score, ranks))

        scored.sort(key=lambda x: x[1], reverse=True)

        candidates = []
        for obj_id, score, ranks in scored[:top_k]:
            candidates.append(FusedCandidate(
                object_id=obj_id,
                fused_score=score,
                clip_score=1.0 / ranks.get("clip", 9999),
                dino_score=1.0 / ranks.get("dino", 9999),
                ulip_score=1.0 / ranks.get("ulip", 9999),
            ))

        method_name = "intersection"
        if candidates:
            logger.info(
                f"Intersection Fusion: Top-{len(candidates)} "
                f"(Bester: {candidates[0].object_id})"
            )

        return FusionResult(
            candidates=candidates,
            method=method_name,
            best_match=candidates[0] if candidates else None,
        )

    # -----------------------------------------------------------------------
    # Methode 3: Reciprocal Rank Fusion (RRF)
    # -----------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        clip_result: Optional[CLIPRetrievalResult],
        dino_result: Optional[DINOReRankingResult],
        shape_result: Optional[ShapeMatchingResult],
        top_k: int,
        k_param: int = 60,
    ) -> FusionResult:
        """Reciprocal Rank Fusion (RRF).

        RRF-Score(obj) = Σ  1 / (k + rank_i(obj))

        wobei die Summe über alle Ranglisten i läuft und k ein
        Glättungsparameter ist (Standard: 60).

        Ref: "Reciprocal Rank Fusion outperforms Condorcet and Individual
              Rank Learning Methods" (Cormack, Clarke & Buettcher, 2009)

        Args:
            k_param: RRF-Glättungsparameter (Standard: 60).
        """
        rrf_scores: Dict[str, float] = {}

        def _add_rrf(candidates_list, label_fn, score_fn):
            for rank, c in enumerate(candidates_list, 1):
                obj_id = label_fn(c)
                rrf_scores[obj_id] = rrf_scores.get(obj_id, 0.0) + 1.0 / (k_param + rank)

        if clip_result:
            _add_rrf(clip_result.candidates, lambda c: c.object_id, lambda c: c.score)
        if dino_result:
            _add_rrf(dino_result.candidates, lambda c: c.object_id, lambda c: c.dino_score)
        if shape_result:
            _add_rrf(shape_result.candidates, lambda c: c.object_id, lambda c: c.shape_score)

        # Sortieren
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        candidates = []
        for obj_id, score in sorted_items[:top_k]:
            candidates.append(FusedCandidate(
                object_id=obj_id,
                fused_score=score,
            ))

        if candidates:
            logger.info(
                f"RRF Fusion: Top-{len(candidates)} "
                f"(Bester: {candidates[0].object_id}, RRF={candidates[0].fused_score:.6f})"
            )

        return FusionResult(
            candidates=candidates,
            method="rank_fusion",
            best_match=candidates[0] if candidates else None,
        )

    # -----------------------------------------------------------------------
    # Methode 4: Majority Voting (SAMURAI-inspired, thesis ablation E6)
    # -----------------------------------------------------------------------

    def _majority_voting(
        self,
        clip_result: Optional[CLIPRetrievalResult],
        dino_result: Optional[DINOReRankingResult],
        shape_result: Optional[ShapeMatchingResult],
        top_k: int,
    ) -> FusionResult:
        """Majority voting fusion (Borda count).

        Each channel produces an independent ranking. The final rank for
        each candidate is the sum of its per-channel ranks (lower = better).
        Candidates not present in a channel receive a penalty rank equal to
        the channel's candidate count + 1.

        This follows the multi-strategy voting approach used by SAMURAI in
        the ROOMELSA setting (Vo et al., 2025).

        Ties are broken by weighted sum of the raw (unnormalised) per-channel
        scores, so the ordering is deterministic.
        """
        # Build per-channel rank maps (1-based)
        rank_maps: List[Dict[str, int]] = []
        score_maps: List[Dict[str, float]] = []
        paths: Dict[str, str] = {}
        view_paths: Dict[str, str] = {}

        if clip_result and clip_result.candidates:
            rm = {}
            sm = {}
            for rank, c in enumerate(clip_result.candidates, 1):
                rm[c.object_id] = rank
                sm[c.object_id] = c.score
            rank_maps.append(rm)
            score_maps.append(sm)

        if dino_result and dino_result.candidates:
            rm = {}
            sm = {}
            for rank, c in enumerate(dino_result.candidates, 1):
                rm[c.object_id] = rank
                sm[c.object_id] = c.dino_score
                if c.best_view_path:
                    view_paths[c.object_id] = c.best_view_path
            rank_maps.append(rm)
            score_maps.append(sm)

        if shape_result and shape_result.candidates:
            rm = {}
            sm = {}
            for rank, c in enumerate(shape_result.candidates, 1):
                rm[c.object_id] = rank
                sm[c.object_id] = c.shape_score
                if c.cad_model_path:
                    paths[c.object_id] = c.cad_model_path
            rank_maps.append(rm)
            score_maps.append(sm)

        if not rank_maps:
            logger.warning("No candidates for majority voting fusion.")
            return FusionResult(candidates=[], method="majority_voting")

        # Collect all candidate IDs
        all_ids = set()
        for rm in rank_maps:
            all_ids.update(rm.keys())

        # Compute Borda rank sum and tie-breaking score
        penalty_ranks = [len(rm) + 1 for rm in rank_maps]
        scored = []
        for obj_id in all_ids:
            rank_sum = sum(
                rm.get(obj_id, penalty) for rm, penalty in zip(rank_maps, penalty_ranks)
            )
            # Tie-break: sum of raw scores (higher = better)
            raw_score_sum = sum(
                sm.get(obj_id, 0.0) for sm in score_maps
            )
            scored.append((obj_id, rank_sum, raw_score_sum))

        # Sort: lowest rank sum first, then highest raw score sum
        scored.sort(key=lambda x: (x[1], -x[2]))

        candidates = []
        for obj_id, rank_sum, raw_score in scored[:top_k]:
            candidates.append(FusedCandidate(
                object_id=obj_id,
                fused_score=-rank_sum,  # negate so higher = better (convention)
                clip_score=rank_maps[0].get(obj_id, 0) if len(rank_maps) > 0 else 0,
                dino_score=rank_maps[1].get(obj_id, 0) if len(rank_maps) > 1 else 0,
                ulip_score=rank_maps[2].get(obj_id, 0) if len(rank_maps) > 2 else 0,
                cad_model_path=paths.get(obj_id, ""),
                best_view_path=view_paths.get(obj_id, ""),
            ))

        if candidates:
            logger.info(
                "Majority Voting Fusion: Top-%d (Best: %s, rank_sum=%d)",
                len(candidates), candidates[0].object_id,
                -int(candidates[0].fused_score),
            )

        return FusionResult(
            candidates=candidates,
            method="majority_voting",
            best_match=candidates[0] if candidates else None,
        )
