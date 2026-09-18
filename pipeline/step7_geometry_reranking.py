#!/usr/bin/env python3
"""Schritt 7 — Geometrischer Check: dGeDi-Re-Ranking der Fusions-Shortlist.

Historie: Bis 2026-09-18 hiess dieses Modul ``step_b2_geometry_reranking.py``
("Sub-step B2") und enthielt eine In-Process-Implementierung gegen den alten
GeDi-Dienst (gedi:5060), den docker-compose.yml nicht mehr definiert. Die
BERICHTETEN Geometrie-Ergebnisse liefen nie darueber: Stage 1 (E2/O1c/O1e)
registriert ueber ``_pair_scores_dgedi`` in experiment1, Stage 3 und Stage 4
rufen ``object_retrieval/dgedi_bridge.dgedi_rerank`` direkt — alle gegen den
dGeDi-HTTP-Dienst (Port 5061). Seit dem Umbau ist dieses Modul Pipeline-
Schritt 7 (es ersetzt die Skalenschaetzung, die kein berichteter Lauf
verwendet hat) und ist ein Client desselben dGeDi-Dienstes:

  - ``geo_rerank(...)`` — die Umsortierungsregel der Shortlist, WOERTLICH aus
    ``object_retrieval/eval_bop_pose._geo_rerank`` hierher gezogen; der
    Stage-3-Treiber importiert sie jetzt von hier (Rangfolgen unveraendert).
  - ``GeometryReRanker.rerank(...)`` — dGeDi ``/rerank`` + ``geo_rerank``
    fuer die interaktive Pipeline (run_pipeline, Schritt 7).
  - ``GeometryReRanker._load_cad_pointcloud`` — UNVERAENDERT: Stage 1 baut
    darauf die CAD-Wolken (UnitSphereReRanker in experiment1), auf deren
    Fingerprint die Deskriptor-Caches aufsetzen.

Der alte In-Process-Pfad (GeDi-Deskriptoren, RANSAC/ICP im Prozess, Signale
``fitness``/``chamfer_*``) wurde entfernt; ``rerank()`` mit dessen Argument
``all_aligned=`` bricht mit klarer Meldung ab. Historische Implementierung:
``git show bd2de45d:pipeline/step_b2_geometry_reranking.py``.
"""

import logging
import hashlib
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .step6_fusion import FusedCandidate

logger = logging.getLogger(__name__)

# Rangkriterium der Umsortierung (Stage-3-Vokabular). Identisch zu
# eval_bop_pose vor dem Umzug: Env STAGE3_GEO_SIGNAL, Default "distance".
GEO_SIGNAL = os.environ.get("STAGE3_GEO_SIGNAL", "distance")   # distance | borda | fitness


def geo_rerank(fused_ranking, geo, top_k, signal: str = None):
    """Re-rank the fused top-K by the dGeDi geometry signal.

    ``signal`` (default ``STAGE3_GEO_SIGNAL`` = **distance**):
      * ``distance`` — rank by the trimmed surface distance after alignment.
        This is the Stage-1 C1 winner (0.6405 vs 0.6362 Borda vs 0.6251 fitness)
        and therefore the **cross-stage-consistent** criterion.
      * ``borda``    — mean-rank of fitness and distance (the pre-2026-08-27
        behaviour; kept so the earlier runs remain reproducible).
      * ``fitness``  — RANSAC inlier fraction only.
    Failed/uncached candidates sort to the back of the shortlist; the tail past
    top_k is untouched."""
    signal = signal or GEO_SIGNAL
    head = fused_ranking[:top_k]
    tail = fused_ranking[top_k:]
    ids = [oid for oid, _ in head]
    NEG = float("-inf")

    def _sig(o, key, sign):
        g = geo.get(o)
        if not g or not g.get("ok"):
            return NEG
        return sign * float(g[key])

    fit = [_sig(o, "ransac_fitness", 1.0) for o in ids]
    dst = [_sig(o, "d_ransac", -1.0) for o in ids]

    def _ranks(vals):
        return np.argsort(np.argsort(-np.asarray(vals), kind="stable"),
                          kind="stable").astype(float)

    if signal == "distance":
        key = _ranks(dst)
    elif signal == "fitness":
        key = _ranks(fit)
    elif signal == "borda":
        key = (_ranks(fit) + _ranks(dst)) / 2.0
    else:
        raise ValueError(f"unknown STAGE3_GEO_SIGNAL {signal!r}")
    order = list(np.argsort(key, kind="stable"))
    head_re = [(ids[i], -float(key[i])) for i in order]
    return head_re + tail


# ---------------------------------------------------------------------------
# Datenstrukturen (Felder unveraendert — Alt-Aufrufer lesen diese Attribute)
# ---------------------------------------------------------------------------

@dataclass
class GeometryCandidate:
    """Kandidat nach dem geometrischen Check (Schritt 7).

    ``chamfer_score``/``d_ransac`` = getrimmte einseitige Oberflaechendistanz
    nach RANSAC(+ICP)-Ausrichtung (mm der Galerie-Skala; niedriger = besser);
    ``ransac_fitness`` = Inlier-Anteil. Die uebrigen Felder stammen aus der
    Fusion bzw. bleiben fuer Altlast-Kompatibilitaet erhalten (der entfernte
    In-Process-Pfad befuellte auch ``d_icp``/``icp_*``)."""
    object_id: str
    fused_score: float = 0.0
    gedi_score: float = 0.0
    chamfer_score: float = float("inf")
    d_ransac: float = float("inf")
    d_icp: float = float("inf")
    geometry_score: float = 0.0
    ransac_transformation: Optional[np.ndarray] = None
    ransac_fitness: float = 0.0
    icp_transformation: Optional[np.ndarray] = None
    icp_fitness: float = 0.0
    icp_inlier_rmse: float = 0.0
    transformation: Optional[np.ndarray] = None
    registration_failed: bool = False
    cad_model_path: str = ""
    best_view_path: str = ""
    clip_score: float = 0.0
    dino_score: float = 0.0
    ulip_score: float = 0.0


@dataclass
class GeometryReRankingResult:
    """Ergebnis von Schritt 7 (Reihenfolge = neue Rangfolge, beste zuerst)."""
    candidates: List[GeometryCandidate]
    signal: str
    best_candidate: Optional[GeometryCandidate] = None
    best_transformation: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Schritt-7-Modul
# ---------------------------------------------------------------------------

class GeometryReRanker:
    """Geometrischer Check der Fusions-Shortlist ueber den dGeDi-Dienst.

    Verwendung (run_pipeline, Schritt 7):
        >>> reranker = GeometryReRanker(config)
        >>> result = reranker.rerank(fused_candidates, observed_pcd)

    Voraussetzung: laufender dGeDi-Dienst mit passender Galerie
    (``docker compose up -d dgedi``; Galerie via ``DGEDI_CACHE_DIR``).
    Die Kandidaten-IDs muessen Schluessel des Galerie-Manifests sein, und die
    Query-Wolke muss in den Einheiten der Galerie vorliegen (BOP-Galerie
    .dgedi_gallery: METER — wie backproject_masked in query_cloud.py liefert).
    """

    # Konfig-Signale des Altbestands -> Stage-3-Rangkriterium
    _LEGACY_TO_STAGE3 = {
        "distance": "distance", "borda": "borda", "fitness": "fitness",
        "gedi": "fitness", "chamfer": "distance", "chamfer_unaligned": "distance",
        "chamfer_ransac": "distance", "chamfer_icp": "distance", "both": "borda",
    }

    def __init__(self, config: PipelineConfig):
        self.config = config

    def rerank(
        self,
        fused_candidates: List[FusedCandidate],
        observed_pcd,
        signal: Optional[str] = None,
        query_id: Optional[str] = None,
        **legacy,
    ) -> GeometryReRankingResult:
        """dGeDi-Registrierung der Top-K + Umsortierung per ``geo_rerank``.

        Parameter wie die berichteten Stage-3-Laeufe (RUN_PROVENANCE.md:
        ``--dgedi-repo`` = 6000 Keypoints / 10000 RANSAC-Iterationen / +ICP).
        ``query_id`` wird nur noch fuer Log-Zwecke akzeptiert.
        """
        if legacy:
            raise RuntimeError(
                "GeometryReRanker: der In-Process-GeDi-Pfad "
                f"({', '.join(sorted(legacy))}=...) wurde am 2026-09-18 entfernt — "
                "dieses Modul ist jetzt ein Client des dGeDi-Dienstes "
                "(docker compose up -d dgedi). Historische Implementierung: "
                "git show bd2de45d:pipeline/step_b2_geometry_reranking.py")
        sig = self._LEGACY_TO_STAGE3.get(
            signal or self.config.geometry_reranking_signal, "distance")
        top_k = int(self.config.geometry_reranking_top_k)
        if not fused_candidates:
            return GeometryReRankingResult(candidates=[], signal=sig)

        # dgedi_bridge liegt in object_retrieval (kein Paket) — Pfad ergaenzen.
        _oret = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "object_retrieval")
        if _oret not in sys.path:
            sys.path.insert(0, _oret)
        from dgedi_bridge import dgedi_rerank

        pts = np.asarray(observed_pcd.points, dtype=np.float32)
        ids = [c.object_id for c in fused_candidates[:top_k]]
        geo = dgedi_rerank(
            pts, ids,
            ransac_keypoints=getattr(self.config, "dgedi_ransac_keypoints", 6000),
            ransac_max_iter=getattr(self.config, "dgedi_ransac_max_iter", 10000),
            use_icp=True)
        if geo is None:
            logger.warning("dGeDi-Dienst nicht erreichbar — Schritt 7 laesst "
                           "die Fusions-Rangfolge unveraendert "
                           "(docker compose up -d dgedi).")

        by_id = {c.object_id: c for c in fused_candidates}
        ranking: List[Tuple[str, float]] = [
            (c.object_id, float(c.fused_score)) for c in fused_candidates]
        n_ok = sum(1 for v in (geo or {}).values() if v.get("ok"))
        if n_ok:
            ranking = geo_rerank(ranking, geo, top_k, signal=sig)

        out: List[GeometryCandidate] = []
        for oid, score in ranking:
            fc = by_id[oid]
            g = (geo or {}).get(oid) or {}
            d = g.get("d_ransac")
            out.append(GeometryCandidate(
                object_id=oid,
                fused_score=float(fc.fused_score),
                geometry_score=float(score),
                ransac_fitness=float(g.get("ransac_fitness", 0.0)),
                chamfer_score=float(d) if d is not None else float("inf"),
                d_ransac=float(d) if d is not None else float("inf"),
                registration_failed=bool(g) and not bool(g.get("ok")),
                cad_model_path=getattr(fc, "cad_model_path", ""),
                best_view_path=getattr(fc, "best_view_path", ""),
                clip_score=getattr(fc, "clip_score", 0.0),
                dino_score=getattr(fc, "dino_score", 0.0),
                ulip_score=getattr(fc, "ulip_score", 0.0)))
        best = out[0] if n_ok else None
        logger.info("Schritt 7: %d/%d Registrierungen ok (Signal %s)%s",
                    n_ok, len(ids), sig,
                    f" — neuer Rang 1: {best.object_id}" if best else "")
        return GeometryReRankingResult(candidates=out, signal=sig,
                                       best_candidate=best,
                                       best_transformation=None)

    @staticmethod
    def _load_cad_pointcloud(cad_path: str, n_points: int = 10000):
        """Load a CAD model and sample a point cloud."""
        import open3d as o3d

        if not cad_path or not os.path.isfile(cad_path):
            # Try common mesh extensions
            for ext in (".obj", ".ply", ".glb", ".stl"):
                alt = cad_path + ext if cad_path else ""
                if os.path.isfile(alt):
                    cad_path = alt
                    break
            else:
                return None

        try:
            mesh = o3d.io.read_triangle_mesh(cad_path)
            if mesh.is_empty():
                return None
            mesh.compute_vertex_normals()
            # Deterministic sampling: sample_points_uniformly() draws from
            # Open3D's GLOBAL RNG and takes no seed argument (0.19), so
            # without this the same CAD yields a different cloud on every
            # call — irreproducible geometry scores, and a descriptor cache
            # that can never hit.  Seed from the CAD path so the cloud is a
            # pure function of the model, not of call order.
            # Masked into the non-negative int32 range — Open3D's seed() is
            # bound to a 32-bit signed int and rejects larger values.
            o3d.utility.random.seed(
                int(hashlib.sha1(os.path.basename(cad_path).encode()
                                 ).hexdigest()[:8], 16) % (2 ** 31 - 1))
            pcd = mesh.sample_points_uniformly(number_of_points=n_points)
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
            return pcd
        except Exception as exc:
            logger.warning("CAD load failed (%s): %s", cad_path, exc)
            return None
