# =============================================================================
# pipeline/run_pipeline.py – Hauptorchestrator der OSCAR+ Pipeline
# =============================================================================
#
# Orchestriert alle 8 Pipeline-Schritte in der richtigen Reihenfolge:
#
#   Schritt 1: Objektlokalisierung      (GroundingDINO + SAM)
#   Schritt 2: Punktwolkenerzeugung      (Open3D)
#   Schritt 3: Semantische Kandidaten    (CLIP)
#   Schritt 4: Bildbasiertes Re-Ranking  (DINOv2)
#   Schritt 5: Shape Matching            (ULIP-2)
#   Schritt 6: Score-Fusion              (Weighted Sum / RRF / Intersection)
#   Schritt 7: Skalenbestimmung          (BBox-Vergleich)
#   Schritt 8: Pose Estimation           (FoundationPose / ICP)
#
# Usage:
#   python -m pipeline.run_pipeline \
#       --rgb path/to/image.png \
#       --depth path/to/depth.png \
#       --prompt "greife nach der Mayonnaisetube" \
#       --descriptions object_database/ycbv_gso/descriptions_attributes.json \
#       --reference_images object_images/ycbv_gso/ \
#       --cad_models object_database/ycbv_gso/
#
# =============================================================================

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from PIL import Image

# --- Pipeline Module ---
from .config import PipelineConfig
from .step1_localization import ObjectLocalizer
from .step2_pointcloud import PointCloudGenerator
from .step3_clip_retrieval import CLIPRetriever
from .step4_dino_reranking import DINOReRanker
from .step5_shape_matching import ShapeMatcher
from .step6_fusion import ScoreFusion
from .step7_scale_estimation import ScaleEstimator
from .step8_pose_estimation import PoseEstimator
from .utils import load_depth_image, ensure_dir
from . import debug_viz as _dbv

# =============================================================================
# Logging konfigurieren
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# =============================================================================
# Datenstruktur für extrahierte Prompt-Elemente
# =============================================================================

@dataclass
class PromptElements:
    """Vom Nutzer-Prompt extrahierte Such- und Detektionselemente.

    Attributes:
        object_name:      Kernobjektname (Nomen).  z.B. "mustard bottle"
        color:            Farbe, wenn erwähnt.      z.B. "yellow"
        shape:            Form-Deskriptor.          z.B. "cylindrical"
        material:         Material.                 z.B. "plastic"
        detection_phrase: Volles Adjektiv+Nomen für GroundingDINO.
                          z.B. "yellow mustard bottle"
        visual_query:     Angereicherter Text für CLIP-Text-Retrieval.
                          z.B. "yellow plastic mustard bottle"
    """
    object_name: str
    color: str = ""
    shape: str = ""
    material: str = ""
    detection_phrase: str = ""   # wird in __post_init__ gesetzt wenn leer
    visual_query: str = ""       # wird in __post_init__ gesetzt wenn leer

    def __post_init__(self):
        attrs = " ".join(x for x in [self.color, self.shape, self.material,
                                      self.object_name] if x)
        if not self.detection_phrase:
            # Kurze Phrase für GroundingDINO: Farbe + Objektname
            self.detection_phrase = " ".join(
                x for x in [self.color, self.object_name] if x
            )
        if not self.visual_query:
            self.visual_query = attrs


# =============================================================================
# Pipeline-Orchestrator
# =============================================================================

class OSCARPlusPipeline:
    """Hauptpipeline: Vom Sprachprompt zur 6D-Pose.

    Verbindet die 8 Pipeline-Schritte zu einem kohärenten Ablauf.
    Jeder Schritt ist ein eigenständiges, testbares Modul.

    Architekturübersicht:
    ┌──────────────────────────────────────────────────────────────────┐
    │                    OSCAR+ Pipeline                               │
    │                                                                  │
    │  RGB-D Image + Prompt                                            │
    │       │                                                          │
    │       ▼                                                          │
    │  ┌─────────────────────┐                                         │
    │  │ 1. Lokalisierung    │ GroundingDINO + SAM                     │
    │  │    → Maske, ROI     │                                         │
    │  └────────┬────────────┘                                         │
    │           │                                                      │
    │           ├──────────────────────────────┐                       │
    │           ▼                              ▼                       │
    │  ┌────────────────────┐     ┌────────────────────┐               │
    │  │ 3. CLIP Retrieval  │     │ 2. Punktwolke       │ nur wenn     │
    │  │    → Top-20        │     │    (lazy, nur wenn  │ Step 5/7/8   │
    │  └────────┬───────────┘     │    5/7/8 aktiv)     │ aktiv        │
    │           ▼                 └──────────┬──────────┘              │
    │  ┌────────────────────┐                │                         │
    │  │ 4. DINOv2 Re-Rank  │                │                         │
    │  │    → Top-5         │                │                         │
    │  └────────┬───────────┘                │                         │
    │           │              ┌─────────────┘                         │
    │           ▼              ▼                                       │
    │      ┌──────────────────────┐                                    │
    │      │ 5. ULIP-2 Shape Match│ re-rankt CLIP-Kandidaten           │
    │      └──────────┬───────────┘                                    │
    │                 │                                                │
    │                 └──┐                                             │
    │                    | (+ CLIP + DINO scores)                      │
    │                    ▼                                             │
    │              ┌────────────┐                                      │
    │              │6. Fusion   │ Gewichtete Summe / RRF / Intersection│
    │              └─────┬──────┘                                      │
    │                    ▼                                             │
    │              ┌────────────┐                                      │
    │              │7. Scale    │ BBox-Vergleich                       │
    │              └─────┬──────┘                                      │
    │                    ▼                                             │
    │              ┌────────────┐                                      │
    │              │8. Pose     │ FoundationPose / ICP                 │
    │              └────────────┘                                      │
    │                    │                                             │
    │                    ▼                                             │
    │              6D Pose [R|t] + skaliertes CAD-Modell               │
    └──────────────────────────────────────────────────────────────────┘

    Usage:
        >>> config = PipelineConfig(
        ...     description_file="object_database/ycbv_gso/descriptions.json",
        ...     reference_images_dir="object_images/ycbv_gso/",
        ...     cad_models_dir="object_database/ycbv_gso/",
        ... )
        >>> pipeline = OSCARPlusPipeline(config)
        >>> result = pipeline.run(rgb_image, depth_image, "greife nach der Mayonnaisetube")
    """

    def __init__(self, config: PipelineConfig, debug_viz: bool = False):
        self.config = config
        self.output_dir = ensure_dir(config.output_dir)
        self.debug_viz = debug_viz

        # --- Module initialisieren (Lazy Loading, Modelle werden bei Bedarf geladen) ---
        self.localizer = ObjectLocalizer(config)
        self.pc_generator = PointCloudGenerator(config)
        self.clip_retriever = CLIPRetriever(config)
        self.dino_reranker = DINOReRanker(config)
        self.shape_matcher = ShapeMatcher(config)
        self.fusion = ScoreFusion(config)
        self.scale_estimator = ScaleEstimator(config)
        self.pose_estimator = PoseEstimator(config)

        self._initialized = False

    def initialize(self):
        """Lädt alle Modelle und Daten vorab (optional, sonst Lazy Loading).

        Nützlich wenn die Pipeline mehrfach ausgeführt wird,
        damit die Modelle nicht bei jedem Run neu geladen werden.
        """
        logger.info("=" * 60)
        logger.info("OSCAR+ Pipeline – Initialisierung")
        logger.info("=" * 60)

        t0 = time.time()

        # CLIP Beschreibungen laden
        if self.config.description_file:
            logger.info("Lade CLIP-Beschreibungen...")
            self.clip_retriever.load_descriptions()

        # DINOv2 Referenzbilder laden
        if self.config.reference_images_dir:
            logger.info("Lade DINOv2-Referenzbilder...")
            self.dino_reranker.load_reference_images()

        # ULIP-2 CAD-Modelle laden
        if self.config.cad_models_dir:
            logger.info("Lade ULIP-2 CAD-Modelle...")
            self.shape_matcher.load_cad_models()

        self._initialized = True
        logger.info(f"Initialisierung abgeschlossen in {time.time()-t0:.1f}s")

    def run(
        self,
        rgb_image: Image.Image,
        depth_image: np.ndarray,
        prompt: str,
        camera_intrinsics: dict = None,
        skip_steps: list = None,
        gt_data=None,
    ) -> dict:
        """Führt die gesamte Pipeline aus.

        Args:
            rgb_image: RGB-Eingabebild (PIL).
            depth_image: Tiefenbild als numpy-Array (H, W), in mm oder m.
            prompt: Natürlichsprachiger Prompt, z.B. "greife nach der Mayonnaisetube".
            camera_intrinsics: Dict mit 'fx', 'fy', 'cx', 'cy' (optional).
            skip_steps: Liste von Schritt-Nummern die übersprungen werden sollen.
            gt_data: Optionales Tuple (scene_gt_dict, label_to_obj_id, img_id)
                     für GT-Wireframe-Overlay im Debug-Modus.

        Returns:
            Dict mit Ergebnissen aller Schritte:
            {
                "localization": LocalizationResult,
                "point_cloud": PointCloudResult,
                "clip_retrieval": CLIPRetrievalResult,
                "dino_reranking": DINOReRankingResult,
                "shape_matching": ShapeMatchingResult,
                "fusion": FusionResult,
                "scale_estimation": ScaleEstimationResult,
                "pose_estimation": PoseEstimationResult,
                "timing": {...},
                "summary": {...},
            }
        """
        skip_steps = skip_steps or []
        results = {}
        timings = {}
        cam = camera_intrinsics or {}
        cam["gt_bbox_center_compensation"] = self.config.gt_bbox_center_compensation

        logger.info("=" * 60)
        logger.info(f"OSCAR+ Pipeline – Start")
        logger.info(f"Prompt: \"{prompt}\"")
        logger.info("=" * 60)
        t_start = time.time()

        # =================================================================
        # Schritt 1: Objektlokalisierung (GroundingDINO + SAM)
        # =================================================================
        if 1 not in skip_steps:
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Schritt 1: Objektlokalisierung")

            # Prompt analysieren: Objekt + visuelle Attribute extrahieren
            prompt_elements = self._extract_prompt_elements(prompt)
            results["prompt_elements"] = prompt_elements
            logger.info(f"  Objekt:   '{prompt_elements.object_name}'")
            logger.info(f"  Farbe:    '{prompt_elements.color}'")
            logger.info(f"  Form:     '{prompt_elements.shape}'")
            logger.info(f"  Material: '{prompt_elements.material}'")
            logger.info(f"  Detektions-Phrase: '{prompt_elements.detection_phrase}'")
            logger.info(f"  CLIP-Query:        '{prompt_elements.visual_query}'")

            loc_result = self.localizer.localize(rgb_image, prompt_elements.visual_query)
            results["localization"] = loc_result
            timings["step1_localization"] = time.time() - t0

            if loc_result is None:
                logger.error("Objekt nicht gefunden – Pipeline abgebrochen.")
                return {"error": "Object not found", "prompt": prompt}

            logger.info(
                f"  ✓ Objekt gefunden (Konfidenz: {loc_result.confidence:.3f})"
            )

            if self.debug_viz and loc_result:
                _dbv.save_debug_step1(
                    rgb_image, loc_result.mask, loc_result.bbox,
                    loc_result.roi_image, prompt,
                    prompt_elements.visual_query, loc_result.confidence,
                    self.output_dir,
                )

        # =================================================================
        # Schritt 3: CLIP Retrieval
        # =================================================================
        if 3 not in skip_steps and "localization" in results:
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Schritt 3: CLIP Retrieval (semantische Kandidaten)")

            if not self._initialized and self.config.description_file:
                self.clip_retriever.load_descriptions()

            loc = results["localization"]
            elements = results.get("prompt_elements")
            visual_query = elements.visual_query if elements else None
            clip_result = self.clip_retriever.retrieve(
                loc.roi_image
            )
            results["clip_retrieval"] = clip_result
            timings["step3_clip"] = time.time() - t0

            logger.info(f"  ✓ {len(clip_result.candidates)} CLIP candidates (S_text, full database)")
            for i, c in enumerate(clip_result.candidates[:5]):
                logger.info(f"    {i+1}. {c.object_id} (S_text={c.score:.4f})")

            if self.debug_viz:
                loc = results["localization"]
                _dbv.save_debug_step3(
                    loc.roi_image, clip_result.candidates,
                    self.config.reference_images_dir, self.output_dir,
                )

        # =================================================================
        # Schritt 4: DINOv2 Re-Ranking
        # =================================================================
        if 4 not in skip_steps and "localization" in results:
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Schritt 4: DINOv2 Re-Ranking")

            if not self._initialized and self.config.reference_images_dir:
                self.dino_reranker.load_reference_images()

            loc = results["localization"]
            clip_res = results.get("clip_retrieval")  # None if Step 3 was skipped
            dino_result = self.dino_reranker.rerank(loc.roi_image, clip_res)
            results["dino_reranking"] = dino_result
            timings["step4_dino"] = time.time() - t0

            encoder_name = "SigLIP" if self.config.appearance_encoder == "siglip" else "DINOv2"
            logger.info(f"  ✓ {len(dino_result.candidates)} {encoder_name} candidates "
                        f"(S_view, top-{self.config.dino_view_topk} softmax, full database)")
            for i, c in enumerate(dino_result.candidates[:5]):
                logger.info(
                    f"    {i+1}. {c.object_id} "
                    f"(S_view={c.dino_score:.4f}, S_text={c.clip_score:.4f})"
                )

            if self.debug_viz:
                loc = results["localization"]
                _dbv.save_debug_step4(
                    loc.roi_image, dino_result.candidates,
                    self.config.reference_images_dir, self.output_dir,
                )

        # =================================================================
        # Schritt 2: Punktwolke erzeugen (lazy — nur wenn Step 5/7/8 aktiv)
        # =================================================================
        _needs_pc = any(s not in skip_steps for s in [5, 7, 8])
        if 2 not in skip_steps and _needs_pc and "localization" in results:
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Schritt 2: Punktwolke erzeugen")

            loc = results["localization"]
            rgb_np = np.array(rgb_image)

            pc_result = self.pc_generator.generate(
                rgb_np, depth_image, loc.mask,
                fx=cam.get("fx"), fy=cam.get("fy"),
                cx=cam.get("cx"), cy=cam.get("cy"),
            )
            results["point_cloud"] = pc_result
            timings["step2_pointcloud"] = time.time() - t0

            if pc_result:
                logger.info(
                    f"  ✓ Punktwolke: {pc_result.num_points} Punkte, "
                    f"Größe: {pc_result.bbox_size}"
                )
                if self.debug_viz:
                    loc = results["localization"]
                    _dbv.save_debug_step2(
                        depth_image, loc.mask,
                        pc_result.points, pc_result.colors,
                        pc_result.num_points, pc_result.bbox_size,
                        self.output_dir,
                    )
                    _dbv.save_pointcloud_interactive(
                        pc_result.points, pc_result.colors, self.output_dir
                    )

        # =================================================================
        # Schritt 5: ULIP-2 Shape Matching
        # =================================================================
        if 5 not in skip_steps and "point_cloud" in results:
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Schritt 5: ULIP-2 Shape Matching")

            if not self._initialized and self.config.cad_models_dir:
                self.shape_matcher.load_cad_models()

            pc = results["point_cloud"]
            if pc:
                # Full-database scoring (thesis Sec. 3.5.2): all three channels
                # score every CAD model; no early pruning by any single channel.
                candidate_ids = None

                query_img = results.get("localization", None)
                shape_result = self.shape_matcher.match(
                    pc,
                    candidate_ids=candidate_ids,
                    query_image=query_img.roi_image if query_img else None,
                )
                results["shape_matching"] = shape_result
                timings["step5_ulip"] = time.time() - t0

                encoder_name = "Uni3D" if self.config.shape_encoder == "uni3d" else "ULIP-2"
                logger.info(f"  ✓ {len(shape_result.candidates)} {encoder_name} candidates "
                            f"(S_shape, mode={self.config.ulip2_mode}, full database)")
                for i, c in enumerate(shape_result.candidates[:5]):
                    logger.info(
                        f"    {i+1}. {c.object_id} (S_shape={c.shape_score:.4f})"
                    )

                if self.debug_viz:
                    clip_res = results.get("clip_retrieval")
                    clip_score_map = (
                        {c.object_id: float(c.score) for c in clip_res.candidates}
                        if clip_res is not None else None
                    )
                    _dbv.save_debug_step5(
                        pc.points, pc.colors,
                        shape_result.candidates,
                        self.config.reference_images_dir, self.output_dir,
                        clip_score_map=clip_score_map,
                    )

        # =================================================================
        # Schritt 6: Score-Fusion
        # =================================================================
        if 6 not in skip_steps:
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Schritt 6: Score-Fusion")

            fusion_result = self.fusion.fuse(
                clip_result=results.get("clip_retrieval"),
                dino_result=results.get("dino_reranking"),
                shape_result=results.get("shape_matching"),
            )
            results["fusion"] = fusion_result
            timings["step6_fusion"] = time.time() - t0

            if fusion_result.best_match:
                bm = fusion_result.best_match
                logger.info(
                    f"  ✓ Best match: {bm.object_id} "
                    f"(fused={bm.fused_score:.4f} | "
                    f"S_text={bm.clip_score:.4f}, S_view={bm.dino_score:.4f}, "
                    f"S_shape={bm.ulip_score:.4f})"
                )
                logger.info(
                    f"  Method: {fusion_result.method} | "
                    f"Weights: clip={self.config.weight_clip}, dino={self.config.weight_dino}, "
                    f"ulip={self.config.weight_ulip}"
                )

            if self.debug_viz:
                loc = results.get("localization")
                _dbv.save_debug_step6(
                    fusion_result.candidates,
                    self.config.reference_images_dir,
                    loc.roi_image if loc else None,
                    self.output_dir,
                )

        # =================================================================
        # Sub-step B2: Geometry Re-ranking (GeDi + Chamfer)
        # =================================================================
        b2_ransac_transform = None  # reused in Step 7 to skip redundant RANSAC
        if (
            getattr(self.config, "geometry_reranking_enabled", False)
            and "fusion" in results
            and results["fusion"].best_match
            and "point_cloud" in results
            and 6 not in skip_steps  # B2 depends on fusion
        ):
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Sub-step B2: Geometry Re-ranking (GeDi + Chamfer)")
            logger.info("  Signal: %s | Top-K: %d",
                        self.config.geometry_reranking_signal,
                        self.config.geometry_reranking_top_k)

            from .step_b2_geometry_reranking import GeometryReRanker
            reranker = GeometryReRanker(self.config)
            pc = results["point_cloud"]

            b2_result = reranker.rerank(
                fused_candidates=results["fusion"].candidates,
                observed_pcd=pc.point_cloud,
            )
            results["geometry_reranking"] = b2_result
            timings["step_b2_geometry"] = time.time() - t0

            # Detailed per-candidate log
            logger.info("  ┌─────────────────────────────────────────────────────────────────┐")
            logger.info("  │  B2 Geometry Re-ranking Results                                 │")
            logger.info("  ├─────┬──────────────────────────┬────────┬────────────┬──────────┤")
            logger.info("  │ Rank│ Object ID                │ GeDi   │ Chamfer    │ Geo Score│")
            logger.info("  ├─────┼──────────────────────────┼────────┼────────────┼──────────┤")
            for i, gc in enumerate(b2_result.candidates, 1):
                chamfer_str = f"{gc.chamfer_score:.6f}" if gc.chamfer_score < float("inf") else "    N/A   "
                marker = " ◄" if i == 1 else ""
                logger.info(
                    "  │ %3d │ %-24s │ %6.0f │ %10s │ %8.4f │%s",
                    i, gc.object_id[:24], gc.gedi_score,
                    chamfer_str, gc.geometry_score, marker,
                )
            logger.info("  └─────┴──────────────────────────┴────────┴────────────┴──────────┘")

            if b2_result.best_candidate:
                best = b2_result.best_candidate
                logger.info(
                    "  ✓ B2 winner: %s (GeDi inliers=%d, fitness=%.4f)",
                    best.object_id, int(best.gedi_score), best.ransac_fitness,
                )
                if best.ransac_transformation is not None:
                    logger.info("  ✓ RANSAC transform forwarded to Step 7 as ICP init")
                b2_ransac_transform = b2_result.best_transformation

        # Für Schritte 7+8 und Debug-Viz werden diese Variablen geteilt
        resolved_mesh = None   # aufgelöster Mesh-Pfad (kein PNG-Fallback)
        scale_result = None
        pose_result = None
        effective_best_model = None  # may differ from fusion best_match after scale gate/B2
        scale_gate_failed = False    # set to True when policy=fail and no candidate passes

        # Apply B2 re-ranking result: override effective_best_model
        if "geometry_reranking" in results and results["geometry_reranking"].best_candidate:
            b2_best = results["geometry_reranking"].best_candidate
            # Wrap as FusedCandidate-like object for downstream compatibility
            from .step6_fusion import FusedCandidate
            effective_best_model = FusedCandidate(
                object_id=b2_best.object_id,
                fused_score=b2_best.geometry_score,
                clip_score=b2_best.clip_score,
                dino_score=b2_best.dino_score,
                ulip_score=b2_best.ulip_score,
                cad_model_path=b2_best.cad_model_path,
                best_view_path=b2_best.best_view_path,
            )

        # =================================================================
        # Scale gate (between fusion and scale estimation)
        # =================================================================
        if (
            self.config.scale_gate_enabled
            and "fusion" in results
            and results["fusion"].best_match
            and "point_cloud" in results
            and 7 not in skip_steps
        ):
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Scale Gate: candidate selection by scale plausibility")

            selected, sg_mesh, sg_rank, sg_log = (
                self._select_candidate_with_scale_gate(
                    results["fusion"], results["point_cloud"],
                )
            )
            fallback_used = selected is not None and sg_rank is None
            results["scale_gate"] = {
                "enabled": True,
                "policy": self.config.scale_gate_reject_policy,
                "selected_object_id": selected.object_id if selected else None,
                "selected_rank": sg_rank,
                "fallback_used": fallback_used,
                "candidates_checked": len(sg_log),
                "rejections": sg_log,
            }
            if selected is not None:
                effective_best_model = selected
                resolved_mesh = sg_mesh
                # scale_result stays None so Step 7 still runs full RANSAC+ICP
                # (which is needed for coarse alignment in Step 8)
            else:
                scale_gate_failed = True
            timings["scale_gate"] = time.time() - t0

        # =================================================================
        # Schritt 7: Skalenbestimmung
        # =================================================================
        if scale_gate_failed:
            logger.warning(
                "Scale gate policy=fail: no candidate passed — skipping Steps 7 and 8."
            )

        if (
            7 not in skip_steps
            and not scale_gate_failed
            and "fusion" in results
            and results["fusion"].best_match
            and "point_cloud" in results
            and scale_result is None  # not already computed by scale gate
        ):
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Schritt 7: Skalenbestimmung")

            best_model = effective_best_model or results["fusion"].best_match
            pc = results["point_cloud"]

            if resolved_mesh is None:
                resolved_mesh = self._resolve_mesh_path_for_candidate(best_model)
                if resolved_mesh:
                    logger.info("  Mesh-Pfad aufgelöst: %s", resolved_mesh)
                else:
                    logger.warning("  Kein gültiger Mesh-Pfad für %s gefunden.",
                                   best_model.object_id)

            if resolved_mesh and pc:
                scale_result = self.scale_estimator.estimate(
                    pc, resolved_mesh,
                    init_transform=b2_ransac_transform,
                )
                results["scale_estimation"] = scale_result
                timings["step7_scale"] = time.time() - t0

                logger.info(
                    f"  ✓ Scale factor: {scale_result.scale_factor:.4f} "
                    f"(confidence={scale_result.confidence:.2f}, "
                    f"method={scale_result.method})"
                )
                if scale_result.visible_axes is not None:
                    logger.info(
                        f"  Per-axis ratios: {np.round(scale_result.scale_per_axis, 3).tolist()} "
                        f"→ used axes {scale_result.visible_axes.tolist()}"
                    )
                if b2_ransac_transform is not None:
                    logger.info("  ICP init: from B2 RANSAC transform (GeDi)")
                else:
                    logger.info("  ICP init: from FPFH RANSAC (no B2 transform)")

        # =================================================================
        # Schritt 8: Pose Estimation
        # =================================================================
        if (
            8 not in skip_steps
            and not scale_gate_failed
            and "fusion" in results
            and results["fusion"].best_match
        ):
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Schritt 8: Pose Estimation")

            best_model = effective_best_model or results["fusion"].best_match
            scale = results.get("scale_estimation")
            scale_factor = scale.scale_factor if scale else 1.0
            loc = results.get("localization")

            # Mesh-Pfad auflösen falls Schritt 7 übersprungen wurde
            if resolved_mesh is None:
                resolved_mesh = self._resolve_mesh_path_for_candidate(best_model)
                if not resolved_mesh:
                    logger.warning("Kein valider Mesh-Pfad gefunden.")

            mesh_to_use = resolved_mesh
            if mesh_to_use:
                pose_result = self.pose_estimator.estimate(
                    rgb_image=np.array(rgb_image),
                    depth_image=depth_image,
                    mask=loc.mask if loc else np.ones_like(depth_image, dtype=bool),
                    cad_model_path=mesh_to_use,
                    scale_factor=scale_factor,
                    observed_pc=results.get("point_cloud"),
                    fx=cam.get("fx"), fy=cam.get("fy"),
                    cx=cam.get("cx"), cy=cam.get("cy"),
                    initial_pose=scale.coarse_alignment if scale is not None else None,
                )
                results["pose_estimation"] = pose_result
                timings["step8_pose"] = time.time() - t0

                logger.info(
                    f"  ✓ Pose estimated (method={pose_result.method}, "
                    f"confidence={pose_result.confidence:.4f})"
                )
                logger.info(
                    f"  Translation: [{pose_result.translation[0]:.4f}, "
                    f"{pose_result.translation[1]:.4f}, "
                    f"{pose_result.translation[2]:.4f}] m"
                )

        # --- Debug-Viz: Schritt 7+8 ---
        if self.debug_viz and "fusion" in results and results["fusion"].best_match:
            best_model = effective_best_model or results["fusion"].best_match
            loc = results.get("localization")
            scale = results.get("scale_estimation")
            scale_factor = scale.scale_factor if scale else 1.0
            obs_size = scale.observed_size if scale else None
            cad_size = scale.cad_size if scale else None

            pose_info = {}
            if pose_result is not None:
                pose_info = {
                    "Methode": pose_result.method,
                    "Konfidenz": f"{pose_result.confidence:.4f}",
                    "t [m]": np.round(pose_result.translation, 3).tolist()
                               if hasattr(pose_result, "translation") else "N/A",
                }

            # GT-Pose-Matrix aufbauen
            gt_pose_matrix = None
            if gt_data is not None:
                scene_gt, label_to_obj_id, frame_id = gt_data
                gt_obj_id = label_to_obj_id.get(best_model.object_id)
                if gt_obj_id is not None:
                    frame_key = str(frame_id)
                    for gt_entry in scene_gt.get(frame_key, []):
                        if gt_entry.get("obj_id") == gt_obj_id:
                            R_gt = np.array(gt_entry["cam_R_m2c"]).reshape(3, 3)
                            t_gt = np.array(gt_entry["cam_t_m2c"]) / 1000.0  # mm → m
                            gt_pose_matrix = np.eye(4)
                            gt_pose_matrix[:3, :3] = R_gt
                            gt_pose_matrix[:3, 3] = t_gt
                            logger.info("  GT Pose gefunden für obj_id=%d (%s)",
                                        gt_obj_id, best_model.object_id)
                            break

            if loc:
                _dbv.save_debug_step7_8(
                    rgb_image, loc.bbox, scale_factor, best_model.object_id,
                    self.config.reference_images_dir, pose_info,
                    obs_size, cad_size, self.output_dir,
                    pose_matrix=pose_result.pose_matrix if pose_result is not None else None,
                    cad_model_path=resolved_mesh or best_model.cad_model_path,
                    cam=cam,
                    pose_method=pose_result.method if pose_result is not None else "icp",
                    gt_pose_matrix=gt_pose_matrix,
                )

        # =================================================================
        # Zusammenfassung
        # =================================================================
        total_time = time.time() - t_start
        timings["total"] = total_time

        results["timing"] = timings
        results["summary"] = self._create_summary(results)

        logger.info("=" * 60)
        logger.info(f"Pipeline abgeschlossen in {total_time:.2f}s")
        logger.info("=" * 60)

        # Ergebnisse speichern
        self._save_results(results)

        if self.debug_viz:
            _dbv._done(self.output_dir)

        return results

    # ------------------------------------------------------------------
    # Scale-gate helpers
    # ------------------------------------------------------------------

    _IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def _resolve_mesh_path_for_candidate(self, candidate):
        """Resolve the real CAD mesh path for a fused candidate.

        Falls back to recursive mesh search when cad_model_path points to an
        image file (which can happen when ULIP did not have the object in
        its Top-K).
        """
        resolved = getattr(candidate, "cad_model_path", "")
        if not resolved or os.path.splitext(resolved)[1].lower() in self._IMG_EXTS:
            resolved = _dbv._find_cad_mesh(
                candidate.object_id, self.config.cad_models_dir
            )
        return resolved or None

    def _select_candidate_with_scale_gate(self, fusion_result, observed_pc):
        """Try fused candidates in rank order; accept the first with plausible scale.

        Uses estimate_fast (sorted-bbox, no ICP) for a fast, deterministic
        gate decision. Step 7 still runs full RANSAC+ICP for coarse alignment.

        Returns:
            (selected_candidate, resolved_mesh, selected_rank, rejection_log)
            selected_rank is 1-based; None signals the fallback-best path.
        """
        rejection_log = []
        max_cands = self.config.scale_gate_max_candidates
        candidates = fusion_result.candidates[:max_cands]

        for rank, cand in enumerate(candidates, start=1):
            mesh_path = self._resolve_mesh_path_for_candidate(cand)
            if not mesh_path:
                rejection_log.append({
                    "rank": rank,
                    "object_id": cand.object_id,
                    "fused_score": round(float(cand.fused_score), 6),
                    "mesh_path": None,
                    "reason": "missing_mesh",
                })
                logger.info(
                    "  Scale gate [%d/%d]: %s — rejected (missing mesh)",
                    rank, len(candidates), cand.object_id,
                )
                continue

            scale_factor, confidence = self.scale_estimator.estimate_fast(
                observed_pc, mesh_path
            )
            ok_scale = self.config.scale_gate_min <= scale_factor <= self.config.scale_gate_max
            ok_conf  = confidence >= self.config.scale_gate_min_confidence

            if ok_scale and ok_conf:
                logger.info(
                    "  Scale gate [%d/%d]: %s — ACCEPTED (scale=%.4f, conf=%.2f)",
                    rank, len(candidates), cand.object_id, scale_factor, confidence,
                )
                return cand, mesh_path, rank, rejection_log

            reason = "scale_out_of_range" if not ok_scale else "low_confidence"
            rejection_log.append({
                "rank": rank,
                "object_id": cand.object_id,
                "fused_score": round(float(cand.fused_score), 6),
                "mesh_path": mesh_path,
                "scale_factor": round(scale_factor, 6),
                "confidence": round(confidence, 4),
                "reason": reason,
            })
            logger.info(
                "  Scale gate [%d/%d]: %s — rejected (%s, scale=%.4f, conf=%.2f)",
                rank, len(candidates), cand.object_id, reason, scale_factor, confidence,
            )

        # No candidate accepted
        logger.warning(
            "  Scale gate: 0/%d candidates passed (policy=%s)",
            len(candidates), self.config.scale_gate_reject_policy,
        )

        if self.config.scale_gate_reject_policy == "fallback_best":
            fallback = fusion_result.candidates[0]
            mesh_path = self._resolve_mesh_path_for_candidate(fallback)
            logger.warning(
                "  Scale gate: falling back to rank-1 fusion candidate (%s)",
                fallback.object_id,
            )
            return fallback, mesh_path, None, rejection_log  # rank=None signals fallback

        return None, None, None, rejection_log

    def _extract_prompt_elements(self, prompt: str) -> "PromptElements":
        """Extrahiert Objekt + visuelle Attribute (Farbe, Form, Material) aus dem Prompt.

        Strategie:
          1. Ollama LLM → strukturierte Ausgabe (object / color / shape / material)
          2. Regelbasierte Heuristik als Fallback

        Beispiele:
            "pick up the yellow mustard bottle"
                → object: mustard bottle | color: yellow
            "greife die zylindrische Plastikflasche"
                → object: Flasche | shape: zylindrisch | material: Plastik

        Returns:
            PromptElements (detection_phrase und visual_query werden automatisch gebaut).
        """
        # ------------------------------------------------------------------
        # 1. LLM-Extraktion via Ollama (strukturierte Ausgabe)
        # ------------------------------------------------------------------
        try:
            import ollama

            system_msg = (
                "You extract object properties from a grasping instruction "
                "(German or English). Reply ONLY in this exact format – "
                "use an empty string when the attribute is not mentioned:\n"
                "object: <noun phrase>\n"
                "color: <color or empty>\n"
                "shape: <shape descriptor or empty>\n"
                "material: <material or empty>"
            )
            user_msg = f"Instruction: {prompt}"

            client = ollama.Client(host=self.config.ollama_host)
            response = client.chat(
                model=self.config.ollama_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                options={"temperature": 0, "num_predict": 40},
            )
            raw = response["message"]["content"].strip()
            parsed = self._parse_ollama_elements(raw)
            if parsed:
                logger.debug("Ollama-Elemente: %s", parsed)
                return parsed

        except Exception as exc:
            logger.debug("Ollama-Extraktion fehlgeschlagen (%s) – nutze Heuristik.", exc)

        # ------------------------------------------------------------------
        # 2. Regelbasierte Heuristik (Fallback)
        # ------------------------------------------------------------------
        return self._extract_prompt_elements_heuristic(prompt)

    @staticmethod
    def _parse_ollama_elements(raw: str) -> "Optional[PromptElements]":
        """Parst die strukturierte LLM-Ausgabe in ein PromptElements-Objekt.

        Erwartet Format::
            object: mustard bottle
            color: yellow
            shape:
            material: plastic
        """
        import re
        fields = {"object": "", "color": "", "shape": "", "material": ""}
        for line in raw.splitlines():
            for key in fields:
                m = re.match(rf"^{key}\s*[:\-]\s*(.*)", line.strip(), re.IGNORECASE)
                if m:
                    fields[key] = m.group(1).strip().lower()
                    break

        if not fields["object"]:
            return None

        return PromptElements(
            object_name=fields["object"],
            color=fields["color"],
            shape=fields["shape"],
            material=fields["material"],
        )

    @staticmethod
    def _extract_prompt_elements_heuristic(prompt: str) -> "PromptElements":
        """Regelbasierter Fallback für _extract_prompt_elements.

        Erkennt gängige Farb-, Form- und Materialwörter (DE + EN) und
        extrahiert den verbleibenden Nomen-Teil als Objektname.
        """
        # --- Bekannte Attribut-Wörter ---
        _COLORS = {
            "red", "green", "blue", "yellow", "orange", "purple", "pink",
            "white", "black", "gray", "grey", "brown", "cyan", "magenta",
            "rot", "grün", "blau", "gelb", "orange", "lila", "rosa",
            "weiß", "schwarz", "grau", "braun",
        }
        _SHAPES = {
            "round", "square", "rectangular", "cylindrical", "flat", "spherical",
            "cubic", "triangular", "oval", "elongated",
            "rund", "eckig", "rechteckig", "zylindrisch", "flach", "kugelig",
        }
        _MATERIALS = {
            "plastic", "metal", "wooden", "glass", "rubber", "cardboard",
            "paper", "fabric", "ceramic", "foam",
            "plastik", "metall", "holz", "glas", "gummi", "pappe", "stoff",
        }
        _VERBS    = {
            # Deutsch
            "greife", "nehme", "hole", "bringe", "hol", "gib", "brauch",
            "brauche", "braucht", "möchte", "möchten", "bitte", "geben",
            # Englisch
            "pick", "grab", "get", "take", "fetch", "bring", "need", "needs",
            "want", "wants", "give", "hand", "pass", "find", "bring", "please",
            "could", "would", "should", "like",
        }
        _PREPS    = {
            # Deutsch
            "nach", "auf", "mit", "vor", "für", "von", "zu", "beim", "bitte",
            "der", "die", "das", "dem", "den", "einer", "einem", "einen", "mir",
            "ich", "du", "er", "sie", "wir", "ihr",
            # Englisch
            "up", "at", "with", "from", "to", "for", "the", "a", "an", "in",
            "on", "of", "me", "i", "you", "we", "us", "my", "your", "our",
        }

        words = prompt.strip().split()
        color = shape = material = ""
        remaining = []

        for w in words:
            wl = w.lower()
            if wl in _VERBS or wl in _PREPS:
                continue
            elif wl in _COLORS and not color:
                color = wl
            elif wl in _SHAPES and not shape:
                shape = wl
            elif wl in _MATERIALS and not material:
                material = wl
            else:
                remaining.append(w)

        object_name = " ".join(remaining).strip().lower() or prompt.lower()
        return PromptElements(
            object_name=object_name,
            color=color,
            shape=shape,
            material=material,
        )

    @staticmethod
    def _extract_object_name_heuristic(prompt: str) -> str:
        """Kompatiblitäts-Wrapper für externe Aufrufer."""
        return OSCARPlusPipeline._extract_prompt_elements_heuristic(prompt).object_name

    def _create_summary(self, results: dict) -> dict:
        """Erstellt eine kompakte Zusammenfassung der Pipeline-Ergebnisse."""
        summary = {
            "timestamp": datetime.now().isoformat(),
        }

        if "localization" in results and results["localization"]:
            loc = results["localization"]
            summary["object_detected"] = True
            summary["detection_confidence"] = loc.confidence
            summary["prompt"] = loc.prompt

        if "fusion" in results and results["fusion"].best_match:
            best = results["fusion"].best_match
            summary["best_model"] = best.object_id
            summary["fusion_score"] = best.fused_score
            summary["fusion_method"] = results["fusion"].method

        if "scale_gate" in results:
            sg = results["scale_gate"]
            summary["scale_gate_selected"] = sg["selected_object_id"]
            summary["scale_gate_rejections"] = len(sg["rejections"])

        if "scale_estimation" in results:
            scale = results["scale_estimation"]
            summary["scale_factor"] = scale.scale_factor

        if "pose_estimation" in results:
            pose = results["pose_estimation"]
            summary["pose_confidence"] = pose.confidence
            summary["pose_method"] = pose.method
            summary["translation"] = pose.translation.tolist()

        if "timing" in results:
            summary["total_time_s"] = results["timing"].get("total", 0)

        return summary

    def _write_ranking_csvs(self, results: dict) -> None:
        """Write per-step ranking CSVs to output_dir for post-hoc analysis."""

        def _write(filename, fieldnames, rows):
            path = os.path.join(self.output_dir, filename)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            logger.info("  [CSV] %s (%d rows)", filename, len(rows))

        # --- CLIP ---
        if "clip_retrieval" in results:
            cands = results["clip_retrieval"].candidates
            _write("rankings_clip.csv",
                   ["rank", "object_id", "score", "description"],
                   [{"rank": i + 1, "object_id": c.object_id,
                     "score": round(float(c.score), 6),
                     "description": getattr(c, "description", "")}
                    for i, c in enumerate(cands)])

        # --- DINOv2 ---
        if "dino_reranking" in results:
            cands = results["dino_reranking"].candidates
            _write("rankings_dino.csv",
                   ["rank", "object_id", "dino_score", "clip_score", "best_view_path"],
                   [{"rank": i + 1, "object_id": c.object_id,
                     "dino_score": round(float(c.dino_score), 6),
                     "clip_score": round(float(getattr(c, "clip_score", 0.0)), 6),
                     "best_view_path": getattr(c, "best_view_path", "")}
                    for i, c in enumerate(cands)])

        # --- ULIP-2 ---
        if "shape_matching" in results:
            cands = results["shape_matching"].candidates
            _write("rankings_ulip.csv",
                   ["rank", "object_id", "shape_score", "best_view_idx",
                    "registration_fitness", "registration_rmse", "cad_model_path"],
                   [{"rank": i + 1, "object_id": c.object_id,
                     "shape_score": round(float(c.shape_score), 6),
                     "best_view_idx": getattr(c, "best_view_idx", -1),
                     "registration_fitness": round(float(getattr(c, "registration_fitness", 0.0)), 6),
                     "registration_rmse": round(float(getattr(c, "registration_rmse", 0.0)), 8),
                     "cad_model_path": getattr(c, "cad_model_path", "")}
                    for i, c in enumerate(cands)])

        # --- Fusion ---
        if "fusion" in results:
            cands = results["fusion"].candidates
            method = getattr(results["fusion"], "method", "")
            _write("rankings_fusion.csv",
                   ["rank", "object_id", "fused_score", "clip_score",
                    "dino_score", "ulip_score", "fusion_method", "cad_model_path"],
                   [{"rank": i + 1, "object_id": c.object_id,
                     "fused_score": round(float(c.fused_score), 6),
                     "clip_score": round(float(getattr(c, "clip_score", 0.0)), 6),
                     "dino_score": round(float(getattr(c, "dino_score", 0.0)), 6),
                     "ulip_score": round(float(getattr(c, "ulip_score", 0.0)), 6),
                     "fusion_method": method,
                     "cad_model_path": getattr(c, "cad_model_path", "")}
                    for i, c in enumerate(cands)])

        # --- B2 Geometry Re-ranking ---
        if "geometry_reranking" in results:
            cands = results["geometry_reranking"].candidates
            _write("rankings_b2_geometry.csv",
                   ["rank", "object_id", "gedi_score", "chamfer_score",
                    "geometry_score", "ransac_fitness", "fused_score", "cad_model_path"],
                   [{"rank": i + 1, "object_id": c.object_id,
                     "gedi_score": round(float(c.gedi_score), 1),
                     "chamfer_score": round(float(c.chamfer_score), 8)
                         if c.chamfer_score < float("inf") else "inf",
                     "geometry_score": round(float(c.geometry_score), 4),
                     "ransac_fitness": round(float(c.ransac_fitness), 6),
                     "fused_score": round(float(c.fused_score), 6),
                     "cad_model_path": getattr(c, "cad_model_path", "")}
                    for i, c in enumerate(cands)])

        # --- Scale gate rejections ---
        if "scale_gate" in results:
            rejections = results["scale_gate"].get("rejections", [])
            if rejections:
                _write("rankings_scale_gate.csv",
                       ["rank", "object_id", "fused_score", "scale_factor",
                        "confidence", "reason", "mesh_path"],
                       [{
                           "rank": r.get("rank", ""),
                           "object_id": r.get("object_id", ""),
                           "fused_score": r.get("fused_score", ""),
                           "scale_factor": r.get("scale_factor", ""),
                           "confidence": r.get("confidence", ""),
                           "reason": r.get("reason", ""),
                           "mesh_path": r.get("mesh_path", ""),
                       } for r in rejections])

    def _save_results(self, results: dict) -> None:
        """Speichert die Pipeline-Zusammenfassung als JSON."""
        summary = results.get("summary", {})
        out_path = os.path.join(self.output_dir, "pipeline_result.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Ergebnisse gespeichert: {out_path}")
        self._write_ranking_csvs(results)


# =============================================================================
# CLI Interface
# =============================================================================

def parse_args():
    """Parst Kommandozeilenargumente."""
    parser = argparse.ArgumentParser(
        description="OSCAR+ Pipeline: Vom Sprachprompt zur 6D-Pose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiel:
  python -m pipeline.run_pipeline \\
      --rgb scene/rgb/000001.png \\
      --depth scene/depth/000001.png \\
      --prompt "greife nach der Mayonnaisetube" \\
      --descriptions object_database/ycbv_gso/descriptions_attributes.json \\
      --reference_images object_images/ycbv_gso/ \\
      --cad_models object_database/ycbv_gso/
        """,
    )
    parser.add_argument("--rgb", required=True, help="Pfad zum RGB-Bild")
    parser.add_argument("--depth", required=True, help="Pfad zum Tiefenbild")
    parser.add_argument("--prompt", required=True, help="Sprachprompt, z.B. 'greife nach der Mayonnaisetube'")
    parser.add_argument("--descriptions", required=True, help="Pfad zur Beschreibungs-JSON")
    parser.add_argument("--reference_images", required=True, help="Pfad zum Referenzbilder-Ordner")
    parser.add_argument("--cad_models", required=True, help="Pfad zum CAD-Modell-Ordner")
    parser.add_argument("--camera", default=None, help="Pfad zu scene_camera.json (BOP-Format)")
    parser.add_argument("--output", default="pipeline_output", help="Ausgabeordner")
    parser.add_argument("--fusion_method", default="weighted_sum", choices=["weighted_sum", "intersection", "rank_fusion", "majority_voting"])
    parser.add_argument("--pose_method", default="icp", choices=["foundationpose", "megapose", "icp"])
    parser.add_argument("--foundationpose_url", default="http://foundationpose:5050", help="URL des FoundationPose HTTP-Service")
    parser.add_argument("--foundationpose_refine_iter", type=int, default=5, help="Refinement-Iterationen fuer FoundationPose register()")
    parser.add_argument("--foundationpose_debug", type=int, default=0, help="FoundationPose Debug-Level (0 = headless)")
    parser.add_argument("--appearance-encoder", choices=["dinov2", "siglip"], default="dinov2",
                        dest="appearance_encoder",
                        help="Appearance encoder for Step 4 re-ranking (ablation E4)")
    parser.add_argument("--shape-encoder", choices=["ulip2", "uni3d"], default="ulip2",
                        dest="shape_encoder",
                        help="Shape encoder for Step 5 matching (ablation E7)")
    parser.add_argument("--num-views", type=int, default=42, dest="num_views",
                        help="Number of rendered views per object to use (ablation O4: 8/16/42)")
    parser.add_argument("--clip_top_k", type=int, default=20)
    parser.add_argument("--dino_top_k", type=int, default=5)
    parser.add_argument("--ulip_top_k", type=int, default=5)
    parser.add_argument("--ulip_repo", default="", help="Pfad zum geklonten ULIP-Repo")
    parser.add_argument("--ulip_checkpoint", default="", help="Pfad zum ULIP-2 Checkpoint (.pt)")
    parser.add_argument(
        "--ulip_mode", default="cross", choices=["pc", "cross", "both"],
        help="ULIP-2 Retrieval-Modus: 'pc' (PC→PC), 'cross' (Image→PC, default), 'both' (gewichteter Mix)"
    )
    parser.add_argument(
        "--ulip_image_weight", type=float, default=0.5,
        help="Gewicht des Image-Embeddings im Modus 'both' (PC-Gewicht = 1 - w)."
    )
    parser.add_argument(
        "--ulip-partial-views", action="store_true", dest="ulip_partial_views",
        help="Use precomputed partial point clouds per view instead of full mesh sampling"
    )
    # Scale gate
    parser.add_argument("--scale-gate", action="store_true", dest="scale_gate_enabled",
                        help="Enable scale-gated candidate selection after fusion")
    parser.add_argument("--scale-gate-min", type=float, default=0.8, dest="scale_gate_min")
    parser.add_argument("--scale-gate-max", type=float, default=1.2, dest="scale_gate_max")
    parser.add_argument("--scale-gate-min-confidence", type=float, default=0.0, dest="scale_gate_min_confidence")
    parser.add_argument("--scale-gate-max-candidates", type=int, default=5, dest="scale_gate_max_candidates")
    parser.add_argument("--scale-gate-reject-policy", choices=["fallback_best", "fail"],
                        default="fallback_best", dest="scale_gate_reject_policy")
    # Geometry re-ranking (Sub-step B2)
    parser.add_argument("--geometry-reranking", action="store_true", dest="geometry_reranking_enabled",
                        help="Enable Sub-step B2 geometry re-ranking (GeDi + Chamfer)")
    parser.add_argument("--geometry-reranking-signal",
                        choices=["fitness", "chamfer_unaligned",
                                 "chamfer_ransac", "chamfer_icp",
                                 # legacy aliases (see step_b2 _SIGNAL_ALIASES)
                                 "gedi", "chamfer", "both"],
                        default="chamfer_ransac", dest="geometry_reranking_signal",
                        help="Geometry signal for B2 re-ranking. "
                             "'chamfer_unaligned' is a diagnostic control and "
                             "should not be used in production.")
    parser.add_argument("--geometry-reranking-top-k", type=int, default=5,
                        dest="geometry_reranking_top_k",
                        help="Number of fused candidates to re-rank in B2")
    parser.add_argument("--gedi-repo", default="", dest="gedi_repo_path",
                        help="Path to cloned fabiopoiesi/gedi repo")
    parser.add_argument("--gedi-checkpoint", default="", dest="gedi_checkpoint",
                        help="Path to GeDi model checkpoint (.tar)")

    # Rotation evaluation
    parser.add_argument("--ulip-rotation-eval", action="store_true", dest="ulip_rotation_eval",
                        help="Run ICP rotation evaluation for ULIP Top-K candidates")
    parser.add_argument("--ulip-rotation-eval-top-k", type=int, default=5, dest="ulip_rotation_eval_top_k")
    parser.add_argument("--ulip-rotation-eval-weight", type=float, default=0.0, dest="ulip_rotation_eval_weight",
                        help="Rerank weight for ICP fitness (0.0 = debug-only)")

    parser.add_argument("--skip_steps", type=int, nargs="*", default=[], help="Schritte überspringen (z.B. --skip_steps 5 8)")
    parser.add_argument("--ollama_model", default="gemma3:4b", help="Ollama-Modell für Prompt-Parsing (default: gemma3:4b)")
    parser.add_argument("--ollama_host", default="http://localhost:11434", help="Ollama-Serveradresse")
    parser.add_argument("--debug-viz", action="store_true", dest="debug_viz",
                        help="Reiche Debug-Bilder (debug_01…debug_07 PNGs + PLY + HTML) speichern")
    parser.add_argument("--until-step", type=int, default=8, dest="until_step",
                        help="Pipeline bis einschließlich Schritt N ausführen (1-8, default: 8)")
    parser.add_argument("--gt-bbox-compensation", action="store_true", dest="gt_bbox_compensation",
                        help="Enable bbox-center compensation for GT wireframe overlay (default: off)")
    return parser.parse_args()


def main():
    """Hauptfunktion für CLI-Ausführung."""
    args = parse_args()

    # --- Config aufbauen ---
    config = PipelineConfig(
        appearance_encoder=args.appearance_encoder,
        shape_encoder=args.shape_encoder,
        description_file=args.descriptions,
        reference_images_dir=args.reference_images,
        cad_models_dir=args.cad_models,
        output_dir=args.output,
        fusion_method=args.fusion_method,
        pose_method=args.pose_method,
        foundationpose_url=args.foundationpose_url,
        foundationpose_est_refine_iter=args.foundationpose_refine_iter,
        foundationpose_debug=args.foundationpose_debug,
        num_views=args.num_views,
        clip_top_k=args.clip_top_k,
        dino_top_k=args.dino_top_k,
        ulip2_top_k=args.ulip_top_k,
        ulip_repo_path=args.ulip_repo,
        ulip2_checkpoint=args.ulip_checkpoint,
        ulip2_mode=args.ulip_mode,
        ulip2_image_weight=args.ulip_image_weight,
        ulip2_use_partial_views=args.ulip_partial_views,
        ulip2_rotation_eval=args.ulip_rotation_eval,
        ulip2_rotation_eval_top_k=args.ulip_rotation_eval_top_k,
        ulip2_rotation_eval_weight=args.ulip_rotation_eval_weight,
        geometry_reranking_enabled=args.geometry_reranking_enabled,
        geometry_reranking_signal=args.geometry_reranking_signal,
        geometry_reranking_top_k=args.geometry_reranking_top_k,
        gedi_repo_path=args.gedi_repo_path,
        gedi_checkpoint=args.gedi_checkpoint,
        scale_gate_enabled=args.scale_gate_enabled,
        scale_gate_min=args.scale_gate_min,
        scale_gate_max=args.scale_gate_max,
        scale_gate_min_confidence=args.scale_gate_min_confidence,
        scale_gate_max_candidates=args.scale_gate_max_candidates,
        scale_gate_reject_policy=args.scale_gate_reject_policy,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        gt_bbox_center_compensation=args.gt_bbox_compensation,
    )

    # --- Bilder laden ---
    logger.info(f"Lade RGB: {args.rgb}")
    rgb_image = Image.open(args.rgb).convert("RGB")

    # --- Kameraintrinsics ---
    camera_intrinsics = None
    if args.camera:
        from .utils import load_camera_intrinsics
        # Image-ID aus dem RGB-Dateinamen ableiten (z.B. "000001.png" → 1)
        image_id = int(os.path.splitext(os.path.basename(args.rgb))[0])
        camera_intrinsics = load_camera_intrinsics(args.camera, image_id=image_id)

    logger.info(f"Lade Depth: {args.depth}")
    depth_image = np.array(Image.open(args.depth)).astype(np.float32)

    # Determine depth_scale: prefer BOP scene_camera.json, fall back to config
    # BOP convention: raw * depth_scale = mm → raw * depth_scale / 1000 = meters
    # Config convention: raw / config.depth_scale = meters
    if camera_intrinsics and camera_intrinsics.get("depth_scale", 0) > 0:
        bop_ds = camera_intrinsics["depth_scale"]
        depth_image = depth_image * bop_ds / 1000.0
        logger.info("Depth: BOP depth_scale=%.4f → raw * %.4f / 1000 = meters", bop_ds, bop_ds)
    else:
        depth_image = depth_image / config.depth_scale
        logger.info("Depth: config depth_scale=%.1f → raw / %.1f = meters", config.depth_scale, config.depth_scale)

    # --- until_step → skip_steps ---
    skip_steps = list(args.skip_steps)
    if args.until_step < 8:
        skip_steps = sorted(set(skip_steps) | set(range(args.until_step + 1, 9)))

    # --- GT-Daten laden (nur bei --debug-viz) ---
    gt_data = None
    if args.debug_viz and args.camera:
        import json as _json
        scene_dir = os.path.dirname(args.camera)
        gt_path = os.path.join(scene_dir, "scene_gt.json")
        id_label_path = os.path.join(scene_dir, "..", "id_to_label.json")
        if os.path.isfile(gt_path) and os.path.isfile(id_label_path):
            try:
                with open(gt_path) as f:
                    scene_gt = _json.load(f)
                with open(id_label_path) as f:
                    id_to_label = _json.load(f)
                label_to_obj_id = {v: int(k) for k, v in id_to_label.items()}
                img_id = int(os.path.splitext(os.path.basename(args.rgb))[0])
                gt_data = (scene_gt, label_to_obj_id, img_id)
                logger.info("GT geladen: %s (%d Labels)", gt_path, len(label_to_obj_id))
            except Exception as e:
                logger.warning("GT laden fehlgeschlagen: %s", e)

    # --- Pipeline ausführen ---
    pipeline = OSCARPlusPipeline(config, debug_viz=args.debug_viz)
    pipeline.initialize()
    result = pipeline.run(
        rgb_image=rgb_image,
        depth_image=depth_image,
        prompt=args.prompt,
        camera_intrinsics=camera_intrinsics,
        skip_steps=skip_steps,
        gt_data=gt_data,
    )

    # --- Zusammenfassung ausgeben ---
    summary = result.get("summary", {})
    print("\n" + "=" * 60)
    print("PIPELINE ERGEBNIS")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
