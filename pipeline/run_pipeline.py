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
    │      ┌────┴────┐                                                 │
    │      ▼         ▼                                                 │
    │  ┌────────┐ ┌────────────┐                                       │
    │  │2. Point│ │3. CLIP     │ Top-K (K=20)                          │
    │  │  Cloud │ │  Retrieval │                                       │
    │  └───┬────┘ └─────┬──────┘                                       │
    │      │            ▼                                              │
    │      │     ┌────────────┐                                        │
    │      │     │4. DINOv2   │ Top-K (K=5)                            │
    │      │     │  Re-Ranking│                                        │
    │      │     └─────┬──────┘                                        │
    │      │           │                                               │
    │      ▼           │                                               │
    │  ┌────────┐      │                                               │
    │  │5. ULIP │      │                                               │
    │  │  Shape │      │                                               │
    │  │  Match │      │                                               │
    │  └───┬────┘      │                                               │
    │      │           │                                               │
    │      └─────┬─────┘                                               │
    │            ▼                                                     │
    │     ┌────────────┐                                               │
    │     │6. Fusion   │ Gewichtete Summe / RRF / Intersection         │
    │     └─────┬──────┘                                               │
    │           ▼                                                      │
    │     ┌────────────┐                                               │
    │     │7. Scale    │ BBox-Vergleich                                │
    │     └─────┬──────┘                                               │
    │           ▼                                                      │
    │     ┌────────────┐                                               │
    │     │8. Pose     │ FoundationPose / MegaPose / ICP               │
    │     └────────────┘                                               │
    │           │                                                      │
    │           ▼                                                      │
    │     6D Pose [R|t] + skaliertes CAD-Modell                        │
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
                    prompt_elements.detection_phrase, loc_result.confidence,
                    self.output_dir,
                )

        # =================================================================
        # Schritt 2: Punktwolke erzeugen
        # =================================================================
        if 2 not in skip_steps and "localization" in results:
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

            logger.info(f"  ✓ {len(clip_result.candidates)} CLIP-Kandidaten")
            for i, c in enumerate(clip_result.candidates[:5]):
                logger.info(f"    {i+1}. {c.object_id} (Score: {c.score:.4f})")

            if self.debug_viz:
                loc = results["localization"]
                _dbv.save_debug_step3(
                    loc.roi_image, clip_result.candidates,
                    self.config.reference_images_dir, self.output_dir,
                )

        # =================================================================
        # Schritt 4: DINOv2 Re-Ranking
        # =================================================================
        if 4 not in skip_steps and "clip_retrieval" in results:
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Schritt 4: DINOv2 Re-Ranking")

            if not self._initialized and self.config.reference_images_dir:
                self.dino_reranker.load_reference_images()

            loc = results["localization"]
            clip_res = results["clip_retrieval"]
            dino_result = self.dino_reranker.rerank(loc.roi_image, clip_res)
            results["dino_reranking"] = dino_result
            timings["step4_dino"] = time.time() - t0

            logger.info(f"  ✓ {len(dino_result.candidates)} DINOv2-Kandidaten")
            for i, c in enumerate(dino_result.candidates[:5]):
                logger.info(
                    f"    {i+1}. {c.object_id} "
                    f"(DINO: {c.dino_score:.4f}, CLIP: {c.clip_score:.4f})"
                )

            if self.debug_viz:
                loc = results["localization"]
                _dbv.save_debug_step4(
                    loc.roi_image, dino_result.candidates,
                    self.config.reference_images_dir, self.output_dir,
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
                # Optional: nur CLIP/DINO-Kandidaten vergleichen
                candidate_ids = None
                if "clip_retrieval" in results:
                    candidate_ids = self.clip_retriever.get_candidate_labels(
                        results["clip_retrieval"]
                    )

                query_img = results.get("localization", None)
                shape_result = self.shape_matcher.match(
                    pc,
                    candidate_ids=candidate_ids,
                    query_image=query_img.roi_image if query_img else None,
                )
                results["shape_matching"] = shape_result
                timings["step5_ulip"] = time.time() - t0

                logger.info(f"  ✓ {len(shape_result.candidates)} Shape-Kandidaten")
                for i, c in enumerate(shape_result.candidates[:5]):
                    logger.info(
                        f"    {i+1}. {c.object_id} (Shape: {c.shape_score:.4f})"
                    )

                if self.debug_viz:
                    _dbv.save_debug_step5(
                        pc.points, pc.colors,
                        shape_result.candidates,
                        self.config.reference_images_dir, self.output_dir,
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
                logger.info(
                    f"  ✓ Bestes Modell: {fusion_result.best_match.object_id} "
                    f"(Fusionierter Score: {fusion_result.best_match.fused_score:.4f})"
                )

            if self.debug_viz:
                loc = results.get("localization")
                _dbv.save_debug_step6(
                    fusion_result.candidates,
                    self.config.reference_images_dir,
                    loc.roi_image if loc else None,
                    self.output_dir,
                )

        # Für Schritte 7+8 und Debug-Viz werden diese Variablen geteilt
        resolved_mesh = None   # aufgelöster Mesh-Pfad (kein PNG-Fallback)
        scale_result = None
        pose_result = None

        # =================================================================
        # Schritt 7: Skalenbestimmung
        # =================================================================
        if (
            7 not in skip_steps
            and "fusion" in results
            and results["fusion"].best_match
            and "point_cloud" in results
        ):
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Schritt 7: Skalenbestimmung")

            best_model = results["fusion"].best_match
            pc = results["point_cloud"]

            # Mesh-Pfad auflösen: cad_model_path kann ein Referenzbild-Pfad (PNG) sein,
            # wenn ULIP das Objekt nicht in seinen Top-K hatte.
            _IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            resolved_mesh = best_model.cad_model_path
            if not resolved_mesh or os.path.splitext(resolved_mesh)[1].lower() in _IMG_EXTS:
                resolved_mesh = _dbv._find_cad_mesh(
                    best_model.object_id, self.config.cad_models_dir
                )
                if resolved_mesh:
                    logger.info("  Mesh-Pfad aufgelöst: %s", resolved_mesh)
                else:
                    logger.warning("  Kein gültiger Mesh-Pfad für %s gefunden.",
                                   best_model.object_id)

            if resolved_mesh and pc:
                scale_result = self.scale_estimator.estimate(pc, resolved_mesh)
                results["scale_estimation"] = scale_result
                timings["step7_scale"] = time.time() - t0

                logger.info(
                    f"  ✓ Skalierungsfaktor: {scale_result.scale_factor:.4f} "
                    f"(Konfidenz: {scale_result.confidence:.2f})"
                )

        # =================================================================
        # Schritt 8: Pose Estimation
        # =================================================================
        if (
            8 not in skip_steps
            and "fusion" in results
            and results["fusion"].best_match
        ):
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Schritt 8: Pose Estimation")

            best_model = results["fusion"].best_match
            scale = results.get("scale_estimation")
            scale_factor = scale.scale_factor if scale else 1.0
            loc = results.get("localization")

            # Mesh-Pfad auflösen falls Schritt 7 übersprungen wurde
            if resolved_mesh is None:
                _IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
                resolved_mesh = best_model.cad_model_path
                if not resolved_mesh or os.path.splitext(resolved_mesh)[1].lower() in _IMG_EXTS:
                    resolved_mesh = _dbv._find_cad_mesh(
                        best_model.object_id, self.config.cad_models_dir
                    )
                if not resolved_mesh:
                    logger.warning("Kein valider Mesh-Pfad gefunden.")
                    mesh_to_use = None

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
                    f"  ✓ Pose geschätzt (Methode: {pose_result.method}, "
                    f"Konfidenz: {pose_result.confidence:.4f})"
                )

        # --- Debug-Viz: Schritt 7+8 ---
        if self.debug_viz and "fusion" in results and results["fusion"].best_match:
            best_model = results["fusion"].best_match
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

    def _save_results(self, results: dict) -> None:
        """Speichert die Pipeline-Zusammenfassung als JSON."""
        summary = results.get("summary", {})
        out_path = os.path.join(self.output_dir, "pipeline_result.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Ergebnisse gespeichert: {out_path}")


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
    parser.add_argument("--fusion_method", default="weighted_sum", choices=["weighted_sum", "intersection", "rank_fusion"])
    parser.add_argument("--pose_method", default="icp", choices=["foundationpose", "megapose", "icp"])
    parser.add_argument("--foundationpose_url", default="http://foundationpose:5050", help="URL des FoundationPose HTTP-Service")
    parser.add_argument("--foundationpose_refine_iter", type=int, default=5, help="Refinement-Iterationen fuer FoundationPose register()")
    parser.add_argument("--foundationpose_debug", type=int, default=0, help="FoundationPose Debug-Level (0 = headless)")
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
        description_file=args.descriptions,
        reference_images_dir=args.reference_images,
        cad_models_dir=args.cad_models,
        output_dir=args.output,
        fusion_method=args.fusion_method,
        pose_method=args.pose_method,
        foundationpose_url=args.foundationpose_url,
        foundationpose_est_refine_iter=args.foundationpose_refine_iter,
        foundationpose_debug=args.foundationpose_debug,
        clip_top_k=args.clip_top_k,
        dino_top_k=args.dino_top_k,
        ulip2_top_k=args.ulip_top_k,
        ulip_repo_path=args.ulip_repo,
        ulip2_checkpoint=args.ulip_checkpoint,
        ulip2_mode=args.ulip_mode,
        ulip2_image_weight=args.ulip_image_weight,
        ulip2_use_partial_views=args.ulip_partial_views,
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
