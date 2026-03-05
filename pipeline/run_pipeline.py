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
from . import visualization as viz

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

    def __init__(self, config: PipelineConfig, visualize: bool = False):
        self.config = config
        self.output_dir = ensure_dir(config.output_dir)
        self.visualize = visualize

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
    ) -> dict:
        """Führt die gesamte Pipeline aus.

        Args:
            rgb_image: RGB-Eingabebild (PIL).
            depth_image: Tiefenbild als numpy-Array (H, W), in mm oder m.
            prompt: Natürlichsprachiger Prompt, z.B. "greife nach der Mayonnaisetube".
            camera_intrinsics: Dict mit 'fx', 'fy', 'cx', 'cy' (optional).
            skip_steps: Liste von Schritt-Nummern die übersprungen werden sollen.

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

            # Prompt bereinigen: Aus "greife nach der Mayonnaisetube"
            # das Zielobjekt extrahieren (vereinfacht: den Prompt direkt nutzen)
            detection_prompt = self._extract_object_name(prompt)
            logger.info(f"  Detektions-Prompt: '{detection_prompt}'")

            loc_result = self.localizer.localize(rgb_image, detection_prompt)
            results["localization"] = loc_result
            timings["step1_localization"] = time.time() - t0

            if loc_result is None:
                logger.error("Objekt nicht gefunden – Pipeline abgebrochen.")
                return {"error": "Object not found", "prompt": prompt}

            logger.info(
                f"  ✓ Objekt gefunden (Konfidenz: {loc_result.confidence:.3f})"
            )

            # --- Visualisierung ---
            if self.visualize and loc_result:
                viz.viz_step1_mask(
                    rgb_image, loc_result.mask, loc_result.bbox,
                    loc_result.confidence, detection_prompt, self.output_dir
                )
                viz.viz_step1_roi(loc_result.roi_image, self.output_dir)

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
                # Optional speichern
                pc_path = os.path.join(self.output_dir, "object_pointcloud.ply")
                self.pc_generator.save_pointcloud(pc_result, pc_path)

                # --- Visualisierung ---
                if self.visualize:
                    loc = results["localization"]
                    viz.viz_step2_depth_masked(depth_image, loc.mask, self.output_dir)
                    viz.viz_step2_pointcloud(
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
            clip_result = self.clip_retriever.retrieve(loc.roi_image)
            results["clip_retrieval"] = clip_result
            timings["step3_clip"] = time.time() - t0

            logger.info(f"  ✓ {len(clip_result.candidates)} CLIP-Kandidaten")
            for i, c in enumerate(clip_result.candidates[:5]):
                logger.info(f"    {i+1}. {c.object_id} (Score: {c.score:.4f})")

            # --- Visualisierung ---
            if self.visualize:
                query_img = results.get("localization", None)
                viz.viz_step3_clip(
                    clip_result, self.config.reference_images_dir,
                    self.output_dir,
                    query_image=query_img.roi_image if query_img else None
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

            # --- Visualisierung ---
            if self.visualize:
                query_img = results.get("localization", None)
                viz.viz_step4_dino(
                    dino_result, self.config.reference_images_dir,
                    self.output_dir,
                    query_image=query_img.roi_image if query_img else None
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

                shape_result = self.shape_matcher.match(pc, candidate_ids=candidate_ids)
                results["shape_matching"] = shape_result
                timings["step5_ulip"] = time.time() - t0

                logger.info(f"  ✓ {len(shape_result.candidates)} Shape-Kandidaten")
                for i, c in enumerate(shape_result.candidates[:5]):
                    logger.info(
                        f"    {i+1}. {c.object_id} (Shape: {c.shape_score:.4f})"
                    )

                # --- Visualisierung ---
                if self.visualize:
                    viz.viz_step5_shape(
                        shape_result, self.config.reference_images_dir,
                        self.output_dir
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

            # --- Visualisierung ---
            if self.visualize:
                query_img = results.get("localization", None)
                viz.viz_step6_fusion(
                    fusion_result, self.config.reference_images_dir,
                    self.output_dir,
                    query_image=query_img.roi_image if query_img else None
                )

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

            if best_model.cad_model_path and pc:
                scale_result = self.scale_estimator.estimate(
                    pc, best_model.cad_model_path
                )
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

            if best_model.cad_model_path:
                pose_result = self.pose_estimator.estimate(
                    rgb_image=np.array(rgb_image),
                    depth_image=depth_image,
                    mask=loc.mask if loc else np.ones_like(depth_image, dtype=bool),
                    cad_model_path=best_model.cad_model_path,
                    scale_factor=scale_factor,
                    observed_pc=results.get("point_cloud"),
                    fx=cam.get("fx"), fy=cam.get("fy"),
                    cx=cam.get("cx"), cy=cam.get("cy"),
                )
                results["pose_estimation"] = pose_result
                timings["step8_pose"] = time.time() - t0

                logger.info(
                    f"  ✓ Pose geschätzt (Methode: {pose_result.method}, "
                    f"Konfidenz: {pose_result.confidence:.4f})"
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

        # --- Summary-Visualisierung ---
        if self.visualize:
            viz.viz_summary(self.output_dir, prompt, timings)
            logger.info(f"Visualisierungen gespeichert in: {self.output_dir}")

        return results

    def _extract_object_name(self, prompt: str) -> str:
        """Extrahiert den Objektnamen aus einem Greif-Prompt.

        Strategie:
          1. Versucht den Objektnamen via Ollama LLM zu extrahieren
             (schnell, lokal, deutsch + englisch nativ unterstützt).
          2. Fällt bei Verbindungsfehler oder Timeout auf die eingebettete
             regelbasierte Heuristik zurück.

        Beispiele:
            "greife nach der Mayonnaisetube" → "Mayonnaisetube"
            "pick up the red cup"            → "red cup"
            "mayonnaise tube"                → "mayonnaise tube"

        Args:
            prompt: Sprachprompt (Deutsch oder Englisch).

        Returns:
            Extrahierter Objektname (Kleinbuchstaben, getrimmt).
        """
        # ------------------------------------------------------------------
        # 1. LLM-Extraktion via Ollama
        # ------------------------------------------------------------------
        try:
            import ollama  # lazy – Paket ist optional zur Laufzeit

            system_msg = (
                "You are a concise extraction assistant. "
                "Given a grasping instruction in German or English, "
                "reply with ONLY the object name (noun phrase) – "
                "no verbs, no articles, no punctuation, no extra words."
            )
            user_msg = f"Instruction: {prompt}"

            # Client-Instanz nutzt den konfigurierten Host (OLLAMA_HOST Env-Var
            # wird als Fallback vom SDK verwendet, wenn kein Host übergeben wird).
            client = ollama.Client(host=self.config.ollama_host)
            response = client.chat(
                model=self.config.ollama_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                options={"temperature": 0, "num_predict": 20},
            )
            extracted = response["message"]["content"].strip().lower()
            if extracted:
                logger.debug(
                    "Ollama (%s) extrahierte Objektname: %r aus %r",
                    self.config.ollama_model, extracted, prompt,
                )
                return extracted

        except Exception as exc:  # noqa: BLE001 – intentional broad catch for robustness
            logger.debug(
                "Ollama-Extraktion fehlgeschlagen (%s) – nutze Heuristik.", exc
            )

        # ------------------------------------------------------------------
        # 2. Regelbasierte Heuristik (Fallback)
        # ------------------------------------------------------------------
        return self._extract_object_name_heuristic(prompt)

    @staticmethod
    def _extract_object_name_heuristic(prompt: str) -> str:
        """Regelbasierter Fallback für _extract_object_name.

        Nimmt alles nach dem letzten Artikel; entfernt andernfalls
        bekannte Verben und Präpositionen.
        """
        articles_de = ["der", "die", "das", "dem", "den", "einer", "einem", "einen"]
        articles_en = ["the", "a", "an"]
        prepositions = [
            "nach", "auf", "mit", "vor",
            "up", "at", "with", "from", "to",
        ]

        words = prompt.strip().split()
        last_article_idx = -1
        for i, word in enumerate(words):
            if word.lower() in articles_de + articles_en:
                last_article_idx = i

        if last_article_idx >= 0 and last_article_idx < len(words) - 1:
            return " ".join(words[last_article_idx + 1:])

        verbs_de = ["greife", "nehme", "hole", "bringe", "pick", "grab", "get"]
        filtered = [
            w for w in words
            if w.lower() not in verbs_de + prepositions + articles_de + articles_en
        ]
        return " ".join(filtered) if filtered else prompt

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
    parser.add_argument("--clip_top_k", type=int, default=20)
    parser.add_argument("--dino_top_k", type=int, default=5)
    parser.add_argument("--ulip_top_k", type=int, default=5)
    parser.add_argument("--ulip_repo", default="", help="Pfad zum geklonten ULIP-Repo")
    parser.add_argument("--ulip_checkpoint", default="", help="Pfad zum ULIP-2 Checkpoint (.pt)")
    parser.add_argument("--skip_steps", type=int, nargs="*", default=[], help="Schritte überspringen (z.B. --skip_steps 5 8)")
    parser.add_argument("--ollama_model", default="mistral-small3.1", help="Ollama-Modell für Prompt-Parsing (default: mistral-small3.1)")
    parser.add_argument("--ollama_host", default="http://localhost:11434", help="Ollama-Serveradresse")
    parser.add_argument("--visualize", action="store_true", help="Zwischenergebnisse als Bilder speichern (Masken, Punktwolken, Kandidaten)")
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
        clip_top_k=args.clip_top_k,
        dino_top_k=args.dino_top_k,
        ulip2_top_k=args.ulip_top_k,
        ulip_repo_path=args.ulip_repo,
        ulip2_checkpoint=args.ulip_checkpoint,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
    )

    # --- Bilder laden ---
    logger.info(f"Lade RGB: {args.rgb}")
    rgb_image = Image.open(args.rgb).convert("RGB")

    logger.info(f"Lade Depth: {args.depth}")
    depth_image = np.array(Image.open(args.depth))
    # Konvertierung in Meter falls nötig (Heuristik)
    if depth_image.max() > 100:
        depth_image = depth_image.astype(np.float32) / config.depth_scale

    # --- Kameraintrinsics ---
    camera_intrinsics = None
    if args.camera:
        from .utils import load_camera_intrinsics
        # Image-ID aus dem RGB-Dateinamen ableiten (z.B. "000001.png" → 1)
        image_id = int(os.path.splitext(os.path.basename(args.rgb))[0])
        camera_intrinsics = load_camera_intrinsics(args.camera, image_id=image_id)

    # --- Pipeline ausführen ---
    pipeline = OSCARPlusPipeline(config, visualize=args.visualize)
    pipeline.initialize()
    result = pipeline.run(
        rgb_image=rgb_image,
        depth_image=depth_image,
        prompt=args.prompt,
        camera_intrinsics=camera_intrinsics,
        skip_steps=args.skip_steps,
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
