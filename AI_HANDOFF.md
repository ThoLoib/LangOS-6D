# AI Handoff – Branch `exp/ulip2-full`

> Zuletzt aktualisiert: 2026-04-02

## Projektziel

Masterarbeit: **Shape-Aware Object Retrieval and 6D Pose Estimation** basierend auf dem OSCAR-Framework ([pullover00/OSCAR](https://github.com/pullover00/OSCAR)).

Kernidee: Das bestehende OSCAR-Retrieval (CLIP + DINOv2) um einen **3D-Shape-Kanal (ULIP-2)** erweitern. Eine partielle Punktwolke aus RGB-D wird per ULIP-2 mit den CAD-Modell-Punktwolken verglichen. Die Scores der drei Kanäle (CLIP, DINOv2, ULIP-2) werden fusioniert.

---

## Branch-Überblick

| Branch | Zweck | Status |
|---|---|---|
| `oscar` | Clean upstream mirror von pullover00/OSCAR | ✅ nie verändert |
| `main` | Thesis-Scaffolding + AI-Docs | ✅ stabil |
| `exp/oscar-repro` | OSCAR baseline reproduziert (d3098bdd) | ✅ abgeschlossen |
| `exp/ulip2` | Shape-Aware Pipeline (PC-ULIP + Fusion) | ✅ stabil |
| **`exp/ulip2-full`** | **ULIP full experiments (PC vs cross-modal image->PC)** | 🟢 aktiv |

---

## Update 2026-04-02 (SAM2.1 migration + audit fixes)

- **SAM → SAM2.1**: Step 1 now uses `facebook/sam2.1-hiera-large` instead of `facebook/sam-vit-large`. Better mask quality, especially at ambiguous boundaries. API change: `Sam2Model`/`Sam2Processor`, simplified `post_process_masks()`.
- **Step 2**: Tightened statistical outlier removal (`std_ratio` 2.0 → 1.0) for cleaner point clouds.
- **Step 1 query**: Localization now uses `visual_query` (LLM-extracted) instead of `detection_phrase`.
- **CLIP text fusion**: Intentionally disabled (`text_query` removed from `retrieve()` call) pending tuning.
- **Mesh path fix**: Null guard in `run_pipeline.py` prevents crash when no valid mesh found.
- **New**: `docs/PIPELINE_AUDIT.md` — comprehensive audit with 20 ranked findings and ablation recommendations.

---

## Update 2026-03-29 (Cleanup: load_object_descriptions → CLIPRetriever)

- Moved `load_object_descriptions()` from `pipeline/utils.py` into `CLIPRetriever._load_object_descriptions()` (static method).
- Aligns Step 3 with Step 4 pattern (data loading as class method, not standalone utility).
- No behavioral change.

---

## Update 2026-03-26 (Partial-to-partial point cloud matching for Step 5)

### Partial views preprocessing
- **New** `rendering/generate_partial_pointclouds.py`: standalone script that generates partial PCs from CAD meshes using front-face culling from 8 camera viewpoints.
  - Uses trimesh for mesh loading/normalization and surface sampling — no Blender needed.
  - Converts texture visuals to per-face colors; replicates the same bbox normalization as `rendering.py`.
  - Output: `{obj_id}_view{N}_partial.npz` files (keys: `points`, `colors`) alongside existing PNGs and camera matrices.
  - Performance: ~1s per object, ~10 min for 1051 YCBV-GSO objects × 8 views.

### Pipeline changes
- **Modified** `pipeline/config.py`: new field `ulip2_use_partial_views: bool = False`.
- **Modified** `pipeline/step5_shape_matching.py`:
  - `ShapeCandidate` gains `best_view_idx: int` field.
  - `ShapeMatcher.load_cad_models()` dual path: if partial views enabled, loads `.npz` files and encodes per-view embeddings `(num_views, embed_dim)`.
  - `match()` uses best-of-N-views scoring (max cosine similarity over views).
  - Separate cache: `.ulip_partial_cache_<hash>.pt` (distinct from full-mesh cache).
  - Fallback: objects without `.npz` files fall back to full mesh sampling.
- **Modified** `pipeline/debug_viz.py`: shows "Best View: N" label and loads matching view thumbnail.
- **Modified** `pipeline/run_pipeline.py`: new `--ulip-partial-views` CLI flag.

### How to use
```bash
# 1. Generate partial point clouds (one-time preprocessing, inside OSCAR container):
python3.11 rendering/generate_partial_pointclouds.py \
    --cad_dir object_database/ycbv_gso/ \
    --images_dir object_images/ycbv_gso/

# 2. Run pipeline with partial views:
python3.11 -m pipeline.run_pipeline \
    --rgb eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
    --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
    --camera eval/datasets/ycbv_gso/test/000048/scene_camera.json \
    --prompt "I need the red mug" \
    --descriptions object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json \
    --reference_images object_images/ycbv_gso/ \
    --cad_models object_database/ycbv_gso/ \
    --ulip_repo /ulip \
    --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt \
    --ulip_mode pc \
    --ulip-partial-views \
    --debug-viz --until-step 6 \
    --output debug_output
```

### All three ULIP modes work with partial views
| Mode | Query embedding | Reference embeddings | Scoring |
|---|---|---|---|
| `pc` | observed PC → ULIP PC encoder | 8 partial PCs → ULIP PC encoder | max over 8 views |
| `cross` | ROI image → OpenCLIP image encoder | 8 partial PCs → ULIP PC encoder | max over 8 views |
| `both` | weighted avg of pc + cross | 8 partial PCs → ULIP PC encoder | max over 8 views |

---

## Update 2026-03-26 (Debug-Visualisierung refactored into main pipeline)

### Refactoring: Debug als optionaler Modus der normalen Pipeline
- **Removed** `pipeline/debug_steps.py` entirely (was ~1473 lines with duplicated pipeline logic).
- **New** `pipeline/debug_viz.py` (~1070 lines): All rich visualization functions extracted as a standalone module.
  - PIL helpers, `save_debug_step1()` … `save_debug_step7_8()`, `_project_cad_wireframe()`, `save_pointcloud_interactive()`, `_done()`.
  - **Bug fix:** `_find_cad_mesh()` moved to module level (was nested inside `save_debug_step7_8`, unreachable from `run_debug()`).
- **Modified** `pipeline/run_pipeline.py`:
  - `OSCARPlusPipeline.__init__()` gains `debug_viz: bool = False` parameter.
  - `OSCARPlusPipeline.run()` gains `gt_data=None` parameter for GT-wireframe overlay.
  - Debug-viz hooks added after each step (only executed when `debug_viz=True`).
  - Mesh-path resolution before step 7: detects image-paths (`.png/.jpg`) in `cad_model_path` and falls back to `_find_cad_mesh()` lookup. Used by both steps 7 and 8.
  - New CLI flags: `--debug-viz`, `--until-step`.
  - `main()` loads GT data from `scene_gt.json` + `id_to_label.json` when `--debug-viz` and `--camera` are set.
  - **Bug fix:** `detection_prompt` (undefined variable) → `prompt_elements.detection_phrase` in step 1 visualization.
- **Modified** `scripts/run_debug_pipeline_foundationpose.sh`: Now calls `pipeline.run_pipeline --debug-viz` instead of `pipeline.debug_steps`.
- **New** `scripts/run_pipeline.sh`: Convenience script for normal pipeline with YCBV-GSO defaults.

### Behavioral changes vs. old `debug_steps.py`
1. CLIP `text_query`: old `run_debug()` called `clip.retrieve(roi)` without text query. The unified pipeline passes `visual_query` from prompt parsing — may give slightly different CLIP rankings.
2. Prompt parsing: old `run_debug()` duplicated the Ollama+heuristic logic; now uses `OSCARPlusPipeline._extract_prompt_elements()` directly.

### Start commands
```bash
# Debug mode (rich PNGs + PLY + HTML):
./scripts/run_debug_pipeline_foundationpose.sh

# Debug mode via run_pipeline.py:
python3.11 -m pipeline.run_pipeline ... --debug-viz --until-step 6

# Normal mode (no debug output):
python3.11 -m pipeline.run_pipeline --rgb ... --depth ... --prompt "..."

# Normal mode + simple viz:
python3.11 -m pipeline.run_pipeline ... --visualize
```

---

## Update 2026-03-24 (GT overlay + intrinsics/depth fixes)

### GT pose overlay in debug_07_scale_pose.png
- GT wireframe overlay (magenta) drawn alongside predicted (green) via `_project_cad_wireframe()`.
- Compensates for mesh bbox_center offset: subtracts `R_gt @ bbox_center` from GT translation.
- Adds "Predicted" / "GT" legend labels; Δt (mm) and ΔR (deg) error metrics to info panel.

### Camera intrinsics priority fix
- Camera loading moved **before** depth conversion so real `fx/fy/cx/cy` from `scene_camera.json` reach `generate()`.
- `config` values used as fallback only when `--camera` is absent.

### BOP depth_scale convention mismatch (gotcha)
- `scene_camera.json` `depth_scale` is a **multiplier** (e.g. 0.1 for this dataset).
- Pipeline divides depth by `config.depth_scale` (default 10000.0) — a **divisor** convention.
- Using the JSON value caused depths to be 100× too large, resulting in ~855mm translation error.
- Decision: always use `config.depth_scale` as divisor; ignore the JSON field entirely.

---

## Update 2026-03-20 (FoundationPose two-container HTTP integration)

Architecture change: FoundationPose now runs as a **separate Docker container** with an HTTP API instead of via subprocess/venv inside the OSCAR container.

- `foundationpose_server.py` in the FoundationPose repo: Flask server with `/health` and `/estimate_pose`
- `pipeline/foundationpose_bridge.py`: rewritten as HTTP client (uses httpx)
- `pipeline/step8_pose_estimation.py`: calls bridge via HTTP, removed subprocess and local-import paths
- `pipeline/config.py`: `foundationpose_url` replaces `foundationpose_python` and `foundationpose_repo_path`
- `docker-compose.yml`: added `foundationpose` service using `shingarey/foundationpose_custom_cuda121`
- FP container mounts OSCAR repo read-only at `/oscar` for CAD model access
- Bridge auto-translates CAD paths from `/app/...` to `/oscar/...`

Why this was done:
- The venv-inside-OSCAR approach failed because the OSCAR container (CUDA 12.2 runtime, Python 3.11) cannot compile pytorch3d/kaolin/nvdiffrast which require CUDA devel headers.
- Two containers with HTTP boundary gives full dependency isolation with no shared Python environment.

Removed (superseded):
- `foundationpose_python` config field and CLI arg
- `foundationpose_repo_path` config field and CLI arg
- Subprocess bridge logic in step8
- Local-import path (`_run_foundationpose_local`) in step8
- `../FoundationPose:/foundationpose` volume mount in oscar service
- MegaPose stub method in step8 (was always NotImplementedError)

Operational pattern:

```bash
# Start FP service (first time loads models ~30s)
docker compose up -d foundationpose

# Run OSCAR with FoundationPose
docker compose run --rm -it oscar bash
./scripts/run_debug_pipeline_foundationpose.sh
# or manually:
python3.11 -m pipeline.run_pipeline \
  --rgb eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
  --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
  --camera eval/datasets/ycbv_gso/test/000048/scene_camera.json \
  --prompt "I need the red mug" \
  --descriptions object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json \
  --reference_images object_images/ycbv_gso/ \
  --cad_models object_database/ycbv_gso/ \
  --ulip_repo /ulip \
  --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt \
  --pose_method foundationpose \
  --output debug_output \
  --debug-viz --until-step 8
```

If FoundationPose service is down or fails, Step 8 falls back to ICP automatically.

## Update 2026-03-19 (FoundationPose split-env integration — superseded)

> This approach was replaced by the two-container HTTP architecture on 2026-03-20.
> The subprocess bridge and venv approach did not work due to CUDA/ABI incompatibilities.
> Kept here for historical context only.

## Update 2026-03-18 (foundationpose prep + step1 cleanup)

- `pipeline/step1_localization.py`:
  - eine doppelte Kommentarzeile im Header entfernt (non-functional cleanup).
- `docker-compose.yml`:
  - zusätzliches Volume-Mount für FoundationPose (superseded by 2026-03-20 two-container setup).
- FoundationPose Setup-Status:
  - Repo lokal geklont (`~/thesis/FoundationPose`)
  - Docker image vorhanden (`foundationpose:latest`)

## Update 2026-03-17 (exp/ulip2-full)

- ULIP Step 5 erweitert um `ulip_mode`:
  - `pc`: nur Shape-Embedding (PointCloud -> CAD-PC)
  - `cross`: Image->PC Cross-Modal (OpenCLIP image branch)
  - `both`: gewichteter Mix (`ulip_image_weight`)
- `debug_steps.py` erweitert:
  - neue CLI-Args `--ulip_mode`, `--ulip_image_weight`
  - `query_image` wird an Step 5 durchgereicht
- GSO-CAD-Laden in Step 5 gefixt:
  - rekursive Mesh-Suche in Unterordnern (`meshes/model.obj`, `textured_simple.obj`)
  - vorher nur 21 Modelle, jetzt 1051 Modelle
- Performance-Fix Step 5:
  - CAD-Embeddings werden als Disk-Cache gespeichert (`.ulip_cache_<hash>.pt`)
  - erste Berechnung bleibt teuer, Folge-Runs laden Cache deutlich schneller
- Step 8 Pose-Fix:
  - falscher Bildpfad (`object_images/...png`) konnte als `cad_model_path` in Fusion landen
  - Fusion trennt jetzt `best_view_path` (DINO-Bild) von echtem `cad_model_path` (Mesh)
  - Debug löst Meshpfad robust auf, damit ICP ein OBJ/PLY/GLB bekommt
- Dependencies ergänzt:
  - `open-clip-torch` (für ULIP cross)
  - `trimesh` (für Overlay/Wireframe-Visualisierung)

---

## Aktueller Stand (exp/ulip2)

### Was funktioniert (End-to-End verifiziert, 2026-03-12)

- **Modulare 8-Schritt-Pipeline** in `pipeline/` – komplett durchgetestet:
  1. `step1_localization.py` – GroundingDINO + SAM → Maske + BBox (Konfidenz 0.847)
  2. `step2_pointcloud.py` – RGB-D + Maske → Open3D Point Cloud (4.201 Punkte bei 2mm Voxel)
  3. `step3_clip_retrieval.py` – Prompt → CLIP → Top-8 Kandidaten (master_chef_can 0.4702)
  4. `step4_dino_reranking.py` – ROI → DINOv2 → Top-5 Re-Ranking (master_chef_can 0.6447)
     - **Batch-Encoding** (32 imgs/pass) + **.pt Disk-Cache** für 9.459 Referenzbilder
     - Erstlauf ~5 Min, danach sofort aus Cache
  5. `step5_shape_matching.py` – **ULIP-2 Point Cloud Encoder** → Shape-Similarity
     - NaN-Scores werden gefiltert (Overflow bei pcd.colors → fix mit `np.clip`)
  6. `step6_fusion.py` – Weighted Sum mit Min-Max-Normalisierung pro Modalität
     - NaN-sichere `_minmax()` Funktion
     - Ergebnis: master_chef_can fused=0.8473
  7. `step7_scale_estimation.py` – RANSAC + ICP Coarse-Alignment → Partial-Aware Scale
     - scale=1.2968, conf=0.63 (2 beste Achsen)
  8. `step8_pose_estimation.py` – FoundationPose (HTTP) oder ICP mit Coarse-Alignment
     - ICP: fitness=0.9895, RMSE=0.007m

- **Debug-Visualisierung** (`pipeline/debug_viz.py`, ~1070 Zeilen, aktiviert via `--debug-viz`):
  - 7 diagnostische PNG-Bilder + interaktiver 3D-Viewer (HTML)
  - 3D-Wireframe-Overlay der CAD-Modell-Pose auf Szenenbild (via trimesh)
  - Automatische Panels: Lokalisierung, Punktwolke, CLIP, DINOv2, ULIP, Fusion, Scale+Pose

- **ULIP-2 Integration** (step5):
  - Lädt nur `point_encoder` + `pc_projection` (~400 MB statt ~5.5 GB für volles OpenCLIP)
  - Backbone: PointBERT Colored (10k Punkte xyzrgb → 1280-dim Embedding)
  - Checkpoint: `ulip2_pointbert_10k.pt` in `/ulip/checkpoints/`
  - ULIP-Repo als Volume gemountet (`../ULIP:/ulip` im Container)

- **LLM-basiertes Prompt Parsing**:
  - Ollama + `gemma3:4b` (localhost:11434, 30s Timeout)
  - Extrahiert Objektname, Farbe, Form, Material aus natürlichem Prompt
  - Fallback: regelbasierter Heuristic-Parser

- **Docker-Konfiguration**:
  - OSCAR Image: `tholoi/oscar-plus` (CUDA 12.2, Python 3.11)
  - FoundationPose Image: `shingarey/foundationpose_custom_cuda121` (CUDA 12.1, Python 3.8)
  - GPU-Support via `deploy.resources.reservations.devices`
  - ULIP-Volume: `../ULIP:/ulip`
  - HuggingFace Cache-Volume

### Bekannte Limitierungen

1. **ULIP-2 Shape Matching (full mesh)** liefert schwache Ergebnisse für partielle Punktwolken (single-view, ~4k Punkte vs. komplette 10k-CAD-Modelle). **Mitigation:** `--ulip-partial-views` schaltet auf partial-to-partial Vergleich um (best-of-8-views).
2. **ICP auf symmetrischen Objekten** (z.B. Dosen) kann Rotation um Symmetrieachse nicht eindeutig bestimmen.

### Was noch fehlt

1. **Evaluation-Script** – Über alle BOP-Szenen laufen, ULIP-2-augmentierte Top-K Accuracy berechnen, mit 75.95% Baseline vergleichen.
2. **MI3DOR Evaluation** – Shape-Retrieval auf MI3DOR testen (ULIP-2 sollte hier besonders helfen).
3. **HouseCat6D** – BOP-Testdaten beschaffen + evaluieren.
4. **Hyperparameter-Tuning** – Fusionsgewichte (aktuell 0.3/0.4/0.3), Top-K je Schritt, Voxelgröße.
5. **Fehlende MI3DOR-Beschreibungen** – 11/21 Kategorien noch nicht generiert.

---

## Datei-Inventar

### Pipeline-Dateien

| Datei | Zeilen | Beschreibung |
|---|---|---|
| `pipeline/__init__.py` | 21 | Package-Init mit Version `0.1.0` |
| `pipeline/config.py` | ~140 | Zentrale `PipelineConfig` Dataclass |
| `pipeline/run_pipeline.py` | ~1045 | Orchestrator + CLI + LLM-Parsing + Debug-Viz-Hooks |
| `pipeline/debug_viz.py` | ~1070 | **Debug-Visualisierung** (7 PNGs + 3D-Viewer) |
| `pipeline/foundationpose_bridge.py` | ~100 | HTTP client for FoundationPose service |
| `pipeline/step1_localization.py` | ~240 | GroundingDINO + SAM |
| `pipeline/step2_pointcloud.py` | ~280 | RGB-D → Point Cloud |
| `pipeline/step3_clip_retrieval.py` | ~280 | CLIP Text-/Bild-Retrieval |
| `pipeline/step4_dino_reranking.py` | ~350 | DINOv2 Re-Ranking + Batch-Cache |
| `pipeline/step5_shape_matching.py` | ~1100 | ULIP-2 Encoder + NaN-Filterung + Partial Views |
| `rendering/generate_partial_pointclouds.py` | ~250 | Partial PC preprocessing (front-face culling) |
| `pipeline/step6_fusion.py` | ~370 | Score-Fusion (weighted_sum, RRF, intersection) |
| `pipeline/step7_scale_estimation.py` | ~300 | RANSAC+ICP Coarse-Alignment + Partial-Aware Scale |
| `pipeline/step8_pose_estimation.py` | ~290 | FoundationPose (HTTP) + ICP fallback |
| `pipeline/utils.py` | ~150 | Hilfsfunktionen |
| `pipeline/visualization.py` | ~375 | Legacy-Visualization |

### Konfiguration (config.py Defaults)

```python
# Punktwolke
voxel_size = 0.002              # Voxel-Downsampling (2mm, ~4000 Punkte)
depth_scale = 10000.0           # BOP depth: 16-bit PNG, 0.1mm Einheiten
depth_trunc = 10.0              # Max Tiefe in Metern

# ULIP-2
ulip2_backbone = "pointbert_colored"
ulip2_num_points = 10000
ulip2_embed_dim = 1280

# Fusion
weight_clip = 0.3
weight_dino = 0.4
weight_ulip = 0.3

# Ollama
ollama_host = "http://localhost:11434"
ollama_model = "gemma3:4b"

# Pose
pose_method = "icp"
foundationpose_url = "http://foundationpose:5050"
```

---

## Architektur

```
Prompt + RGB-D Image
       │
       ▼
┌──────────────────┐
│ 1. Lokalisierung │ GroundingDINO + SAM → Maske + BBox
└────────┬─────────┘
    ┌────┴────────────────┐
    ▼                     ▼
┌──────────┐    ┌──────────────────┐
│ 2. Point │    │ 3. CLIP Retrieval│ Prompt → Text Embeddings
│   Cloud  │    │    → Top-20      │
└────┬─────┘    └────────┬─────────┘
     │                   ▼
     │          ┌──────────────────┐
     │          │ 4. DINOv2 ReRank │ ROI → Image Embeddings
     │          │    → Top-5       │ (Batch + Disk-Cache)
     │          └────────┬─────────┘
     ▼                   │
┌──────────┐             │
│ 5. ULIP-2│ PC Embed    │
│  → Top-5 │ (NaN-safe)  │
└────┬─────┘             │
     └───────┬───────────┘
             ▼
    ┌──────────────────┐
    │ 6. Score Fusion  │ Weighted Sum (0.3 / 0.4 / 0.3)
    │    → Top-1       │ (NaN-safe Min-Max Norm.)
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │ 7. Scale Est.    │ RANSAC+ICP → Partial-Aware Scale
    └────────┬─────────┘ (2 beste Achsen)
             ▼
    ┌──────────────────┐           ┌──────────────────────┐
    │ 8. Pose Est.     │── HTTP ──>│ FoundationPose       │
    │  (OSCAR cont.)   │<── JSON ──│ (separate container)  │
    └──────────────────┘           └──────────────────────┘
      ↓ fallback: ICP              → 4×4 Pose Matrix
```

---

## How to Run

### Container starten
```bash
docker compose up -d foundationpose   # optional: start FP service
docker compose run --rm -it oscar bash
```

### Debug-Modus (empfohlen zum Testen)
```bash
./scripts/run_debug_pipeline_foundationpose.sh
# → debug_output/debug_01_localization.png ... debug_07_scale_pose.png

# oder manuell:
python3.11 -m pipeline.run_pipeline \
    --rgb eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
    --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
    --prompt "i need the blue coffee can" \
    --descriptions object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json \
    --reference_images object_images/ycbv_gso/ \
    --cad_models object_database/ycbv_gso/ \
    --camera eval/datasets/ycbv_gso/test/000048/scene_camera.json \
    --ulip_repo /ulip \
    --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt \
    --output debug_output \
    --debug-viz --until-step 8
```

### Volle Pipeline
```bash
./scripts/run_pipeline.sh
# oder:
python3.11 -m pipeline.run_pipeline \
    --rgb eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
    --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
    --prompt "pick up the mustard bottle" \
    --descriptions object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json \
    --reference_images object_images/ycbv_gso/ \
    --cad_models object_database/ycbv_gso/ \
    --camera eval/datasets/ycbv_gso/test/000048/scene_camera.json \
    --ulip_repo /ulip \
    --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt
```

### OSCAR Baseline (Vergleich)
```bash
python retrieval_combi_eval.py  # → 75.95% Top-1
```

---

## Bekannte Bugs & Workarounds

| Problem | Lösung | Datei |
|---|---|---|
| `knn_cuda` nicht installierbar | try/except + Warning | `ULIP/models/pointbert/dvae.py` |
| `pointnet2_ops` nicht installierbar | Optional import + `_fps_pytorch()` | `ULIP/models/pointbert/misc.py` |
| PyTorch 2.6 `weights_only` Error | `torch.load(..., weights_only=False)` | `step5_shape_matching.py` |
| `np.asarray(pcd.colors)` Overflow | `np.clip(raw, 0.0, 1.0)` | `step5_shape_matching.py` |
| NaN in ULIP Similarity Scores | `torch.where(nan_mask, -1.0, sims)` | `step5_shape_matching.py` |
| NaN in Fusion Min-Max Norm. | Filter NaN vor min/max | `step6_fusion.py` |
| Camera intrinsics KeyError | Fallback auf ersten Key | `pipeline/utils.py` |
| BOP `depth_scale` multiplier vs divisor mismatch | Always use `config.depth_scale` as divisor; ignore `scene_camera.json` field (it uses multiplier 0.1, not divisor) | `pipeline/run_pipeline.py` |
| Stale .pyc im Docker | `rm -rf /app/pipeline/__pycache__` nach Edits | manuell |

---

## Baseline-Ergebnisse (exp/oscar-repro)

| Datensatz | Methode | Top-1 Acc. | Paper | Anmerkung |
|---|---|---|---|---|
| YCBV_GSO | OSCAR full pipeline | **75.95%** | ~60% | GT-Masken statt GroundedSAM |
| MI3DOR | OSCAR full pipeline | NN=77.95% | NN=89.4% | Descriptions nur 10/21 Kat. |
