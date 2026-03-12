# AI Handoff – Branch `exp/ulip2`

> Zuletzt aktualisiert: 2026-03-12

---

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
| **`exp/ulip2`** | **Shape-Aware Pipeline (dieser Branch)** | 🟢 End-to-End funktioniert (8 Schritte) |

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
  8. `step8_pose_estimation.py` – ICP mit Coarse-Alignment aus Step 7 als Startpose
     - fitness=0.9895, RMSE=0.007m

- **Debug-Visualisierung** (`pipeline/debug_steps.py`, ~1200 Zeilen):
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
  - Image: `tholoi/oscar-plus`
  - GPU-Support via `deploy.resources.reservations.devices`
  - ULIP-Volume: `../ULIP:/ulip`
  - HuggingFace Cache-Volume
  - Alle Python-Dependencies inkl. trimesh

### Bekannte Limitierungen

1. **ULIP-2 Shape Matching** liefert schwache Ergebnisse für partielle Punktwolken (single-view, ~4k Punkte vs. komplette 10k-CAD-Modelle). Fusion kompensiert durch CLIP+DINO.
2. **ICP auf symmetrischen Objekten** (z.B. Dosen) kann Rotation um Symmetrieachse nicht eindeutig bestimmen.
3. **FoundationPose** ist als `NotImplementedError` markiert – Pipeline nutzt ICP als Default.

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
| `pipeline/run_pipeline.py` | ~750 | Orchestrator + CLI + LLM-Parsing |
| `pipeline/debug_steps.py` | ~1200 | **Debug-Visualisierung** (7 PNGs + 3D-Viewer) |
| `pipeline/step1_localization.py` | ~240 | GroundingDINO + SAM |
| `pipeline/step2_pointcloud.py` | ~280 | RGB-D → Point Cloud |
| `pipeline/step3_clip_retrieval.py` | ~280 | CLIP Text-/Bild-Retrieval |
| `pipeline/step4_dino_reranking.py` | ~350 | DINOv2 Re-Ranking + Batch-Cache |
| `pipeline/step5_shape_matching.py` | ~680 | ULIP-2 Encoder + NaN-Filterung |
| `pipeline/step6_fusion.py` | ~370 | Score-Fusion (weighted_sum, RRF, intersection) |
| `pipeline/step7_scale_estimation.py` | ~300 | RANSAC+ICP Coarse-Alignment + Partial-Aware Scale |
| `pipeline/step8_pose_estimation.py` | ~460 | ICP mit initial_pose Support |
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
ollama_timeout = 30.0

# Pose
pose_method = "icp"             # FoundationPose ist NotImplemented
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
    ┌──────────────────┐
    │ 8. Pose Est.     │ ICP mit Coarse-Alignment als Startpose
    └──────────────────┘ → 4×4 Pose Matrix
```

---

## How to Run

### Container starten
```bash
docker compose build
docker compose run --rm -it oscar bash
```

### Debug-Modus (empfohlen zum Testen)
```bash
python -m pipeline.debug_steps \
    --prompt "i need the blue coffee can" \
    --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt \
    --until_step 8
# → debug_output/debug_01_localization.png ... debug_07_scale_pose.png
```

### Volle Pipeline
```bash
python -m pipeline.run_pipeline \
    --rgb eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
    --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
    --prompt "pick up the mustard bottle" \
    --descriptions object_database/ycbv_gso/descriptions_attributes.json \
    --reference_images object_images/ycbv_gso/ \
    --cad_models object_database/ycbv_gso/ \
    --camera_json eval/datasets/ycbv_gso/test/000048/scene_camera.json \
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
| FoundationPose Fallback ohne initial_pose | `initial_pose` durchreichen | `step8_pose_estimation.py` |
| Camera intrinsics KeyError | Fallback auf ersten Key | `pipeline/utils.py` |
| Stale .pyc im Docker | `rm -rf /app/pipeline/__pycache__` nach Edits | manuell |

---

## Baseline-Ergebnisse (exp/oscar-repro)

| Datensatz | Methode | Top-1 Acc. | Paper | Anmerkung |
|---|---|---|---|---|
| YCBV_GSO | OSCAR full pipeline | **75.95%** | ~60% | GT-Masken statt GroundedSAM |
| MI3DOR | OSCAR full pipeline | NN=77.95% | NN=89.4% | Descriptions nur 10/21 Kat. |