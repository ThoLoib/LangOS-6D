# AI Handoff – Branch `exp/ulip2`

> Zuletzt aktualisiert: 2026-03-05

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
| **`exp/ulip2`** | **Shape-Aware Pipeline (dieser Branch)** | 🟡 implementiert, End-to-End-Test ausstehend |

---

## Aktueller Stand (exp/ulip2)

### Was ist fertig

- **Modulare 8-Schritt-Pipeline** in `pipeline/` (17 Dateien):
  1. `step1_localization.py` – GroundingDINO + SAM → Maske + BBox
  2. `step2_pointcloud.py` – RGB-D + Maske → Open3D Point Cloud
  3. `step3_clip_retrieval.py` – Prompt → CLIP → Top-K Kandidaten
  4. `step4_dino_reranking.py` – ROI → DINOv2 → Re-Ranking
  5. `step5_shape_matching.py` – **ULIP-2 Point Cloud Encoder** → Shape-Similarity
  6. `step6_fusion.py` – Weighted Sum / RRF / Intersection
  7. `step7_scale_estimation.py` – BBox-Vergleich → Skalenfaktor
  8. `step8_pose_estimation.py` – FoundationPose / ICP

- **ULIP-2 Integration** (step5):
  - Lädt nur `point_encoder` + `pc_projection` (~400 MB statt ~5.5 GB für volles OpenCLIP)
  - Backbone: PointBERT Colored (10k Punkte xyzrgb → 1280-dim Embedding)
  - Checkpoint: `ulip2_pointbert_10k.pt` in `/ulip/checkpoints/`
  - ULIP-Repo als Volume gemountet (`../ULIP:/ulip` im Container)

- **ULIP-Patches** (im ULIP-Repo, nicht in OSCAR):
  - `ULIP/models/pointbert/dvae.py` → `knn_cuda` optional per try/except
  - `ULIP/models/pointbert/misc.py` → `pointnet2_ops` optional, Pure-PyTorch `_fps_pytorch()` Fallback

- **LLM-basiertes Prompt Parsing** (run_pipeline.py):
  - Ollama + `mistral-small3.1` → extrahiert Objektname aus natürlichem Prompt
  - Fallback: regelbasierter Heuristic-Parser wenn LLM nicht verfügbar

- **Visualisierung** (`pipeline/visualization.py`):
  - `--visualize` Flag erzeugt PNG-Bilder nach jedem Schritt
  - Masken-Overlay, Depth-Maps, Point-Cloud-Projektionen, Top-K-Grids, Summary

- **Docker-Konfiguration**:
  - GPU-Support via `deploy.resources.reservations.devices`
  - ULIP-Volume: `../ULIP:/ulip`
  - Ollama im Container installiert + `mistral-small3.1` gepullt beim Start
  - Alle Python-Dependencies in `requirements.txt`

### Was noch fehlt

1. **End-to-End Test** – Pipeline komplett durchlaufen (alle 8 Schritte) und Ergebnis verifizieren.
2. **Evaluation-Script** – Analog zu `retrieval_combi_eval.py` (OSCAR baseline), aber mit ULIP-2: über alle BOP-Szenen laufen, Top-K Accuracy berechnen, mit 75.95% Baseline vergleichen.
3. **MI3DOR Evaluation** – Shape-Retrieval auf MI3DOR testen (hier sollte ULIP-2 besonders helfen).
4. **HouseCat6D** – BOP-Testdaten beschaffen + evaluieren.
5. **Hyperparameter-Tuning** – Fusionsgewichte (aktuell 0.3/0.4/0.3), Top-K je Schritt.

---

## Datei-Inventar

### Neue Dateien (pipeline/)

| Datei | Zeilen | Beschreibung |
|---|---|---|
| `pipeline/__init__.py` | 21 | Package-Init mit Version `0.1.0` |
| `pipeline/config.py` | 133 | Zentrale `PipelineConfig` Dataclass (alle Hyperparameter) |
| `pipeline/run_pipeline.py` | 747 | Orchestrator + CLI (`argparse`) + LLM-Parsing |
| `pipeline/step1_localization.py` | ~240 | GroundingDINO + SAM Wrapper |
| `pipeline/step2_pointcloud.py` | ~280 | RGB-D → Open3D Point Cloud |
| `pipeline/step3_clip_retrieval.py` | ~280 | CLIP Text-/Bild-Retrieval |
| `pipeline/step4_dino_reranking.py` | ~320 | DINOv2 Re-Ranking |
| `pipeline/step5_shape_matching.py` | 669 | **ULIP-2 Encoder** + Shape Matching |
| `pipeline/step6_fusion.py` | ~380 | Score-Fusion (weighted_sum, RRF, intersection) |
| `pipeline/step7_scale_estimation.py` | ~280 | BBox-/PC-basierte Skalenbestimmung |
| `pipeline/step8_pose_estimation.py` | ~450 | FoundationPose / ICP Wrapper |
| `pipeline/utils.py` | 142 | Hilfen: `crop_with_mask`, `load_depth_image`, `load_camera_intrinsics`, `load_object_descriptions`, `ensure_dir` |
| `pipeline/visualization.py` | 375 | Viz für alle Schritte + Summary |
| `pipeline/requirements.txt` | ~45 | Pipeline-spezifische Requirements |

### Modifizierte Dateien

| Datei | Änderung |
|---|---|
| `docker-compose.yml` | Volume `../ULIP:/ulip`, GPU `deploy.resources.reservations` |
| `requirements.txt` | +`ollama`, `open3d`, `easydict`, `timm`, `pyyaml_env_tag`, `termcolor` |

### ULIP-Patches (im ULIP-Repo, NICHT in OSCAR)

| Datei | Änderung |
|---|---|
| `ULIP/models/pointbert/dvae.py` | `from knn_cuda import KNN` → try/except mit Warning |
| `ULIP/models/pointbert/misc.py` | `pointnet2_ops` optional + `_fps_pytorch()` Fallback |

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
     │          │    → Top-5       │
     │          └────────┬─────────┘
     ▼                   │
┌──────────┐             │
│ 5. ULIP-2│ PC Embed    │
│  → Top-5 │             │
└────┬─────┘             │
     └───────┬───────────┘
             ▼
    ┌──────────────────┐
    │ 6. Score Fusion  │ Weighted Sum (0.3 / 0.4 / 0.3)
    │    → Top-1       │
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │ 7. Scale Est.    │ BBox-Vergleich PC vs CAD
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │ 8. Pose Est.     │ FoundationPose / ICP → 4×4 Pose Matrix
    └──────────────────┘
```

---

## Konfiguration (config.py)

Wichtige Felder und Defaults:

```python
# ULIP-2
ulip_repo_path = ""            # Im Container: "/ulip"
ulip2_checkpoint = ""          # Im Container: "/ulip/checkpoints/ulip2_pointbert_10k.pt"
ulip2_backbone = "pointbert_colored"
ulip2_num_points = 10000
ulip2_use_colors = True
ulip2_embed_dim = 1280         # ViT-bigG-14 aligned

# Fusion
weight_clip = 0.3
weight_dino = 0.4
weight_ulip = 0.3
fusion_top_k = 1

# Ollama (Prompt Parsing)
ollama_host = "http://localhost:11434"
ollama_model = "mistral-small3.1"
ollama_timeout = 5.0
```

---

## How to Run

### 1. Container starten

```bash
docker compose build          # Dependencies + Ollama installieren
docker compose run --rm -it oscar bash
```

### 2. Pipeline ausführen (einzelnes Bild)

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
    --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt \
    --visualize
```

### 3. OSCAR Baseline (zum Vergleich)

```bash
cd /app/object_retrieval
python retrieval_combi_eval.py
# → results_topk_eval_ycbv_gso/accuracy_summary_topk_15.json
```

---

## Daten-Layout (lokal, gitignored)

| Pfad | Beschreibung | Größe |
|---|---|---|
| `eval/datasets/ycbv_gso/test/` | 12 BOP-Szenen (000048–000059), je rgb/, depth/, mask_visib/, scene_camera.json, scene_gt.json | ~900 Bilder |
| `eval/datasets/mi3dor/image/test/` | MI3DOR Testbilder (21 Kategorien × 500) | 10.500 |
| `object_database/ycbv_gso/` | 1051 CAD-Modelle (OBJ) + `descriptions_attributes.json` | ~2 GB |
| `object_database/MI3DOR/` | 3848 CAD-Modelle + Descriptions (10/21 Kat.) | ~3 GB |
| `object_database/housecat6d/` | 194 CAD-Modelle + Descriptions | ~500 MB |
| `object_images/ycbv_gso/` | Gerenderte Referenzbilder (1050 × 8 Views) | ~4 GB |
| `object_images/MI3DOR/` | Gerenderte Referenzbilder (3848 Objekte) | ~5 GB |
| `/ulip/checkpoints/ulip2_pointbert_10k.pt` | ULIP-2 Checkpoint (PointBERT Colored, 10k) | ~402 MB |

---

## Baseline-Ergebnisse (exp/oscar-repro)

| Datensatz | Methode | Top-1 Acc. | Paper | Anmerkung |
|---|---|---|---|---|
| YCBV_GSO | OSCAR full pipeline | **75.95%** | ~60% | GT-Masken statt GroundedSAM |
| MI3DOR | OSCAR full pipeline | NN=77.95% | NN=89.4% | Descriptions nur 10/21 Kat. |

---

## Bekannte Bugs & Workarounds

| Problem | Lösung | Datei |
|---|---|---|
| `knn_cuda` nicht installierbar (GPU-Build) | try/except + Warning | `ULIP/models/pointbert/dvae.py` |
| `pointnet2_ops` nicht installierbar | Optional import + `_fps_pytorch()` | `ULIP/models/pointbert/misc.py` |
| PyTorch 2.6 `weights_only` Error | `torch.load(..., weights_only=False)` | `pipeline/step5_shape_matching.py` |
| Camera intrinsics KeyError | Fallback auf ersten Key | `pipeline/utils.py` |
| Ollama-Modell default war `llama3.2` | Korrigiert zu `mistral-small3.1` | `pipeline/config.py` |

---

## Offene Fragen / Risiken

- **End-to-End-Test** noch nicht vollständig durchgelaufen — bisher nur Einzelschritte debugged.
- **Fusionsgewichte** (0.3/0.4/0.3) sind Initial-Werte, nicht tuned.
- **MI3DOR Descriptions** unvollständig (10/21 Kategorien) — beeinflusst CLIP-Kanal.
- **HouseCat6D BOP-Testdaten** nicht vorhanden — kein HouseCat6D eval möglich.
- **GroundingDINO + SAM** noch nicht mit LangSAM getestet (Step 1 braucht Modelle).
- **Evaluation-Script** fehlt — muss über alle Szenen loopen und Top-K Accuracy berechnen.

---

## Key Decisions

- ULIP-2 Checkpoint wird nur partiell geladen (point_encoder + pc_projection), nicht das volle Modell (~5.5 GB mit OpenCLIP).
- Ollama läuft im selben Container, `start.sh` pullt beim Start.
- Pure-PyTorch Fallbacks statt knn_cuda/pointnet2_ops (etwas langsamer, aber portabel).
- Pipeline ist modular: jeder Schritt hat eigene Klasse, eigenes Result-Dataclass, eigene Tests möglich.
