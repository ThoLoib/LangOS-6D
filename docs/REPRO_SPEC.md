# Reproduktions-Spezifikation — jeder Konfigurationswert, je Lauf

*Die vollständige Grundlage für die späteren Repro-Skripte: **jeder** Wert, der ein
Ergebnis beeinflusst, mit Quelle. Aufgebaut in drei Schichten, weil genau so auch die
CLI-Skripte aufgebaut sein werden:*

```
Schicht 1  PipelineConfig-Defaults      (pipeline/config.py — gilt überall)
Schicht 2  Experiment-Overrides         (EvalConfig + pipeline_overrides je Treiber)
Schicht 3  Lauf-Variablen               (Env-Variablen + CLI-Flags je Aufruf)
```

*Ein künftiges Skript `repro.py --experiment stage1 --arm E1c_full_fusion` muss genau
Schicht 2 + 3 als Argumente tragen; Schicht 1 liegt im Code. Erstellt 2026-09-06 aus dem
Code, nicht aus dem Gedächtnis.*

---

## Schicht 1 — PipelineConfig-Defaults (gelten überall, sofern nicht überschrieben)

Quelle: `pipeline/config.py`, 103 Felder. Hier vollständig, gruppiert:

### Segmentierung (Schritt 1)
| Feld | Wert |
|---|---|
| `grounding_dino_model` | `IDEA-Research/grounding-dino-base` |
| `sam_model` | `facebook/sam2.1-hiera-large` |
| `detection_confidence` | 0.3 |
| `mask_largest_cc` | True |
| `mask_dilation_kernel` / `_iterations` | 5 / 1 |

### Punktwolke (Schritt 2)
| Feld | Wert |
|---|---|
| `camera_fx/fy/cx/cy` | 591.0 / 591.0 / 320.0 / 240.0 *(nur Fallback; BOP-Läufe lesen `scene_camera.json`)* |
| `depth_scale` / `depth_trunc` | 10000.0 / 2.0 |
| `voxel_size` | 0.002 |
| `depth_gate_enabled` / `_tolerance` | True / 0.3 |
| `sor_nb_neighbors` / `sor_std_ratio` | 10 / 1.0 |
| `ror_enabled` | False |

### S_text — CLIP (Schritt 3)
| Feld | Wert |
|---|---|
| `clip_model_name` | `ViT-B/32` |
| `clip_top_k` / `clip_threshold` | 20 / 0.25 |
| Objekt-Score | **max** über die 42 Per-View-Beschreibungen |

### S_view — DINOv2 (Schritt 4)
| Feld | Wert |
|---|---|
| `appearance_encoder` / `dino_model_name` | `dinov2` / `facebook/dinov2-base` |
| `dino_pooling` | **`cls` (Default!)** — alle Experimente überschreiben auf `mean` |
| `num_views` | 42 |
| `dino_view_aggregation` / `_topk` / `_temperature` | topk_softmax / 5 / 0.5 |

### S_shape — ULIP-2 / Uni3D (Schritt 5)
| Feld | Wert |
|---|---|
| `shape_encoder` | `ulip2` |
| `ulip2_backbone` / `_embed_dim` | `pointbert_colored` / 1280 |
| `ulip2_checkpoint` | `/ulip/checkpoints/ulip2_pointbert_10k.pt` *(via EvalConfig)* |
| `ulip2_num_points` / `_jitter_std` | 10000 / 0.001 |
| `ulip2_use_colors` | True |
| `ulip2_mode` | `cross` *(Stage 1/3-pc überschreiben auf `pc`)* |
| `ulip2_use_partial_views` | **False (Default!)** — Partial muss aktiv gesetzt werden |
| `ulip_view_aggregation` / `_topk` / `_temperature` | topk_softmax / 5 / 0.5 |
| `uni3d_model_name` / `_checkpoint` | `uni3d-g` / `/uni3d/modelzoo/uni3d-g/model.pt` |
| `uni3d_num_points` / `_embed_dim` | 10000 / 1024 |
| ULIP-XYZ-Variante (O5) | `pointbert`, 8192 Punkte, 512-d, ohne Farbe |

### Fusion (Schritt 6)
| Feld | Wert |
|---|---|
| `fusion_method` | `weighted_sum` (min-max-normiert) |
| `weight_clip` / `weight_dino` / `weight_ulip` | **0.3 / 0.4 / 0.3** |
| ⚠ EvalConfig-Default | **0 / 0.5 / 0.5** — Treiber MÜSSEN die Gewichte explizit setzen |

### Geometrie (Sub-Schritt B2)
| Feld | Wert |
|---|---|
| `geometry_reranking_signal` | `chamfer_ransac` |
| `geometry_reranking_top_k` | 5 *(Stage 1 überschreibt per `--geom-k 50`)* |
| `chamfer_trim_ratio` | 0.1 |
| `geometry_skip_icp` | False |
| dGeDi-Repo-Parameter | 6000 Keypoints, 10000 RANSAC-Iterationen, ICP an |
| `GEOM_VOXEL` (Stage 1) | 0.02 (Einheitskugel-Skala); Inlier-Schwelle = 1.5 %·Durchmesser |
| Alt-GeDi (ungenutzt seit dGeDi) | `gedi_dim=32, r_lrf=0.5, num_keypoints=5000` |

### Skala + Pose (Schritte 7–8)
| Feld | Wert |
|---|---|
| `scale_gate_enabled` | **False** — Schritt 7 ist verworfen, läuft nirgends |
| `pose_method` | `icp` als Fallback; Experimente nutzen `foundationpose` |
| `foundationpose_url` / `_est_refine_iter` | `http://foundationpose:5050` / 5 |
| `icp_max_iterations` / `_threshold` | 50 / 0.02 |
| Einheiten | FP rechnet in m; BOP-Meshes `scale=0.001`, Rückgabe t×1000 → mm |

### Determinismus
`PYTHONHASHSEED=0` in jedem Lauf · `seed=42` (PipelineConfig) · Stage-4-Sampling `--seed 0`
· Surface-Sampling geseedet. **Nicht bitreproduzierbar:** FoundationPose-Hypothesen und
open3d-RANSAC — deshalb speichert Stage 3 die rohen Posen in `records.json`.

---

## Schicht 2+3 — Preprocessing je Datensatz

Ein künftiges `repro_preprocess.py --dataset <ds> --step <step>` braucht genau diese Werte.

### P1 · Rendern — `rendering/rendering.py` (Blender, über Env-Variablen)
| Parameter | Wert | Anmerkung |
|---|---|---|
| Blender | **3.4.1** (`/home/tessa/Cap3D/…/blender-3.4.1-linux-x64/`) | 3.3.1 hat kein PIL und scheitert **still mit rc=0** |
| `OBJECT_FOLDER` / `OBJECT_IMAGES` | je Datensatz | siehe P-Tabelle unten |
| `NUM_VIEWS` | 42 | Ikosphäre Subdivision 1, **FPS-geordnet** → die ersten N sind ein gültiges N-View-Set |
| Kameradistanz | bbox-Maximum × **1.15** | |
| Renderer | CYCLES, **16 Samples**, GPU/CUDA, View-Transform `Standard` | |
| je View gespeichert | `<id>_<v>.png`, `<id>_bg.png`, `<id>_view<v>_CamMatrix.npy` | CamMatrix ist Pflicht für P2 |
| `SHARD_INDEX`/`SHARD_TOTAL` | 0/1 | nur Parallelisierung, ergebnisneutral |

### P2 · Teilwolken — `rendering/generate_partial_pointclouds.py`
| Parameter | Wert |
|---|---|
| `--num_points` | 10000 |
| `--hpr-param` | 2.8 *(MI3DOR-Renderings entstanden mit 3.2/0 — siehe Provenienz-Memory)* |
| `--jitter-std` | 0.001 |
| Verfahren | Mesh in Kameraframe rotieren → Hidden Point Removal je View |
| Aufruf | `--cad_dir <db>/<ds> --images_dir <img>/<ds>` (Konvention `<cad_dir>/<obj_id>/`); **`--mesh-glob` nur für MI3DOR-Layout** (obj_id = Dateistamm!) |

### P3 · Beschreibungen — `rendering/generate_descriptions.py`
| Parameter | Wert |
|---|---|
| Modell | `llava-hf/llava-1.5-7b-hf`, float16 |
| `--prompt` | "Extract visual attributes of the object in the image: object type, …" (Default im Skript) |
| `max_new_tokens` / `--batch-size` | 100 / 8 |
| `--images_dir` | Ordner **mit Objektunterordnern** — der Objektordner selbst ergibt „0 objects", rc=0 |

### P4 · Embeddings — `tools/precompute_embeddings.py --passes <p>`
| Pass | Kanäle | Modus | Gallery | Overrides |
|---|---|---|---|---|
| `base` | clip+dino+shape | cross | partial | — |
| `siglip` | dino | — | partial | `appearance_encoder=siglip` |
| `ulip_fullmesh` | shape | cross | **full-mesh** | — |
| `ulip_pc_rgb` | shape | pc | partial | nutzt den `base`-Cache |
| `ulip_pc_xyz` | shape | pc | partial | `pointbert`, 8192 pts, 512-d, ohne Farbe |
| `uni3d` | shape | pc | partial | Uni3D-g, 1024-d |

Cache-Fingerprint: Encoder-Config + **Dateigrößen** (nicht Inhalte!) — nach dem Farb-Fix
mussten die Full-Mesh-Caches deshalb von Hand beiseite (siehe `RUN_PROVENANCE.md`).

### P5 · GeDi-Deskriptoren — `dgedi_service/precompute_gallery.py`
| Parameter | Wert |
|---|---|
| Aufruf | `docker compose run --rm --no-deps dgedi python3 /oscar/dgedi_service/precompute_gallery.py --manifest <m> --out <dir>` |
| `--n-points` / `--mode` | 10000 / `multi_scale` |
| Manifest | `{namespaced_id: mesh_relpath}` — **muss im Repo liegen** (Container mountet nur `.:/oscar`) |
| **NICHT** | `tools/precompute_gedi_descriptors.py` — zielt auf den toten Alt-`gedi`-Dienst |

### P-Tabelle · Datensatz-Pfade und Eigenheiten
| Datensatz | CADs | Mesh-Pfad | ID-Regel | Farbe im Mesh |
|---|---|---|---|---|
| shrec18_v2 | 3308 | `eval/datasets/shrec18/shrec18_full/cad/*.obj` | stem | Textur (≈70 % auslesbar) |
| MI3DOR | 3848 | `object_database/MI3DOR/model/test/*/*.obj` | **Dateistamm** | **keine** (uniform 0.4) |
| ycbv | 21 | `object_database/ycbv/*/textured_simple.obj` | parent | Textur |
| tless | 30 | `object_database/tless/*/model.ply` | parent | **keine** |
| lmo | 8 | `object_database/lmo/*/model.ply` | parent | Vertexfarben |
| gso | 1030 | `object_database/gso/*/meshes/model.obj` | grandparent | Textur; **Einheit: m** |
| housecat6d | 199 | `object_database/housecat6d/*/*.obj` | **stem** (Kategorie-Ordner!) | fast keine |
| itodd | 28 | `object_database/itodd/*/model.ply` | parent | keine; Einheit: mm |

⚠ Für Full-Mesh-Embeddings gilt `_FULLMESH_ID_MODE` in `stage3_gallery.py` — **nicht**
`DATASET_LAYOUT["id_mode"]` (HouseCat6D-Kollaps, `AI_LOG` 2026-08-31). Farb-Sampling seit
2026-09-01 via `face_colors → to_color().vertex_colors → None`.

---

## Schicht 2+3 — Experimente

### E1 · Stage 1 — `experiments/experiment1_shrec18_stage1.py`

**Feste Pfad-Flags** (jeder Aufruf):
```
--data-root eval/datasets/shrec18/shrec18_full
--images-dir object_images/shrec18_v2
--desc-file object_database/shrec18_v2/descriptions_attributes.json
--results-root object_retrieval/results_shrec18_v2_stage1_42v_k5
```

**Overrides gegenüber Schicht 1:** `BASE_WEIGHTS=(0.3,0.4,0.3)` · `SHAPE_AGG_VIEWS=42` ·
`TOP_F=20` · Geometrie: `--with-geometry --geom-k 50`, `GEOM_VOXEL=0.02` ·
`clip/dino/ulip/fusion_top_k=10^6` (volle Datenbank) · `num_views=None` beim Scoring
(alle Views cachen, Trimmen zur Ableitungszeit).

**Lauf-Variablen (Schicht 3) — die vier Fallen:**
| Variable | Wert | Regel |
|---|---|---|
| `SHREC_DINO_POOLING` | `mean` | immer |
| `SHREC_FORCE_PARTIAL_CACHE` | Cache-Datei je Encoder | **NUR bei Partial-Pässen** (fehlt → still Full-Mesh; bei Full-Mesh gesetzt → still Partial). Coloured `…c3b88090d599c522.pt`, XYZ `…641102dfbaf4e90c.pt`, Uni3D `…eabcf9b9096553c9.pt` |
| `STAGE1_GEOMETRY_BACKEND` | `dgedi` | bei Geometriearmen (sonst still übersprungen, rc=0) |
| `DGEDI_CACHE_DIR` | `.dgedi_gallery_shrec` | dGeDi vorher per `docker compose up -d --force-recreate dgedi` umschalten, `n_gallery==3308` prüfen; danach zurück |

**Ein Arm = `--ablations <name>`.** Die Arm→Pass/Gewichte/Geometrie-Zuordnung steht
generiert in `CONFIG_TO_RESULT.md`; die Pass-Definitionen in `PASS_DEFS` des Treibers.

### E2 · Stage 2 — `object_retrieval/retrieval_mi3dor_eval_oscarplus.py`

**EvalConfig (explizit im Treiber):** `clip_prune_mode=threshold, clip_tau=0.37,
clip_fallback_k=20, clip_top_k=20, TOP_F=20, dino/ulip/fusion_top_k=9999,
weights=0.3/0.4/0.3, ulip_query_cache_path=ulip_query_cache_mi3dor.pt`.
`pipeline_overrides`: 42 Views, topk_softmax k=5 τ=0.5 (beide View-Kanäle), DINO mean.

**Lauf-Variablen:**
| Variable | Default | gültige Werte |
|---|---|---|
| `MI3DOR_MODES` | — | `fullmesh` *(das einzige, was auf dieser Maschine real läuft — keine `*_partial.npz`)* |
| `MI3DOR_NUM_VIEWS` | 42 | 8 für den OSCAR-Legacy-Vergleich |
| `MI3DOR_DINO_POOLING` | mean | |
| `MI3DOR_RESULT_FOLDER` | `results_mi3dor_oscarplus_v2_tau037` | |
| `MI3DOR_MAX_QUERIES_PER_CAT` | 0 (= alle, n=10500) | |

Sieben Arme fallen aus **einem** Pass: clip_only · dino_only_full · ulip_only_full ·
clip_dino_ulip_full · oscar_maxview · oscar_softmax · clip_pruned_dino_ulip.
Weight-Sweep: `mi3dor_weight_sweep.py`, Simplex-Schritt 0.05, Selbsttest FT@BASE≈0.682.

### E3 · Stage 3 — `object_retrieval/eval_bop_pose.py`

**Fest:** `--datasets all` (ycbv+tless+lmo, `test_targets_bop19`, n=12284, GT `mask_visib`)
· Gallery via `assemble_gallery` (G_proxy 1257 ∪ Ziel-CADs → 1316) · Gewichte 0.3/0.4/0.3
· 42 Views · topk_softmax k=5 τ=0.5 · DINO mean · `MIN_CLOUD_PTS`-Schwelle für pc-Query
· D_sym: 10000 Oberflächenpunkte, geseedet · F-Score bei 1 %/5 % Durchmesser.

**Achsen als CLI-Flags:**
| Flag | Wirkung |
|---|---|
| `--mode 3a\|3b\|3c` | Retrieval / Proxy-Pose / Zerlegung |
| *(kein Flag)* | cross-Query (ULIP-Bildturm) |
| `--pc-query` | Punktwolken-Query |
| `--fullmesh` | Full-Mesh-Gallery (IDs via `_FULLMESH_ID_MODE`, Deckungs-Gate ≥95 %) |
| `--oscar-baseline` | E5: τ=0.37-Kaskade → DINO Best-View, ohne Shape |
| `--dgedi --dgedi-repo --dgedi-top-k 5` | Geometrie (6000 kp / 10k RANSAC / ICP) |
| `--gt-records …/gt/combined_gt.json` | 3b — **Datei**, nicht Ordner |
| `--from-3a <dir>` | 3c-Quelle |
| Env `STAGE3_GEO_SIGNAL` | `distance` (Default) \| `fitness` \| `borda` |

dGeDi-Gallery: **BOP** (`.dgedi_gallery`, 1316) — Vorabprüfung `n_gallery`, sonst 17-h-Leerlauf.

### E4 · Stage 4 — `experiments/experiment4_{query_latency,onboarding}.py`

Wrapper: `scripts/stage4_query.sh` / `scripts/stage4_onboarding.sh`.

| | Query | Onboarding |
|---|---|---|
| Kern-Flags | `--dataset ycbv --n-queries 50 --views 16,42 --warmup 2 --seed 0` | `--stages mesh,partial,describe,embed --reuse-renders --num-views 16,42 --measure-invalidation --inv-sample 15` |
| optional | `--geometry --geo-k 5` · `--no-pose` · `--proxy-only` · `--refine-iter 5` | `--stages render` (Host, Blender 3.4.1, n=5) · `--stages dgedi` (Host, n=3) · `--num-points 8192` |
| Messung | CUDA-sync um jeden Schritt · Median/IQR/p95 · kalt/warm getrennt · Kanäle via Methoden-Wrapping | dito; Cache-Anhängen echt (load+insert+save); Invalidierung = Stückkosten × 1257 |
| Prompt | erster Satz der gespeicherten Beschreibung, LLaVA-Floskel entfernt | — |

---

## Für die künftigen Repro-Skripte

Aus dieser Spezifikation folgt die Skript-Struktur direkt:

```
repro_preprocess.py --dataset ycbv --step render|partial|describe|embed|dgedi
repro_experiment.py --stage 1 --arm E1c_full_fusion
repro_experiment.py --stage 3 --mode 3a --query cross --gallery fullmesh [--geo distance]
repro_experiment.py --stage 4 --side query|onboarding [--views 16,42]
```

Drei Anforderungen an die Skripte, gelernt aus den Fehlern dieser Evaluation:
1. **Schicht 3 wird vom Skript gesetzt, nie vom Aufrufer erwartet** — die vier
   Stage-1-Variablen und die dGeDi-Gallery sind abgeleitete Größen (aus Arm-Name und
   Datensatz), keine freien Parameter.
2. **Jede Stufe prüft ihr Ergebnis, nicht den Rückgabewert** (Blender rc=0 bei Fehler,
   „0 objects" bei falschem `--images_dir`, Geometrie „skipped" bei totem Backend).
3. **Jeder Lauf schreibt seine volle Konfiguration ins Ergebnis** — auch die
   Env-Variablen, die heute fehlen. Dann kann `CONFIG_TO_RESULT.md` vollständig
   generiert werden statt teilweise von Hand.

### Anforderung 4 — Kanal-Scores persistieren

Zwischen zwei Fusionsvarianten ändert sich oft nur **ein** Kanal; die anderen sind
bitgleich. Trotzdem kostet heute jede Variante einen vollen Lauf, weil nirgends die
Score-Matrizen abgelegt werden. Die Ergebnisdateien halten je Query nur die Top-5 je Kanal
plus die Positionen der relevanten Objekte — daraus lässt sich **keine** neue Fusion ableiten.

`mi3dor_weight_sweep.py` macht bereits das Richtige (jede Query einmal scoren, dann 231
Gewichtungen im Speicher fusionieren) und wirft den Cache am Ende weg.

**Anforderung:** `repro_experiment.py` schreibt je Lauf die drei normalisierten
Per-Kanal-Score-Maps als Artefakt neben die Metriken. Dann sind Gewichts-Sweeps, Fusionsmethoden
und einzelne Kanaltausche eine Ableitung von Minuten statt eines Laufs von Stunden.

*Konkreter Anlass:* der fusionierte MI3DOR-Partial-Arm (2026-09-06) brauchte ~5 h, obwohl sich
gegenüber dem Full-Mesh-Lauf nur der ULIP-Kanal ändert und dessen beide Eingänge — Gallery-Cache
und Query-Embeddings — fertig auf der Platte lagen.
