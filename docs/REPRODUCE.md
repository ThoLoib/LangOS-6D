# Reproduktion — Terminalzeile je Ergebnis

Jede Zahl in `final_results/RESULTS.md` entsteht aus genau einem der folgenden
Aufrufe. Alle Befehle laufen **vom Repo-Root des Hosts**; was den
oscar-Container braucht, wrappt sich selbst (`docker compose run`). Die beiden
Einstiege sind reine Python-CLIs:

- `repro_preprocess.py` — Galerien bauen (einmalig je Datensatz)
- `repro_experiment.py` — Experimente; setzt alle stillen Env-Variablen selbst,
  schreibt `run_config.json` neben die Ergebnisse und druckt die Headline-Metrik
  inklusive des RESULTS.md-Referenzwerts.

Voraussetzungen: Daten laut `docs/DATASETS.md`; Dienste `foundationpose` (nur
Stage 3b/gt und Stage-4-Query) und `dgedi` (nur Geometrie-Läufe) gestartet.
Reproduktionsläufe schreiben in **neue** Ordner (`results_repro_*`) — die
Original-Ergebnisordner werden nie überschrieben.

Determinismus: Retrieval-Metriken (Stage 1, 2, 3a, Verdeckung) reproduzieren
exakt. FoundationPose-Hypothesen und RANSAC (3b/gt, Geometrie) sind nicht
bitreproduzierbar — dort gelten die publizierten `records.json` als Referenz,
Wiederholungsläufe treffen die Mediane auf ~Zehntel-mm.

---

## 0 · Preprocessing (einmalig je Datensatz; entfällt mit den Drive-Galerien)

Für jeden Datensatz `<ds>` ∈ {shrec18_v2, MI3DOR, ycbv, tless, lmo, gso,
housecat6d, itodd} in dieser Reihenfolge:

```
python3 repro_preprocess.py --dataset <ds> --step check      # was fehlt?
python3 repro_preprocess.py --dataset <ds> --step render     # Host, Blender 3.4.1
python3 repro_preprocess.py --dataset <ds> --step partial
python3 repro_preprocess.py --dataset <ds> --step describe
python3 repro_preprocess.py --dataset <ds> --step embed --passes base,ulip_fullmesh
```

Abweichungen der Evaluation (Provenienz, absichtlich beibehalten):
- **MI3DOR-Partialwolken**: `--step partial --hpr-param 3.2 --jitter-std 0`
- **shrec18_v2**: zusätzlich `--step embed --passes siglip,ulip_pc_rgb,ulip_pc_xyz,uni3d`
- **dGeDi-Galerien**: `python3 repro_preprocess.py --dataset shrec18_v2 --step dgedi`
  (→ `.dgedi_gallery_shrec`, 3308) und für Stage 3 die BOP-Gallery `.dgedi_gallery` (1316).

## 1 · Stage 1 — SHREC'18 (RESULTS.md §1)

Ein Arm = ein Aufruf; Arm-Namen exakt wie in RESULTS.md 1.8:

```
python3 repro_experiment.py --stage 1 --arm <ARM>
```

| RESULTS.md | `<ARM>` | erwartet |
|---|---|---|
| 1.1/1.4 BASE | `E1c_full_fusion` | nDCG 0.5868 / hit@1 0.341 |
| 1.1 stärkster Arm ohne Geometrie | `E2b_fullmesh` | 0.5935 / 0.360 |
| 1.5 C1 bestes Geometriesignal | `E2_chamfer_ransac` | 0.6405 / 0.472 |
| 1.1 stärkster Arm insgesamt | `E2b_fullmesh_geo` | 0.6417 / 0.481 |
| 1.3 A1 | `E1_view_only`, `E4_siglip_only` | 0.5506 / 0.5165 |
| 1.3 A3 | `E1_shape_only`, `E7_uni3d_shape_only` | 0.5353 / 0.5337 |
| 1.3 A4b cross-Zellen | `E7_ulip2_cross[…]` | s. Tabelle 1.8 |
| 1.4 B1 | `E6_rrf` | 0.5744 |
| 1.4 B3 Kaskade | `E1_oscar_cascade` | 0.4561 |
| … alle übrigen 40 Arme | Tabelle 1.8 | Spalten nDCG/hit@1 |

Geometrie-Arme (`E2_*`, `O1c/O1e`, `E2b_fullmesh_geo`) verlangen den
dGeDi-Dienst mit SHREC-Gallery; das Skript prüft das und nennt sonst den
Startbefehl. Sonderfälle:

```
# 1.2 Track-Vergleich (649 Test-Queries, identische BASE):
python3 repro_experiment.py --stage 1 --arm E1c_full_fusion --subset object_retrieval/shrec18_splits/test_split_ids.json
python3 repro_experiment.py --stage 1 --arm E2_chamfer_ransac --subset object_retrieval/shrec18_splits/test_split_ids.json
#   erwartet: nDCG 0.5945 bzw. 0.6434 (RESULTS.md 1.2)

# 1.4 B2 Gewichtskarten (66 Punkte je Modus; Treiber direkt, Sweep-Flags):
docker compose run --rm oscar python3 experiments/experiment1_shrec18_stage1.py --weight-sweep --weight-step 0.1 --sweep-shape-pass ulip_pc_rgb --results-root object_retrieval/results_repro_stage1
docker compose run --rm oscar python3 experiments/experiment1_shrec18_stage1.py --weight-sweep --weight-step 0.1 --sweep-shape-pass base --results-root object_retrieval/results_repro_stage1

# 1.5 C2 Shortlist-Tiefe: identischer Arm mit kleinerem K (nutzt den K=50-Registrierungs-Cache):
docker compose run --rm -e SHREC_DINO_POOLING=mean -e PYTHONHASHSEED=0 -e STAGE1_GEOMETRY_BACKEND=dgedi -e DGEDI_CACHE_DIR=.dgedi_gallery_shrec -e GEOM_VOXEL=0.02 oscar python3 experiments/experiment1_shrec18_stage1.py --data-root eval/datasets/shrec18/shrec18_full --images-dir object_images/shrec18_v2 --desc-file object_database/shrec18_v2/descriptions_attributes.json --ablations E2_chamfer_ransac --with-geometry --geom-k 20 --results-root object_retrieval/results_repro_stage1_k20
#   (K=5 analog mit --geom-k 5; erwartet 0.6279 bzw. 0.6022)

# 1.6 Per-Query-Bilanzen:
docker compose run --rm oscar python3 object_retrieval/paired_significance.py --results-root object_retrieval/results_shrec18_v2_stage1_42v_k5

# 1.7 Kategorien (Kanäle + Fusion-vs-bester-Einzelkanal):
python3 tools/compare_arms_by_category.py --preset channels
python3 tools/compare_arms_by_category.py E1c_full_fusion E2_chamfer_ransac
```

## 2 · Stage 2 — MI3DOR (RESULTS.md §2)

```
python3 repro_experiment.py --stage 2 --gallery partial     # BASE: NN 88.44 / FT 0.6918
python3 repro_experiment.py --stage 2 --gallery fullmesh    # NN 86.57 / FT 0.6818
python3 repro_experiment.py --stage 2 --gallery fullmesh --views 8   # OSCAR-Legacy (2.5): Fusion NN 86.62
python3 repro_experiment.py --stage 2 --sweep               # Gewichtskarte 2.3 (231 Punkte, Optimum FT 0.6902)
python3 tools/mi3dor_categories.py                          # Kategorientabelle 2.6 (9 von 21 negativ)
```

Ein Lauf erzeugt alle sieben Arme der Tabelle 2.1 (clip_only, dino_only_full,
ulip_only_full, clip_dino_ulip_full, oscar_maxview, oscar_softmax,
clip_pruned_dino_ulip) in `metrics_summary_topk_15.json`. Der partial-Lauf
erzwingt den Partial-Cache automatisch und ist damit gegen den stillen
Full-Mesh-Fallback abgesichert (RESULTS.md 2.4 „Wie dieser Lauf abgesichert ist“).

## 3 · Stage 3 — BOP (RESULTS.md §3)

```
# 3a Retrieval (Tabelle 3a, Zeilen 1–5):
python3 repro_experiment.py --stage 3 --mode 3a --query cross --gallery fullmesh   # R@1 0.5151
python3 repro_experiment.py --stage 3 --mode 3a --query cross --gallery partial    # R@1 0.4818
python3 repro_experiment.py --stage 3 --mode 3a --query pc    --gallery partial    # R@1 0.4636
python3 repro_experiment.py --stage 3 --mode 3a --query pc    --gallery fullmesh   # R@1 0.3878
python3 repro_experiment.py --stage 3 --mode 3a --query cross --oscar-baseline     # R@1 0.3198

# 3a Geometrie-Re-Ranking (vier Zellen):
python3 repro_experiment.py --stage 3 --mode 3a --query cross --geo distance       # 0.4229
python3 repro_experiment.py --stage 3 --mode 3a --query cross --geo fitness        # 0.4278
python3 repro_experiment.py --stage 3 --mode 3a --query pc    --geo distance       # 0.3725
python3 repro_experiment.py --stage 3 --mode 3a --query pc    --geo fitness        # 0.3820

# 3b Pose (FoundationPose-Dienst nötig; D_sym-Mediane):
python3 repro_experiment.py --stage 3 --mode gt                                    # 1.72 mm (Referenz)
python3 repro_experiment.py --stage 3 --mode 3b --query cross --gallery partial    # 18.37 mm
python3 repro_experiment.py --stage 3 --mode 3b --query cross --gallery fullmesh   # 18.91 mm
python3 repro_experiment.py --stage 3 --mode 3b --query cross --oscar-baseline     # 21.73 mm
python3 repro_experiment.py --stage 3 --mode 3b --query cross --geo distance       # 28.79 mm

# 3c Zerlegung (liest den jeweiligen 3a-Lauf):
python3 repro_experiment.py --stage 3 --mode 3c --query cross --from-3a results_repro_stage3            # 15.34 mm
python3 repro_experiment.py --stage 3 --mode 3c --query cross --gallery fullmesh --from-3a <3a-fullmesh-Ordner>  # 13.51 mm

# 3d Verdeckung (Re-Analyse, bricht bei Verknüpfungsfehler ab):
python3 tools/occlusion_analysis.py
```

## 4 · Stage 4 — Latenz (RESULTS.md §4)

```
python3 repro_experiment.py --stage 4 --side query                                  # 4.1: Anfrage 16/42 Views
python3 repro_experiment.py --stage 4 --side query --shape-source fullmesh          # 4.5: ulip 38 ms
python3 repro_experiment.py --stage 4 --side query --geometry                       # 4.1: +1.84 s (K=5)
python3 repro_experiment.py --stage 4 --side onboarding                             # 4.2: mesh…embed + Invalidierung (n=59)
python3 repro_experiment.py --stage 4 --side onboarding --shape-source fullmesh     # 4.5: Onboarding −8–10 %
python3 repro_experiment.py --stage 4 --side onboarding --stages render             # 4.2: Host, Blender, n=59
python3 repro_experiment.py --stage 4 --side onboarding --stages dgedi              # 4.2: Host, n klein
```

Latenzen sind hardwaregebunden (RTX 4090); reproduziert werden Größenordnungen
und Verhältnisse, nicht Millisekunden.

## 5 · End-to-End-Lauf (eine Anfrage, komplette Pipeline)

Eine Szene aus YCB-V gegen die Stage-3a-Gallery, bis zur Pose:

```
docker compose up -d foundationpose
docker compose run --rm oscar python3 -m pipeline.run_pipeline --rgb eval/datasets/ycbv/test/000048/rgb/000001.png --depth eval/datasets/ycbv/test/000048/depth/000001.png --camera eval/datasets/ycbv/test/000048/scene_camera.json --prompt "the blue coffee can" --descriptions object_database/ycbv/descriptions_attributes.json --reference_images object_images/ycbv/ --cad_models object_database/ycbv/ --ulip_repo /ulip --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt --ulip_mode cross --pose_method foundationpose
```

`--pose_method icp` läuft ohne FoundationPose-Dienst (Fallback). Erwartetes
Ergebnis des Beispiels: `best_model: obj_000001` (Master-Chef-Dose),
`pose_method: foundationpose`, ~20 s inklusive Kaltstart. Das Ergebnis einer
einzelnen freien Anfrage ist promptabhängig — die belastbaren Zahlen liefern
die Stage-Läufe oben.

## 6 · Eigene Gallery, eigene Queries

Eine neue Gallery ist ein Ordner nach Standard-Layout — CAD-Dateien unter
`object_database/<name>/<objekt_id>/<datei>.obj|.ply`. Dann genügen zwei Zeilen:

```
python3 repro_preprocess.py --dataset <name> --cad-dir object_database/<name> --mesh-glob "*/*.obj" --id-mode parent --step all
python3 repro_preprocess.py --dataset <name> --cad-dir object_database/<name> --mesh-glob "*/*.obj" --id-mode parent --step check
```

`--step all` fährt render → partial → describe → embed nacheinander (Render auf
dem Host via Blender, der Rest wrappt sich selbst in den Container) und prüft
jede Stufe. Optional `--views 16` für schnelleres Onboarding (Stage 4: halbe
Kosten für −0.005 nDCG).

Danach beliebige Queries gegen diese Gallery — `--gallery <name>` ersetzt die
drei Pfad-Flags:

```
docker compose run --rm oscar python3 -m pipeline.run_pipeline --gallery <name> --rgb <bild>.png --depth <tiefe>.png --camera <scene_camera.json> --prompt "..." --pose_method icp
```

Query-Format: RGB-PNG + Tiefen-PNG plus `scene_camera.json` im BOP-Format
(`cam_K` + `depth_scale`); ohne `--camera` gelten die Default-Intrinsics aus
`pipeline/config.py`. `--pose_method foundationpose` braucht den gestarteten
FP-Dienst, `icp` läuft ohne. Fehlt etwas an der Gallery, bricht die Pipeline
mit einer Anleitung ab statt still zurückzufallen.

Verifiziertes Beispiel: `object_database/demo_gallery/` (zwei YCB-V-Objekte,
exakt mit den zwei Zeilen oben gebaut, `--views 16`); die Query „the blue
coffee can" gegen Szene 48 liefert `best_model: coffee_can` in ~18 s (ICP).
Grenze freier Prompts: die Objekt-*Lokalisierung* (GroundingDINO) ist bei
kleinen/verdeckten Zielobjekten die schwächste Stufe — greift sie das falsche
Objekt, rankt das Retrieval den falschen Crop korrekt. Genau deshalb läuft die
Evaluation (Stage 3) mit GT-Masken; `pipeline_output/rankings_*.csv` zeigt je
Kanal, was das Retrieval gesehen hat.

## Übersichten regenerieren

```
python3 tools/results_overview.py -o docs/RESULTS_OVERVIEW.md
python3 tools/run_provenance.py   -o docs/RUN_PROVENANCE.md
```
