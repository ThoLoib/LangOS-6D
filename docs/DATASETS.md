# Datensätze und Modelle beschaffen

Was man braucht, um die OSCAR+-Evaluation nachzurechnen — Quellen, erwartetes
Layout und die Abkürzung über unsere fertig vorverarbeiteten Galerien. Kein
Skript dieses Repos lädt Daten herunter; die Beschaffung ist immer manuell.

## Abkürzung: fertig vorverarbeitete Galerien

Alle Renderings, Partialwolken, Beschreibungen und Embedding-Caches der
Evaluation liegen gespiegelt auf `gdrive:Masterthesis/OSCAR/` (rclone-Remote):

| Drive-Ordner | Inhalt | Ziel im Repo |
|---|---|---|
| `object_images/<ds>/` | Renderings + `*_partial.npz` + `.…cache….pt` | `object_images/<ds>/` |
| `object_database/<ds>/` | CADs + `descriptions_attributes.json` | `object_database/<ds>/` |
| `eval/datasets/` | Roh-Testdaten (SHREC, BOP) | `eval/datasets/` |
| `object_retrieval/results_*` | die finalen Ergebnisordner | `object_retrieval/` |

Mit diesen Ordnern entfällt das komplette Preprocessing (mehrere Tage GPU).
Wer von Null startet: Rohdaten unten beschaffen, dann `repro_preprocess.py`
(Reihenfolge render → partial → describe → embed → dgedi; Befehle in
`docs/REPRODUCE.md` §0).

## Rohdatensätze

| Datensatz | Rolle | Quelle | erwartetes Layout (Repo) |
|---|---|---|---|
| **SHREC'18 RGB-D-to-CAD** | Stage 1 (2101 Queries / 3308 CADs) | SHREC 2018 „RGB-D Object-to-CAD Retrieval“-Track (Pham et al., 3DOR 2018, DOI 10.2312/3dor.20181052); Daten beim Track-Organisator | `eval/datasets/shrec18/shrec18_full/{cad/*.obj, rgbd…, results/}` + `rgbd.csv`, `cad.csv`. `results/` (Beispiel-Rankinglisten) definiert den 70/30-Split → `object_retrieval/shrec18_splits/` |
| **MI3DOR** | Stage 2 (10 500 Bilder / 3848 CADs, 21 Kategorien) | MI3DOR-Benchmark (Monocular Image based 3D Object Retrieval; SHREC'19-Track) | `object_database/MI3DOR/model/test/*/*.obj` + Query-Bilder; **Objekt-ID = Dateistamm** |
| **YCB-V** | Stage 3/4 | BOP-Hub (bop.felk.cvut.cz): Base + `models_eval` + Test (BOP19-Targets) | `eval/datasets/ycbv/test/…` mit `scene_gt_info.json`; CADs `object_database/ycbv/*/textured_simple.obj` (mm) |
| **T-LESS** | Stage 3 | BOP-Hub, Primesense-Test | `eval/datasets/tless/test_primesense/…`; CADs `object_database/tless/*/model.ply` |
| **LM-O** | Stage 3 | BOP-Hub | `eval/datasets/lmo/test/…`; CADs `object_database/lmo/*/model.ply` |
| **GSO** (1030 Proxys) | Stage-3-Proxy-Gallery | Google Scanned Objects | `object_database/gso/*/meshes/model.obj` — **Einheit Meter**, ID = Großeltern-Ordner |
| **HouseCat6D** (199) | Stage-3-Proxy-Gallery | HouseCat6D-Release | `object_database/housecat6d/<kategorie>/*.obj` — ID = Dateistamm |
| **ITODD** (28) | Stage-3-Proxy-Gallery | BOP-Hub (nur Modelle) | `object_database/itodd/*/model.ply` (mm) |

Erwartete Objektzahlen prüft `python3 repro_preprocess.py --dataset <ds> --step check`.

## Modelle / Checkpoints

| Modell | Bezug | Pfad im Container |
|---|---|---|
| CLIP ViT-B/32, DINOv2-base, SigLIP-base/16, GroundingDINO, SAM 2.1, LLaVA-1.5-7b | automatisch via HuggingFace beim ersten Lauf | HF-Cache-Volume |
| ULIP-2 farbig (`ulip2_pointbert_10k.pt`, 1280-d) und XYZ (`ulip2_pointbert_8k_xyz.pt`, 512-d) | ULIP-Repo (Salesforce) + Release-Checkpoints | `/ulip/checkpoints/…` |
| Uni3D-g (`model.pt`) | Uni3D-Release | `/uni3d/modelzoo/uni3d-g/` |
| dGeDi (distilled GeDi) | dGeDi-Repo als Schwesterverzeichnis `../dGeDi` (Compose mountet es) | `/dgedi` |
| FoundationPose | eigenes Repo `~/thesis/FoundationPose` + Gewichte; läuft als Compose-Dienst `foundationpose` (Port 5050) | — |
| Blender **3.4.1** (Rendern, Host) | blender.org-Archiv — **exakt 3.4.1**: 3.3.x scheitert still (rc=0, kein PIL) | Host, `--blender`-Flag |

## Stage 5 (Greifstudie)

Braucht die BOP-Testsplits von YCB-V, T-LESS (Primesense) und LM-O **mit `depth/` und
`mask_visib/`** (wie Stage 3), die CADs der drei Proxy-Quellen (GSO, HouseCat6D, ITODD unter
`object_database/` bzw. `eval/datasets/itodd/models`) und die BOP-CADs der Ziele (`models_cad`
bei T-LESS, `models` bei LM-O, das texturierte YCB-Mesh bei YCB-V; `models_eval` wo vorhanden).
Die Instanzlisten liegen im Repo (`grasping/proxy_grasp_instances.json`); zum Neubauen braucht es
die 3b-Records: `rclone copy gdrive:Masterthesis/OSCAR/object_retrieval/results_bop_stage3_v2/3b_cross/ object_retrieval/results_bop_stage3_v2/3b_cross/`.
Kein Blender, keine Galerie-Embeddings — die Proxys sind fixiert.

## Dienste

```
docker compose build oscar            # Haupt-Image
docker compose up -d foundationpose   # Pose-Dienst (Port 5050)
DGEDI_CACHE_DIR=.dgedi_gallery docker compose up -d dgedi   # Geometrie (Port 5061)
```

`DGEDI_CACHE_DIR` wählt die Deskriptor-Gallery des dGeDi-Dienstes:
`.dgedi_gallery` (BOP, 1316 Objekte, Stage 3) bzw. `.dgedi_gallery_shrec`
(3308, Stage-1-Geometriearme). `repro_experiment.py` prüft vor jedem
Geometrie-Lauf per Health-Endpoint, dass die richtige Gallery geladen ist.
