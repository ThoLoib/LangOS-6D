# Provenienz aller Läufe — welches Skript, welche Konfiguration

*Für jedes Ergebnis: mit welchem Skript es entstand und mit **welchem einzelnen gesetzten
Wert**. Angelegt am 2026-09-06, nachdem ein still auf Full-Mesh zurückgefallener Lauf nur
deshalb auffiel, weil zwei Arme bitgleiche Zahlen lieferten.*

> **Wozu.** Die Treiber protokollieren ihre Fusion-Config in `metrics_summary.json`, aber
> **nicht die Umgebungsvariablen** — und genau dort sitzen die Schalter, die das Verhalten
> still ändern. Diese Tabelle schließt die Lücke. Vor jedem neuen Lauf gegenlesen; nach
> jedem Lauf ergänzen.

---

## 0. Die vier Schalter, die still wirken

| Variable | wirkt auf | wenn nicht gesetzt |
|---|---|---|
| `SHREC_FORCE_PARTIAL_CACHE` | Stage 1, Shape-Gallery | **SHREC hat null `*_partial.npz`** (nur den Cache). Ohne die Variable findet `_collect_partial_items()` nichts → **stiller Fallback auf Full-Mesh**. Ein `partial`-Pass rechnet dann Full-Mesh. |
| `SHREC_DINO_POOLING` | Stage 1, DINO-View-Embeddings | Default `cls`; für Vergleichbarkeit mit Stage 2/3 muss `mean` gesetzt sein. Cache ist nach Pooling gekeyt, kollidiert also nicht. |
| `STAGE1_GEOMETRY_BACKEND` | Stage 1, Geometrie | Default probt den **alten** `gedi`-Dienst (Port 5060, läuft nicht) → **Geometriearme werden still übersprungen, rc=0**. Muss `dgedi` sein. |
| `DGEDI_CACHE_DIR` | dGeDi-Dienst | Default `.dgedi_gallery` (BOP, 1316). SHREC braucht `.dgedi_gallery_shrec` (3308). Falsche Gallery → 0 Registrierungen. |

`PYTHONHASHSEED=0` ist in allen Läufen gesetzt.

---

## 1. Stage 1 — SHREC'18

**Gemeinsam für alle Läufe** (sofern nicht abweichend vermerkt):

```
--data-root  eval/datasets/shrec18/shrec18_full
--images-dir object_images/shrec18_v2
--desc-file  object_database/shrec18_v2/descriptions_attributes.json
--results-root object_retrieval/results_shrec18_v2_stage1_42v_k5
PYTHONHASHSEED=0   SHREC_DINO_POOLING=mean
```
Fusion-Config je Arm in `metrics_summary.json` → `config`: Gewichte (0.3, 0.4, 0.3),
`fusion_method=weighted_sum`, `scope=full`, `geometry_k=5`, `clip_prune_k=20`,
`geom_voxel=0.02`, `inlier_threshold_pct_diameter=1.5`.

| Arme | Datum | Skript | `FORCE_PARTIAL` | `GEOMETRY_BACKEND` | `DGEDI_CACHE_DIR` | Besonderheiten |
|---|---|---|---|---|---|---|
| 28 Arme des Grids (E1*, E4*, E6, O2*, O4*, A2_V8/16/32, A7_V8/16/32, E7_uni3d*, O5*, E7_ulip2_cross_shape_only) | 26.08. 13:14–13:39 | `run_stage1_full.sh` | **✓ gesetzt** | dgedi | shrec | `--resume --allow-partial-gallery` |
| `A2_view_only_V42`, `A7_shape_only_V42` | 26.08. 15:03 | `run_a7.sh` | **✓ gesetzt** | — | — | Cache explizit: `.ulip_partial_cache_c3b88090d599c522.pt` |
| `E2_*` (5 Geometriearme) | 27.08. 01:46–01:53 | `run_stage1_full.sh` Phase D | ✓ | dgedi | shrec | `--with-geometry --geom-k 50` |
| `O1c`, `O1e` | 27.08. 06:30 | `run_stage1_full.sh` Phase D | ✓ | dgedi | shrec | `--with-geometry --geom-k 50` |
| `E2b_fullmesh`, `E2b_fullmesh_shape_only` | 03.09. 10:17 | `run_stage1_fullmesh_color.sh` | ✗ | — | — | nach dem Textur-Farb-Fix; **Full-Mesh-Arme, Fallback folgenlos** |
| `E7_ulip2_cross_fullmesh`, `..._shape_only` | 04.09. 04:00 | `run_stage1_cross_fullmesh.sh` | ✗ | — | — | Staging `results_stage1_cross_fullmesh`; **Full-Mesh-Arme, folgenlos** |
| `E2b_fullmesh_geo` | 04.09. 14:44 | `run_stage1_geo_on_best.sh` | ✗ | **dgedi** | **shrec** | Gate wählte Grundlage selbst; `--with-geometry --geom-k 50` |
| ⚠️ `E7_ulip2_cross` | 06.09. 17:59 | `run_stage1_cross_fullmesh.sh` | ✗ | — | — | **UNGÜLTIG** — `partial`-Pass ohne die Variable → still Full-Mesh gerechnet, Werte bitgleich zu `E7_ulip2_cross_fullmesh` |

### Was der Fallback wirklich getroffen hat

Nur **ein** Arm. Die Grid-Läufe vom August setzen die Variable; alle September-Arme sind
Full-Mesh **by design**, für die ist der Fallback der gewollte Pfad. `E7_ulip2_cross` ist der
einzige Partial-Arm, der ohne die Variable lief — und damit ungültig.

**Erkennungsmerkmal für die Zukunft:** im Lauflog steht `[init] Encoding 3308 ULIP CAD
meshes...` statt `[init] ULIP partial-view cache loaded`. Bei einem `partial`-Pass ist das
der Fallback.

---

## 2. Stage 2 — MI3DOR

**Treiber:** `object_retrieval/retrieval_mi3dor_eval_oscarplus.py`

| Ergebnisordner | Datum | Skript | `MI3DOR_MODES` | `MI3DOR_NUM_VIEWS` | `MI3DOR_DINO_POOLING` | Gewichte |
|---|---|---|---|---|---|---|
| `..._v2_tau037` | 07.08. | manuell | partial | 42 | *(cls, Default)* | 0/0.5/0.5 |
| `..._v2_tau037_dinomean` | 08.08. | manuell | partial ¹ | 42 | mean | 0/0.5/0.5 |
| `..._v2_tau037_dinomean_fixedw` | 25.08. | `run_stage2_after_stage1.sh` | fullmesh | 42 | mean | **0.3/0.4/0.3** |
| `..._v2_tau037_dinomean_ulipfix` | 26.08. | `run_mi3dor_ulipfix.sh` | partial ¹ | 42 | mean | 0.3/0.4/0.3 |
| `results_mi3dor_oscar_legacy_v8` | 27.08. | `run_mi3dor_oscar_legacy.sh` | fullmesh | **8** | mean | *(Kaskade, Gewichte unbenutzt)* |
| `results_mi3dor_wsweep` | 27.08. | `mi3dor_weight_sweep.py` | — | 42 | mean | Simplex-Sweep, Schrittweite 0.05 |

¹ **`MI3DOR_MODES=partial` griff nie.** MI3DOR hat keine `*_partial.npz`; jeder Lauf fiel
still auf Full-Mesh zurück (Meldung: `no partial PCs found … Falling back to full-mesh`).
Die berichteten Stage-2-Zahlen sind **durchgehend Full-Mesh**. Der `_dinomean`-Lauf ist der
einzige, in dem `ulip2_use_partial_views=True` tatsächlich wirkte — daraus stammt der
A4-Transfer-Vergleich.

**τ = 0.37** in allen Läufen. Zusätzlich: die MI3DOR-Meshes tragen keinerlei Farbe
(uniform 0.4/0.4/0.4), siehe `STAGE2_RESULTS.md`.

---

## 3. Stage 3 — BOP

**Treiber:** `object_retrieval/eval_bop_pose.py`, gemeinsame Flags `--datasets all`,
`PYTHONHASHSEED=0`. Gallery: `assemble_gallery`, Gewichte 0.3/0.4/0.3, 42 Views,
top-k-softmax k=5 τ=0.5, DINO mean.

| Ergebnis | Datum | Skript | Modus + Flags | `STAGE3_GEO_SIGNAL` | dGeDi-Gallery |
|---|---|---|---|---|---|
| `3a_pc` | 17.08. | manuell | `3a --pc-query` | — | — |
| `3a_pc_geo` | 18.08. | manuell | `3a --pc-query --dgedi --dgedi-repo --dgedi-top-k 5` | *(Default distance)* | BOP 1316 |
| `3a_cross` | 18.08. | manuell | `3a` | — | — |
| `3a_cross_geo` | 18.08. | manuell | `3a --dgedi --dgedi-repo --dgedi-top-k 5` | *(Default)* | BOP 1316 |
| `3b_cross` | 19.08. | manuell | `3b --gt-records …/gt/combined_gt.json` | — | — |
| `3b_cross_geo` | 20.08. | `run_3b_geo.sh` | `3b --dgedi --dgedi-repo --dgedi-top-k 5 --gt-records …` | *(Default)* | BOP 1316 |
| `3c_cross` | 25.08. | `run_stage3_3c.sh` | `3c --from-3a …/3a_cross` | — | — |
| `3a_oscar` | 28.08. | `run_remaining_chain.sh` | `3a --oscar-baseline` | — | — |
| `3a_cross_geo_borda` | 28.08. | `run_stage3_geo_redo.sh` | `3a --dgedi …` | `borda` | BOP 1316 ⚠️ Arm gekillt, 0 % Deckung |
| `3a_cross_geo_distance` | 29.08. | `run_stage3_geo_redo.sh` | `3a --dgedi …` | `distance` | BOP 1316 |
| `3a_cross_geo_fitness` | 29.08. | `run_stage3_geo_redo.sh` | `3a --dgedi …` | `fitness` | BOP 1316 |
| `3a_pc_geo_distance` | 29.08. | `run_stage3_rest.sh` | `3a --pc-query --dgedi …` | `distance` | BOP 1316 |
| `3a_pc_geo_fitness` | 30.08. | `run_stage3_rest.sh` | `3a --pc-query --dgedi …` | `fitness` | BOP 1316 |
| `3b_oscar` | 30.08. | `run_stage3_rest.sh` | `3b --oscar-baseline --gt-records …` | — | — |
| `3a_fullmesh` | 30.08. | `run_stage3_rest.sh` | `3a --fullmesh` | — | — |
| `3a_pc_fullmesh` | 30.08. | `run_stage3_pc_fullmesh.sh` | `3a --fullmesh --pc-query` | — | — |
| `3a_pc_v2` | 01.09. | `run_fullmesh_color_redo.sh` | `3a --pc-query` | — | — |
| `3a_cross_v2` | 01.09. | `run_fullmesh_color_redo.sh` | `3a` | — | — |
| `3a_pc_fullmesh_v2` | 02.09. | `run_fullmesh_color_redo.sh` | `3a --fullmesh --pc-query` | — | — |
| `3a_cross_fullmesh_v2` | 02.09. | `run_fullmesh_color_redo.sh` | `3a --fullmesh` | — | — |
| `3b_cross_fullmesh` | 03.09. | `run_stage3_fullmesh_pose.sh` | `3b --fullmesh --gt-records …` | — | — |
| `3c_cross_fullmesh` | 04.09. | `run_stage3_fullmesh_pose.sh` | `3c --fullmesh --from-3a …/3a_cross_fullmesh_v2` | — | — |

**dGeDi-Repo-Parameter** bei allen `--dgedi-repo`-Läufen: 6000 Keypoints, 10 000
RANSAC-Iterationen, ICP an, Shortlist K=5.

**Ab 01.09.** schreiben alle Läufe zusätzlich `arm_ranks` je Query (Ziel-Rang je Kanal) —
die `_v2`-Läufe und alles danach. Ältere Läufe haben das Feld nicht.

**Der Farb-Fix (01.09.)** trennt die Läufe: alles ab `_v2` nutzt texturierte Mesh-Farben
(`to_color().vertex_colors`), alles davor nicht. Die vier `_v2`-Läufe ersetzen deshalb
`3a_pc`, `3a_cross`, `3a_pc_fullmesh`, `3a_fullmesh`.

---

## 4. Stage 4 — Latenz

**Treiber:** `experiments/experiment4_query_latency.py`, `experiments/experiment4_onboarding.py`

| Ergebnis | Datum | Skript | Flags |
|---|---|---|---|
| `query_latency_ycbv.json` | 04.09. 19:02 | `stage4_query.sh` via `run_stage4_full.sh` | `-d ycbv -n 50 -v 16,42` (mit Pose, `--refine-iter 5`) |
| `query_latency_ycbv_geo.json` | 04.09. 19:04 | `stage4_query.sh` | `-d ycbv -n 25 -v 42 --geometry --no-pose`, K=5 |
| `onboarding.json` | 04.09. 19:39 | `stage4_onboarding.sh` | `--stages mesh,partial,describe,embed --reuse-renders -v 16,42 --measure-invalidation --inv-sample 15` |
| `onboarding_render.json` | 04.09. 19:39 | `stage4_onboarding.sh`, **Host** | `--stages render --render-objects 5`, Blender 3.4.1 |
| `onboarding_dgedi.json` | 06.09. | `experiment4_onboarding.py`, **Host** | `--stages dgedi -v 16,42 --max-objects 3` |

**Warm-up** 2 Queries je View-Zahl, nicht gewertet. **Seed** 0. **Gallery** 1278
(ycbv-Ziele + Proxy). `--num-points 8192` für Teilwolken.

**Zwei Stufen laufen auf dem Host**, nicht im Container: `render` (Blender liegt unter
`/home/tessa/Cap3D/…/blender-3.4.1-linux-x64/`, nicht im Compose-Mount) und `dgedi`
(braucht `docker`, das im Container fehlt; Manifest und Ausgabe müssen zudem **innerhalb
des Repos** liegen, weil der dGeDi-Container nur `.:/oscar` mountet).

---

## 5. Regenerieren

Die Datums- und Config-Spalten lassen sich aus den Ergebnisdateien nachziehen:

```bash
python3 tools/run_provenance.py            # Arm → Datum → gespeicherte Config
```

Die Skript- und Umgebungsspalten stammen aus den Skripten selbst und werden **von Hand**
gepflegt — die Treiber schreiben sie nicht mit. Genau das war die Lücke, durch die der
Full-Mesh-Fallback vom 06.09. schlüpfen konnte.
