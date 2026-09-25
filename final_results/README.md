# final_results — Manifest

Konsolidierte Endergebnisse der OSCAR+-Evaluation (Masterarbeit TU Wien), Stand 2026-09-07.
Einstieg: **[RESULTS.md](RESULTS.md)** — alle vier Stages als durchgehende Story, mit
vollständigen Tabellen, Diskussion je Experiment und übergreifender Diskussion.
Die interaktiven Fassungen (mit allen Diagrammen) liegen unter [artifacts/](artifacts/).

Jede Zahl in RESULTS.md ist durch eine Datei in diesem Ordner gedeckt. Die Original-Läufe
(inkl. `results_per_query.json` je Arm) bleiben unter `object_retrieval/results_*` bzw.
`results_stage4/` im Repo-Arbeitsverzeichnis und auf `gdrive:Masterthesis/OSCAR/` — dieser
Ordner enthält die **Summary-Ebene**, nicht die Rohdaten je Query.

## Zuordnung RESULTS.md → Dateien → Ursprung

| RESULTS.md | Datei hier | Ursprung (Repo) |
|---|---|---|
| 1.1, 1.3–1.6, 1.8 (alle Stage-1-Arme) | **`stage1/stage1_summary_arms.csv`** (die 40 berichteten Arme, frisch aus den `metrics_summary.json` generiert und gegen §1.8 abgeglichen; offizielle + Tiefen-Metriken in einer Datei), `stage1/arms_overview.csv` (Tabelle §1.8 mit Gruppen/Bemerkungen) | `object_retrieval/results_shrec18_v2_stage1_42v_k5/<arm>/metrics_summary.json` |
| 1.2 Track-Vergleich | `stage1/stage1_summary_testsplit.csv` | `object_retrieval/results_shrec18_v2_stage1_42v_k5_testsplit/` (649 Test-Queries) |
| 1.4 B2 Gewichtskarten **v2 (PRODUKTIONSIDENTISCH: BASE == Produktionsarm exakt)** | `stage1/weight_sweep_66_pc.csv`, `stage1/weight_sweep_66_cross.csv` (je 66 Simplex-Punkte, Spalten nDCG/hit1/NN_cat/MRR) + Manifeste (Wächter, Bestpunkte) | `experiments/stage1_weight_sweep_v2.py` auf den Score-Stores `_cache/scores_*.pt`, Produktions-Config 42v/k5, GEOM_K=5; Wächter pc = E1c_full_fusion 0.5868/0.3413, cross = E7_ulip2_cross 0.5588/0.3289 |
| 1.4 B2 Gewichtskarten v1 (ALT: 16v/k8-Config, nur historisch — Shape-Kanal weicht von der Produktion ab) | `stage1/weightmap_pc.csv`, `stage1/weightmap_cross.csv` | Weight-Sweep 16v/k8 (Tier-2-Ableitung) |
| 1.3 A4b Kategorien-Kipp-Analyse | `stage1/category_fullmesh_vs_partial.csv` (4 Zellen × 20 Klassen; Δ > 0 = partial besser) | `tools/compare_arms_by_category.py` über die vier Arm-Paare |
| 1.7 Kategorien-Kanäle | `stage1/category_channels.csv` | dito, isolierte Kanäle |
| 6.2.3/6.8 Subkategorie-Split (hit@1 je Arm, mit/ohne echte Subkategorie; 426/2101 Queries kodieren "keine Sub" als wiederholte Kategorie) | `stage1/stage1_subcategory_split.csv` (40 Arme; Alias-Arme A2_view_only_V42/A7_shape_only_V42 aus ihren alias_of-Zielen) | per-Query `NN_sub` aus `object_retrieval/results_shrec18_v2_stage1_42v_k5/<arm>/results_per_query.json` x Subkategorie-Kriterium aus `eval/shrec18_official/rgbd.csv` |
| 1.6 Per-Query-Bilanzen | `stage1/paired_significance_nDCG.csv`, `stage1/paired_significance_NN_sub.csv` — Kopien, bereinigt um die Vergleichszeile gegen das gestrichene fusionierte Uni3D; Original unverändert im Repo | `scripts/run_significance_after_stage1.sh` |
| 2.1–2.4 Stage-2-Arme | `stage2/metrics_summary_fused_partial.json` (**BASE**: fusioniert × partial, NN 88.44 — Variante `clip_dino_ulip_full`), `stage2/metrics_summary_fused_fullmesh.json` (86.57) | `results_mi3dor_oscarplus_v2_tau037_dinomean_partialforce/partial/` bzw. `…_ulipfix/fullmesh/` |
| Offizielle MI3DOR-Metriken (Benchmark-Definitionen: ST@2C-1, F mikro@20, DCG mit Off-by-one-Diskont ueber C, ANMRR mit S=min(4C,2*T_max) + Positions-Logik, AUC) | **Vollstaendig**: `stage2/metrics_official_traces.csv` (21 Zeilen = 7 Arme x 3 Laeufe: partial 42v, fullmesh 42v, fullmesh 8v) + Manifest; Vorstufe `stage2/metrics_official.csv` (4 Arme partial + 2 fullmesh aus Kanal-Caches) bleibt bestehen — beide Wege treffen sich auf 4 Stellen | Port `object_retrieval/mi3dor_official_metrics.py` von cross_performance.m (Commit 4325c24c, 2023-10-24); `mi3dor_official_from_traces.py` auf den `eval_trace.rel_positions` der drei `results_topk_15.json` (W1: NN/FT je Arm exakt 1e-9 auf Produktion) |
| 2.3 Gewichtskarte v2 (PRODUKTIONSIDENTISCH: BASE == Produktionslauf exakt, 300/300 Fusions-Stichproben byte-gleich) | `stage2/weight_sweep_66.csv` (66 Punkte, Schritt 0.1 wie Stage 1) + `stage2/weight_sweep_66_manifest.json` (Selfchecks, Konfig) | `object_retrieval/mi3dor_weight_sweep_v2.py`; Kanal-Score-Matrizen persistiert in `results_mi3dor_wsweep_v2/channel_scores.npz` (373 MB, Repo-Ordner + Drive — kuenftige Sweeps ohne GPU) |
| 2.3 Gewichtskarte v1 (ALT: eigene Refusion, BASE 0.6851 statt 0.6918 — nur historisch) | `stage2/weight_sweep_mi3dor.csv` (231 Punkte) | `object_retrieval/results_mi3dor_wsweep/` |
| 2.5 Legacy V=8 | `stage2/metrics_summary_legacy_v8.json` | `results_mi3dor_oscar_legacy_v8/fullmesh/` |
| 2.6 Kategorien | `stage2/category_table.csv` | aus `results_topk_15.json` (Fusionsspalte = `clip_dino_ulip_full`, Full-Mesh) |
| 6.21 Spalten "Shape alone"/Einzelkanaele (R@1 je Arm x Datensatz, aus `arm_ranks` der per-Instanz-Records) | `stage3/stage3a_arm_recall1.csv` (4 Laeufe x 7 Arme x lmo/tless/ycbv/all) | `object_retrieval/results_bop_stage3_v2/{3a_cross_fullmesh_v2,3a_cross_v2,3a_pc_v2,3a_pc_fullmesh_v2}/*/records.json`. ACHTUNG: die Kaskaden-Zeile 0.3198 der Tabelle stammt NICHT aus arm_ranks (oscar_maxview dort 0.3675), sondern aus dem separaten E5-Lauf `3a_oscar` (archiviert als `3a_oscar_combined_stage3a.json`) |
| 3a Retrieval | `stage3/3a_*.json` (je Lauf `recall@1/5/10`, `mrr`, je Datensatz) | `object_retrieval/results_bop_stage3_v2/<lauf>/combined_stage3a.json` |
| 3b Pose | `stage3/gt_combined_gt.json`, `stage3/3b_*.json` | dito |
| Herkunft x Datensatz in 3c (Median D_sym mm + normiert, je dataset x nb_provenance + ALL) | `stage3/stage3c_origin_by_dataset.csv` (3c_cross + 3c_cross_fullmesh) | aus `d_posed`/`d_sym_norm` der `results_bop_stage3_v2/3c_cross*/*/records.json`; ALL-Zeilen treffen die archivierten provenance-Mediane (10.354 / 20.102) exakt |
| 3c Zerlegung | `stage3/3c_*.json` | dito |
| 6.28 Verdeckung, diameter-normalisiert (Median d_sym_norm je Sichtbarkeitsklasse, Grenzen 0.5/0.8/0.95) | `stage3/occlusion_by_visibility_norm.csv` (3b_cross, ALLE + je Datensatz; mm-Spalte identisch zur bestehenden Tabelle, Gegenprobe 27.7/18.8/14.4/18.8 und n 1184/2394/2992/5714 bestanden) | `tools/occlusion_analysis.collect` (d_sym_norm aus den 3b-Records) |
| 3d Verdeckung | `stage3/occlusion_by_visibility.csv` | `tools/occlusion_analysis.py` (mit Selbstprüfung gegen publizierte R@1) |
| 4.1 Anfrage | `stage4/query_latency_ycbv.json` (partial), `…_fullmesh.json`, `…_geo.json` | `scripts/stage4_query.sh` → `experiments/experiment4_query_latency.py` |
| 4.2 Onboarding | `stage4/onboarding.json` (n=59, partial), `stage4/onboarding_render_n59.json` (Render-Vollerhebung), `stage4/onboarding_render.json` (alte n=5-Messung, als Beleg der Korrektur), `stage4/onboarding_dgedi.json` (n=3, KALT: je Objekt eigener docker-compose-Aufruf inkl. Container-/Python-/Modellstart), `stage4/onboarding_dgedi_warm_n59.json` (**n=59 wie die uebrige Onboarding-Tabelle**, WARM: median 1.414 s je Objekt, je Datensatz aufgeschluesselt; einmaliger Start getrennt), `stage4/onboarding_dgedi_warm.json` (n=21, nur YCB-V — Vorstufe) | `scripts/stage4_onboarding.sh`, `scripts/run_stage4_render_full.sh`; warm: `dgedi_service/precompute_gallery.py --timing-json` (99180da9) |
| 4.4 Invalidierung | `stage4/inv_test.json`, `stage4/clip_test.json` | `experiments/experiment4_onboarding.py` |
| 4.5 Repräsentationskosten | `stage4/onboarding_fullmesh.json`, `stage4/partial_16_42.json`, `stage4/views_16_42.json` | `--shape-source fullmesh`-Läufe |

| Stage 5 (Proxy-Greifstudie) | `stage5/trials.csv` (180 Trials, eine Zeile je Trial), `stage5/REPORT.md` (generierte Tabellen), `stage5/manifest.json` (Args, Protokoll, Git-Revision) | `_s5_out/proxy_grasp/` (Drive: `_s5_out/proxy_grasp_tessa/`); (Alt-Stand; im aktuellen `docs/STAGE5_RESULTS.md` nicht mehr gefuehrt) Protokoll `docs/STAGE5_PROTOCOL.md`; Laptop-Replikat (µ=2.4) `_s5_out/proxy_grasp_laptop_2026-09-11/` |

| **Stage 5 final — solo_full (52 Objekte, gt / 3b-Proxy / 3c-Substitut)** | `stage5_full/trials.csv` (1560 Trials), `stage5_full/plan.json`, `stage5_full/TABELLEN.md`, `stage5_full/STAGE5_RESULTS_KOPIE.md` (= `docs/STAGE5_RESULTS.md`), `stage5_full/viz/` (Videos + Overlays, Drive) | `_s5_out/solo_full/` (Drive: `solo_full_tessa`); Plan+Lauf `grasping/stage_5_full.py`, Einzelserien `grasping/stage_5.py` (frueher solo_trial.py), Videos `grasping/stage_5_viz.py` |
| Solo v2 (Einzelobjekt, stehend + Yaw; Alt-Stand) | `stage5_solo/trials.csv` (618 Trials, inkl. Eggbox-Nachserie), `stage5_solo/plan.json`, `stage5_solo/AUSWERTUNG.md` | `_s5_out/solo_v2/` (Drive: `solo_v2_tessa`); Treiber `grasping/solo_trial.py` |

## Stolperfallen, festgehalten

- **`…_ulipfix/partial` ist NICHT der Partial-Lauf.** Dort fiel der Pass mangels
  `SHREC_FORCE_PARTIAL_CACHE` still auf Full-Mesh zurück (Werte identisch zu
  `ulipfix/fullmesh`). Der echte fusionierte Partial-Lauf ist
  `…_partialforce/partial` — nur der liegt hier.
- Stage-2-Metriknamen in den JSONs: `NN_accuracy` (in %), `FT_mean`, `ST_mean`, `F1_mean`,
  `nDCG@2R_mean`, `mAP`, `ANMRR_mean`; Arme unter `variants`, `primary` ist die Kaskade
  `clip_pruned_dino_ulip`.
- Der Stage-2-Gewichts-Sweep lief auf der **Full-Mesh**-Gallery; sein Optimum (0.6902 FT)
  ist deshalb nicht mit dem Partial-BASE (0.6918) vergleichbar.
- **Drei Stage-1-Arme sind bewusst gestrichen** und tauchen weder in RESULTS.md noch in den
  CSVs hier auf, obwohl ihre Ordner im Ergebnisverzeichnis existieren: `E2_chamfer_icp` und
  `E2_chamfer_unaligned` (byte-identische Aliasse von `chamfer_ransac` bzw. `fitness` — das
  Backend liefert nur zwei Signale, → RESULTS.md 1.5 C1) sowie `E7_uni3d` (fusioniertes
  Uni3D; nur der isolierte A3-Vergleich wird berichtet).
- In den offiziellen Stage-1-Metriken gilt precision = recall = F1 = NNT1 = NNT2 by design
  (Top-f-Kürzung); die Tiefen-Familie steht in denselben Zeilen (hit@1 = `NN_sub`).
- Vier still wirkende Env-Variablen der Läufe sind in `docs/RUN_PROVENANCE.md` dokumentiert
  (`SHREC_FORCE_PARTIAL_CACHE`, `SHREC_DINO_POOLING`, `STAGE1_GEOMETRY_BACKEND`,
  `DGEDI_CACHE_DIR`).

## Reproduktion

- Übersichtstabellen: `python3 tools/results_overview.py -o docs/RESULTS_OVERVIEW.md`
- Verdeckungsanalyse: `python3 tools/occlusion_analysis.py [--mode 3a|3b|both] [--csv …]`
- Stage-4-Messungen: `scripts/stage4_onboarding.sh [--shape-source fullmesh]`,
  `scripts/stage4_query.sh`, `scripts/run_stage4_render_full.sh`
- Stage-2-Partial-Arm: `scripts/run_stage2_partial_fused.sh` (erzwingt den Partial-Cache und
  verifiziert ihn dreifach)
- Stage-1-Minimaltreiber: `stage1_reproduce.py`

Upload-Ziel dieses Ordners: `gdrive:Masterthesis/OSCAR/final_results/`
