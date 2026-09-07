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
| 1.4 B2 Gewichtskarten | `stage1/weightmap_pc.csv`, `stage1/weightmap_cross.csv` (je 66 Simplex-Punkte) | Weight-Sweep 16v/k8 (Tier-2-Ableitung) |
| 1.3 A4b Kategorien-Kipp-Analyse | `stage1/category_fullmesh_vs_partial.csv` (4 Zellen × 20 Klassen; Δ > 0 = partial besser) | `tools/compare_arms_by_category.py` über die vier Arm-Paare |
| 1.7 Kategorien-Kanäle | `stage1/category_channels.csv` | dito, isolierte Kanäle |
| 1.6 Per-Query-Bilanzen | `stage1/paired_significance_nDCG.csv`, `stage1/paired_significance_NN_sub.csv` — Kopien, bereinigt um die Vergleichszeile gegen das gestrichene fusionierte Uni3D; Original unverändert im Repo | `scripts/run_significance_after_stage1.sh` |
| 2.1–2.4 Stage-2-Arme | `stage2/metrics_summary_fused_partial.json` (**BASE**: fusioniert × partial, NN 88.44 — Variante `clip_dino_ulip_full`), `stage2/metrics_summary_fused_fullmesh.json` (86.57) | `results_mi3dor_oscarplus_v2_tau037_dinomean_partialforce/partial/` bzw. `…_ulipfix/fullmesh/` |
| 2.3 Gewichtskarte | `stage2/weight_sweep_mi3dor.csv` (231 Punkte) | `object_retrieval/results_mi3dor_wsweep/` |
| 2.5 Legacy V=8 | `stage2/metrics_summary_legacy_v8.json` | `results_mi3dor_oscar_legacy_v8/fullmesh/` |
| 2.6 Kategorien | `stage2/category_table.csv` | aus `results_topk_15.json` (Fusionsspalte = `clip_dino_ulip_full`, Full-Mesh) |
| 3a Retrieval | `stage3/3a_*.json` (je Lauf `recall@1/5/10`, `mrr`, je Datensatz) | `object_retrieval/results_bop_stage3_v2/<lauf>/combined_stage3a.json` |
| 3b Pose | `stage3/gt_combined_gt.json`, `stage3/3b_*.json` | dito |
| 3c Zerlegung | `stage3/3c_*.json` | dito |
| 3d Verdeckung | `stage3/occlusion_by_visibility.csv` | `tools/occlusion_analysis.py` (mit Selbstprüfung gegen publizierte R@1) |
| 4.1 Anfrage | `stage4/query_latency_ycbv.json` (partial), `…_fullmesh.json`, `…_geo.json` | `scripts/stage4_query.sh` → `experiments/experiment4_query_latency.py` |
| 4.2 Onboarding | `stage4/onboarding.json` (n=59, partial), `stage4/onboarding_render_n59.json` (Render-Vollerhebung), `stage4/onboarding_render.json` (alte n=5-Messung, als Beleg der Korrektur), `stage4/onboarding_dgedi.json` (n=3) | `scripts/stage4_onboarding.sh`, `scripts/run_stage4_render_full.sh` |
| 4.4 Invalidierung | `stage4/inv_test.json`, `stage4/clip_test.json` | `experiments/experiment4_onboarding.py` |
| 4.5 Repräsentationskosten | `stage4/onboarding_fullmesh.json`, `stage4/partial_16_42.json`, `stage4/views_16_42.json` | `--shape-source fullmesh`-Läufe |

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
