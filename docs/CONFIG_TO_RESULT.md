# Konfiguration → Ergebnis (generiert)

> Erzeugt von `tools/run_provenance.py --markdown`. Nicht von Hand aendern.
> Die Skript- und Variablenspalten stammen aus `S1_RUNS` im Werkzeug,
> alles andere aus den Ergebnisdateien.


**Kurzcodes:** `FP✓` = `SHREC_FORCE_PARTIAL_CACHE` gesetzt · `FP✓col` = mit dem coloured-Cache (1280-d) · `FP—` = bewusst nicht gesetzt · `mean` = `SHREC_DINO_POOLING=mean` · `geo:dgedi/shrec` = `STAGE1_GEOMETRY_BACKEND=dgedi` + `DGEDI_CACHE_DIR=.dgedi_gallery_shrec` · `K=50` = `--geom-k 50`. `PYTHONHASHSEED=0` überall.


## Stage 1 — SHREC'18

| Arm | Shape-Pass | Gewichte | Geo | Skript | Variablen | nDCG | NN_sub |
|---|---|---|---|---|---|---|---|
| `A2_view_only_V16` | `—` | 0.0/1.0/0.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5481 | 0.3260 |
| `A2_view_only_V32` | `—` | 0.0/1.0/0.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5426 | 0.3208 |
| `A2_view_only_V42` | `—` | 0.0/1.0/0.0 | — | `run_a7.sh` | FP✓col · mean | 0.5506 | 0.3337 |
| `A2_view_only_V8` | `—` | 0.0/1.0/0.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5302 | 0.3027 |
| `A7_shape_only_V16` | `ulip_pc_rgb_v16` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5227 | 0.3079 |
| `A7_shape_only_V32` | `ulip_pc_rgb_v32` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5300 | 0.3170 |
| `A7_shape_only_V42` | `ulip_pc_rgb` | 0.0/0.0/1.0 | — | `run_a7.sh` | FP✓col · mean | 0.5353 | 0.3275 |
| `A7_shape_only_V8` | `ulip_pc_rgb_v8` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5119 | 0.2913 |
| `A7f_full_fusion_shape_V42` | `ulip_pc_rgb_v42` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5868 | 0.3413 |
| `E1_oscar_cascade` | `—` | 0.0/1.0/0.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.4561 | 0.2347 |
| `E1_shape_only` | `ulip_pc_rgb` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5353 | 0.3275 |
| `E1_view_only` | `—` | 0.0/1.0/0.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5506 | 0.3337 |
| `E1a_text_only` | `—` | 1.0/0.0/0.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.4218 | 0.1304 |
| `E1b_text_view` | `—` | 0.43/0.57/0.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5519 | 0.3122 |
| `E1c_full_fusion` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5868 | 0.3413 |
| `E1d_clip_pruned` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.4566 | 0.2308 |
| `E2_both` | `ulip_pc_rgb` | 0.3/0.4/0.3 | both_borda | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.6362 | 0.4645 |
| `E2_chamfer_icp` | `ulip_pc_rgb` | 0.3/0.4/0.3 | chamfer_icp | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.6405 | 0.4717 |
| `E2_chamfer_ransac` | `ulip_pc_rgb` | 0.3/0.4/0.3 | chamfer_ransac | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.6405 | 0.4717 |
| `E2_chamfer_unaligned` | `ulip_pc_rgb` | 0.3/0.4/0.3 | chamfer_unaligned | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.6251 | 0.4393 |
| `E2_fitness` | `ulip_pc_rgb` | 0.3/0.4/0.3 | fitness | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.6251 | 0.4393 |
| `E2b_fullmesh` | `ulip_pc_fullmesh` | 0.3/0.4/0.3 | — | `run_stage1_fullmesh_color.sh` | FP— (FM gewollt) · mean | 0.5935 | 0.3598 |
| `E2b_fullmesh_geo` | `ulip_pc_fullmesh` | 0.3/0.4/0.3 | chamfer_ransac | `run_stage1_geo_on_best.sh` | geo:dgedi/shrec · K=50 · mean | 0.6417 | 0.4807 |
| `E2b_fullmesh_shape_only` | `ulip_pc_fullmesh` | 0.0/0.0/1.0 | — | `run_stage1_fullmesh_color.sh` | FP— (FM gewollt) · mean | 0.4956 | 0.2822 |
| `E4_siglip` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5659 | 0.3051 |
| `E4_siglip_only` | `—` | 0.0/1.0/0.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5165 | 0.2642 |
| `E6_rrf` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5744 | 0.3175 |
| `E7_ulip2_cross` | `ulip_cross_rgb` | 0.3/0.4/0.3 | — | `run_stage1_cross_fullmesh.sh` | FP✓col · mean | 0.5588 | 0.3289 |
| `E7_ulip2_cross_fullmesh` | `ulip_cross_fullmesh` | 0.3/0.4/0.3 | — | `run_stage1_cross_fullmesh.sh` | FP— (FM gewollt) · mean | 0.5511 | 0.3084 |
| `E7_ulip2_cross_fullmesh_shape_only` | `ulip_cross_fullmesh` | 0.0/0.0/1.0 | — | `run_stage1_cross_fullmesh.sh` | FP— (FM gewollt) · mean | 0.4569 | 0.2028 |
| `E7_ulip2_cross_shape_only` | `ulip_cross_rgb` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.4809 | 0.2637 |
| `E7_uni3d` | `uni3d` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5913 | 0.3455 |
| `E7_uni3d_shape_only` | `uni3d` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5337 | 0.3094 |
| `O1c_gedi_post_fusion` | `—` | 0.43/0.57/0.0 | fitness | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5961 | 0.4055 |
| `O1e_gedi_with_base` | `ulip_pc_rgb` | 0.3/0.4/0.3 | both_borda_base | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.6287 | 0.4588 |
| `O2_clip_threshold` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.4561 | 0.2294 |
| `O2_clip_threshold_cal` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5186 | 0.2808 |
| `O2_visual_first` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5570 | 0.3408 |
| `O4_V16` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5820 | 0.3327 |
| `O4_V32` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5800 | 0.3289 |
| `O4_V8` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5714 | 0.3251 |
| `O5_xyz_only` | `ulip_pc_xyz` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5880 | 0.3541 |
| `O5_xyz_shape_only` | `ulip_pc_xyz` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FP✓ · mean · geo:dgedi/shrec | 0.5422 | 0.3598 |

## Stage 2 — MI3DOR

| Ordner | Modus | Views | Arm | NN | FT | mAP |
|---|---|---|---|---|---|---|
| `results_mi3dor_oscar_legacy_v8` | fullmesh | 8 | `clip_only` | 67.95 | 0.575 | 0.000 |
| `results_mi3dor_oscar_legacy_v8` | fullmesh | 8 | `dino_only_full` | 81.96 | 0.591 | 0.000 |
| `results_mi3dor_oscar_legacy_v8` | fullmesh | 8 | `ulip_only_full` | 78.10 | 0.510 | 0.000 |
| `results_mi3dor_oscar_legacy_v8` | fullmesh | 8 | `clip_dino_ulip_full` | 86.62 | 0.665 | 0.000 |
| `results_mi3dor_oscar_legacy_v8` | fullmesh | 8 | `oscar_maxview` | 84.40 | 0.575 | 0.000 |
| `results_mi3dor_oscar_legacy_v8` | fullmesh | 8 | `oscar_softmax` | 84.42 | 0.575 | 0.000 |
| `results_mi3dor_oscar_legacy_v8` | fullmesh | 8 | `clip_pruned_dino_ulip` | 85.78 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037` | fullmesh | 42 | `clip_only` | 67.95 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037` | fullmesh | 42 | `dino_only_full` | 78.01 | 0.587 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037` | fullmesh | 42 | `ulip_only_full` | 78.10 | 0.510 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037` | fullmesh | 42 | `clip_dino_ulip_full` | 83.42 | 0.620 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037` | fullmesh | 42 | `oscar_maxview` | 84.79 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037` | fullmesh | 42 | `oscar_softmax` | 84.51 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037` | fullmesh | 42 | `clip_pruned_dino_ulip` | 85.93 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037` | partial | 42 | `clip_only` | 67.95 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037` | partial | 42 | `dino_only_full` | 78.01 | 0.587 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037` | partial | 42 | `ulip_only_full` | 68.11 | 0.453 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037` | partial | 42 | `clip_dino_ulip_full` | 84.11 | 0.620 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037` | partial | 42 | `oscar_maxview` | 84.79 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037` | partial | 42 | `oscar_softmax` | 84.51 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037` | partial | 42 | `clip_pruned_dino_ulip` | 85.52 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | fullmesh | 42 | `clip_only` | 67.95 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | fullmesh | 42 | `dino_only_full` | 83.03 | 0.629 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | fullmesh | 42 | `ulip_only_full` | 78.10 | 0.510 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | fullmesh | 42 | `clip_dino_ulip_full` | 85.17 | 0.639 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | fullmesh | 42 | `oscar_maxview` | 84.88 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | fullmesh | 42 | `oscar_softmax` | 85.04 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | fullmesh | 42 | `clip_pruned_dino_ulip` | 86.16 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | partial | 42 | `clip_only` | 67.95 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | partial | 42 | `dino_only_full` | 83.03 | 0.629 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | partial | 42 | `ulip_only_full` | 68.11 | 0.453 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | partial | 42 | `clip_dino_ulip_full` | 87.05 | 0.648 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | partial | 42 | `oscar_maxview` | 84.88 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | partial | 42 | `oscar_softmax` | 85.04 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | partial | 42 | `clip_pruned_dino_ulip` | 85.22 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw` | partial | 42 | `clip_only` | 67.95 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw` | partial | 42 | `dino_only_full` | 83.03 | 0.629 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw` | partial | 42 | `ulip_only_full` | 0.00 | 0.000 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw` | partial | 42 | `clip_dino_ulip_full` | 85.21 | 0.674 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw` | partial | 42 | `oscar_maxview` | 84.88 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw` | partial | 42 | `oscar_softmax` | 85.04 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw` | partial | 42 | `clip_pruned_dino_ulip` | 85.04 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | fullmesh | 42 | `clip_only` | 67.95 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | fullmesh | 42 | `dino_only_full` | 83.03 | 0.629 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | fullmesh | 42 | `ulip_only_full` | 78.10 | 0.510 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | fullmesh | 42 | `clip_dino_ulip_full` | 86.57 | 0.682 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | fullmesh | 42 | `oscar_maxview` | 84.88 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | fullmesh | 42 | `oscar_softmax` | 85.04 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | fullmesh | 42 | `clip_pruned_dino_ulip` | 86.52 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | partial | 42 | `clip_only` | 67.95 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | partial | 42 | `dino_only_full` | 83.03 | 0.629 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | partial | 42 | `ulip_only_full` | 78.10 | 0.510 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | partial | 42 | `clip_dino_ulip_full` | 86.57 | 0.682 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | partial | 42 | `oscar_maxview` | 84.88 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | partial | 42 | `oscar_softmax` | 85.04 | 0.575 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | partial | 42 | `clip_pruned_dino_ulip` | 86.52 | 0.575 | 0.000 |

## Stage 3 — BOP

| Lauf | Modus | Gallery | arm_ranks | R@1 / D_sym | MRR |
|---|---|---|---|---|---|
| `3a_cross` | 3a | 1316 | nein | 0.4818 | 0.5971 |
| `3a_cross_fullmesh_v2` | 3a | 1316 | ja | 0.5151 | 0.6379 |
| `3a_cross_geo` | 3a | 1316 | nein | 0.4582 | 0.5787 |
| `3a_cross_geo_borda` | 3a | 1316 | nein | 0.4818 | 0.5971 |
| `3a_cross_geo_distance` | 3a | 1316 | nein | 0.4229 | 0.5576 |
| `3a_cross_geo_fitness` | 3a | 1316 | nein | 0.4278 | 0.5569 |
| `3a_cross_v2` | 3a | 1316 | ja | 0.4818 | 0.5971 |
| `3a_fullmesh` | 3a | 1316 | nein | 0.4639 | 0.6021 |
| `3a_oscar` | 3a | 1316 | nein | 0.3198 | 0.4043 |
| `3a_pc` | 3a | 1316 | nein | 0.4636 | 0.5844 |
| `3a_pc_fullmesh` | 3a | 1316 | nein | 0.3504 | 0.4592 |
| `3a_pc_fullmesh_v2` | 3a | 1316 | ja | 0.3878 | 0.4899 |
| `3a_pc_geo` | 3a | 1316 | nein | 0.4131 | 0.5471 |
| `3a_pc_geo_distance` | 3a | 1316 | nein | 0.3725 | 0.5215 |
| `3a_pc_geo_fitness` | 3a | 1316 | nein | 0.3820 | 0.5249 |
| `3a_pc_v2` | 3a | 1316 | ja | 0.4636 | 0.5844 |
| `3b_cross` | 3b | 1257 | — | 18.37 mm | — |
| `3b_cross_fullmesh` | 3b | 1257 | — | 18.91 mm | — |
| `3b_cross_geo` | 3b | 1257 | — | 28.79 mm | — |
| `3b_oscar` | 3b | 1257 | — | 21.73 mm | — |
| `3c_cross` | 3c | — | — | 15.34 mm | — |
| `3c_cross_fullmesh` | 3c | — | — | 13.51 mm | — |
| `3c_smoke` | 3c | — | — | 17.13 mm | — |

## Stage 4 — Latenz

| Datei | Stufen / Views | Gallery | Median je Einheit |
|---|---|---|---|
| `clip_test.json` | ['describe', 'embed'], 42 Views | — | 18.375 s (n=1) |
| `inv_test.json` | ['embed'], 42 Views | — | 2.447 s (n=2) |
| `onboarding.json` | ['mesh', 'partial', 'describe', 'embed'], 16 Views | — | 12.731 s (n=59) |
| `onboarding.json` | ['mesh', 'partial', 'describe', 'embed'], 42 Views | — | 18.293 s (n=59) |
| `onboarding_dgedi.json` | ['dgedi'], 16 Views | — | 11.296 s (n=3) |
| `onboarding_dgedi.json` | ['dgedi'], 42 Views | — | 10.209 s (n=3) |
| `onboarding_render.json` | ['render'], 16 Views | — | 14.452 s (n=5) |
| `onboarding_render.json` | ['render'], 42 Views | — | 34.681 s (n=5) |
| `partial_16_42.json` | ['partial', 'embed'], 16 Views | — | 2.981 s (n=3) |
| `partial_16_42.json` | ['partial', 'embed'], 42 Views | — | 5.777 s (n=3) |
| `query_latency_ycbv.json` | [16, 42], 42 Views | 1278 | 2.602 s (n=50) |
| `query_latency_ycbv.json` | [16, 42], 16 Views | 1278 | 2.184 s (n=50) |
| `query_latency_ycbv_geo.json` | [42], 42 Views | 1278 | 2.890 s (n=25) |
| `views_16_42.json` | ['describe', 'embed'], 16 Views | — | 9.938 s (n=3) |
| `views_16_42.json` | ['describe', 'embed'], 42 Views | — | 14.045 s (n=3) |
