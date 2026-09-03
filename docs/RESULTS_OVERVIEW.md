# Ergebnisübersicht — alle Stages

> **Generiert** von `tools/results_overview.py`. Nicht von Hand ändern —
> nach jedem Lauf neu erzeugen. Handgepflegte Tabellen driften.

Stand: 2026-09-04 01:12


## Stage 1 — SHREC'18 (39 Arme)

| Arm | nDCG | NN_sub | Ort |
|---|---|---|---|
| `A2_view_only_V16` | 0.5481 | 0.3260 | object_retrieval/ |
| `A2_view_only_V32` | 0.5426 | 0.3208 | object_retrieval/ |
| `A2_view_only_V42` | 0.5506 | 0.3337 | object_retrieval/ |
| `A2_view_only_V8` | 0.5302 | 0.3027 | object_retrieval/ |
| `A7_shape_only_V16` | 0.5227 | 0.3079 | object_retrieval/ |
| `A7_shape_only_V32` | 0.5300 | 0.3170 | object_retrieval/ |
| `A7_shape_only_V42` | 0.5353 | 0.3275 | object_retrieval/ |
| `A7_shape_only_V8` | 0.5119 | 0.2913 | object_retrieval/ |
| `A7f_full_fusion_shape_V42` | 0.5868 | 0.3413 | object_retrieval/ |
| `E1_oscar_cascade` | 0.4561 | 0.2347 | object_retrieval/ |
| `E1_shape_only` | 0.5353 | 0.3275 | object_retrieval/ |
| `E1_view_only` | 0.5506 | 0.3337 | object_retrieval/ |
| `E1a_text_only` | 0.4218 | 0.1304 | object_retrieval/ |
| `E1b_text_view` | 0.5519 | 0.3122 | object_retrieval/ |
| `E1c_full_fusion` | 0.5868 | 0.3413 | object_retrieval/ |
| `E1d_clip_pruned` | 0.4566 | 0.2308 | object_retrieval/ |
| `E2_both` | 0.6362 | 0.4645 | object_retrieval/ |
| `E2_chamfer_icp` | 0.6405 | 0.4717 | object_retrieval/ |
| `E2_chamfer_ransac` | 0.6405 | 0.4717 | object_retrieval/ |
| `E2_chamfer_unaligned` | 0.6251 | 0.4393 | object_retrieval/ |
| `E2_fitness` | 0.6251 | 0.4393 | object_retrieval/ |
| `E2b_fullmesh` | 0.5935 | 0.3598 | object_retrieval/ |
| `E2b_fullmesh_shape_only` | 0.4956 | 0.2822 | object_retrieval/ |
| `E4_siglip` | 0.5659 | 0.3051 | object_retrieval/ |
| `E4_siglip_only` | 0.5165 | 0.2642 | object_retrieval/ |
| `E6_rrf` | 0.5744 | 0.3175 | object_retrieval/ |
| `E7_ulip2_cross_shape_only` | 0.4809 | 0.2637 | object_retrieval/ |
| `E7_uni3d` | 0.5913 | 0.3455 | object_retrieval/ |
| `E7_uni3d_shape_only` | 0.5337 | 0.3094 | object_retrieval/ |
| `O1c_gedi_post_fusion` | 0.5961 | 0.4055 | object_retrieval/ |
| `O1e_gedi_with_base` | 0.6287 | 0.4588 | object_retrieval/ |
| `O2_clip_threshold` | 0.4561 | 0.2294 | object_retrieval/ |
| `O2_clip_threshold_cal` | 0.5186 | 0.2808 | object_retrieval/ |
| `O2_visual_first` | 0.5570 | 0.3408 | object_retrieval/ |
| `O4_V16` | 0.5820 | 0.3327 | object_retrieval/ |
| `O4_V32` | 0.5800 | 0.3289 | object_retrieval/ |
| `O4_V8` | 0.5714 | 0.3251 | object_retrieval/ |
| `O5_xyz_only` | 0.5880 | 0.3541 | object_retrieval/ |
| `O5_xyz_shape_only` | 0.5422 | 0.3598 | object_retrieval/ |

## Stage 2 — MI3DOR (56 Zeilen)

| Ordner | Modus | Arm | NN | FT | mAP |
|---|---|---|---|---|---|
| `results_mi3dor_oscar_legacy_v8` | fullmesh | `clip_dino_ulip_full` | 86.62 | 0.665 | 0.688 |
| `results_mi3dor_oscar_legacy_v8` | fullmesh | `clip_only` | 67.95 | 0.575 | 0.580 |
| `results_mi3dor_oscar_legacy_v8` | fullmesh | `clip_pruned_dino_ulip` | 85.78 | 0.575 | 0.593 |
| `results_mi3dor_oscar_legacy_v8` | fullmesh | `dino_only_full` | 81.96 | 0.591 | 0.607 |
| `results_mi3dor_oscar_legacy_v8` | fullmesh | `oscar_maxview` | 84.40 | 0.575 | 0.592 |
| `results_mi3dor_oscar_legacy_v8` | fullmesh | `oscar_softmax` | 84.42 | 0.575 | 0.592 |
| `results_mi3dor_oscar_legacy_v8` | fullmesh | `ulip_only_full` | 78.10 | 0.510 | 0.518 |
| `results_mi3dor_oscarplus_v2_tau037` | fullmesh | `clip_dino_ulip_full` | 83.42 | 0.620 | 0.635 |
| `results_mi3dor_oscarplus_v2_tau037` | fullmesh | `clip_only` | 67.95 | 0.575 | 0.580 |
| `results_mi3dor_oscarplus_v2_tau037` | fullmesh | `clip_pruned_dino_ulip` | 85.93 | 0.575 | 0.593 |
| `results_mi3dor_oscarplus_v2_tau037` | fullmesh | `dino_only_full` | 78.01 | 0.587 | 0.597 |
| `results_mi3dor_oscarplus_v2_tau037` | fullmesh | `oscar_maxview` | 84.79 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037` | fullmesh | `oscar_softmax` | 84.51 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037` | fullmesh | `ulip_only_full` | 78.10 | 0.510 | 0.518 |
| `results_mi3dor_oscarplus_v2_tau037` | partial | `clip_dino_ulip_full` | 84.11 | 0.620 | 0.640 |
| `results_mi3dor_oscarplus_v2_tau037` | partial | `clip_only` | 67.95 | 0.575 | 0.580 |
| `results_mi3dor_oscarplus_v2_tau037` | partial | `clip_pruned_dino_ulip` | 85.52 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037` | partial | `dino_only_full` | 78.01 | 0.587 | 0.597 |
| `results_mi3dor_oscarplus_v2_tau037` | partial | `oscar_maxview` | 84.79 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037` | partial | `oscar_softmax` | 84.51 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037` | partial | `ulip_only_full` | 68.11 | 0.453 | 0.451 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | fullmesh | `clip_dino_ulip_full` | 85.17 | 0.639 | 0.657 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | fullmesh | `clip_only` | 67.95 | 0.575 | 0.580 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | fullmesh | `clip_pruned_dino_ulip` | 86.16 | 0.575 | 0.593 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | fullmesh | `dino_only_full` | 83.03 | 0.629 | 0.647 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | fullmesh | `oscar_maxview` | 84.88 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | fullmesh | `oscar_softmax` | 85.04 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | fullmesh | `ulip_only_full` | 78.10 | 0.510 | 0.518 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | partial | `clip_dino_ulip_full` | 87.05 | 0.648 | 0.671 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | partial | `clip_only` | 67.95 | 0.575 | 0.580 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | partial | `clip_pruned_dino_ulip` | 85.22 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | partial | `dino_only_full` | 83.03 | 0.629 | 0.647 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | partial | `oscar_maxview` | 84.88 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | partial | `oscar_softmax` | 85.04 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean` | partial | `ulip_only_full` | 68.11 | 0.453 | 0.451 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw` | partial | `clip_dino_ulip_full` | 85.21 | 0.674 | 0.699 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw` | partial | `clip_only` | 67.95 | 0.575 | 0.580 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw` | partial | `clip_pruned_dino_ulip` | 85.04 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw` | partial | `dino_only_full` | 83.03 | 0.629 | 0.647 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw` | partial | `oscar_maxview` | 84.88 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw` | partial | `oscar_softmax` | 85.04 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw` | partial | `ulip_only_full` | 0.00 | 0.000 | 0.000 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | fullmesh | `clip_dino_ulip_full` | 86.57 | 0.682 | 0.705 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | fullmesh | `clip_only` | 67.95 | 0.575 | 0.580 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | fullmesh | `clip_pruned_dino_ulip` | 86.52 | 0.575 | 0.593 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | fullmesh | `dino_only_full` | 83.03 | 0.629 | 0.647 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | fullmesh | `oscar_maxview` | 84.88 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | fullmesh | `oscar_softmax` | 85.04 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | fullmesh | `ulip_only_full` | 78.10 | 0.510 | 0.518 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | partial | `clip_dino_ulip_full` | 86.57 | 0.682 | 0.705 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | partial | `clip_only` | 67.95 | 0.575 | 0.580 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | partial | `clip_pruned_dino_ulip` | 86.52 | 0.575 | 0.593 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | partial | `dino_only_full` | 83.03 | 0.629 | 0.647 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | partial | `oscar_maxview` | 84.88 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | partial | `oscar_softmax` | 85.04 | 0.575 | 0.592 |
| `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix` | partial | `ulip_only_full` | 78.10 | 0.510 | 0.518 |

## Stage 3 — BOP (22 Arme)

### 3a — Retrieval

| Arm | R@1 | MRR | Shape allein | YCB-V | T-LESS | LM-O |
|---|---|---|---|---|---|---|
| `3a_cross` | 0.4818 | 0.5971 | — | 0.732 | 0.332 | 0.464 |
| `3a_cross_fullmesh_v2` | 0.5151 | 0.6379 | 0.2272 | 0.726 | 0.394 | 0.478 |
| `3a_cross_geo` | 0.4582 | 0.5787 | — | 0.602 | 0.360 | 0.504 |
| `3a_cross_geo_borda` | 0.4818 | 0.5971 | — | 0.732 | 0.332 | 0.464 |
| `3a_cross_geo_distance` | 0.4229 | 0.5576 | — | 0.542 | 0.338 | 0.480 |
| `3a_cross_geo_fitness` | 0.4278 | 0.5569 | — | 0.558 | 0.337 | 0.480 |
| `3a_cross_v2` | 0.4818 | 0.5971 | 0.1997 | 0.732 | 0.332 | 0.464 |
| `3a_fullmesh` | 0.4639 | 0.6021 | — | 0.566 | 0.396 | 0.490 |
| `3a_oscar` | 0.3198 | 0.4043 | — | 0.498 | 0.214 | 0.304 |
| `3a_pc` | 0.4636 | 0.5844 | — | 0.671 | 0.350 | 0.400 |
| `3a_pc_fullmesh` | 0.3504 | 0.4592 | — | 0.635 | 0.157 | 0.436 |
| `3a_pc_fullmesh_v2` | 0.3878 | 0.4899 | 0.0089 | 0.740 | 0.159 | 0.446 |
| `3a_pc_geo` | 0.4131 | 0.5471 | — | 0.534 | 0.336 | 0.426 |
| `3a_pc_geo_distance` | 0.3725 | 0.5215 | — | 0.477 | 0.305 | 0.387 |
| `3a_pc_geo_fitness` | 0.3820 | 0.5249 | — | 0.490 | 0.314 | 0.390 |
| `3a_pc_v2` | 0.4636 | 0.5844 | 0.0211 | 0.671 | 0.350 | 0.400 |

### 3b / 3c — Pose

| Arm | Modus | D_sym Median (mm) | Δ Median (mm) | Deckung |
|---|---|---|---|---|
| `3b_cross` | 3b | 18.37 | 15.79 | 1.000 |
| `3b_cross_fullmesh` | 3b | 18.91 | 16.63 | 1.000 |
| `3b_cross_geo` | 3b | 28.79 | 26.07 | 1.000 |
| `3b_oscar` | 3b | 21.73 | 18.86 | 1.000 |
| `3c_cross` | 3c | 15.34 | 12.39 | 1.000 |
| `3c_smoke` | 3c | 17.13 | 12.58 | 1.000 |

## Stage 4 — Latenz


**`query_latency_ycbv`** — NVIDIA GeForce RTX 4090

- 42 Views — gesamt 6.719 s · io_load 7 ms, segment 264 ms, pointcloud 1 ms, encode_query 39 ms, clip 16 ms, dino 606 ms, ulip 226 ms, fusion 12 ms, retrieval_total 1032 ms, geometry 5451 ms

**`onboarding`** — NVIDIA GeForce RTX 4090

- 16 Views — gesamt 11.085 s · mesh 104 ms, describe 10179 ms, io_load_images 58 ms, embed_dino 119 ms, io_load_clouds 13 ms, embed_ulip 580 ms, cache_write 1 ms
- 42 Views — gesamt 20.297 s · mesh 98 ms, describe 18119 ms, io_load_images 142 ms, embed_dino 290 ms, io_load_clouds 34 ms, embed_ulip 1532 ms, cache_write 1 ms

**`onboarding_render`** — ?

- 16 Views — gesamt 14.187 s · render 14187 ms
- 42 Views — gesamt 34.963 s · render 34963 ms

## Bekannte Lücken

- **Stage 1 hat keine Zelle cross × full-mesh.** Vorhanden sind pc×partial (`E1_shape_only`), pc×full-mesh (`E2b_fullmesh_shape_only`) und cross×partial (`E7_ulip2_cross_shape_only`). Auf BOP ist genau die fehlende Kombination der beste Arm (R@1 0.5151).
- **`3c_cross_fullmesh` fehlt** — Zerlegung dazu.
