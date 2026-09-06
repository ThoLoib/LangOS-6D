# Konfiguration → Ergebnis (generiert)

> Erzeugt von `tools/run_provenance.py --markdown`. Nicht von Hand aendern.
> Die Skript- und Variablenspalten stammen aus `S1_RUNS` im Werkzeug,
> alles andere aus den Ergebnisdateien.


## Stage 1 — SHREC'18

| Arm | Shape-Pass | Gewichte | Geo | Skript | Variablen | nDCG | NN_sub |
|---|---|---|---|---|---|---|---|
| `A2_view_only_V16` | `—` | 0.0/1.0/0.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5481 | 0.3260 |
| `A2_view_only_V32` | `—` | 0.0/1.0/0.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5426 | 0.3208 |
| `A2_view_only_V42` | `—` | 0.0/1.0/0.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5506 | 0.3337 |
| `A2_view_only_V8` | `—` | 0.0/1.0/0.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5302 | 0.3027 |
| `A7_shape_only_V16` | `ulip_pc_rgb_v16` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5227 | 0.3079 |
| `A7_shape_only_V32` | `ulip_pc_rgb_v32` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5300 | 0.3170 |
| `A7_shape_only_V42` | `ulip_pc_rgb` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5353 | 0.3275 |
| `A7_shape_only_V8` | `ulip_pc_rgb_v8` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5119 | 0.2913 |
| `A7f_full_fusion_shape_V42` | `ulip_pc_rgb_v42` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5868 | 0.3413 |
| `E1_oscar_cascade` | `—` | 0.0/1.0/0.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.4561 | 0.2347 |
| `E1_shape_only` | `ulip_pc_rgb` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5353 | 0.3275 |
| `E1_view_only` | `—` | 0.0/1.0/0.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5506 | 0.3337 |
| `E1a_text_only` | `—` | 1.0/0.0/0.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.4218 | 0.1304 |
| `E1b_text_view` | `—` | 0.43/0.57/0.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5519 | 0.3122 |
| `E1c_full_fusion` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5868 | 0.3413 |
| `E1d_clip_pruned` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.4566 | 0.2308 |
| `E2_both` | `ulip_pc_rgb` | 0.3/0.4/0.3 | both_borda | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.6362 | 0.4645 |
| `E2_chamfer_icp` | `ulip_pc_rgb` | 0.3/0.4/0.3 | chamfer_icp | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.6405 | 0.4717 |
| `E2_chamfer_ransac` | `ulip_pc_rgb` | 0.3/0.4/0.3 | chamfer_ransac | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.6405 | 0.4717 |
| `E2_chamfer_unaligned` | `ulip_pc_rgb` | 0.3/0.4/0.3 | chamfer_unaligned | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.6251 | 0.4393 |
| `E2_fitness` | `ulip_pc_rgb` | 0.3/0.4/0.3 | fitness | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.6251 | 0.4393 |
| `E2b_fullmesh` | `ulip_pc_fullmesh` | 0.3/0.4/0.3 | — | `run_stage1_fullmesh_color.sh` | FORCE_PARTIAL=— (Full-Mesh gewollt) · DINO=mean | 0.5935 | 0.3598 |
| `E2b_fullmesh_geo` | `ulip_pc_fullmesh` | 0.3/0.4/0.3 | chamfer_ransac | `run_stage1_geo_on_best.sh` | GEO_BACKEND=dgedi · DGEDI=shrec · --geom-k 50 · DINO=mean | 0.6417 | 0.4807 |
| `E2b_fullmesh_shape_only` | `ulip_pc_fullmesh` | 0.0/0.0/1.0 | — | `run_stage1_fullmesh_color.sh` | FORCE_PARTIAL=— (Full-Mesh gewollt) · DINO=mean | 0.4956 | 0.2822 |
| `E4_siglip` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5659 | 0.3051 |
| `E4_siglip_only` | `—` | 0.0/1.0/0.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5165 | 0.2642 |
| `E6_rrf` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5744 | 0.3175 |
| `E7_ulip2_cross` | `ulip_cross_rgb` | 0.3/0.4/0.3 | — | `run_stage1_cross_fullmesh.sh` | FORCE_PARTIAL=✓ (colored) · DINO=mean | 0.5588 | 0.3289 |
| `E7_ulip2_cross_fullmesh` | `ulip_cross_fullmesh` | 0.3/0.4/0.3 | — | `run_stage1_cross_fullmesh.sh` | FORCE_PARTIAL=— (Full-Mesh gewollt) · DINO=mean | 0.5511 | 0.3084 |
| `E7_ulip2_cross_fullmesh_shape_only` | `ulip_cross_fullmesh` | 0.0/0.0/1.0 | — | `run_stage1_cross_fullmesh.sh` | FORCE_PARTIAL=— (Full-Mesh gewollt) · DINO=mean | 0.4569 | 0.2028 |
| `E7_ulip2_cross_shape_only` | `ulip_cross_rgb` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.4809 | 0.2637 |
| `E7_uni3d` | `uni3d` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5913 | 0.3455 |
| `E7_uni3d_shape_only` | `uni3d` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5337 | 0.3094 |
| `O1c_gedi_post_fusion` | `—` | 0.43/0.57/0.0 | fitness | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5961 | 0.4055 |
| `O1e_gedi_with_base` | `ulip_pc_rgb` | 0.3/0.4/0.3 | both_borda_base | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.6287 | 0.4588 |
| `O2_clip_threshold` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.4561 | 0.2294 |
| `O2_clip_threshold_cal` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5186 | 0.2808 |
| `O2_visual_first` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5570 | 0.3408 |
| `O4_V16` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5820 | 0.3327 |
| `O4_V32` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5800 | 0.3289 |
| `O4_V8` | `ulip_pc_rgb` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5714 | 0.3251 |
| `O5_xyz_only` | `ulip_pc_xyz` | 0.3/0.4/0.3 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5880 | 0.3541 |
| `O5_xyz_shape_only` | `ulip_pc_xyz` | 0.0/0.0/1.0 | — | `run_stage1_full.sh` | FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec | 0.5422 | 0.3598 |

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
