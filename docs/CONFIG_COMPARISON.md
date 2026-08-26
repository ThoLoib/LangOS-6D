# OSCAR+ — Cross-Stage Configuration Comparison

*Exhaustive line-out of every parameter set in the three experiment drivers/scripts and the
shared `pipeline/config.py`, compared across Stage 1 (SHREC'18), Stage 2 (MI3DOR) and Stage 3
(BOP). Companion to `EVALUATION_STORY_AND_PLAN.md`, `PIPELINE_IMPLEMENTATION.md`,
`EXPERIMENTS_IMPLEMENTATION.md`. Last updated 2026-08-26.*

**Sources:** `pipeline/config.py` (shared defaults) · `experiments/experiment1_shrec18_stage1.py`
+ `scripts/run_stage1_full.sh` (S1) · `object_retrieval/retrieval_mi3dor_eval_oscarplus.py` (S2) ·
`object_retrieval/stage3_gallery.py` + `object_retrieval/eval_bop_pose.py` + `scripts/run_stage3*.sh` (S3).

**Legend:** ✅ identical · ◆ intended difference (the axis that stage tests) · ⚠️ difference to state explicitly.

---

## 1 · Dataset & query
| param | Stage 1 | Stage 2 | Stage 3 | |
|---|---|---|---|---|
| dataset | SHREC'18 | MI3DOR | YCB-V · T-LESS · LM-O (BOP) | ◆ |
| queries | 2,101 RGB-D scans | ~10,500 RGB images | 12,284 RGB-D crops | ◆ |
| query crop | segmentation mask | object-centric | **GT** bbox + mask | ◆ |
| depth available | yes | no | yes | ◆ |

## 2 · Encoders (all frozen, training-free)
| param | S1 | S2 | S3 | |
|---|---|---|---|---|
| CLIP | ViT-B/32 | ViT-B/32 | ViT-B/32 | ✅ |
| appearance encoder | DINOv2-base | DINOv2-base | DINOv2-base | ✅ |
| shape encoder | ULIP-2 | ULIP-2 | ULIP-2 | ✅ |
| ULIP backbone | pointbert_colored | pointbert_colored | pointbert_colored | ✅ |
| ULIP num_points | 10,000 | 10,000 | 10,000 | ✅ |
| ULIP embed_dim | 1280 (ViT-bigG) | 1280 | 1280 | ✅ |
| ULIP colors | xyzrgb (True) | xyzrgb | xyzrgb | ✅ |
| ULIP checkpoint | `ulip2_pointbert_10k.pt` (**explicit**) | config default | config default | ⚠️ S1 pins it; S2/S3 resolve via `build_pipeline` — pin them too (see §Loose ends) |

## 3 · View aggregation
*(config default `dino_pooling=cls` is overridden to `mean` in every stage)*
| param | S1 | S2 | S3 | |
|---|---|---|---|---|
| num_views | 42 | 42 | 42 | ✅ |
| dino_pooling | **mean** (`SHREC_DINO_POOLING`) | **mean** (`MI3DOR_DINO_POOLING`) | **mean** (`_base_cfg`) | ✅ |
| dino_view_aggregation | topk_softmax | topk_softmax | topk_softmax | ✅ |
| dino_view_topk | **5** | **5** | **5** | ✅ |
| dino_view_temperature | 0.5 | 0.5 | 0.5 | ✅ |
| ulip_view_aggregation | topk_softmax | topk_softmax | topk_softmax | ✅ |
| ulip_view_topk | **5** *(was 8; fixed 2026-08-26)* | **5** | **5** | ✅ |
| ulip_view_temperature | 0.5 | 0.5 | 0.5 | ✅ |
| shape views pooled | `SHAPE_AGG_VIEWS = 42` *(was 16; fixed)* | 42 (prod `step5`) | 42 (prod `step5`) | ✅ |

## 4 · Shape mode & reference
| param | S1 | S2 | S3 | |
|---|---|---|---|---|
| ulip2_mode | **pc** (query point cloud) | **cross** (query image) | **cross** (pc via `--pc-query`) | ◆ depth availability |
| ulip2_use_partial_views | True | True | True | ✅ (full-mesh via A4 / `--fullmesh`) |
| S_text aggregation | max over 42 view descriptions | max | max | ✅ (OSCAR original; not softmax) |

## 5 · Fusion & scope
| param | S1 | S2 | S3 | |
|---|---|---|---|---|
| fusion_method | weighted_sum | weighted_sum | weighted_sum | ✅ |
| normalisation | min–max per channel | min–max | min–max | ✅ |
| weight_clip / dino / ulip | **0.3 / 0.4 / 0.3** | 0.3 / 0.4 / 0.3 | 0.3 / 0.4 / 0.3 | ✅ |
| BASE scope | full database | full database (auto-expand) | full database (`top_k = 10⁶`) | ✅ same effect |

## 6 · CLIP pruning / OSCAR cascade (only the cascade arms)
| param | S1 | S2 | S3 | |
|---|---|---|---|---|
| clip_top_k | 20 (`CLIP_PRUNE_K`) | 20 | 10⁶ (no prune) | ⚠️ S3 BASE full-DB; τ-prune only in E5 baseline |
| clip_prune_mode | threshold / topk (cascade arms) | threshold | none (threshold in E5) | ⚠️ same τ, applied per-arm |
| clip_tau (τ) | **0.37** (`CLIP_TAU_TEXT`) | **0.37** | **0.37** (E5 `oscar_cascade`) | ✅ |
| clip_fallback_k | 20 | 20 | 20 (E5) | ✅ |

## 7 · Geometry re-ranking (dGeDi) — the one real config difference
| param | S1 | S2 | S3 | |
|---|---|---|---|---|
| backend | **dGeDi** (`STAGE1_GEOMETRY_BACKEND=dgedi`) | — (no geometry) | **dGeDi** (`--dgedi`) | ◆ S2 has no query clouds |
| **shortlist depth K** | **50** (`--geom-k 50`) | — | **5** (`--dgedi-top-k 5`) | ⚠️ **differs** — cost (6× queries) + geometry-hurts-pose |
| ransac_keypoints | 6000 (`DGEDI_KP`) | — | 6000 (`--dgedi-repo`) | ✅ |
| ransac_max_iter | 10,000 (`DGEDI_MAXIT`) | — | 10,000 | ✅ |
| use_icp (refine) | True (`DGEDI_USE_ICP`) | — | True (`--dgedi-repo`) | ✅ |
| scale handling | unit-diameter (unitless CADs, diameters.json=1.0) | — | native metric (BOP mm) | ◆ dataset-driven |
| geometry signals | fitness / unaligned / chamfer_ransac / chamfer_icp / both-Borda | — | fitness-based re-rank | ⚠️ S1 sweeps all signals; S3 uses the re-rank diagnostically |

*Descriptor + RANSAC config is identical (6000 kp / 10k iter / ICP); only the shortlist depth K
differs (50 vs 5). See `EVALUATION_STORY_AND_PLAN.md` §5 for why K=5 in S3.*

## 8 · Pose estimation (Stage 3 only)
| param | S1 | S2 | S3 | |
|---|---|---|---|---|
| estimator | — | — | FoundationPose + ICP fallback | ◆ |
| refine-iter | — | — | 5 | — |
| proxy scale | — | — | true metric size (m→mm, no learned scale) | — |
| pose metric | — | — | D_sym (mm & /diam), F@{1%,5%} | ◆ |

## 9 · Determinism & reporting
| param | S1 | S2 | S3 | |
|---|---|---|---|---|
| PYTHONHASHSEED | 0 | 0 | 0 | ✅ |
| resample seed | SHA-256 (stable) | SHA-256 | SHA-256 | ✅ |
| pose/RANSAC seed | seeded | seeded | `--seed 0` | ✅ |
| reported depth | SHREC official + depth-K family | top-k = 15 | R@{1,5,10}, MRR, D_sym | ◆ metric convention |
| TOP_F (precision/recall cut) | — | 20 | — | — |

---

## Bottom line

**The retrieval stack is identical across all three stages** — encoders, 42 views, top-5-softmax
(k=5, τ=0.5), mean DINO pooling, weights (0.3, 0.4, 0.3), τ=0.37, dGeDi RANSAC config
(6000 kp / 10k iter / ICP), and the RNG seeds. This was confirmed by the 2026-08-26
config-comparability audit, which found and fixed the one arm that was ever out of spec:
Stage-1 shape ran at **16 views + top-8**, now corrected to **42 + top-5** to match S2/S3.

**Intended differences (◆):** shape mode (pc/cross), dataset, and the presence of a pose stage.

**Two differences to state explicitly in the thesis (⚠️):**
1. **Geometry shortlist depth K = 50 (S1) vs 5 (S3).** Deliberate: S3 has 6× the queries (K=50
   would be impractical) *and* geometry is net-negative for pose, so a shallow re-rank limits the
   damage. Everything else in the dGeDi config is identical.
2. **CLIP-pruning expression.** S1/S2 carry `clip_top_k = 20` for their OSCAR-cascade arms; S3's
   BASE is full-DB (`10⁶`) with τ-pruning only in the E5 baseline. The threshold **τ = 0.37 is
   the same**; it is just applied per-arm rather than as a global scope.

## Loose ends
- **⚠️ ULIP checkpoint pinning.** S1 sets `ulip2_checkpoint = ulip2_pointbert_10k.pt` explicitly;
  S2/S3 rely on `build_pipeline` resolving the same colored 10k checkpoint from
  `ulip2_backbone = pointbert_colored`. It *does* resolve to the same weights, but pinning it
  explicitly in the S2/S3 configs (as S1 does) would remove any chance of a silent divergence —
  recommended before the final thesis tables.
