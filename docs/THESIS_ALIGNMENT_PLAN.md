# Thesis Alignment Plan

**Created:** 2026-07-09
**Branch:** `thesis-approach` (from `exp/ulip2v2`)
**Thesis:** "Language and Shape based 3D-Object Retrieval for Pose Estimation" (T. Loibelsberger, TU Wien)
**Thesis source:** `/mnt/c/Users/tholo/OneDrive - TU Wien/Masterarbeit/WS/thesis/`

---

## 1. Thesis Pipeline vs. Current Codebase

The thesis describes OSCAR+ as a three-step pipeline: **Step A** (Scene Grounding), **Step B** (Multi-Signal CAD Retrieval, split into B1 and B2), and **Step C** (CAD-to-Pose Estimation). The current codebase uses an 8-step sequential naming (Steps 1–8). The table below maps the thesis methodology to the current code state.

### Mapping: Thesis Steps → Code Steps

| Thesis Step | Description | Current Code | Status |
|---|---|---|---|
| **Step A** | Object localisation (GroundingDINO + SAM2.1) | `step1_localization.py` | ✅ Implemented |
| **Step A** | Mask post-processing: largest connected component + dilation | — | ❌ **Not implemented** |
| **Step A** | Point cloud extraction (depth gating, voxel, SOR) | `step2_pointcloud.py` | ✅ Implemented |
| **Step B1** | Semantic channel S_text (CLIP) | `step3_clip_retrieval.py` | ✅ Implemented |
| **Step B1** | Appearance channel S_view (DINOv2, CLS token, softmax top-k) | `step4_dino_reranking.py` | ⚠️ Uses mean pooling, not CLS token |
| **Step B1** | Shape channel S_shape (ULIP-2, partial views) | `step5_shape_matching.py` | ✅ Implemented |
| **Step B1** | Score fusion (weighted sum + majority voting) | `step6_fusion.py` | ⚠️ Weighted sum only; majority voting missing |
| **Step B1** | Full-database scoring (all channels, all candidates) | `run_pipeline.py` | ⚠️ CLIP cascade is default; full-database variant exists in eval |
| **Step B2** | GeDi correspondence score (RANSAC inlier count) | — | ❌ **Not implemented** |
| **Step B2** | Trimmed one-sided Chamfer score | — | ❌ **Not implemented** |
| **Step C** | Coarse alignment via GeDi + ICP | `step7_scale_estimation.py` | ⚠️ Uses FPFH, not GeDi |
| **Step C** | 6D pose (FoundationPose + ICP fallback) | `step8_pose_estimation.py` | ✅ Implemented |
| Scale gate | Post-fusion scale-based candidate filtering | `run_pipeline.py` | ✅ Implemented (thesis keeps as ablation only) |

### New Components Required by Thesis

| Component | Thesis Reference | Priority | Complexity |
|---|---|---|---|
| **GeDi geometric descriptors** | Sec. 3.3 (B2), Sec. 3.4 (C) | **Critical** | Medium — requires `gedi` Python package |
| **Trimmed one-sided Chamfer** | Sec. 3.3 (B2) | **Critical** | Low — nearest-neighbour distance, trim top 10% |
| **Sub-step B2 geometry re-ranking** | Sec. 3.3 | **Critical** | Medium — RANSAC on GeDi descriptors, re-rank top-k |
| **Mask post-processing** | Sec. 3.2 (Step A) | High | Low — OpenCV morphology, connected components |
| **Majority voting fusion** | Sec. 3.3 (E6) | High | Low — rank aggregation |
| **DINOv2 CLS token** | Sec. 3.5 | High | Trivial — 1 line change |
| **SigLIP encoder** | Sec. 3.5 (E4) | Medium | Low — drop-in replacement for DINOv2 |
| **Uni3D encoder** | Sec. 3.5 (E7) | Medium | Medium — different model loading |
| **SHREC'18 ObjectNN+ eval** | Sec. 5.2 (Stage 1) | **Critical** | Medium — new dataset wrapper + metric functions |
| **BOP-core eval** (YCB-V, T-LESS, LM-O) | Sec. 5.3/5.4 (Stages 3a/3b) | **Critical** | Medium — BOP target list iteration, pose metrics |
| **Full-database scoring mode** | Sec. 3.3 (B1) | High | Low — already possible via eval_common |
| **Isaac Sim grasping** | Sec. 5.6 (Stage 5) | Low | High — separate environment |
| **Antipodal grasp sampler** | Sec. 4.3 | Low | Medium |
| **Random seed determinism** | Sec. 3.4 | Medium | Low — thread seed through RANSAC, sampling |

### Parameter Mismatches (Current Code vs. Thesis Table)

| Parameter | Code (config.py) | Thesis (Table 4.1) | Action |
|---|---|---|---|
| View top-k (DINOv2) | `dino_view_topk = 8` | k_v = 5 (CNOS) | **Thesis says 5** — needs alignment |
| Softmax temperature | `dino_view_temperature = 0.5` | τ = 0.5 | ✅ Matches |
| DINOv2 pooling | mean(dim=1) | CLS token | **Change to CLS** |
| Geometric descriptor | FPFH (step7/8) | GeDi | **Change to GeDi** |
| ICP corr. distance (Step C) | 3×vox (step7), 0.02m (step8) | 3×vox | **Align step8** |
| B2 shortlist size | — | k=5 | **Add** |
| Chamfer trim | — | top 10% | **Add** |
| RANSAC config | 100k / 0.999 / 1.5×vox | 3-point / 1.5×vox / 10^5 / 0.999 | ✅ Matches |

---

## 2. Implementation Plan

### Phase A: Foundation (no GPU needed, no data dependencies)

**A1. DINOv2 CLS token** (trivial)
- File: `pipeline/step4_dino_reranking.py`
- Change: `.mean(dim=1)` → `[:, 0]`
- Note: Invalidates DINOv2 disk cache — document in commit
- Add config flag `dino_pooling: str = "cls"` with "mean" as legacy option

**A2. Mask post-processing in Step A** (low complexity)
- File: `pipeline/step1_localization.py`
- Add after mask generation:
  1. Largest connected component retention (`cv2.connectedComponentsWithStats`)
  2. Mask dilation (`cv2.dilate`, 5×5 kernel, 1 iteration)
- Add config fields: `mask_largest_cc: bool = True`, `mask_dilation_kernel: int = 5`, `mask_dilation_iterations: int = 1`

**A3. View top-k alignment** (trivial)
- File: `pipeline/config.py`
- Change: `dino_view_topk = 5` (CNOS default per thesis)
- Note: This is an ablation variable (O4), so make it easy to change

**A4. Majority voting fusion** (low complexity)
- File: `pipeline/step6_fusion.py`
- Add `majority_voting` fusion method alongside existing `weighted_sum` and `rank_fusion`
- Each channel produces independent ranking; final rank = sum of per-channel ranks (Borda count), with tie-breaking by weighted sum
- Add config: `fusion_method = "weighted_sum"` (already exists), add `"majority_voting"` option

**A5. Trimmed one-sided Chamfer distance** (low complexity)
- New function in a new module or in `step7_scale_estimation.py`
- For each query point, find nearest CAD point; discard top 10% distances; mean of rest
- Input: two Open3D point clouds
- Output: float score (lower = better geometric fit)

### Phase B: GeDi Integration (requires `gedi` package)

**B1. Install/integrate GeDi**
- GeDi: [poiesi/gedi](https://github.com/poiesi/gedi) — "Learning General and Distinctive 3D Local Deep Descriptors"
- Add to `requirements.txt`
- Verify compatibility with CUDA 12.2 / Python 3.11 / PyTorch in OSCAR container
- If incompatible: may need a lightweight reimplementation or a separate container (like FoundationPose)

**B2. GeDi descriptor computation**
- New module: `pipeline/gedi_descriptors.py`
- Compute GeDi keypoint descriptors on:
  1. Query partial point cloud (per-query, Step A output)
  2. CAD partial views (onboarding-time, cached per view)
- Cache format: `.gedi_cache_<hash>.pt` per CAD model

**B3. Sub-step B2: Geometry re-ranking**
- New module: `pipeline/step_b2_geometry_reranking.py`
- Input: top-k fused candidates from Step B1, query partial PC
- For each candidate:
  1. Load best-matching partial CAD view
  2. Compute GeDi descriptors (or load from cache)
  3. Run RANSAC in GeDi descriptor space → inlier count = S_GeDi
  4. Compute trimmed one-sided Chamfer → S_chamfer
  5. Re-rank by chosen signal (GeDi only, Chamfer only, or combined)
- Output: re-ranked candidate list
- Config: `geometry_reranking_enabled: bool = True`, `geometry_reranking_signal: str = "gedi"` (`"gedi"`, `"chamfer"`, `"both"`)

**B4. Step C: Replace FPFH with GeDi for coarse alignment**
- File: `pipeline/step7_scale_estimation.py`
- Replace FPFH feature computation with GeDi descriptors
- RANSAC + ICP refinement logic stays the same
- Reuse GeDi descriptors from B2 when available (avoid recomputation)

### Phase C: Evaluation Infrastructure

**C1. SHREC'18 ObjectNN+ evaluation wrapper** (Stage 1)
- New file: `object_retrieval/retrieval_shrec18_eval_oscarplus.py`
- Dataset: 2101 RGB-D query crops, 3308 ShapeNetSem CAD models, 20 categories
- Metrics: NN, FT, ST, E-measure, DCG, mAP, Recall@1, Recall@5
- Runs the full ablation grid (E1–E7, O1–O5)
- No Step A (queries are pre-cropped)
- Needs: SHREC'18 ObjectNN+ dataset download + preparation

**C2. BOP-core evaluation (Stages 3a/3b)**
- New file: `object_retrieval/eval_bop_pose.py`
- Datasets: YCB-V, T-LESS, LM-O
- Uses BOP target list (`test_targets_bop19.json`) — bypasses Step A
- Metrics: BOP-AR (VSD+MSSD+MSPD), ADD, ADD-S, Pose Success@K
- Retrieved-CAD vs. oracle-CAD configurations
- Needs: BOP dataset downloads, FoundationPose running

**C3. MI3DOR/MI3DOR2 evaluation update (Stage 2)**
- File: existing `object_retrieval/retrieval_mi3dor_eval_oscarplus.py`
- Run the frozen Stage 1 best configuration
- Compare directly with reproduced OSCAR baseline

### Phase D: Encoder Alternatives (Ablations) ✅

**D1. SigLIP encoder (E4)** ✅
- Added SigLIP as alternative appearance encoder in Step B1
- Config: `appearance_encoder: str = "dinov2"` → `"dinov2"` | `"siglip"`
- File: `pipeline/step4_dino_reranking.py` — `_load_model()` dispatches by encoder type
- CLI: `--appearance-encoder siglip`
- SigLIP model: `google/siglip-base-patch16-224` (via HuggingFace transformers)
- Separate cache files (.siglip_cache_* vs .dino_cache_*)

**D2. Uni3D encoder (E7)** ✅
- Added Uni3D as alternative shape encoder in Step B1
- Config: `shape_encoder: str = "ulip2"` → `"ulip2"` | `"uni3d"`
- File: `pipeline/step5_shape_matching.py` — `Uni3DEncoder` class, dispatch in `encode_pointcloud()`
- CLI: `--shape-encoder uni3d`
- Uni3D model: `BAAI/Uni3D` (torch.hub or HuggingFace)
- PC-only mode (no cross-modal image encoder)

### Phase E: Grasping Demo (Stage 5)

**E1. Antipodal grasp sampler**
- New module (location TBD)
- Sample surface points + normals from retrieved proxy CAD mesh
- Antipodal pair selection, collision rejection, geometric scoring

**E2. Isaac Sim integration**
- Separate environment / container
- Scene generation, grasp execution, success measurement
- Out of scope for initial implementation

---

## 3. Execution Order (Recommended)

```
Priority 1 (immediate, no external deps):
  A1 → DINOv2 CLS token
  A2 → Mask post-processing
  A3 → View top-k = 5
  A4 → Majority voting fusion
  A5 → Trimmed Chamfer distance

Priority 2 (requires GeDi package):
  B1 → GeDi installation / compatibility check
  B2 → GeDi descriptor computation + caching
  B3 → Sub-step B2 geometry re-ranking
  B4 → Replace FPFH with GeDi in Step C

Priority 3 (evaluation infrastructure):
  C1 → SHREC'18 ObjectNN+ eval wrapper
  C2 → BOP-core pose evaluation
  C3 → MI3DOR update

Priority 4 (encoder alternatives for ablations):
  D1 → SigLIP
  D2 → Uni3D

Priority 5 (grasping demo):
  E1 → Antipodal grasp sampler
  E2 → Isaac Sim
```

---

## 4. Validation Strategy

Each change should be validated as far as possible without GPU/Docker/full datasets:

| Change | Validation |
|---|---|
| DINOv2 CLS token | `python -c "import pipeline.step4_dino_reranking"` — syntax check |
| Mask post-processing | Unit test with a synthetic binary mask |
| Majority voting | Unit test with synthetic rank lists |
| Trimmed Chamfer | Unit test with known point cloud pairs |
| GeDi descriptors | `python -c "import gedi"` — import check |
| B2 geometry re-ranking | Dry run with `--until-step 6 --debug-viz` |
| SHREC'18 eval | Dataset loading check (no GPU needed) |

For full validation: Docker + GPU + datasets required. Document any step that cannot be validated without these.

---

## 5. Files Expected to Change

| File | Type of Change |
|---|---|
| `pipeline/config.py` | Add new config fields (mask, GeDi, B2, pooling, view-k) |
| `pipeline/step1_localization.py` | Add mask post-processing |
| `pipeline/step4_dino_reranking.py` | CLS token option |
| `pipeline/step6_fusion.py` | Add majority voting |
| `pipeline/step7_scale_estimation.py` | Replace FPFH with GeDi |
| `pipeline/run_pipeline.py` | Wire B2 into pipeline, update CLI args |
| `pipeline/gedi_descriptors.py` | **New** — GeDi descriptor computation + caching |
| `pipeline/step_b2_geometry_reranking.py` | **New** — geometry re-ranking module |
| `object_retrieval/retrieval_shrec18_eval_oscarplus.py` | **New** — SHREC'18 eval wrapper |
| `object_retrieval/eval_bop_pose.py` | **New** — BOP pose evaluation |
| `requirements.txt` | Add `gedi` (if pip-installable) |
| `docs/THESIS_ALIGNMENT_PLAN.md` | This file (updated as work progresses) |
| `README.md` | Updated to reflect new pipeline structure |
| `CLAUDE.md` | Updated goals and current problem |
| `AI_HANDOFF.md` | Updated with new changes |

---

## 6. Open Questions / Blockers

1. **GeDi package compatibility** — Does `gedi` work with CUDA 12.2 / Python 3.11 / PyTorch 2.x? Needs testing.
2. **SHREC'18 ObjectNN+ dataset** — Is it downloaded? Where is it stored? Need to verify availability.
3. **BOP-core datasets** — Are T-LESS and LM-O downloaded? YCB-V is available under `eval/datasets/ycbv_gso/`.
4. **View top-k = 5 vs 8** — Thesis says 5 (CNOS), code has 8. The thesis also lists O4 ablation for V ∈ {8, 16, 32}. Confirm the default.
5. **Full-database vs. cascade** — The thesis default is full-database scoring. The pipeline currently uses CLIP cascade (top-20 → DINO/ULIP). The eval suite already supports both. Need to wire full-database mode into the pipeline proper.
6. **Isaac Sim** — Not available in the current Docker setup. Stage 5 is lower priority.

---

## 7. Completed Work (Updated as Progress is Made)

- [x] Phase 1: Documentation updated (README.md, CLAUDE.md, AI_HANDOFF.md)
- [x] Branch `thesis-approach` created from `exp/ulip2v2`
- [x] Thesis chapters read and compared with codebase
- [x] This plan created
- [x] A1: DINOv2 CLS token (`step4_dino_reranking.py` — `_pool_features()` with config `dino_pooling`)
- [x] A2: Mask post-processing (`step1_localization.py` — `_refine_mask()`: largest CC + dilation)
- [x] A3: View top-k alignment (`config.py` — `dino_view_topk = 5`, CNOS default per thesis Table 4.1)
- [x] A4: Majority voting fusion (`step6_fusion.py` — `_majority_voting()` Borda count)
- [x] A5: Trimmed Chamfer distance (`utils.py` — `trimmed_chamfer_distance()`, scipy cKDTree, 10% trim)
- [x] B1: GeDi descriptor module (`gedi_descriptors.py` — wrapper with caching, Open3D Feature format)
- [x] B2: Geometry re-ranking module (`step_b2_geometry_reranking.py` — GeDi RANSAC + trimmed Chamfer)
- [x] B3: FPFH→GeDi in Step 7 (`step7_scale_estimation.py` — GeDi primary, FPFH fallback, B2 transform reuse)
- [x] B4: B2 wired into pipeline (`run_pipeline.py` — between fusion and scale gate, CLI flags added)
- [x] C1: SHREC'18 ObjectNN+ eval wrapper (`retrieval_shrec18_eval_oscarplus.py` — full-database scoring, 20 categories)
- [x] C2: BOP-core pose eval (`eval_bop_pose.py` — YCB-V/T-LESS/LM-O, ADD/ADD-S metrics, Stages 3a/3b)
- [x] C3: MI3DOR eval already aligned — thesis defaults propagate via PipelineConfig (CLS token, topk_softmax k=5). Just needs to be run.
- [x] D1: SigLIP encoder alternative (`step4_dino_reranking.py` — `--appearance-encoder siglip`)
- [x] D2: Uni3D encoder alternative (`step5_shape_matching.py` — `--shape-encoder uni3d`)
- [x] GeDi Docker container built and verified (Dockerfile.gedi — PyTorch 2.0.1+cu118, Open3D 0.18.0)
- [ ] E1–E2: Grasping demo
