# OSCAR Pipeline Audit Report

**Date**: 2026-03-29
**Scope**: All 8 pipeline steps, config, CLI, rendering assumptions
**Files audited**: `pipeline/config.py`, `step1_localization.py`, `step2_pointcloud.py`, `step3_clip_retrieval.py`, `step4_dino_reranking.py`, `step5_shape_matching.py`, `step6_fusion.py`, `step7_scale_estimation.py`, `step8_pose_estimation.py`, `run_pipeline.py`, `utils.py`, `visualization.py`, `debug_viz.py`, `rendering/rendering.py`

---

## Ranked Findings

Each finding is tagged: **[H]** = High impact, **[M]** = Medium, **[L]** = Low.
Impact reflects potential change in retrieval accuracy, pose quality, or end-to-end correctness.

---

### 1. [H] Fusion weights (0.3 / 0.4 / 0.3) are untuned defaults

- **File**: `config.py:101-103`
- **Current**: `weight_clip=0.3, weight_dino=0.4, weight_ulip=0.3`
- **Issue**: These weights determine the final object ID selection. There is no evidence they were tuned on any evaluation set. The DINO weight being highest bakes in an assumption that visual appearance is always more informative than semantics (CLIP) or shape (ULIP). For shape-ambiguous objects (bowls, cups) ULIP may deserve more weight; for textureless objects, CLIP may dominate.
- **Risk**: Wrong object selected → everything downstream (scale, pose) fails.
- **Recommendation**: Grid search or Bayesian optimization over a validation set. Start with 0.2/0.4/0.4 as an alternative.

### 2. [H] Depth heuristic `if depth.max() > 100` applied twice

- **File**: `step2_pointcloud.py:138`, `run_pipeline.py:999`
- **Current**: Both the CLI loader and `PointCloudGenerator.generate()` check `depth.max() > 100` and divide by `depth_scale`. If the CLI converts first and the resulting max is still > 100 (because `depth_scale` is wrong or the scene is far away), step 2 divides **again**.
- **Impact**: Double-division produces incorrect point clouds (100x too small), breaking scale estimation and pose.
- **Recommendation**: Convert exactly once. Remove the heuristic from `generate()` and make the CLI conversion authoritative, or add a flag marking whether conversion already happened.

### 3. [H] `detection_confidence` used for both box and text thresholds

- **File**: `step1_localization.py:147-148`
- **Current**: `threshold=self.config.detection_confidence, text_threshold=self.config.detection_confidence` — both set to 0.3.
- **Issue**: GroundingDINO's box threshold and text threshold control different things. The text threshold filters token-level phrase confidences; the box threshold filters detection confidences. Using the same value means you can't independently tune text recall vs. detection precision. For noisy/ambiguous prompts, a lower text threshold but higher box threshold may be better.
- **Recommendation**: Split into `detection_box_threshold` and `detection_text_threshold`.

### 4. [H] CLIP `visual_query` is extracted but never passed to `retrieve()`

- **File**: `run_pipeline.py:371-375`
- **Current**: `visual_query` is extracted from the prompt (line 372) but `retrieve()` is called without `text_query=visual_query` or `text_query_weight`. The `retrieve()` method supports text-image fusion (`text_query`, `text_query_weight` params) but they are never used.
- **Impact**: The entire LLM-based prompt parsing (Ollama) produces attributes that are used only for the detection phrase but **not** for CLIP retrieval, wasting the attribute information. Text-image fusion in CLIP could significantly improve retrieval for ambiguous queries.
- **Recommendation**: Pass `text_query=visual_query, text_query_weight=0.3` (tunable). This was likely the original intent.

### 5. [H] Min-max normalization in fusion is sensitive to outliers and candidate set composition

- **File**: `step6_fusion.py:196-215`
- **Current**: Scores are min-max normalized per modality across the candidate set. If ULIP only has scores for 3 out of 20 candidates (because only 3 were in its top-K), the others get ULIP=0.0 after normalization. But if all 3 ULIP candidates have similar scores, min-max stretches them to [0, 1], massively amplifying small differences.
- **Impact**: Score scale depends on the candidate composition, making fusion unstable across scenes.
- **Recommendation**: Consider z-score normalization or raw cosine scores (which are already roughly calibrated). At minimum, handle the case where a modality has very few candidates by clamping the range.

### 6. [H] Step 5 candidate filtering: ULIP only compares against CLIP top-K, not DINO top-K

- **File**: `run_pipeline.py:453-457`
- **Current**: `candidate_ids` for shape matching comes from `clip_retriever.get_candidate_labels(results["clip_retrieval"])`, i.e., the 20 CLIP candidates. DINO re-ranking narrows this to 5, but ULIP still compares against all 20. This means objects that DINO rejected can still score high in ULIP and re-enter the fusion.
- **Issue**: This is both a design decision and a potential confounder. If ULIP's purpose is to provide an independent shape signal, comparing against CLIP's 20 makes sense. But it means fusion can override DINO's visual filtering.
- **Recommendation**: Document as deliberate. Consider experimenting with restricting ULIP to DINO's top-K.

### 7. [M] Statistical outlier removal hardcoded in step 2

- **File**: `step2_pointcloud.py:170-172`
- **Current**: `nb_neighbors=10, std_ratio=2.0` — hardcoded, not in config.
- **Impact**: Too aggressive removal can clip object edges; too lenient keeps noise. For small objects or thin structures (forks, markers), these defaults may remove valid points.
- **Recommendation**: Make configurable or at least document the assumption.

### 8. [M] CLIP model `ViT-B/32` is the smallest CLIP variant

- **File**: `config.py:54`
- **Current**: `clip_model_name: str = "ViT-B/32"`
- **Impact**: `ViT-L/14` or `ViT-L/14@336px` have significantly better retrieval quality. On a 4050 (6 GB), `ViT-L/14` fits comfortably for inference.
- **Recommendation**: Ablation: ViT-B/32 vs. ViT-L/14 vs. ViT-L/14@336px. Expect 2-5% recall improvement.

### 9. [M] DINOv2-base vs. DINOv2-large/giant

- **File**: `config.py:65`
- **Current**: `dino_model_name: str = "facebook/dinov2-base"` (86M params, 768-dim)
- **Impact**: `dinov2-large` (300M, 1024-dim) or `dinov2-giant` are known to be significantly better for visual matching tasks. On a 4050, `dinov2-large` fits for inference.
- **Recommendation**: Ablation: base vs. large. Expect noticeable re-ranking improvement.

### 10. [M] ROI background color (205, 205, 205) affects CLIP and DINO embeddings

- **File**: `step1_localization.py:284`
- **Current**: Background is set to gray (205, 205, 205) for the ROI crop. This is a design choice from OSCAR.
- **Impact**: Both CLIP and DINOv2 see this gray background. If the rendered reference images use a different background, there's a systematic domain shift. The rendered images from `rendering.py` use Blender's default background (likely transparent or white depending on compositing).
- **Recommendation**: Verify that ROI background matches rendering background. If not, align them.

### 11. [M] `voxel_size = 0.002` affects both step 2 and steps 7/8

- **File**: `config.py:48`, used in `step2_pointcloud.py:162`, `step7_scale_estimation.py:187`, `step8_pose_estimation.py:271`
- **Current**: 2mm voxel size used everywhere. In step 7/8, it's used with `or 0.005` fallback.
- **Issue**: 2mm is quite fine for RANSAC feature matching — FPFH features at `voxel_size * 5 = 0.01m` radius. For larger objects this is fine, but for the RANSAC correspondence distance (`voxel_size * 1.5 = 3mm`), this is very tight and may reject valid correspondences.
- **Recommendation**: Consider separate voxel sizes for point cloud generation vs. registration.

### 12. [M] ICP `max_correspondence_distance` differs between step 7 and step 8

- **File**: `step7_scale_estimation.py:239` uses `voxel_size * 3`, `step8_pose_estimation.py:318` uses `config.icp_threshold` (0.02m)
- **Issue**: With `voxel_size=0.002`, step 7 ICP uses 6mm and step 8 ICP uses 20mm. This inconsistency means step 7 alignment can fail (too tight) while step 8 uses the result from step 7 as initial pose with a wider tolerance.
- **Recommendation**: Make step 7's ICP distance configurable or tied to `icp_threshold`.

### 13. [M] RANSAC convergence criteria: 100000 iterations, 0.999 confidence

- **File**: `step7_scale_estimation.py:223`, `step8_pose_estimation.py:310`
- **Current**: `RANSACConvergenceCriteria(100000, 0.999)` — same in both places.
- **Impact**: These are reasonable defaults but runtime-heavy for large point clouds. The 0.999 confidence with 100k max iterations can take seconds. For evaluation at scale, this adds up.
- **Recommendation**: Make configurable. Consider 50000/0.99 for faster evaluation runs.

### 14. [M] `depth_scale = 10000.0` default doesn't match BOP standard

- **File**: `config.py:46`
- **Current**: `depth_scale: float = 10000.0` (raw / 10000 = meters)
- **Issue**: BOP datasets use `depth_scale` as a multiplier in `scene_camera.json` (typically 0.1 or 1.0), meaning `raw_depth * depth_scale = mm`. The pipeline's `depth_scale` is a divisor. This inversion is confusing and the default value (10000) doesn't match standard BOP convention.
- **Impact**: If `scene_camera.json` has a `depth_scale`, it's loaded but ignored: `run_pipeline.py:996-1000` always uses `config.depth_scale`.
- **Recommendation**: Use the BOP `depth_scale` from `scene_camera.json` when available. The camera intrinsics loading function already extracts it (`utils.py:114`).

### 15. [M] DINOv2 uses average pooling over patch tokens (not CLS token)

- **File**: `step4_dino_reranking.py:157`
- **Current**: `features = outputs.last_hidden_state.mean(dim=1)` — averages all patch tokens.
- **Issue**: DINOv2 is designed to be used with the CLS token (`outputs.last_hidden_state[:, 0]`) for global image representation. Average pooling dilutes the representation with background patches.
- **Impact**: Potentially reduces re-ranking quality, especially when the object is small relative to the ROI crop.
- **Recommendation**: Compare CLS token vs. mean pooling. CLS is the documented approach for DINOv2.

### 16. [L] Ollama prompt parsing `temperature=0, num_predict=40`

- **File**: `run_pipeline.py:734`
- **Issue**: `num_predict=40` tokens is very tight. Complex prompts with compound nouns (German) might get truncated.
- **Recommendation**: Increase to 60-80 for safety.

### 17. [L] Single best detection used (no multi-instance support in pipeline)

- **File**: `step1_localization.py:208`
- **Current**: `best_idx = det["scores"].argmax().item()` — only the highest-confidence detection is used.
- **Impact**: If the target object appears multiple times or the wrong instance is selected, there's no recovery.
- **Note**: `localize_all()` exists but the pipeline only calls `localize()`.

### 18. [L] RRF k-parameter = 60 is hardcoded

- **File**: `step6_fusion.py:326`
- **Current**: `k_param: int = 60` — standard IR default.
- **Impact**: With only 3 ranking lists and short lists (5-20 items), this value may overly smooth differences. Lower k (e.g., 10-20) would give more weight to top-ranked items.
- **Recommendation**: Make configurable if RRF is used for evaluation.

### 19. [L] CAD mesh sampling in steps 7 and 8: always 10000 points

- **File**: `step7_scale_estimation.py:264`, `step8_pose_estimation.py:255`
- **Current**: `number_of_points=10000` hardcoded.
- **Impact**: For very small or very large meshes, 10k may be too few or too many. Affects FPFH quality and ICP convergence.

### 20. [L] Normal estimation radius 0.01m hardcoded in steps 7 and 8

- **File**: `step7_scale_estimation.py:266`, `step8_pose_estimation.py:257`
- **Current**: `KDTreeSearchParamHybrid(radius=0.01, max_nn=30)` for normals on the raw (not downsampled) point cloud.
- **Impact**: 10mm radius may be too small for coarse objects or too large for fine details. Should ideally scale with object size.

---

## Top 10 Parameters Shortlist

| # | Parameter | Location | Current | Experiment Range | Expected Impact |
|---|-----------|----------|---------|------------------|-----------------|
| 1 | `weight_clip / weight_dino / weight_ulip` | `config.py:101-103` | 0.3/0.4/0.3 | Grid: {0.1,0.2,0.3,0.4,0.5} with sum=1 | High — controls final selection |
| 2 | `clip_model_name` | `config.py:54` | ViT-B/32 | ViT-B/32, ViT-L/14, ViT-L/14@336px | Medium-High — CLIP recall |
| 3 | `dino_model_name` | `config.py:65` | dinov2-base | dinov2-base, dinov2-large | Medium-High — re-ranking quality |
| 4 | `text_query_weight` (currently unused) | `step3_clip_retrieval.py:230` | 0.0 (disabled) | 0.0, 0.1, 0.2, 0.3 | High — enables prompt attributes |
| 5 | DINOv2 pooling method | `step4_dino_reranking.py:157` | mean | mean, CLS token | Medium — embedding quality |
| 6 | `ulip2_mode` | `config.py:90` | cross | pc, cross, both | Medium — shape signal type |
| 7 | `ulip2_use_partial_views` | `config.py:92` | False | True, False | Medium — domain mismatch |
| 8 | `fusion_method` | `config.py:100` | weighted_sum | weighted_sum, rank_fusion | Medium — ranking strategy |
| 9 | `detection_confidence` | `config.py:35` | 0.3 | 0.2, 0.3, 0.4, 0.5 | Medium — detection recall/precision |
| 10 | `voxel_size` | `config.py:48` | 0.002 | 0.001, 0.002, 0.005 | Medium — PC quality & registration |

---

## Minimal Ablation Matrix

For a thesis experiment section, run this 2-phase ablation:

### Phase 1 — Retrieval accuracy (steps 1-6, fast, no GPU-heavy pose)

| Experiment | What changes | Metric |
|---|---|---|
| Baseline | Default config | Recall@1, Recall@5 |
| +text_query | `text_query_weight=0.2` in step 3 | Recall@1, Recall@5 |
| +ViT-L/14 | `clip_model_name=ViT-L/14` | Recall@1, Recall@5 |
| +dinov2-large | `dino_model_name=facebook/dinov2-large` | Recall@1, Recall@5 |
| +CLS pooling | DINOv2 CLS instead of mean | Recall@1, Recall@5 |
| +partial-views | `--ulip-partial-views --ulip_mode pc` | Recall@1, Recall@5 |
| +weights-tuned | Optimized fusion weights | Recall@1, Recall@5 |

### Phase 2 — Pose quality (on best retrieval config)

| Experiment | What changes | Metric |
|---|---|---|
| ICP default | voxel_size=0.002, icp_threshold=0.02 | ADD-S, translation error |
| FoundationPose | `--pose_method foundationpose` | ADD-S, translation error |

---

## High-Leverage Parameters

1. **Fusion weights** — Biggest single lever. Wrong ID = zero pose quality. Tune first.
2. **text_query_weight** — Currently 0% text influence in CLIP. Enabling it is a free feature already coded but unused.
3. **CLIP model size** — ViT-L/14 is a near-drop-in upgrade. Memory fits on 4050.
4. **DINOv2 CLS vs mean** — Single line change, potential significant gain.

## Hidden Confounders

1. **Double depth conversion** (Finding #2) — Can silently break point clouds and all downstream steps. Must fix before evaluation.
2. **ROI background color mismatch** between query (gray 205) and reference renderings (may be white/black) — Systematically biases CLIP and DINO similarity.
3. **BOP depth_scale from scene_camera.json is loaded but ignored** — The pipeline extracts it but `run_pipeline.py` only uses `config.depth_scale`.
4. **DINOv2 cache fingerprint** uses file count + newest mtime — Adding an unrelated file (e.g., `.npz`) to an object folder would change the hash but `.npz` isn't counted (only `.png/.jpg`). However, deleting/renaming images would invalidate the cache correctly.

## Design Decisions to Document

1. **ULIP compares against CLIP top-20, not DINO top-5** — Deliberate: provides independent shape signal.
2. **Single detection per prompt** — Pipeline uses best detection only.
3. **Fusion uses min-max normalization** — Chosen over z-score or raw scores.
4. **Step 7 uses top-2 axes for partial-aware scale** — Assumes one axis is always under-observed.
5. **Step 8 ICP uses step 7's coarse alignment as initial pose** — Different correspondence thresholds between steps.

## Architectural Opportunity: Unified OpenCLIP Model for Steps 3 and 5

When `ulip2_mode=cross`, step 5 already loads **OpenCLIP ViT-bigG-14** and encodes the ROI image to produce the query embedding. Step 3 currently loads a **separate ViT-B/32** model for the same ROI image.

These two image encodings are redundant if both steps use the same model. If step 3 were switched from ViT-B/32 to OpenCLIP ViT-bigG-14:

1. The ROI image embedding computed in step 3 would be **identical** to what step 5 produces.
2. Step 5 could accept the precomputed embedding as a parameter instead of re-encoding — no model reload, no second forward pass.
3. Only one image encoder would need to live in VRAM simultaneously (~2 GB for ViT-bigG-14 vs. ~340 MB ViT-B/32 + ~2 GB ViT-bigG-14 today).

**Side effect**: CLIP retrieval quality in step 3 improves significantly (ViT-bigG-14 >> ViT-B/32).

**Prerequisite**: CAD text description embeddings must be regenerated using ViT-bigG-14's text encoder (different embedding space from ViT-B/32 — existing description caches would be invalid).

**When this is not helpful**: If `ulip2_mode=pc` (cross encoder never loaded), the sharing opportunity disappears and you'd pay the full ViT-bigG-14 cost just for step 3.

**Recommended condition for implementation**: `ulip2_mode in ("cross", "both")` — share the model; `ulip2_mode="pc"` — keep ViT-B/32 or upgrade independently.

---

## Quick Wins (< 1 hour each)

1. **Enable text_query in CLIP retrieval** — Pass `visual_query` to `retrieve()`, add `--clip-text-weight` CLI arg. ~20 lines changed.
2. **Switch DINOv2 to CLS token** — Change line 157 in `step4_dino_reranking.py` from `.mean(dim=1)` to `[:, 0]`. Invalidate cache. 1 line.
3. **Fix double depth conversion** — Remove `if depth.max() > 100` from `step2_pointcloud.py:138-139`, make CLI conversion authoritative. ~5 lines.
4. **Split detection thresholds** — Add `detection_text_threshold` to config. ~10 lines.
5. **Use BOP depth_scale from scene_camera.json** — Already loaded, just need to wire it through. ~5 lines.

## Risky Changes (need careful testing)

1. **Changing CLIP model to ViT-L/14** — Invalidates all CLIP caches, may change description embedding quality. Need to regenerate.
2. **Changing fusion weights** — Requires proper eval framework first. Grid search without metrics is guesswork.
3. **Changing voxel_size** — Affects point cloud density, FPFH quality, and registration in three different steps. Cascading effects.
