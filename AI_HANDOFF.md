# AI Handoff – Branch `thesis-approach`

> Last updated: 2026-07-09

## Update 2026-07-09 (thesis-approach: align codebase with thesis methodology)

Branch `thesis-approach` created from `exp/ulip2v2` to implement all remaining thesis methodology components.

### Phase A: Foundation
- **A1**: DINOv2 CLS token pooling (`dino_pooling: "cls"` in config, `_pool_features()` uses `[:, 0]`)
- **A2**: Mask post-processing — largest connected component + dilation (`_refine_mask()` in step1)
- **A3**: View top-k alignment (`dino_view_topk = 5`, CNOS default per thesis Table 4.1)
- **A4**: Majority voting fusion (`_majority_voting()` Borda count in step6)
- **A5**: Trimmed Chamfer distance (`trimmed_chamfer_distance()` in utils.py, scipy cKDTree)

### Phase B: GeDi Integration (two-container HTTP architecture)
- **B1**: GeDi descriptor module (`pipeline/gedi_descriptors.py` — HTTP client to GeDi service)
- **B2**: Sub-step B2 geometry re-ranking (`pipeline/step_b2_geometry_reranking.py` — GeDi RANSAC + trimmed Chamfer)
- **B3**: GeDi replaces FPFH in Step 7 coarse alignment (`_ransac_with_descriptors()` — GeDi primary, FPFH fallback)
- **B4**: B2 wired into pipeline between fusion and scale gate (`run_pipeline.py`)
- **GeDi Docker**: Separate container (`Dockerfile.gedi`) — PyTorch 2.0.1+cu118, Open3D 0.18.0, pointnet2_ops compiled with CUDA 11.8
- **GeDi server**: `gedi_server.py` — Flask HTTP server with `/health` and `/compute_descriptors` endpoints

### Phase C: Evaluation Infrastructure
- **C1**: SHREC'18 ObjectNN+ eval wrapper (`retrieval_shrec18_eval_oscarplus.py`)
- **C2**: BOP-core pose eval (`eval_bop_pose.py` — YCB-V, T-LESS, LM-O, ADD/ADD-S metrics)
- **C3**: MI3DOR — already aligned, thesis defaults propagate via PipelineConfig

### Phase D: Encoder Alternatives
- **D1**: SigLIP as alternative appearance encoder (`--appearance-encoder siglip`, `google/siglip-base-patch16-224`)
- **D2**: Uni3D as alternative shape encoder (`--shape-encoder uni3d`, `BAAI/Uni3D`)

### ICP alignment fix
- Step 8 ICP correspondence distance changed from `icp_threshold=0.02` (10×vox) to `3×voxel_size` (thesis spec)

### New CLI flags
```
--appearance-encoder {dinov2,siglip}
--shape-encoder {ulip2,uni3d}
--geometry-reranking
--geometry-reranking-signal {gedi,chamfer,both}
--geometry-reranking-top-k N
--gedi-repo PATH
--gedi-checkpoint PATH
```

### New files
| File | Description |
|---|---|
| `pipeline/gedi_descriptors.py` | GeDi HTTP client with disk caching |
| `pipeline/step_b2_geometry_reranking.py` | Sub-step B2 geometry re-ranking |
| `Dockerfile.gedi` | GeDi Docker container |
| `gedi_server.py` | GeDi Flask HTTP server |
| `scripts/install_gedi.sh` | Standalone GeDi install script |
| `object_retrieval/retrieval_shrec18_eval_oscarplus.py` | SHREC'18 evaluation |
| `object_retrieval/eval_bop_pose.py` | BOP-core pose evaluation |

### Docker services
```
oscar           — OSCAR pipeline (PyTorch 2.x, CUDA 12.2)
foundationpose  — FoundationPose service (port 5050)
gedi            — GeDi descriptor service (port 5060)
```

### Remaining
- [ ] E1–E2: Grasping demo (lowest priority)
- [ ] Download SHREC'18, T-LESS, LM-O datasets
- [ ] Regenerate DINOv2 caches (CLS token changes embeddings)

---

## Update 2026-04-23 (OSCAR+ evaluation suite: shared eval_common + per-dataset wrappers + MI3DOR partial PCs)

### Motivation

The pre-existing `retrieval_mi3dor_eval_oscarplus.py` (~880 lines) mixed MI3DOR-specific iteration logic with shared pipeline init, metric accumulation, and result serialization. It was not reusable for YCBV-GSO or HouseCat6D, and the per-query JSON mixed "full-set" and "CLIP-gated" DINO/ULIP rankings under config-dependent names (`dino_only`, `ulip_only`, `fusion_all`, `fusion_clip_ulip`) that forced a reader to consult the config to know what each field represented.

The partial-pointcloud workflow was also only wired for datasets whose CAD layout matches `<cad_dir>/<obj_id>/`; MI3DOR uses `model/test/<category>/<obj_id>.obj`.

### What was delivered

**Shared evaluator** (`object_retrieval/eval_common.py`, new, ~680 lines):
- `EvalConfig` dataclass — pipeline+output settings shared across datasets (no dataset paths hard-coded).
- Metric helpers (`dcg_at_k`, `ideal_dcg_at_k`, `average_precision_from_binary`, `compute_anmrr`) and constant-memory incremental accumulators (`make_accum` / `update_accum` / `finalize_accum`).
- `build_pipeline(cfg, cad_mesh_items=None)` — loads CLIP descriptions, DINOv2 references, `ScoreFusion`, and optionally `ShapeMatcher`. Accepts pre-built `[(obj_id, mesh_path)]` for datasets where basename ≠ obj_id (e.g. YCBV's `<name>/meshes/model.obj`). When `cfg.ulip2_use_partial_views=True`, routes through the existing `_collect_partial_items` / partial cache infrastructure in `ShapeMatcher`.
- `run_query(...)` — runs CLIP once, **DINO once** at full depth, **ULIP once** at full depth, then derives CLIP-pruned DINO/ULIP variants by id-intersection on the full rankings. Preserves original stage scores; does not re-rank. See DECISIONS for the equivalence argument.
- `run_evaluation(cfg, to_label_fn, query_factory, components, ulip_cache=None)` — iterates queries, updates six metric accumulators, writes two JSON files per topk (`results_topk_K.json` and `metrics_summary_topk_K.json`).
- Helpers: `empty_pointcloud_result`, `query_has_depth`, `load_ulip_query_cache`, `pre_encode_ulip_queries`, `crop_by_bbox`, `crop_with_mask`, `_filter_dino_result_by_ids`, `_filter_shape_result_by_ids`.

**Dataset wrappers** (each ~100–170 lines, all in `object_retrieval/`):
- `retrieval_mi3dor_eval_oscarplus.py` — category-level, splits `airplane_test_0001` → `airplane`. Uses `cad_mesh_glob="../object_database/MI3DOR/model/test/*/*.obj"` and `_collect_filtered_cad_mesh_items()` to restrict CAD meshes to categories that have CLIP descriptions.
- `retrieval_ycbv_eval_oscarplus.py` — BOP format, instance-level (`to_label_fn = lambda x: x`). Obj_id extracted from grandparent directory (`<name>/meshes/model.obj` → `<name>`). Crops by visible bbox from `scene_gt_info.json`.
- `retrieval_housecat6d_eval_oscarplus.py` — BOP format, instance-level, mask-crop via `mask_visib/`. Excludes `bg/` and `collision/` CAD subdirs.

**Six explicit, unambiguous result variants** (replaces old `dino_only`/`ulip_only`/`fusion_all`/`fusion_clip_ulip`):
```
clip_only
dino_only_full
ulip_only_full
dino_only_clip_pruned       ← id-filter of dino_only_full by CLIP candidate set
ulip_only_clip_pruned       ← id-filter of ulip_only_full by CLIP candidate set
clip_pruned_dino_ulip       ← fusion(dino_pruned, ulip_pruned); also declared primary
```
All six are computed and written in one run — no config flag toggles behaviour.

**Auto-expanded full-ranking depth:**
`run_evaluation` computes `dino_full_top_k = max(cfg.dino_top_k, len(dino_rer._ref_embeddings))` and `ulip_full_top_k = max(cfg.ulip2_top_k, len(shape_m._cad_embeddings))` once at startup, logs whether the depth was auto-expanded, and threads both into `run_query`. `cfg.dino_top_k` / `cfg.ulip2_top_k` now control the reported top-k, not the ranking depth. The used depths are recorded in `metrics_summary_topk_K.json` under `config.dino_full_top_k_used` / `ulip_full_top_k_used`.

**Partial-pointcloud workflow for MI3DOR** (`rendering/generate_partial_pointclouds.py`):
- New `--mesh-glob` CLI arg + `_build_mesh_map_from_glob(pattern)` helper for datasets where `<cad_dir>/<obj_id>/` does not match (MI3DOR: `model/test/<category>/<obj_id>.obj`).
- `process_object()` now accepts an explicit `mesh_path=` kwarg; the CLI skips auto-discovery when it is provided.
- `main()` tracks and reports `obj_ids` that appear in `images_dir` but have no matching mesh.

**Query-side cache**: `precompute_ulip_query_embeddings.py` (new) loads only OpenCLIP ViT-bigG-14 in float16 (~5 GB, fits on a 6 GB GPU) and batch-encodes all query images to a `.pt` cache. The main eval script detects the cache (`ulip_query_cache_path`) and skips per-query image-encoder forward passes entirely.

### Per-query JSON schema (new)

```
category, filename, gt, pred,
clip_candidates, dino_candidates_full, dino_candidates_clip_pruned,
ulip_candidates_full, ulip_candidates_clip_pruned, matched_files,
clip_pruned_dino_ulip_pred, clip_pruned_dino_ulip_top5
```

`pred` ≡ `clip_pruned_dino_ulip_pred`. No `fusion_pred` / `fusion_top5` — removed to eliminate ambiguity about whether fused DINO/ULIP came from the full set or CLIP-pruned set.

### Runtime

Per query: 1× CLIP + 1× DINO (full, top_k = reference count) + 1× ULIP matmul (full, top_k = CAD count) + 1× fusion. The deeper top_k adds negligible cost (only the final sort widens; the underlying matmul over all references was already happening). The ULIP image encoder is called once; the encoded query is reused by the single ULIP `match` call. Compared to an earlier double-run design that ran DINO and ULIP twice (once full, once CLIP-gated), this saves one DINO pass + one ULIP matmul per query.

### How to use

```bash
cd OSCAR/object_retrieval

# 1. Pre-encode ULIP query embeddings (one-off, 6 GB GPU friendly):
python precompute_ulip_query_embeddings.py

# 2. (MI3DOR only, optional) generate partial-view CAD PCs:
python ../rendering/generate_partial_pointclouds.py \
    --mesh-glob "../object_database/MI3DOR/model/test/*/*.obj" \
    --images_dir ../object_images/MI3DOR/ \
    --num_points 10000

# 3. Run evaluations (produces six-variant JSON summaries):
python retrieval_mi3dor_eval_oscarplus.py
python retrieval_ycbv_eval_oscarplus.py
python retrieval_housecat6d_eval_oscarplus.py

# To enable partial-view ULIP for MI3DOR, set ulip2_use_partial_views=True in the EvalConfig.
```

Outputs land in `result_folder` as `results_topk_K.json` (per-query records) and `metrics_summary_topk_K.json` (metrics + config). Result directories and the ULIP query cache are now gitignored.

### Minor collateral

- `object_retrieval/retrieval_mi3dor_eval.py` (baseline, untouched otherwise): one-line fix — `bop_root` now points to `eval/datasets/mi3dor/image/test` instead of the parent dataset root.
- `.gitignore` extended to exclude `object_retrieval/results_*/`, `object_retrieval/top*_rankings_*.json`, `object_retrieval/ulip_query_cache_*.pt`, and `debug_output/`.

---

## Projektziel

Masterarbeit: **Shape-Aware Object Retrieval and 6D Pose Estimation** basierend auf dem OSCAR-Framework ([pullover00/OSCAR](https://github.com/pullover00/OSCAR)).

Kernidee: Das bestehende OSCAR-Retrieval (CLIP + DINOv2) um einen **3D-Shape-Kanal (ULIP-2)** erweitern. Eine partielle Punktwolke aus RGB-D wird per ULIP-2 mit den CAD-Modell-Punktwolken verglichen. Die Scores der drei Kanäle (CLIP, DINOv2, ULIP-2) werden fusioniert.

---

## Branch-Überblick

| Branch | Purpose | Status |
|---|---|---|
| `oscar` | Clean upstream mirror of pullover00/OSCAR | ✅ unchanged |
| `main` | Thesis scaffolding + AI docs | ✅ stable |
| `exp/oscar-repro` | OSCAR baseline reproduced (d3098bdd) | ✅ completed |
| `exp/ulip2` | Shape-aware pipeline (PC-ULIP + Fusion) | ✅ stable |
| `exp/ulip2-full` | ULIP full experiments (PC vs cross-modal image->PC) | ✅ stable |
| **`exp/ulip2v2`** | **Scale gate, rotation eval, eval suite** | 🟢 active |

---

## Update 2026-04-13 (Branch `exp/ulip2v2`: Scale Gate + Rotation Variance Evaluation)

### Motivation

Two weaknesses of the previous ULIP-2 shape matching were identified and addressed:

1. **Scale is not encoded in the ULIP embedding.** Both query and CAD point clouds are normalized to a unit sphere before encoding, so objects with similar shape but different physical size receive similar scores. The scale estimator already existed (Step 7) but was always applied to the top-1 fusion candidate without any opportunity to try alternatives.
2. **Rotation sensitivity in pc→pc mode.** `normalize_pointcloud()` removes translation and scale but not rotation. The observed partial PC is in the camera frame; CAD partial views were rendered in their own canonical frame. ULIP-2 is not guaranteed to be rotation-invariant, so orientation mismatch can degrade similarity scores.

### Scale Gate (`pipeline/config.py`, `pipeline/run_pipeline.py`)

A new post-fusion candidate selection mechanism iterates over the top fused candidates (in descending `fused_score` order) and accepts the first one whose scale estimate falls within a configurable range.

**Config fields added:**
```python
scale_gate_enabled: bool = False        # disabled by default
scale_gate_min: float = 0.8
scale_gate_max: float = 1.2
scale_gate_min_confidence: float = 0.0
scale_gate_max_candidates: int = 5
scale_gate_reject_policy: str = "fallback_best"  # "fallback_best" | "fail"
```

**CLI flags:** `--scale-gate`, `--scale-gate-min`, `--scale-gate-max`, `--scale-gate-min-confidence`, `--scale-gate-max-candidates`, `--scale-gate-reject-policy`

**Behavior:**
- Runs between Step 6 (fusion) and Step 7 (scale estimation).
- For each candidate in rank order: resolve mesh path → run `ScaleEstimator.estimate()` → check `scale_gate_min ≤ scale_factor ≤ scale_gate_max` and `confidence ≥ scale_gate_min_confidence`.
- First candidate that passes becomes `effective_best_model` for Steps 7, 8, and debug viz.
- If none pass: `fallback_best` returns the top-1 fusion candidate with a warning; `fail` skips Steps 7 and 8.
- All checked candidates are logged with rank, object_id, fused_score, scale_factor, confidence, and reason.
- Result stored in `results["scale_gate"]` with `selected_object_id`, `selected_rank`, `fallback_used`, `policy`, `candidates_checked`, and `rejections`.

**New helpers in `OSCARPlusPipeline`:**
- `_resolve_mesh_path_for_candidate(candidate)` — DRY mesh-path resolution (image-path fallback → `_find_cad_mesh()`). Replaces duplicated logic previously in Steps 7 and 8.
- `_select_candidate_with_scale_gate(fusion_result, observed_pc)` — iterates candidates, runs scale estimation, returns `(selected, scale_result, mesh_path, selected_rank, rejection_log)`.

**Known constraint:** Scale gate requires CAD meshes to have consistent real-world scale. If meshes are arbitrarily normalized, the gate will reject correct objects. Kept disabled by default for this reason.

### Rotation Variance Evaluation (`pipeline/step5_shape_matching.py`)

After ULIP scoring, for the top-K candidates only, a lightweight ICP registration is run between the observed partial point cloud and the candidate's best matching partial reference point cloud. Results are recorded per candidate but do not affect ranking by default.

**Config fields added:**
```python
ulip2_rotation_eval: bool = False       # disabled by default
ulip2_rotation_eval_top_k: int = 5
ulip2_rotation_eval_method: str = "icp"
ulip2_rotation_eval_weight: float = 0.0  # 0.0 = debug-only; >0 = optional rerank
```

**CLI flags:** `--ulip-rotation-eval`, `--ulip-rotation-eval-top-k`, `--ulip-rotation-eval-weight`

**`ShapeCandidate` extended:**
```python
best_partial_pc_path: str = ""      # path to best matching partial PC .npz
registration_fitness: float = 0.0  # ICP fitness (0..1)
registration_rmse: float = 0.0     # ICP inlier RMSE
```

**`ShapeMatcher` changes:**
- `_partial_view_paths: Dict[str, List[Tuple[int, str]]]` — stored during `_load_cad_models_partial()` so the per-view `.npz` paths are available for registration.
- `match()` populates `best_partial_pc_path` on each `ShapeCandidate` based on `best_view_idx`.
- `_run_rotation_eval()` — runs ICP for top-K candidates, logs fitness/RMSE, optionally adjusts `shape_score` if `ulip2_rotation_eval_weight > 0`.

**`_register_partial_pointclouds_icp()` (module-level function):**
- Loads reference partial PC from `.npz`.
- Normalizes both point clouds to unit sphere (same as ULIP preprocessing).
- Downsamples, estimates normals, runs Open3D point-to-plane ICP (50 iterations).
- Returns `(fitness, inlier_rmse, 4×4 transform)`.
- Operates only on partial views; requires `--ulip-partial-views` to be effective.

**Design rationale — why not random SO(3) augmentation:**
- Random rotations can reward accidental alignments and are non-deterministic.
- In `both` mode, rotating the query PC would mix one fixed image embedding with many rotated PC embeddings.
- ICP on known partial views answers the research question directly: does geometric alignment to the candidate's reference view improve confidence?

### Scale gate reliability fixes (second session 2026-04-13)

The scale gate and Step 7 both had reliability issues with partial/cut-off objects.

**Scale gate now uses `estimate_fast()`:**
- `_select_candidate_with_scale_gate()` rewritten to call `self.scale_estimator.estimate_fast(observed_pc, mesh_path)` (returns `(float, float)`) instead of the full RANSAC+ICP `estimate()`.
- Returns 4-tuple `(candidate, mesh_path, selected_rank, rejection_log)` — no `scale_result`. Step 7 always runs its full `estimate()` for the coarse alignment transform needed by Step 8.
- Rejection log entries include `rank`, `mesh_path`, `scale_factor`, `confidence`, `reason`.
- `results["scale_gate"]` enriched: `policy`, `selected_rank`, `fallback_used`, `candidates_checked`.
- `scale_gate_failed` flag (initialized `False`) set to `True` when `policy=fail` and no candidate passes; Steps 7 and 8 are then skipped with a warning.

**Step 7 ICP confidence fallback:**
- After RANSAC+ICP computes `confidence`, if `confidence < config.scale_icp_min_confidence` (default 0.15), scale factor is overridden with `estimate_fast()` result. ICP transform T is kept for coarse alignment.
- New config field: `scale_icp_min_confidence: float = 0.15`
- Observed case this fixes: scissors cut off at image boundary → ICP confidence 0.00 → scale 2.25× → FoundationPose received wrong-scaled mesh.

**Debug output additions:**
- `debug_viz.py` `save_debug_step5()`: ULIP top-3 → **top-5** (figure height 9, row spacing 0.175, ICP registration_fitness shown in score label when > 0).
- `run_pipeline.py` `_write_ranking_csvs()`: writes per-step ranking CSVs to `output_dir` at end of every run (regardless of `--debug-viz`):
  - `rankings_clip.csv` — rank, object_id, score, description
  - `rankings_dino.csv` — rank, object_id, dino_score, clip_score, best_view_path
  - `rankings_ulip.csv` — rank, object_id, shape_score, best_view_idx, registration_fitness, registration_rmse, cad_model_path
  - `rankings_fusion.csv` — rank, object_id, fused_score, clip_score, dino_score, ulip_score, fusion_method, cad_model_path
  - `rankings_scale_gate.csv` — written only when scale gate has rejections

**FoundationPose CUDA error:**
- Symptom: `RuntimeError: CUDA error: unknown error` on first FP call after container idle → silent ICP fallback.
- Cause: stale GPU context in the FP container. Not a code issue.
- Fix: `docker compose restart foundationpose` before running the pipeline.

### Other changes

- `scripts/run_debug_pipeline_foundationpose.sh`: Fixed trailing whitespace after backslashes (caused shell to treat `\ ` as a literal argument instead of line continuation).

### How to use

```bash
# Scale gate only
python3.11 -m pipeline.run_pipeline ... \
    --scale-gate --scale-gate-min 0.8 --scale-gate-max 1.2 --scale-gate-max-candidates 5

# Rotation eval (debug-only, no ranking change)
python3.11 -m pipeline.run_pipeline ... \
    --ulip-partial-views --ulip-rotation-eval --ulip-rotation-eval-top-k 5

# Rotation eval with optional reranking
python3.11 -m pipeline.run_pipeline ... \
    --ulip-partial-views --ulip-rotation-eval --ulip-rotation-eval-weight 0.1

# Both features together
bash /app/scripts/run_debug_pipeline_foundationpose.sh  # includes both flags
```

---

## Update 2026-04-09 (SAM2 warning fix, GT bbox compensation toggle, README file reference)

- **SAM2.1 model_type warning fix**: `step1_localization.py` now loads `Sam2Config` explicitly and overrides `model_type = "sam2"` before `Sam2Model.from_pretrained()`. This suppresses the spurious "model of type sam2_video to instantiate a model of type sam2" warning caused by HuggingFace metadata mismatch in `facebook/sam2.1-hiera-large`.
- **GT bbox_center compensation toggle**: New config flag `gt_bbox_center_compensation` (default: `False`) and CLI arg `--gt-bbox-compensation`. When OFF, GT wireframe overlay uses the pose directly without adjusting for mesh bbox-center offset. Made optional because the tuna_can mesh is near-centered (4.2mm offset) and the compensation was introducing visible error.
- **README pipeline file reference**: Added a table listing all 15 `pipeline/*.py` files with one-line descriptions, including `debug_viz.py`, `visualization.py`, `utils.py`, and `foundationpose_bridge.py`.

---

## Update 2026-04-03 (Multi-view aggregation for Steps 4 & 5)

- **Query-conditioned multi-view scoring** (inspired by OPEN, Chu et al. TCSVT 2024): Steps 4 and 5 now aggregate multiple reference views per object instead of relying on a single hard-max winner.
- **Step 4 (DINOv2)**: Object-level DINOv2 score is computed by selecting the top-k best-matching views for each candidate, applying a softmax with temperature over their cosine similarities, and computing a weighted sum. This replaces the previous hard-max over views.
- **Step 5 (ULIP-2)**: Same aggregation strategy applied to partial point cloud views in partial-view mode. The hard `max(dim=0)` is replaced by configurable multi-view aggregation.
- **Config**: New parameters `dino_view_aggregation`, `dino_view_topk`, `dino_view_temperature` (Step 4) and `ulip_view_aggregation`, `ulip_view_topk`, `ulip_view_temperature` (Step 5). Default: `topk_softmax` with k=4, τ=0.1.
- **Legacy mode**: Setting aggregation to `"max"` restores previous behavior.
- **Debuggability**: Best single view is still tracked for each object (`best_view_path` / `best_view_idx`).

---

## Update 2026-04-03 (Step 2 point cloud quality)

- **Depth conversion fix**: `run_pipeline.py` now prefers BOP `depth_scale` from `scene_camera.json` when available (raw × depth_scale / 1000 = meters), falling back to `config.depth_scale` (raw / depth_scale = meters). Removes the `if depth.max() > 100` heuristic — conversion is now deterministic and happens once before `pipeline.run()`.
- **Depth gating**: New 2D pre-filter (`depth_gate_enabled`, `depth_gate_tolerance=0.3`) removes depth outliers within the mask using median-relative gating before backprojection.
- **Configurable SOR/ROR**: Statistical outlier removal params (`sor_nb_neighbors`, `sor_std_ratio`) and optional radius outlier removal (`ror_enabled`, `ror_nb_points`, `ror_radius`) are now config-driven.
- **depth_trunc=2.0m**: Reduced from 10.0m default — 2m covers tabletop scenes without passing far-plane noise.
- **step2_pointcloud.py**: Removed internal `if depth.max() > 100` heuristic. Caller guarantees float32 meters.

---

## Update 2026-04-02 (SAM2.1 migration + audit fixes)

- **SAM → SAM2.1**: Step 1 now uses `facebook/sam2.1-hiera-large` instead of `facebook/sam-vit-large`. Better mask quality, especially at ambiguous boundaries. API change: `Sam2Model`/`Sam2Processor`, simplified `post_process_masks()`.
- **Step 2**: Tightened statistical outlier removal (`std_ratio` 2.0 → 1.0) for cleaner point clouds.
- **Step 1 query**: Localization now uses `visual_query` (LLM-extracted) instead of `detection_phrase`.
- **CLIP text fusion**: Intentionally disabled (`text_query` removed from `retrieve()` call) pending tuning.
- **Mesh path fix**: Null guard in `run_pipeline.py` prevents crash when no valid mesh found.
- **New**: `docs/PIPELINE_AUDIT.md` — comprehensive audit with 20 ranked findings and ablation recommendations.

---

## Update 2026-03-29 (Cleanup: load_object_descriptions → CLIPRetriever)

- Moved `load_object_descriptions()` from `pipeline/utils.py` into `CLIPRetriever._load_object_descriptions()` (static method).
- Aligns Step 3 with Step 4 pattern (data loading as class method, not standalone utility).
- No behavioral change.

---

## Update 2026-03-26 (Partial-to-partial point cloud matching for Step 5)

### Partial views preprocessing
- **New** `rendering/generate_partial_pointclouds.py`: standalone script that generates partial PCs from CAD meshes using front-face culling from 8 camera viewpoints.
  - Uses trimesh for mesh loading/normalization and surface sampling — no Blender needed.
  - Converts texture visuals to per-face colors; replicates the same bbox normalization as `rendering.py`.
  - Output: `{obj_id}_view{N}_partial.npz` files (keys: `points`, `colors`) alongside existing PNGs and camera matrices.
  - Performance: ~1s per object, ~10 min for 1051 YCBV-GSO objects × 8 views.

### Pipeline changes
- **Modified** `pipeline/config.py`: new field `ulip2_use_partial_views: bool = False`.
- **Modified** `pipeline/step5_shape_matching.py`:
  - `ShapeCandidate` gains `best_view_idx: int` field.
  - `ShapeMatcher.load_cad_models()` dual path: if partial views enabled, loads `.npz` files and encodes per-view embeddings `(num_views, embed_dim)`.
  - `match()` uses best-of-N-views scoring (max cosine similarity over views).
  - Separate cache: `.ulip_partial_cache_<hash>.pt` (distinct from full-mesh cache).
  - Fallback: objects without `.npz` files fall back to full mesh sampling.
- **Modified** `pipeline/debug_viz.py`: shows "Best View: N" label and loads matching view thumbnail.
- **Modified** `pipeline/run_pipeline.py`: new `--ulip-partial-views` CLI flag.

### How to use
```bash
# 1. Generate partial point clouds (one-time preprocessing, inside OSCAR container):
python3.11 rendering/generate_partial_pointclouds.py \
    --cad_dir object_database/ycbv_gso/ \
    --images_dir object_images/ycbv_gso/

# 2. Run pipeline with partial views:
python3.11 -m pipeline.run_pipeline \
    --rgb eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
    --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
    --camera eval/datasets/ycbv_gso/test/000048/scene_camera.json \
    --prompt "I need the red mug" \
    --descriptions object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json \
    --reference_images object_images/ycbv_gso/ \
    --cad_models object_database/ycbv_gso/ \
    --ulip_repo /ulip \
    --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt \
    --ulip_mode pc \
    --ulip-partial-views \
    --debug-viz --until-step 6 \
    --output debug_output
```

### All three ULIP modes work with partial views
| Mode | Query embedding | Reference embeddings | Scoring |
|---|---|---|---|
| `pc` | observed PC → ULIP PC encoder | 8 partial PCs → ULIP PC encoder | max over 8 views |
| `cross` | ROI image → OpenCLIP image encoder | 8 partial PCs → ULIP PC encoder | max over 8 views |
| `both` | weighted avg of pc + cross | 8 partial PCs → ULIP PC encoder | max over 8 views |

---

## Update 2026-03-26 (Debug-Visualisierung refactored into main pipeline)

### Refactoring: Debug als optionaler Modus der normalen Pipeline
- **Removed** `pipeline/debug_steps.py` entirely (was ~1473 lines with duplicated pipeline logic).
- **New** `pipeline/debug_viz.py` (~1070 lines): All rich visualization functions extracted as a standalone module.
  - PIL helpers, `save_debug_step1()` … `save_debug_step7_8()`, `_project_cad_wireframe()`, `save_pointcloud_interactive()`, `_done()`.
  - **Bug fix:** `_find_cad_mesh()` moved to module level (was nested inside `save_debug_step7_8`, unreachable from `run_debug()`).
- **Modified** `pipeline/run_pipeline.py`:
  - `OSCARPlusPipeline.__init__()` gains `debug_viz: bool = False` parameter.
  - `OSCARPlusPipeline.run()` gains `gt_data=None` parameter for GT-wireframe overlay.
  - Debug-viz hooks added after each step (only executed when `debug_viz=True`).
  - Mesh-path resolution before step 7: detects image-paths (`.png/.jpg`) in `cad_model_path` and falls back to `_find_cad_mesh()` lookup. Used by both steps 7 and 8.
  - New CLI flags: `--debug-viz`, `--until-step`.
  - `main()` loads GT data from `scene_gt.json` + `id_to_label.json` when `--debug-viz` and `--camera` are set.
  - **Bug fix:** `detection_prompt` (undefined variable) → `prompt_elements.detection_phrase` in step 1 visualization.
- **Modified** `scripts/run_debug_pipeline_foundationpose.sh`: Now calls `pipeline.run_pipeline --debug-viz` instead of `pipeline.debug_steps`.
- **New** `scripts/run_pipeline.sh`: Convenience script for normal pipeline with YCBV-GSO defaults.

### Behavioral changes vs. old `debug_steps.py`
1. CLIP `text_query`: old `run_debug()` called `clip.retrieve(roi)` without text query. The unified pipeline passes `visual_query` from prompt parsing — may give slightly different CLIP rankings.
2. Prompt parsing: old `run_debug()` duplicated the Ollama+heuristic logic; now uses `OSCARPlusPipeline._extract_prompt_elements()` directly.

### Start commands
```bash
# Debug mode (rich PNGs + PLY + HTML):
./scripts/run_debug_pipeline_foundationpose.sh

# Debug mode via run_pipeline.py:
python3.11 -m pipeline.run_pipeline ... --debug-viz --until-step 6

# Normal mode (no debug output):
python3.11 -m pipeline.run_pipeline --rgb ... --depth ... --prompt "..."

# Normal mode + simple viz:
python3.11 -m pipeline.run_pipeline ... --visualize
```

---

## Update 2026-03-24 (GT overlay + intrinsics/depth fixes)

### GT pose overlay in debug_07_scale_pose.png
- GT wireframe overlay (magenta) drawn alongside predicted (green) via `_project_cad_wireframe()`.
- Compensates for mesh bbox_center offset: subtracts `R_gt @ bbox_center` from GT translation.
- Adds "Predicted" / "GT" legend labels; Δt (mm) and ΔR (deg) error metrics to info panel.

### Camera intrinsics priority fix
- Camera loading moved **before** depth conversion so real `fx/fy/cx/cy` from `scene_camera.json` reach `generate()`.
- `config` values used as fallback only when `--camera` is absent.

### BOP depth_scale convention mismatch (gotcha)
- `scene_camera.json` `depth_scale` is a **multiplier** (e.g. 0.1 for this dataset).
- Pipeline divides depth by `config.depth_scale` (default 10000.0) — a **divisor** convention.
- Using the JSON value caused depths to be 100× too large, resulting in ~855mm translation error.
- Decision: always use `config.depth_scale` as divisor; ignore the JSON field entirely.

---

## Update 2026-03-20 (FoundationPose two-container HTTP integration)

Architecture change: FoundationPose now runs as a **separate Docker container** with an HTTP API instead of via subprocess/venv inside the OSCAR container.

- `foundationpose_server.py` in the FoundationPose repo: Flask server with `/health` and `/estimate_pose`
- `pipeline/foundationpose_bridge.py`: rewritten as HTTP client (uses httpx)
- `pipeline/step8_pose_estimation.py`: calls bridge via HTTP, removed subprocess and local-import paths
- `pipeline/config.py`: `foundationpose_url` replaces `foundationpose_python` and `foundationpose_repo_path`
- `docker-compose.yml`: added `foundationpose` service using `shingarey/foundationpose_custom_cuda121`
- FP container mounts OSCAR repo read-only at `/oscar` for CAD model access
- Bridge auto-translates CAD paths from `/app/...` to `/oscar/...`

Why this was done:
- The venv-inside-OSCAR approach failed because the OSCAR container (CUDA 12.2 runtime, Python 3.11) cannot compile pytorch3d/kaolin/nvdiffrast which require CUDA devel headers.
- Two containers with HTTP boundary gives full dependency isolation with no shared Python environment.

Removed (superseded):
- `foundationpose_python` config field and CLI arg
- `foundationpose_repo_path` config field and CLI arg
- Subprocess bridge logic in step8
- Local-import path (`_run_foundationpose_local`) in step8
- `../FoundationPose:/foundationpose` volume mount in oscar service
- MegaPose stub method in step8 (was always NotImplementedError)

Operational pattern:

```bash
# Start FP service (first time loads models ~30s)
docker compose up -d foundationpose

# Run OSCAR with FoundationPose
docker compose run --rm -it oscar bash
./scripts/run_debug_pipeline_foundationpose.sh
# or manually:
python3.11 -m pipeline.run_pipeline \
  --rgb eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
  --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
  --camera eval/datasets/ycbv_gso/test/000048/scene_camera.json \
  --prompt "I need the red mug" \
  --descriptions object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json \
  --reference_images object_images/ycbv_gso/ \
  --cad_models object_database/ycbv_gso/ \
  --ulip_repo /ulip \
  --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt \
  --pose_method foundationpose \
  --output debug_output \
  --debug-viz --until-step 8
```

If FoundationPose service is down or fails, Step 8 falls back to ICP automatically.

## Update 2026-03-19 (FoundationPose split-env integration — superseded)

> This approach was replaced by the two-container HTTP architecture on 2026-03-20.
> The subprocess bridge and venv approach did not work due to CUDA/ABI incompatibilities.
> Kept here for historical context only.

## Update 2026-03-18 (foundationpose prep + step1 cleanup)

- `pipeline/step1_localization.py`:
  - eine doppelte Kommentarzeile im Header entfernt (non-functional cleanup).
- `docker-compose.yml`:
  - zusätzliches Volume-Mount für FoundationPose (superseded by 2026-03-20 two-container setup).
- FoundationPose Setup-Status:
  - Repo lokal geklont (`~/thesis/FoundationPose`)
  - Docker image vorhanden (`foundationpose:latest`)

## Update 2026-03-17 (exp/ulip2-full)

- ULIP Step 5 erweitert um `ulip_mode`:
  - `pc`: nur Shape-Embedding (PointCloud -> CAD-PC)
  - `cross`: Image->PC Cross-Modal (OpenCLIP image branch)
  - `both`: gewichteter Mix (`ulip_image_weight`)
- `debug_steps.py` erweitert:
  - neue CLI-Args `--ulip_mode`, `--ulip_image_weight`
  - `query_image` wird an Step 5 durchgereicht
- GSO-CAD-Laden in Step 5 gefixt:
  - rekursive Mesh-Suche in Unterordnern (`meshes/model.obj`, `textured_simple.obj`)
  - vorher nur 21 Modelle, jetzt 1051 Modelle
- Performance-Fix Step 5:
  - CAD-Embeddings werden als Disk-Cache gespeichert (`.ulip_cache_<hash>.pt`)
  - erste Berechnung bleibt teuer, Folge-Runs laden Cache deutlich schneller
- Step 8 Pose-Fix:
  - falscher Bildpfad (`object_images/...png`) konnte als `cad_model_path` in Fusion landen
  - Fusion trennt jetzt `best_view_path` (DINO-Bild) von echtem `cad_model_path` (Mesh)
  - Debug löst Meshpfad robust auf, damit ICP ein OBJ/PLY/GLB bekommt
- Dependencies ergänzt:
  - `open-clip-torch` (für ULIP cross)
  - `trimesh` (für Overlay/Wireframe-Visualisierung)

---

## Aktueller Stand (exp/ulip2)

### Was funktioniert (End-to-End verifiziert, 2026-03-12)

- **Modulare 8-Schritt-Pipeline** in `pipeline/` – komplett durchgetestet:
  1. `step1_localization.py` – GroundingDINO + SAM → Maske + BBox (Konfidenz 0.847)
  2. `step2_pointcloud.py` – RGB-D + Maske → Open3D Point Cloud (4.201 Punkte bei 2mm Voxel)
  3. `step3_clip_retrieval.py` – Prompt → CLIP → Top-8 Kandidaten (master_chef_can 0.4702)
  4. `step4_dino_reranking.py` – ROI → DINOv2 → Top-5 Re-Ranking (master_chef_can 0.6447)
     - **Batch-Encoding** (32 imgs/pass) + **.pt Disk-Cache** für 9.459 Referenzbilder
     - Erstlauf ~5 Min, danach sofort aus Cache
  5. `step5_shape_matching.py` – **ULIP-2 Point Cloud Encoder** → Shape-Similarity
     - NaN-Scores werden gefiltert (Overflow bei pcd.colors → fix mit `np.clip`)
  6. `step6_fusion.py` – Weighted Sum mit Min-Max-Normalisierung pro Modalität
     - NaN-sichere `_minmax()` Funktion
     - Ergebnis: master_chef_can fused=0.8473
  7. `step7_scale_estimation.py` – RANSAC + ICP Coarse-Alignment → Partial-Aware Scale
     - scale=1.2968, conf=0.63 (2 beste Achsen)
  8. `step8_pose_estimation.py` – FoundationPose (HTTP) oder ICP mit Coarse-Alignment
     - ICP: fitness=0.9895, RMSE=0.007m

- **Debug-Visualisierung** (`pipeline/debug_viz.py`, ~1070 Zeilen, aktiviert via `--debug-viz`):
  - 7 diagnostische PNG-Bilder + interaktiver 3D-Viewer (HTML)
  - 3D-Wireframe-Overlay der CAD-Modell-Pose auf Szenenbild (via trimesh)
  - Automatische Panels: Lokalisierung, Punktwolke, CLIP, DINOv2, ULIP, Fusion, Scale+Pose

- **ULIP-2 Integration** (step5):
  - Lädt nur `point_encoder` + `pc_projection` (~400 MB statt ~5.5 GB für volles OpenCLIP)
  - Backbone: PointBERT Colored (10k Punkte xyzrgb → 1280-dim Embedding)
  - Checkpoint: `ulip2_pointbert_10k.pt` in `/ulip/checkpoints/`
  - ULIP-Repo als Volume gemountet (`../ULIP:/ulip` im Container)

- **LLM-basiertes Prompt Parsing**:
  - Ollama + `gemma3:4b` (localhost:11434, 30s Timeout)
  - Extrahiert Objektname, Farbe, Form, Material aus natürlichem Prompt
  - Fallback: regelbasierter Heuristic-Parser

- **Docker-Konfiguration**:
  - OSCAR Image: `tholoi/oscar-plus` (CUDA 12.2, Python 3.11)
  - FoundationPose Image: `shingarey/foundationpose_custom_cuda121` (CUDA 12.1, Python 3.8)
  - GPU-Support via `deploy.resources.reservations.devices`
  - ULIP-Volume: `../ULIP:/ulip`
  - HuggingFace Cache-Volume

### Bekannte Limitierungen

1. **ULIP-2 Shape Matching (full mesh)** liefert schwache Ergebnisse für partielle Punktwolken (single-view, ~4k Punkte vs. komplette 10k-CAD-Modelle). **Mitigation:** `--ulip-partial-views` schaltet auf partial-to-partial Vergleich um (best-of-8-views).
2. **ICP auf symmetrischen Objekten** (z.B. Dosen) kann Rotation um Symmetrieachse nicht eindeutig bestimmen.

### Was noch fehlt

1. **Evaluation-Script** – Über alle BOP-Szenen laufen, ULIP-2-augmentierte Top-K Accuracy berechnen, mit 75.95% Baseline vergleichen.
2. **MI3DOR Evaluation** – Shape-Retrieval auf MI3DOR testen (ULIP-2 sollte hier besonders helfen).
3. **HouseCat6D** – BOP-Testdaten beschaffen + evaluieren.
4. **Hyperparameter-Tuning** – Fusionsgewichte (aktuell 0.3/0.4/0.3), Top-K je Schritt, Voxelgröße.
5. **Fehlende MI3DOR-Beschreibungen** – 11/21 Kategorien noch nicht generiert.

---

## Datei-Inventar

### Pipeline-Dateien

| File | Lines | Description |
|---|---|---|
| `pipeline/__init__.py` | 19 | Package init with version `0.1.0` |
| `pipeline/config.py` | ~192 | Central `PipelineConfig` dataclass |
| `pipeline/run_pipeline.py` | ~1249 | Orchestrator + CLI + LLM parsing + debug-viz hooks + scale gate + ranking CSVs |
| `pipeline/debug_viz.py` | ~1247 | Debug visualization (7 PNGs + 3D viewer, ULIP top-5) |
| `pipeline/foundationpose_bridge.py` | ~130 | HTTP client for FoundationPose service |
| `pipeline/step1_localization.py` | ~323 | GroundingDINO + SAM2.1 |
| `pipeline/step2_pointcloud.py` | ~341 | RGB-D → Point Cloud (depth gating, SOR/ROR) |
| `pipeline/step3_clip_retrieval.py` | ~322 | CLIP text/image retrieval |
| `pipeline/step4_dino_reranking.py` | ~532 | DINOv2 re-ranking + multi-view aggregation + disk cache |
| `pipeline/step5_shape_matching.py` | ~1408 | ULIP-2 encoder + partial views + multi-view aggregation + rotation eval |
| `rendering/generate_partial_pointclouds.py` | ~332 | Partial PC preprocessing (front-face culling, mesh-glob support) |
| `pipeline/step6_fusion.py` | ~375 | Score fusion (weighted_sum, RRF, intersection) |
| `pipeline/step7_scale_estimation.py` | ~422 | RANSAC+ICP coarse alignment + partial-aware scale + fast bbox fallback |
| `pipeline/step8_pose_estimation.py` | ~357 | FoundationPose (HTTP) + ICP fallback |
| `pipeline/utils.py` | ~115 | Helper functions (camera loading, BOP format) |

### Konfiguration (config.py Defaults)

```python
# Punktwolke
voxel_size = 0.002              # Voxel-Downsampling (2mm, ~4000 Punkte)
depth_scale = 10000.0           # BOP depth: 16-bit PNG, 0.1mm Einheiten
depth_trunc = 2.0               # Max Tiefe in Metern (tabletop)

# Depth gating (2D pre-filter)
depth_gate_enabled = True
depth_gate_tolerance = 0.3      # ±30% around median

# SOR / ROR (3D post-filter)
sor_nb_neighbors = 10
sor_std_ratio = 1.0
ror_enabled = False

# ULIP-2
ulip2_backbone = "pointbert_colored"
ulip2_num_points = 10000
ulip2_embed_dim = 1280

# Multi-view aggregation (Steps 4 & 5)
dino_view_aggregation = "topk_softmax"
dino_view_topk = 5              # thesis Table 4.1 (CNOS default)
dino_view_temperature = 0.5
ulip_view_aggregation = "topk_softmax"
ulip_view_topk = 8
ulip_view_temperature = 0.5

# Encoder alternatives (ablations E4, E7)
appearance_encoder = "dinov2"    # "dinov2" | "siglip"
shape_encoder = "ulip2"          # "ulip2" | "uni3d"

# Fusion
weight_clip = 0.3
weight_dino = 0.4
weight_ulip = 0.3

# Geometry re-ranking (Sub-step B2)
geometry_reranking_enabled = True
geometry_reranking_signal = "gedi"  # "gedi" | "chamfer" | "both"
gedi_url = "http://gedi:5060"

# Ollama
ollama_host = "http://localhost:11434"
ollama_model = "gemma3:4b"

# Pose
pose_method = "icp"
foundationpose_url = "http://foundationpose:5050"

# Debug
gt_bbox_center_compensation = False
```

---

## Architektur

```
Prompt + RGB-D Image
       │
       ▼
┌──────────────────┐
│ 1. Lokalisierung │ GroundingDINO + SAM → Maske + BBox
└────────┬─────────┘
    ┌────┴────────────────┐
    ▼                     ▼
┌──────────┐    ┌──────────────────┐
│ 2. Point │    │ 3. CLIP Retrieval│ Prompt → Text Embeddings
│   Cloud  │    │    → Top-20      │
└────┬─────┘    └────────┬─────────┘
     │                   ▼
     │          ┌──────────────────┐
     │          │ 4. DINOv2 ReRank │ ROI → Image Embeddings
     │          │    → Top-5       │ (Batch + Disk-Cache)
     │          └────────┬─────────┘
     ▼                   │
┌──────────┐             │
│ 5. ULIP-2│ PC Embed    │
│  → Top-5 │ (NaN-safe)  │
└────┬─────┘             │
     └───────┬───────────┘
             ▼
    ┌──────────────────┐
    │ 6. Score Fusion  │ Weighted Sum / Majority Voting
    │    → Top-K       │ (NaN-safe Min-Max Norm.)
    └────────┬─────────┘
             ▼
    ┌──────────────────┐           ┌──────────────────────┐
    │ B2. Geometry     │── HTTP ──>│ GeDi Descriptors     │
    │  Re-ranking      │<── JSON ──│ (separate container)  │
    └────────┬─────────┘           └──────────────────────┘
             ▼                      RANSAC inlier + Chamfer
    ┌──────────────────┐
    │ 7. Scale Est.    │ GeDi/FPFH RANSAC+ICP → Scale
    └────────┬─────────┘ (reuses B2 transform)
             ▼
    ┌──────────────────┐           ┌──────────────────────┐
    │ 8. Pose Est.     │── HTTP ──>│ FoundationPose       │
    │  (OSCAR cont.)   │<── JSON ──│ (separate container)  │
    └──────────────────┘           └──────────────────────┘
      ↓ fallback: ICP              → 4×4 Pose Matrix
```

---

## How to Run

### Container starten
```bash
docker compose up -d foundationpose   # optional: start FP service
docker compose run --rm -it oscar bash
```

### Debug-Modus (empfohlen zum Testen)
```bash
./scripts/run_debug_pipeline_foundationpose.sh
# → debug_output/debug_01_localization.png ... debug_07_scale_pose.png

# oder manuell:
python3.11 -m pipeline.run_pipeline \
    --rgb eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
    --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
    --prompt "i need the blue coffee can" \
    --descriptions object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json \
    --reference_images object_images/ycbv_gso/ \
    --cad_models object_database/ycbv_gso/ \
    --camera eval/datasets/ycbv_gso/test/000048/scene_camera.json \
    --ulip_repo /ulip \
    --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt \
    --output debug_output \
    --debug-viz --until-step 8
```

### Volle Pipeline
```bash
./scripts/run_pipeline.sh
# oder:
python3.11 -m pipeline.run_pipeline \
    --rgb eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
    --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
    --prompt "pick up the mustard bottle" \
    --descriptions object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json \
    --reference_images object_images/ycbv_gso/ \
    --cad_models object_database/ycbv_gso/ \
    --camera eval/datasets/ycbv_gso/test/000048/scene_camera.json \
    --ulip_repo /ulip \
    --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt
```

### OSCAR Baseline (Vergleich)
```bash
python retrieval_combi_eval.py  # → 75.95% Top-1
```

---

## Bekannte Bugs & Workarounds

| Problem | Lösung | Datei |
|---|---|---|
| `knn_cuda` nicht installierbar | try/except + Warning | `ULIP/models/pointbert/dvae.py` |
| `pointnet2_ops` nicht installierbar | Optional import + `_fps_pytorch()` | `ULIP/models/pointbert/misc.py` |
| PyTorch 2.6 `weights_only` Error | `torch.load(..., weights_only=False)` | `step5_shape_matching.py` |
| `np.asarray(pcd.colors)` Overflow | `np.clip(raw, 0.0, 1.0)` | `step5_shape_matching.py` |
| NaN in ULIP Similarity Scores | `torch.where(nan_mask, -1.0, sims)` | `step5_shape_matching.py` |
| NaN in Fusion Min-Max Norm. | Filter NaN vor min/max | `step6_fusion.py` |
| Camera intrinsics KeyError | Fallback auf ersten Key | `pipeline/utils.py` |
| BOP `depth_scale` multiplier vs divisor mismatch | Always use `config.depth_scale` as divisor; ignore `scene_camera.json` field (it uses multiplier 0.1, not divisor) | `pipeline/run_pipeline.py` |
| Stale .pyc im Docker | `rm -rf /app/pipeline/__pycache__` nach Edits | manuell |

---

## Baseline-Ergebnisse (exp/oscar-repro)

| Datensatz | Methode | Top-1 Acc. | Paper | Anmerkung |
|---|---|---|---|---|
| YCBV_GSO | OSCAR full pipeline | **75.95%** | ~60% | GT-Masken statt GroundedSAM |
| MI3DOR | OSCAR full pipeline | NN=77.95% | NN=89.4% | Descriptions nur 10/21 Kat. |
