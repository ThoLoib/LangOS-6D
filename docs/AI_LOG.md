# AI Log

## 2026-03-18 Step 1 localization cleanup

Goal
- Verify what changed in `pipeline/step1_localization.py` and document it.

Changes
- Confirmed a non-functional cleanup in Step 1:
  - removed one duplicated comment line in the module header.
- No runtime logic, model call, threshold, or output schema changed.

Impact
- Behavior unchanged.
- Improves readability and avoids confusion during code review.

## 2026-03-18 FoundationPose setup and compose update

Goal
- Prepare a reproducible local setup for FoundationPose and document current switch status.

Changes
- Host setup:
  - cloned `NVlabs/FoundationPose` to `~/thesis/FoundationPose`
  - installed Docker image `foundationpose:latest`
- OSCAR integration prep:
  - updated `docker-compose.yml` volumes with `../FoundationPose:/foundationpose`
- Codebase check:
  - verified `pipeline/step8_pose_estimation.py` still uses a FoundationPose template path (`NotImplementedError`) and falls back to ICP.

Impact
- FoundationPose assets are available from the OSCAR container path `/foundationpose`.
- Runtime behavior of Step 8 is unchanged until estimator integration is implemented.

## 2026-03-17 ULIP Full Mode, CAD Cache, and Pose Path Fixes

Goal
- Enable side-by-side experiments for ULIP `pc` vs ULIP `cross` (full cross-modal) in the debug pipeline.
- Fix slow Step 5 by caching CAD embeddings.
- Fix Step 8 failures caused by image paths being passed as CAD mesh paths.

Changes
- Modified `pipeline/step5_shape_matching.py`:
  - added ULIP cross-modal image encoding support (`open-clip-torch`)
  - recursive CAD mesh discovery (supports `meshes/model.obj` style layouts)
  - added CAD embedding disk cache (`.ulip_cache_<hash>.pt`)
  - stores cached CAD embeddings on CPU to reduce repeated GPU work
- Modified `pipeline/debug_steps.py`:
  - added CLI args `--ulip_mode` and `--ulip_image_weight`
  - forwards `query_image` to Step 5
  - robust CAD mesh path resolution before Step 7/8
- Modified `pipeline/step6_fusion.py`:
  - separated DINO `best_view_path` (image) from true `cad_model_path` (mesh)
  - prevents Step 8 from trying to load PNG as mesh
- Modified dependencies:
  - root `requirements.txt`: added `open-clip-torch`, `trimesh`

Results
- `open_clip` import error resolved.
- CAD loading count corrected from 21 to 1051 models for ycbv_gso.
- Step 8 no longer fails with `CAD-Mesh leer: ...png` due to wrong path propagation.
- Step 5 subsequent runs are faster due to cache reuse.

## 2026-03-05 ULIP-2 Pipeline Integration + Visualization

Goal
- Implement full 8-step shape-aware retrieval pipeline on branch exp/ulip2.
- Integrate real ULIP-2 point cloud encoder (PointBERT Colored, 10k points, 1280-dim).
- Add LLM-based prompt parsing via Ollama.
- Add visualization module for intermediate results.
- Debug and fix all runtime errors during initial test runs.

Changes

### New: pipeline/ module (17 files)
- Created `pipeline/__init__.py`: Package init, version 0.1.0.
- Created `pipeline/config.py`: Central `PipelineConfig` dataclass with all hyperparameters for 8 steps, ULIP-2 fields (repo_path, checkpoint, backbone, num_points, embed_dim), Ollama fields (host, model, timeout), fusion weights.
- Created `pipeline/run_pipeline.py` (747 lines): Main orchestrator with CLI (argparse). Handles all 8 steps sequentially. LLM-based object name extraction using `ollama.Client.chat()` with heuristic fallback `_extract_object_name_heuristic()`. Camera intrinsics loaded from BOP scene_camera.json with image_id extracted from filename. Visualization calls integrated when `--visualize` flag set.
- Created `pipeline/step1_localization.py`: GroundingDINO + SAM wrapper → mask + BBox.
- Created `pipeline/step2_pointcloud.py`: RGB-D + mask → Open3D point cloud generation.
- Created `pipeline/step3_clip_retrieval.py`: CLIP text/image retrieval → Top-K candidates.
- Created `pipeline/step4_dino_reranking.py`: DINOv2 image-based re-ranking.
- Created `pipeline/step5_shape_matching.py` (669 lines): **Full ULIP-2 integration.** `ULIP2PointEncoder(nn.Module)` wraps point_encoder + pc_projection from checkpoint. `ShapeMatcher` class loads model, encodes CAD library, computes cosine similarity. Only loads ~400 MB (point encoder + projection), NOT full OpenCLIP ViT-bigG-14 (~5.5 GB). Supports pointbert_colored, pointbert, pointnext backbones. Uses `sys.path.insert(0, ulip_repo_path)` for ULIP imports. Fixed `torch.load(..., weights_only=False)` for PyTorch 2.6 compatibility.
- Created `pipeline/step6_fusion.py`: Score fusion (weighted_sum, rank_fusion, intersection).
- Created `pipeline/step7_scale_estimation.py`: BBox comparison for scale estimation.
- Created `pipeline/step8_pose_estimation.py`: FoundationPose / ICP wrapper.
- Created `pipeline/utils.py` (142 lines): `crop_with_mask()`, `load_depth_image()`, `load_camera_intrinsics()` (with fallback to first key), `load_object_descriptions()`, `ensure_dir()`.
- Created `pipeline/visualization.py` (375 lines): Visualization for all steps. `viz_step1_mask()` (RGB + green mask overlay + bbox), `viz_step1_roi()` (cropped ROI), `viz_step2_depth_masked()` (depth falsecolor), `viz_step2_pointcloud()` (XY/XZ/YZ projections), `viz_step3_clip()`, `viz_step4_dino()`, `viz_step5_shape()`, `viz_step6_fusion()` (all top-5 grids with reference thumbnails + scores), `viz_summary()` (all steps in one grid).
- Created `pipeline/requirements.txt`: Pipeline-specific dependencies.

### Modified: docker-compose.yml
- Added volume `../ULIP:/ulip` to mount ULIP repo into container.
- Added `deploy.resources.reservations.devices` with `driver: nvidia, count: all, capabilities: [gpu]` for GPU access.

### Modified: requirements.txt (root)
- Added: `ollama`, `open3d`, `easydict`, `timm`, `pyyaml_env_tag`, `termcolor`.

### Patched: ULIP repo (separate repo, not in OSCAR)
- `ULIP/models/pointbert/dvae.py`: Wrapped `from knn_cuda import KNN` in try/except ImportError with warning. knn_cuda requires custom CUDA build that's not available in our Docker image; it's only needed for DGCNN, not PointTransformer.
- `ULIP/models/pointbert/misc.py`: Made `pointnet2_ops` optional via try/except. Added `_fps_pytorch(xyz, npoint)` — pure PyTorch Farthest Point Sampling fallback using iterative distance updates. `fps()` function now uses pointnet2_ops when available, falls back to `_fps_pytorch()`.

Bugs Fixed
- KeyError '0' in `load_camera_intrinsics()`: BOP scene_camera.json uses string keys like "1", not "0". Fixed to extract image_id from filename and fallback to first available key.
- `open3d` missing: Added to requirements.txt.
- ULIP repo not found at host path `/home/tholoi/thesis/ULIP`: Added Docker volume mount `../ULIP:/ulip`. Pipeline uses `/ulip` paths inside container.
- GPU not detected in container: Added `deploy.resources.reservations.devices` block.
- `easydict`, `timm`, `pyyaml_env_tag` missing (ULIP transitive deps): Added to requirements.txt.
- `termcolor` missing: Added to requirements.txt.
- PyTorch 2.6 `weights_only` default changed to True: Fixed `torch.load()` call in step5 with `weights_only=False`.
- Ollama model default mismatch: Config had `llama3.2`, but `start.sh` pulls `mistral-small3.1`. Corrected config default.

Commands Run
- docker compose build
- docker compose run --rm -it oscar bash
- python -m pipeline.run_pipeline --rgb ... --depth ... --prompt ... (multiple test iterations)
- Various pip installs during debugging

Files Touched
- pipeline/__init__.py (new)
- pipeline/config.py (new)
- pipeline/run_pipeline.py (new)
- pipeline/step1_localization.py (new)
- pipeline/step2_pointcloud.py (new)
- pipeline/step3_clip_retrieval.py (new)
- pipeline/step4_dino_reranking.py (new)
- pipeline/step5_shape_matching.py (new)
- pipeline/step6_fusion.py (new)
- pipeline/step7_scale_estimation.py (new)
- pipeline/step8_pose_estimation.py (new)
- pipeline/utils.py (new)
- pipeline/visualization.py (new)
- pipeline/requirements.txt (new)
- docker-compose.yml (modified)
- requirements.txt (modified)
- AI_HANDOFF.md (rewritten)
- docs/AI_LOG.md (updated)

TODOs
- Run full end-to-end test (all 8 steps) and verify output.
- Create evaluation script: loop over all BOP scenes, compute ULIP-2-augmented Top-K accuracy, compare to 75.95% baseline.
- Evaluate on MI3DOR (shape-focused dataset, ULIP-2 should improve results).
- Tune fusion weights via grid search or Bayesian optimization.
- Generate missing MI3DOR descriptions (11/21 categories).
- Obtain HouseCat6D BOP test scenes.

## 2026-02-06

Goal
- Document the repository state after resetting main and creating the thesis scaffold.

Changes
- Reset main to a clean scaffold and removed OSCAR baseline files from this branch.
- Added README with thesis goal, research questions, approach, and branching strategy.
- Created placeholder directories with gitkeep files.
- Added AI documentation files.

Commands Run
- Unknown or not found in repository evidence.

## 2026-02-06 GPU and docs update

Goal
- Record GPU related compose decision and align AI docs.

Changes
- Documented GPU support intent for OSCAR docker compose setup.
- Updated AI docs to include this status.

Commands Run
- Unknown or not found in repository evidence.

## 2026-02-08 YCB-V and GSO repro setup

Goal
- Move reproduction work to exp/oscar-repro, set up local YCB-V plus GSO data layout, and fix git tracking behavior for large files.

Changes
- Prepared YCB-V test folder under eval/datasets/ycbv_gso/test from downloaded archives.
- Downloaded GSO assets and extracted to per object folders under home tholoi thesis datasets gso extracted_by_zip.
- Populated local OSCAR paths for GSO models and images, and prepared object_images/ycbv_gso.
- Aborted an oversized push that contained dataset history.
- Reset local branch tip to origin exp/oscar-repro and recommitted only gitignore.
- Pushed b70f4063 to origin exp/oscar-repro.

Files Touched
- gitignore on exp/oscar-repro to exclude eval datasets, object_database, object_images, and local artifacts.
- Local runtime directories populated but not committed:
  - eval/datasets/ycbv_gso/test
  - object_database/gso/models_orig
  - object_images/gso
  - object_images/ycbv_gso

Commands Run
- Verified from terminal transcript:
  - git reset --mixed origin/exp/oscar-repro
  - git add .gitignore
  - git commit -m Add gitignore for datasets/assets
  - git push origin exp/oscar-repro

TODOs
- Finalize and verify eval/datasets/ycbv_gso/test/id_to_label.json for current references.
- Verify YCB-V reference images in object_images/ycbv_gso before running retrieval.
- Run and document first successful object_retrieval/i2i_bbox_dino.py execution.

## 2026-02-19 to 2026-02-23 Rendering and data pipeline

Goal
- Render all 3D models for YCBV_GSO, HouseCat6D, and MI3DOR datasets using Blender.
- Fix data consistency issues across models, renderings, and descriptions.

Changes
- Modified rendering/rendering.py: added multi-dataset config block, use_folder_name flag for YCBV/GSO naming, seen_names dedup set, skip_patterns for textured_simple.
- Created download_missing_gso.py: targeted GSO model downloader from Gazebo Fuel API. Fixed ZIP extraction bug (Fuel ZIPs have no top-level folder).
- Created check_consistency.py: cross-dataset model/rendering/description consistency checker.
- Deleted 1535 stray JPG thumbnails from object_images/ycbv_gso/ and removed 60 JPG entries from descriptions JSON.
- Placed descriptions_attributes.json from descriptions_tessa/ycbv_gso/ into object_database/ycbv_gso/.
- Copied helper_files/id_to_label_ext.json to eval/datasets/ycbv_gso/test/id_to_label.json.

Rendering Results
- YCBV_GSO: 1050/1051 rendered (CAR_CARRIER_TRAIN had import issue in Blender).
- HouseCat6D: 194 real objects rendered (199 total includes 5 bg test_scene OBJs — correct).
- MI3DOR: 3848/3848 rendered (already done prior).

## 2026-03-04 Retrieval evaluation

Goal
- Run full OSCAR retrieval pipeline on YCBV_GSO and MI3DOR, compare to paper.

Changes
- Configured retrieval_combi_eval.py for YCBV_GSO (ref_dir, bop_root, desc_file, result_folder).
- Configured retrieval_mi3dor_eval.py for MI3DOR.
- Fixed bug in txt_img_wacv2.py line 326: similarities.items() on a list — changed to sort by x["score"].
- Created docs/OSCAR_Retrieval_Guide.md: comprehensive pipeline documentation.

Results
- YCBV_GSO full pipeline (retrieval_combi_eval.py): 75.95% top-1 accuracy.
  - Paper reports ~60% for YCBV_GSO. Difference: we use GT masks, paper uses GroundedSAM.
- MI3DOR full pipeline (retrieval_mi3dor_eval.py): NN=77.95%, mAP=0.534, ANMRR=0.410.
  - Paper reports NN=89.4%. Gap: descriptions only cover 10/21 categories (~52% of queries have no matching descriptions).

Files Touched
- rendering/rendering.py (modified)
- object_retrieval/retrieval_combi_eval.py (modified)
- object_retrieval/retrieval_mi3dor_eval.py (modified)
- object_retrieval/txt_img_wacv2.py (modified)
- download_missing_gso.py (new)
- download_collection.py (new)
- check_consistency.py (new)
- docs/OSCAR_Retrieval_Guide.md (new)

Commands Run
- git add (10 files) && git commit && git push origin exp/oscar-repro (d3098bdd)

TODOs
- Generate missing MI3DOR descriptions for 11 categories (2031 models).
- Obtain HouseCat6D BOP test scenes from dataset authors.
- Begin exp/ulip2 shape-aware retrieval experiments.


## 2026-03-12 Pipeline Debugging, ULIP NaN Fix, Batch Cache, ICP Alignment

Goal
- Fix all runtime bugs in the 8-step pipeline after initial end-to-end test.
- Improve DINOv2 speed (Step 4) from serial encoding to batch + disk cache.
- Fix ULIP-2 NaN scores (Step 5) caused by Open3D color overflow.
- Fix score fusion NaN propagation (Step 6).
- Fix ICP pose estimation not using coarse alignment from Step 7.
- Add 3D wireframe overlay for debug visualization (Step 7+8).

Changes

### Modified: pipeline/step4_dino_reranking.py (rewritten)
- Replaced serial 1-by-1 DINOv2 encoding with batch encoding (32 images/pass).
- Added `.pt` disk cache keyed by model name + fingerprint (file count + newest mtime).
- `CACHE_VERSION = 1`, `BATCH_SIZE = 32`.
- First run: ~5 min for 9,459 reference images. Subsequent runs: instant from cache.
- Cache file: `.dino_cache_{model}_{fingerprint}.pt`.

### Modified: pipeline/step5_shape_matching.py
- Fixed overflow bug: `np.asarray(pcd.colors, dtype=np.float32)` → `np.clip(raw, 0.0, 1.0).astype(np.float32)`. Open3D colors can exceed [0,1] range after processing, causing float32 overflow → inf → NaN embeddings.
- Added NaN filtering in `match()`: `torch.isnan(sims)` detected, replaced with -1.0 before `topk()`. Previously NaN rose to top of rankings.

### Modified: pipeline/step6_fusion.py
- Made `_minmax()` NaN-safe: filters `NaN` values before computing min/max, returns 0.0 for NaN entries.
- Added NaN guard in ULIP score collection: skips `c.shape_score` if NaN.

### Modified: pipeline/step7_scale_estimation.py (rewritten previously)
- Two-stage approach: `_coarse_align()` (RANSAC + ICP) → Partial-Aware Scale (2 best-visible axes).
- Returns `ScaleEstimationResult` with `coarse_alignment` (4×4 matrix) and `visible_axes`.
- fitness=0.63–0.90, scale=1.30–1.35, conf=0.63–0.90 depending on voxel size.

### Modified: pipeline/step8_pose_estimation.py
- Added `initial_pose` parameter to `_estimate_foundationpose()` signature.
- Fixed FoundationPose fallback: passes `initial_pose` to `_estimate_icp()` (was missing).
- Fixed main `estimate()`: passes `initial_pose` through when calling FoundationPose path.
- ICP now uses coarse alignment from Step 7 as initial transform, skipping redundant RANSAC.

### Modified: pipeline/config.py
- Changed `pose_method` default from `"foundationpose"` to `"icp"` (FoundationPose is NotImplemented, always fell back to ICP without initial_pose).
- Changed `voxel_size` from `0.005` (5mm, ~810 pts) to `0.002` (2mm, ~4200 pts) for denser point clouds.
- Changed `ollama_model` from `"mistral-small3.1"` to `"gemma3:4b"`.
- Changed `ollama_timeout` from `5.0` to `30.0`.
- Changed `depth_scale` to `10000.0` and `depth_trunc` to `10.0`.

### Modified: pipeline/debug_steps.py (~1200 lines)
- Added `_project_cad_wireframe()`: projects CAD mesh edges onto scene image using pose matrix + camera intrinsics (trimesh).
- Updated `save_debug_step7_8()`: adds `pose_matrix`, `cad_model_path`, `cam` parameters for 3D wireframe overlay (falls back to thumbnail if not available).
- Fixed wireframe projection: inverted pose matrix (ICP returns camera→CAD, wireframe needs CAD→camera).
- Fixed wireframe scaling: use `vertex_mean` (matching Open3D `get_center()`) instead of `bbox_center`.
- Pipeline calls pass `sr.coarse_alignment` as `initial_pose` to Step 8.

### Installed in Docker: trimesh
- `pip install trimesh` in container for 3D wireframe overlay rendering.

Bugs Fixed
- NaN ULIP scores (foam_brick at #1 with NaN): Open3D pcd.colors overflow → np.clip fix.
- NaN in topk rankings: replaced NaN with -1.0 before torch.topk().
- NaN in fusion normalization: _minmax() now filters NaN values.
- FoundationPose fallback not passing initial_pose: added parameter forwarding.
- Stale .pyc files in Docker: cleared `/app/pipeline/__pycache__` (owned by root).
- Wireframe projected to wrong position: pose matrix needed inversion (np.linalg.inv).
- Wireframe scaling mismatch: bbox_center vs vertex_mean consistency fix.

Pipeline Test Results (2026-03-12, scene 000048/000001, "i need the blue coffee can")
- Step 1: Localization — confidence 0.847, BBox [207, 141, 335, 328]
- Step 2: Point Cloud — 4,201 points (2mm voxel), BBox [0.094, 0.142, 0.062]m
- Step 3: CLIP — #1 master_chef_can 0.4702
- Step 4: DINOv2 — #1 master_chef_can 0.6447 (from cache, instant)
- Step 5: ULIP-2 — #1 foam_brick 0.1977 (weak but no NaN; partial view limitation)
- Step 6: Fusion — master_chef_can fused=0.8473 (CLIP+DINO compensate weak ULIP)
- Step 7: Scale — factor=1.2968, conf=0.63
- Step 8: ICP — fitness=0.9895, RMSE=0.007m (using coarse alignment as start)

Files Touched
- pipeline/config.py (modified)
- pipeline/debug_steps.py (heavily modified)
- pipeline/step4_dino_reranking.py (rewritten)
- pipeline/step5_shape_matching.py (modified)
- pipeline/step6_fusion.py (modified)
- pipeline/step7_scale_estimation.py (rewritten previously)
- pipeline/step8_pose_estimation.py (modified)
- AI_HANDOFF.md (rewritten)
- docs/AI_LOG.md (updated)
- docs/DECISIONS.md (updated)
- Readme.md (updated)

TODOs
- Create evaluation script for ULIP-2-augmented retrieval accuracy across all BOP scenes.
- Evaluate on MI3DOR dataset (shape-focused, ULIP-2 should help).
- Tune fusion weights and voxel size via grid search.
- Investigate ULIP-2 performance gap (partial vs complete point clouds).
- Generate missing MI3DOR descriptions (11/21 categories).
- Obtain HouseCat6D BOP test scenes.

## 2026-03-12 Pipeline Debugging, ULIP NaN Fix, Batch Cache, ICP Alignment

Goal
- Fix all runtime bugs in the 8-step pipeline after initial end-to-end test.
- Improve DINOv2 speed (Step 4) from serial encoding to batch + disk cache.
- Fix ULIP-2 NaN scores (Step 5) caused by Open3D color overflow.
- Fix score fusion NaN propagation (Step 6).
- Fix ICP pose estimation not using coarse alignment from Step 7.
- Add 3D wireframe overlay for debug visualization (Step 7+8).

Changes

### Modified: pipeline/step4_dino_reranking.py (rewritten)
- Replaced serial 1-by-1 DINOv2 encoding with batch encoding (32 images/pass).
- Added `.pt` disk cache keyed by model name + fingerprint (file count + newest mtime).
- `CACHE_VERSION = 1`, `BATCH_SIZE = 32`.
- First run: ~5 min for 9,459 reference images. Subsequent runs: instant from cache.
- Cache file: `.dino_cache_{model}_{fingerprint}.pt`.

### Modified: pipeline/step5_shape_matching.py
- Fixed overflow bug: `np.asarray(pcd.colors, dtype=np.float32)` → `np.clip(raw, 0.0, 1.0).astype(np.float32)`. Open3D colors can exceed [0,1] range after processing, causing float32 overflow → inf → NaN embeddings.
- Added NaN filtering in `match()`: `torch.isnan(sims)` detected, replaced with -1.0 before `topk()`. Previously NaN rose to top of rankings.

### Modified: pipeline/step6_fusion.py
- Made `_minmax()` NaN-safe: filters `NaN` values before computing min/max, returns 0.0 for NaN entries.
- Added NaN guard in ULIP score collection: skips `c.shape_score` if NaN.

### Modified: pipeline/step7_scale_estimation.py (rewritten previously)
- Two-stage approach: `_coarse_align()` (RANSAC + ICP) → Partial-Aware Scale (2 best-visible axes).
- Returns `ScaleEstimationResult` with `coarse_alignment` (4×4 matrix) and `visible_axes`.
- fitness=0.63–0.90, scale=1.30–1.35, conf=0.63–0.90 depending on voxel size.

### Modified: pipeline/step8_pose_estimation.py
- Added `initial_pose` parameter to `_estimate_foundationpose()` signature.
- Fixed FoundationPose fallback: passes `initial_pose` to `_estimate_icp()` (was missing).
- Fixed main `estimate()`: passes `initial_pose` through when calling FoundationPose path.
- ICP now uses coarse alignment from Step 7 as initial transform, skipping redundant RANSAC.

### Modified: pipeline/config.py
- Changed `pose_method` default from `"foundationpose"` to `"icp"` (FoundationPose is NotImplemented, always fell back to ICP without initial_pose).
- Changed `voxel_size` from `0.005` (5mm, ~810 pts) to `0.002` (2mm, ~4200 pts) for denser point clouds.
- Changed `ollama_model` from `"mistral-small3.1"` to `"gemma3:4b"`.
- Changed `ollama_timeout` from `5.0` to `30.0`.
- Changed `depth_scale` to `10000.0` and `depth_trunc` to `10.0`.

### Modified: pipeline/debug_steps.py (~1200 lines)
- Added `_project_cad_wireframe()`: projects CAD mesh edges onto scene image using pose matrix + camera intrinsics (trimesh).
- Updated `save_debug_step7_8()`: adds `pose_matrix`, `cad_model_path`, `cam` parameters for 3D wireframe overlay (falls back to thumbnail if not available).
- Fixed wireframe projection: inverted pose matrix (ICP returns camera→CAD, wireframe needs CAD→camera).
- Fixed wireframe scaling: use `vertex_mean` (matching Open3D `get_center()`) instead of `bbox_center`.
- Pipeline calls pass `sr.coarse_alignment` as `initial_pose` to Step 8.

### Installed in Docker: trimesh
- `pip install trimesh` in container for 3D wireframe overlay rendering.

Bugs Fixed
- NaN ULIP scores (foam_brick at #1 with NaN): Open3D pcd.colors overflow → np.clip fix.
- NaN in topk rankings: replaced NaN with -1.0 before torch.topk().
- NaN in fusion normalization: _minmax() now filters NaN values.
- FoundationPose fallback not passing initial_pose: added parameter forwarding.
- Stale .pyc files in Docker: cleared `/app/pipeline/__pycache__` (owned by root).
- Wireframe projected to wrong position: pose matrix needed inversion (np.linalg.inv).
- Wireframe scaling mismatch: bbox_center vs vertex_mean consistency fix.

Pipeline Test Results (2026-03-12, scene 000048/000001, "i need the blue coffee can")
- Step 1: Localization — confidence 0.847, BBox [207, 141, 335, 328]
- Step 2: Point Cloud — 4,201 points (2mm voxel), BBox [0.094, 0.142, 0.062]m
- Step 3: CLIP — #1 master_chef_can 0.4702
- Step 4: DINOv2 — #1 master_chef_can 0.6447 (from cache, instant)
- Step 5: ULIP-2 — #1 foam_brick 0.1977 (weak but no NaN; partial view limitation)
- Step 6: Fusion — master_chef_can fused=0.8473 (CLIP+DINO compensate weak ULIP)
- Step 7: Scale — factor=1.2968, conf=0.63
- Step 8: ICP — fitness=0.9895, RMSE=0.007m (using coarse alignment as start)

Files Touched
- pipeline/config.py (modified)
- pipeline/debug_steps.py (heavily modified)
- pipeline/step4_dino_reranking.py (rewritten)
- pipeline/step5_shape_matching.py (modified)
- pipeline/step6_fusion.py (modified)
- pipeline/step7_scale_estimation.py (rewritten previously)
- pipeline/step8_pose_estimation.py (modified)
- AI_HANDOFF.md (rewritten)
- docs/AI_LOG.md (updated)
- docs/DECISIONS.md (updated)
- Readme.md (updated)

TODOs
- Create evaluation script for ULIP-2-augmented retrieval accuracy across all BOP scenes.
- Evaluate on MI3DOR dataset (shape-focused, ULIP-2 should help).
- Tune fusion weights and voxel size via grid search.
- Investigate ULIP-2 performance gap (partial vs complete point clouds).
- Generate missing MI3DOR descriptions (11/21 categories).
- Obtain HouseCat6D BOP test scenes.