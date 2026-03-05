# AI Log

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
