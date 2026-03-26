# AI Log

## 2026-03-26 Debug visualization refactored into main pipeline

Goal
- Eliminate the duplicated pipeline logic in `debug_steps.py` by making debug visualization an optional mode of the normal pipeline.

Changes
- Deleted `pipeline/debug_steps.py` (~1473 lines, contained a full copy of the 8-step pipeline in `run_debug()`).
- New `pipeline/debug_viz.py` (~1070 lines): all visualization functions extracted from the old file. `_find_cad_mesh()` promoted to module level (was nested inside `save_debug_step7_8()`, causing a NameError at runtime).
- Modified `pipeline/run_pipeline.py`:
  - `OSCARPlusPipeline.__init__()`: new `debug_viz: bool = False` parameter.
  - `OSCARPlusPipeline.run()`: new `gt_data=None` parameter for GT wireframe overlay.
  - Debug-viz hooks (calls to `_dbv.save_debug_step*()`) added after each of the 8 steps, guarded by `if self.debug_viz`.
  - Mesh-path resolution added before step 7: detects image-paths (`.png/.jpg`) in `cad_model_path` and resolves via `_find_cad_mesh()`. Result shared with step 8.
  - GT pose matrix built from `gt_data` parameter (same logic as old `run_debug()` lines 1294-1312).
  - New CLI flags: `--debug-viz` (rich debug images), `--until-step N` (converted to `skip_steps`).
  - `main()`: loads GT data from `scene_gt.json` + `id_to_label.json` when `--debug-viz` and `--camera` are set.
  - Bug fix: `detection_prompt` (undefined) → `prompt_elements.detection_phrase` in step 1 viz call.
- Modified `scripts/run_debug_pipeline_foundationpose.sh`: calls `pipeline.run_pipeline --debug-viz` with full YCBV-GSO defaults.
- New `scripts/run_pipeline.sh`: convenience script for normal pipeline execution.

Behavioral changes vs. old `debug_steps.py`
1. CLIP retrieval now receives `text_query=visual_query` from prompt parsing (old code omitted it) — may produce slightly different rankings.
2. Prompt parsing uses `_extract_prompt_elements()` (Ollama + heuristic) instead of duplicated logic.
3. `_find_cad_mesh` bug fixed — was unreachable in old code due to nested scope.

Impact
- Single source of truth for pipeline logic (no more `run_debug()` copy).
- `git grep "def run_debug"` returns no results.
- Debug shell script remains compatible (same output files, same CLI flags via `"$@"`).

## 2026-03-24 GT overlay + intrinsics/depth fixes

Goal
- Add ground truth pose wireframe overlay to debug_07_scale_pose.png for visual pose validation

Changes
- pipeline/debug_steps.py: load scene_gt.json + id_to_label.json in run_debug(); build 4x4 GT pose matrix; draw magenta GT wireframe via second _project_cad_wireframe() call; compensate for mesh bbox_center offset (subtract R_gt @ bbox_center from GT translation before projection); add "Predicted"/"GT" legend to Panel A; add Δt/ΔR metrics to Panel C; Panel C height +90px when GT shown
- pipeline/debug_steps.py + run_pipeline.py: moved camera loading before depth conversion so real fx/fy/cx/cy reach generate(); depth_scale always taken from config (BOP JSON field uses multiplier convention incompatible with pipeline divisor convention)
- pipeline/step2_pointcloud.py: identified dead code — PinholeCameraIntrinsic object created but never used; depth_scale param in generate() never exercised from pipeline call sites

Key finding
- BOP scene_camera.json depth_scale=0.1 is a multiplier; pipeline divides by config.depth_scale=10000.0. Using the JSON value caused depths to be 100× too large, resulting in ~855mm translation error in predicted pose. Always use config value.
- GT wireframe shift (~8px) caused by mesh bbox_center offset from origin; compensated by adjusting GT translation by -R_gt @ bbox_center before projection.

## 2026-03-20 FoundationPose two-container HTTP integration

Goal
- Replace the broken venv/subprocess FoundationPose integration with a clean two-container HTTP architecture.

Changes
- New file: `FoundationPose/foundationpose_server.py`
  - Minimal Flask server with `/health` and `/estimate_pose` endpoints.
  - Runs inside the FP container's conda env (Python 3.8, torch 2.1.0+cu121, pytorch3d, kaolin, nvdiffrast).
  - Lazy-loads scorer, refiner, and GL context on first request.
  - Accepts base64-encoded numpy arrays + camera matrix + CAD path via JSON POST.
  - Returns 4x4 pose matrix + confidence as JSON.

- Rewritten: `pipeline/foundationpose_bridge.py`
  - Now an HTTP client using httpx (was: subprocess launcher).
  - Encodes RGB/depth/mask as base64 numpy blobs.
  - Auto-translates CAD paths from `/app/...` (OSCAR container) to `/oscar/...` (FP container).
  - Configurable timeout (120s read, 10s connect).

- Rewritten: `pipeline/step8_pose_estimation.py`
  - Removed `_run_foundationpose_local()` (local import path — never worked in OSCAR container).
  - Removed `_run_foundationpose_subprocess()` (subprocess path — broken due CUDA mismatch).
  - Removed `_estimate_megapose()` (was always NotImplementedError).
  - Single FoundationPose path now calls `foundationpose_bridge.call_foundationpose()`.
  - ICP fallback preserved and unchanged.

- Modified: `pipeline/config.py`
  - Replaced `foundationpose_python` (str) with `foundationpose_url` (str, default `http://foundationpose:5050`).
  - Removed `foundationpose_repo_path` (no longer needed — FP container manages its own repo).

- Modified: `pipeline/debug_steps.py`, `pipeline/run_pipeline.py`
  - Replaced `--foundationpose_python` and `--foundationpose_repo` CLI args with `--foundationpose_url`.

- Modified: `docker-compose.yml`
  - Added `foundationpose` service using `shingarey/foundationpose_custom_cuda121:latest`.
  - FP service mounts `../FoundationPose:/workspace` and `.:/oscar:ro`.
  - Entrypoint activates conda env and runs `foundationpose_server.py`.
  - Healthcheck on `/health` endpoint.
  - Removed `../FoundationPose:/foundationpose` volume mount from oscar service (no longer needed).

- Updated: `README.md`, `AI_HANDOFF.md`, `docs/DECISIONS.md`
  - Replaced venv setup instructions with two-container startup instructions.
  - Removed references to `/foundationpose/.venv/bin/python`.
  - Updated command examples.

Diagnosis that motivated this change
- OSCAR container: `nvidia/cuda:12.2.0-runtime-ubuntu22.04`, Python 3.11, no CUDA dev headers.
- FoundationPose needs: CUDA devel image, Python 3.8, torch 2.0/2.1+cu118/cu121, pytorch3d/kaolin/nvdiffrast (all require compilation).
- A venv inside the OSCAR container cannot bridge this gap: no nvcc, wrong Python ABI, wrong CUDA version.
- The pre-built `shingarey/foundationpose_custom_cuda121` image has everything pre-compiled.

Options evaluated
1. HTTP API between two containers (chosen) — simplest, no Docker socket, no shared Python.
2. Shared-volume CLI handoff via `docker compose exec` — viable but requires Docker socket in OSCAR container.
3. Fix the venv inside OSCAR — not viable due CUDA runtime vs devel mismatch.
4. Install CUDA devel in OSCAR image — bloats image, fragile compilation chain.

Removed items
- `foundationpose_python` config field and `--foundationpose_python` CLI arg (replaced by `foundationpose_url`).
- `foundationpose_repo_path` config field and `--foundationpose_repo` CLI arg (FP container manages its own repo).
- `_run_foundationpose_local()` in step8 (never worked in OSCAR container).
- `_run_foundationpose_subprocess()` in step8 (broken due CUDA mismatch).
- `_estimate_megapose()` in step8 (was always NotImplementedError).
- `../FoundationPose:/foundationpose` volume mount in oscar service.

Impact
- FoundationPose can now actually run from the OSCAR pipeline (was previously broken).
- ICP fallback remains intact and robust.
- `docker compose up -d foundationpose` + `docker compose run --rm -it oscar bash` is the new startup pattern.

Manual follow-up needed
- Delete obsolete 11 GB venv: `rm -rf ~/thesis/FoundationPose/.venv`
- Test end-to-end with `--pose_method foundationpose` to validate the HTTP path.

## 2026-03-19 FoundationPose integration and split-environment execution (superseded)

> Superseded by 2026-03-20 two-container HTTP architecture.
> The subprocess bridge and venv approach did not work due to CUDA/ABI incompatibilities.

Goal
- Run FoundationPose in Step 8 without breaking OSCAR runtime dependencies.

Changes
- Added subprocess execution path for FoundationPose in step8.
- Created `pipeline/foundationpose_bridge.py` as standalone subprocess script.
- Added `foundationpose_python` config field and CLI arg.
- Added persistent volumes for Ollama, Torch, and CLIP caches.

Why superseded
- The dedicated venv at `/foundationpose/.venv` (created inside OSCAR's CUDA 12.2 runtime container) could not compile pytorch3d, kaolin, or nvdiffrast due to missing CUDA dev headers and Python ABI mismatch.

## 2026-03-18 FoundationPose setup and compose update

Goal
- Prepare a reproducible local setup for FoundationPose and document current switch status.

Changes
- Host setup:
  - cloned `NVlabs/FoundationPose` to `~/thesis/FoundationPose`
  - installed Docker image `foundationpose:latest`
- OSCAR integration prep:
  - updated `docker-compose.yml` volumes with `../FoundationPose:/foundationpose` (superseded by 2026-03-20)
- Codebase check:
  - verified `pipeline/step8_pose_estimation.py` still uses a FoundationPose template path and falls back to ICP.

Impact
- FoundationPose assets are available locally.
- Runtime behavior of Step 8 was unchanged until 2026-03-20 HTTP integration.

## 2026-03-18 Step 1 localization cleanup

Goal
- Verify what changed in `pipeline/step1_localization.py` and document it.

Changes
- Confirmed a non-functional cleanup in Step 1:
  - removed one duplicated comment line in the module header.
- No runtime logic, model call, threshold, or output schema changed.

Impact
- Behavior unchanged.

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
- First run: ~5 min for 9,459 reference images. Subsequent runs: instant from cache.

### Modified: pipeline/step5_shape_matching.py
- Fixed overflow bug: `np.asarray(pcd.colors)` -> `np.clip(raw, 0.0, 1.0)`.
- Added NaN filtering in `match()`: replaced with -1.0 before `topk()`.

### Modified: pipeline/step6_fusion.py
- Made `_minmax()` NaN-safe: filters NaN values before computing min/max.

### Modified: pipeline/step7_scale_estimation.py (rewritten previously)
- Two-stage approach: RANSAC + ICP -> Partial-Aware Scale (2 best-visible axes).

### Modified: pipeline/step8_pose_estimation.py
- Added `initial_pose` parameter forwarding to ICP.
- ICP now uses coarse alignment from Step 7 as initial transform.

### Modified: pipeline/config.py
- Changed defaults: `pose_method` to `"icp"`, `voxel_size` to `0.002`, `ollama_model` to `"gemma3:4b"`.

### Modified: pipeline/debug_steps.py
- Added 3D wireframe overlay projection using trimesh.

Bugs Fixed
- NaN ULIP scores, NaN in topk rankings, NaN in fusion normalization.
- FoundationPose fallback not passing initial_pose.
- Wireframe projection and scaling issues.

Pipeline Test Results (scene 000048/000001, "i need the blue coffee can")
- Step 1: confidence 0.847
- Step 6: master_chef_can fused=0.8473
- Step 8: ICP fitness=0.9895, RMSE=0.007m

## 2026-03-05 ULIP-2 Pipeline Integration + Visualization

Goal
- Implement full 8-step shape-aware retrieval pipeline on branch exp/ulip2.
- Integrate real ULIP-2 point cloud encoder (PointBERT Colored, 10k points, 1280-dim).
- Add LLM-based prompt parsing via Ollama.
- Add visualization module for intermediate results.

Changes

### New: pipeline/ module (17 files)
- Created full pipeline package: config, orchestrator, 8 step modules, utils, visualization.
- ULIP-2 integration in step5 (loads ~400 MB point encoder + projection).
- LLM prompt parsing via Ollama with heuristic fallback.

### Modified: docker-compose.yml
- Added volume `../ULIP:/ulip` and GPU device reservation.

### Modified: requirements.txt (root)
- Added: `ollama`, `open3d`, `easydict`, `timm`, `pyyaml_env_tag`, `termcolor`.

### Patched: ULIP repo (separate repo)
- Made `knn_cuda` and `pointnet2_ops` optional with fallbacks.

Bugs Fixed
- KeyError in camera intrinsics, missing packages, PyTorch 2.6 weights_only change.

## 2026-03-04 Retrieval evaluation

Goal
- Run full OSCAR retrieval pipeline on YCBV_GSO and MI3DOR, compare to paper.

Results
- YCBV_GSO: 75.95% top-1 accuracy (paper ~60%, difference: GT masks vs GroundedSAM).
- MI3DOR: NN=77.95% (paper NN=89.4%, gap: descriptions only 10/21 categories).

## 2026-02-19 to 2026-02-23 Rendering and data pipeline

Goal
- Render all 3D models for YCBV_GSO, HouseCat6D, and MI3DOR datasets using Blender.

Rendering Results
- YCBV_GSO: 1050/1051 rendered.
- HouseCat6D: 194 real objects rendered.
- MI3DOR: 3848/3848 rendered.

## 2026-02-08 YCB-V and GSO repro setup

Goal
- Move reproduction work to exp/oscar-repro, set up local YCB-V plus GSO data layout.

Changes
- Prepared YCB-V test folder, downloaded GSO assets, fixed git tracking for large files.

## 2026-02-06 Repository scaffold and GPU setup

Goal
- Document the repository state after resetting main and creating the thesis scaffold.

Changes
- Reset main to a clean scaffold.
- Added README, placeholder directories, AI documentation files.
- Documented GPU support intent for Docker compose.
