# Decisions

## 2026-03-24 BOP depth_scale convention — always use config divisor

Decision
- Always use `config.depth_scale` (default 10000.0) as the divisor when converting raw depth pixels to metres.
- Do not use the `depth_scale` field from `scene_camera.json`.

Rationale
- BOP `scene_camera.json` defines `depth_scale` as a **multiplier** (e.g. 0.1 for this dataset: raw × 0.1 = depth in mm).
- The pipeline divides raw depth by `config.depth_scale` (a **divisor** convention).
- Using the JSON value (0.1) as a divisor gave depths 100× too large, producing a translation error of ~855mm in the predicted pose.
- The config value (10000.0) is correct: it converts 16-bit PNG depth (0.1mm units) to metres.

Alternatives Considered
- Detect and adapt to the BOP convention at runtime — rejected; fragile, adds edge-case logic, and the config value is already correct for the target dataset.

## 2026-03-24 GT wireframe bbox-center compensation

Decision
- When projecting the GT wireframe overlay, subtract `R_gt @ bbox_center` from the GT translation vector before rendering.

Rationale
- BOP ground truth poses are annotated with models centered at the mesh bounding-box origin.
- The pipeline's OBJ files have a non-zero `bbox_center` offset (e.g. mug: ~8.3mm in X → ~7.8px shift at scene depth).
- Without this correction the GT wireframe is visibly misaligned even when the pose is geometrically correct.

Alternatives Considered
- Re-centre the OBJ meshes at the origin — too invasive, affects all downstream steps.
- Apply no correction and accept the visual offset — rejected; defeats the purpose of the overlay.

## 2026-03-20 two-container HTTP architecture for FoundationPose

Decision
- Run FoundationPose as a separate Docker compose service with a Flask HTTP API.
- OSCAR calls `http://foundationpose:5050/estimate_pose` from Step 8.
- Replace the subprocess bridge and venv-inside-OSCAR approach.

Rationale
- The OSCAR container (CUDA 12.2 runtime, Python 3.11) cannot compile pytorch3d, kaolin, or nvdiffrast which require a CUDA devel image.
- A virtual environment inside the OSCAR container cannot bridge this CUDA/ABI gap.
- HTTP over the Docker compose network gives full dependency isolation with zero shared Python state.
- The pre-built `shingarey/foundationpose_custom_cuda121` image already has all compiled dependencies.

Alternatives Considered
- Venv inside OSCAR container (previous approach): failed due CUDA runtime vs devel mismatch and Python 3.11 vs 3.8 ABI conflicts.
- Two-container with shared-volume CLI handoff (`docker compose exec`): viable but requires Docker socket in OSCAR container or host-side orchestration.
- Install CUDA devel toolkit in OSCAR image: bloats image by 10+ GB, fragile compilation chain, ongoing maintenance.
- HTTP API (chosen): simplest inter-container call, no Docker socket needed, JSON in/out, healthcheck support.

## 2026-03-19 persist model and embedding caches via compose volumes

Decision
- Persist Ollama data and model caches in docker compose with named volumes.

Rationale
- Prevent repeated model downloads and cache warmups across `docker compose run --rm` sessions.
- Keep runtime reproducible while reducing setup latency.

Alternatives Considered
- Keep cache only in ephemeral container filesystem; rejected due repeated startup cost.

## 2026-03-19 run FoundationPose in a separate Python environment (superseded)

> Superseded by 2026-03-20 two-container HTTP architecture.

Decision
- Execute FoundationPose from Step 8 via subprocess bridge using a configurable interpreter.

Rationale
- Single-env installation caused repeated dependency conflicts.
- Subprocess bridge allows one end-to-end pipeline call while preserving stability of both stacks.

Why superseded
- The venv approach could not work because the OSCAR container lacks CUDA devel headers needed to compile pytorch3d/kaolin/nvdiffrast. The two-container approach eliminates this class of problem entirely.

## 2026-03-18 staged FoundationPose switch

Decision
- Use a staged migration path for FoundationPose:
- first install FoundationPose and expose it via Docker volume,
- then keep Step 8 on ICP fallback until API integration is implemented and validated.

Rationale
- Reduces risk of breaking the current end-to-end pipeline while environment dependencies are prepared.
- Allows iterative verification (setup, weights, extension build, API wiring, evaluation).

Alternatives Considered
- Immediate hard switch from ICP to FoundationPose in Step 8; rejected due incomplete integration and higher regression risk.

## 2026-03-17 enable ULIP mode switch in debug and pipeline

Decision
- Expose ULIP retrieval mode as a runtime option (`pc`, `cross`, `both`) instead of hardcoding point-cloud-only behavior.

Rationale
- Needed for direct thesis ablation: shape-only vs full ULIP cross-modal retrieval on identical scenes.

Alternatives Considered
- Keep single `pc` mode only — rejected, prevents controlled comparison.

## 2026-03-17 recursive CAD mesh discovery for ycbv_gso

Decision
- Use recursive mesh lookup in CAD object folders and prefer known mesh filenames in `meshes/`.

Rationale
- ycbv_gso object layouts are nested; non-recursive lookup found only 21 models.
- Recursive lookup resolves 1051 models and stabilizes Step 5 coverage.

Alternatives Considered
- Enforce one flat file layout per object — rejected, too invasive for downloaded assets.

## 2026-03-17 cache ULIP CAD embeddings on disk

Decision
- Save/reload CAD embeddings in `.ulip_cache_<hash>.pt` keyed by model+config+mesh inventory.

Rationale
- Step 5 over 1000+ CAD models is the dominant runtime; repeated runs should not recompute unchanged embeddings.

Alternatives Considered
- In-memory cache only — rejected, not persistent across process/container restarts.

## 2026-03-17 separate image view paths from CAD mesh paths in fusion

Decision
- Keep DINO `best_view_path` separate from `cad_model_path` in fusion output.

Rationale
- Passing image paths as mesh paths caused Step 8 ICP to read `.png` as CAD mesh and fail.

Alternatives Considered
- Force Step 8 to ignore fusion path and always search filesystem — kept as fallback only.

## 2026-03-12 default pose_method to icp

Decision
- Changed `pose_method` default from `"foundationpose"` to `"icp"` in config.py.

Rationale
- FoundationPose is marked `NotImplementedError`. It always fell back to ICP anyway, but the fallback path did not forward `initial_pose` from Step 7's coarse alignment. Using ICP directly ensures the coarse alignment is used as the initial transform.

Alternatives Considered
- Implement FoundationPose wrapper — deferred, not critical for thesis prototype.
- Keep foundationpose default and fix the fallback — done as well, but direct ICP is cleaner.

## 2026-03-12 reduce voxel_size from 5mm to 2mm

Decision
- Changed `voxel_size` from `0.005` to `0.002` in config.py.

Rationale
- At 5mm, the observed point cloud had only ~810 points — too sparse for reliable ULIP-2 shape matching (expects 10,000 points). At 2mm, ~4,200 points are retained from a single depth view, providing much better surface coverage.

Alternatives Considered
- 0.001m (1mm, ~10k+ points): too dense, slower without significant quality gain.
- 0.003m (3mm, ~2-3k points): considered as middle ground, 2mm chosen for better ULIP coverage.

## 2026-03-12 DINOv2 batch encoding with disk cache

Decision
- Rewrote step4_dino_reranking.py with batch encoding (32 images/forward pass) and `.pt` disk cache.

Rationale
- Serial encoding of 9,459 reference images took ~45 minutes (1 forward pass per image). Batch encoding reduces this to ~5 minutes. Disk cache makes subsequent runs instant.
- Cache keyed by model name + fingerprint (hash of file count + newest modification time) to auto-invalidate when reference images change.

Alternatives Considered
- Pre-compute embeddings offline and store as a separate file — less flexible, manual step.
- Use FAISS index — overkill for ~10k vectors, simple cosine similarity is fast enough.

## 2026-03-12 NaN handling in ULIP and fusion

Decision
- Added explicit NaN detection and replacement throughout Step 5 and Step 6.

Rationale
- Open3D `pcd.colors` can produce values outside [0,1] (e.g. from depth-to-color mapping), causing float32 overflow -> inf -> NaN embeddings -> NaN cosine similarity. NaN silently propagated through topk() and corrupted fusion normalization.
- Fix: clip colors to [0,1], replace NaN similarities with -1.0, skip NaN in min-max normalization.

Alternatives Considered
- Discard objects with NaN entirely — too aggressive, could lose valid partial matches.
- Use nanmean/nanmin — less explicit, harder to debug.

## 2026-03-12 switch LLM to gemma3:4b

Decision
- Changed `ollama_model` from `"mistral-small3.1"` to `"gemma3:4b"`.

Rationale
- gemma3:4b fits in 6GB VRAM alongside the other models (GroundingDINO, SAM, CLIP, DINOv2, ULIP-2). Responds within 5-10 seconds for prompt parsing.
- mistral-small3.1 required more VRAM and was slower on the RTX 4050 Laptop GPU.

Alternatives Considered
- CPU-only inference for LLM — too slow (30+ seconds).
- Skip LLM entirely, use only heuristic parser — less robust for complex prompts.

## 2026-03-12 wireframe overlay via trimesh

Decision
- Installed trimesh in Docker for 3D wireframe overlay in debug visualization.

Rationale
- The debug image Step 7+8 previously showed a 2D thumbnail pasted onto the scene, which didn't convey pose orientation. Projecting CAD mesh edges using the estimated pose + camera intrinsics gives visual verification of alignment quality.

Alternatives Considered
- Use Open3D offscreen rendering — harder to integrate, requires display server.
- Use matplotlib 3D projection — less precise, no mesh topology awareness.

## 2026-03-04 use GT masks for retrieval eval

Decision
- Run retrieval_combi_eval.py with ground-truth segmentation masks from BOP data rather than GroundedSAM predictions.

Rationale
- Isolates retrieval accuracy from segmentation errors. Gives upper-bound performance.
- Paper's full pipeline uses GroundedSAM which adds segmentation noise. Our 75.95% vs paper's ~60% is consistent with this difference.

Alternatives Considered
- Run GroundedSAM for fair 1:1 comparison — possible future work but not primary focus.

## 2026-03-04 focus on full OSCAR pipeline only

Decision
- Skip running individual baselines (i2i_bbox_clip, i2i_seg_clip, etc.) that require ycbv_test_bop19 data. Focus on retrieval_combi_eval.py as the main evaluation script.

Rationale
- Most baseline scripts need ycbv_test_bop19 (21-object YCBV BOP test set) which is not downloaded.
- The full OSCAR pipeline is what the thesis aims to improve, not the individual baselines.

Alternatives Considered
- Download ycbv_test_bop19 and run all baselines — deferred, not critical for thesis progress.

## 2026-02-23 download only missing GSO models

Decision
- Create download_missing_gso.py that checks existing folders and downloads only absent models from Gazebo Fuel API.

Rationale
- Full re-download of 1030 models wastes bandwidth. Script checks folder existence and downloads only the ~722 missing ones.
- Fixed ZIP extraction: Fuel ZIPs have no top-level directory so must extract into named subfolder.

Alternatives Considered
- Re-download everything — rejected, too slow.
- Manual download — rejected, 722 models.

## 2026-02-19 rendering multi-dataset config

Decision
- Add a dataset config block at the top of rendering.py with use_folder_name flag rather than separate scripts per dataset.

Rationale
- YCBV uses textured.obj and GSO uses model.obj — both produce duplicate model_name when derived from filename. Using parent folder name (use_folder_name=True) avoids collisions.
- Single script with config section is easier to maintain than duplicating.

Alternatives Considered
- Separate rendering scripts per dataset — rejected to avoid duplication.
- Renaming model files inside each folder — too invasive on downloaded data.

## 2026-02-08 exclude datasets from git tracking

Decision
- Add gitignore rules to stop tracking heavy local data and generated assets.

Rationale
- Large local dataset commits caused very slow push and upload size issues.
- Reproduction data should remain local runtime state, not repository history.

Alternatives Considered
- Keep tracking data in git, rejected due size and performance.

## 2026-02-06 reset main to scaffold

Decision
- Keep main as a clean thesis workspace scaffold rather than the OSCAR baseline code.

Rationale
- README defines OSCAR as benchmark baseline while thesis workflow and integration notes belong on main.

Alternatives Considered
- Unknown or not found in repository evidence.

## 2026-02-06 branch strategy

Decision
- Use oscar as baseline mirror and exp branches for ablations and reproduction work.

Rationale
- Separates pristine upstream baseline from experimental and thesis specific changes.

Alternatives Considered
- Unknown or not found in repository evidence.

## 2026-02-06 enable GPU access in compose

Decision
- Add GPU device reservation for the oscar service in compose during setup.

Rationale
- Required to access NVIDIA GPU inside container.

Alternatives Considered
- CPU only execution.
