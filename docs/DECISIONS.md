# Decisions

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

## 2026-02-08 exclude datasets from git tracking

Decision
- Add gitignore rules to stop tracking heavy local data and generated assets.

Rationale
- Large local dataset commits caused very slow push and upload size issues.
- Reproduction data should remain local runtime state, not repository history.

Alternatives Considered
- Keep tracking data in git, rejected due size and performance.

## 2026-02-19 rendering multi-dataset config

Decision
- Add a dataset config block at the top of rendering.py with use_folder_name flag rather than separate scripts per dataset.

Rationale
- YCBV uses textured.obj and GSO uses model.obj — both produce duplicate model_name when derived from filename. Using parent folder name (use_folder_name=True) avoids collisions.
- Single script with config section is easier to maintain than duplicating.

Alternatives Considered
- Separate rendering scripts per dataset — rejected to avoid duplication.
- Renaming model files inside each folder — too invasive on downloaded data.

## 2026-02-23 download only missing GSO models

Decision
- Create download_missing_gso.py that checks existing folders and downloads only absent models from Gazebo Fuel API.

Rationale
- Full re-download of 1030 models wastes bandwidth. Script checks folder existence and downloads only the ~722 missing ones.
- Fixed ZIP extraction: Fuel ZIPs have no top-level directory so must extract into named subfolder.

Alternatives Considered
- Re-download everything — rejected, too slow.
- Manual download — rejected, 722 models.

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
- Tradeoff: FPFH computation and ICP are slightly slower with more points, but runtime is still under 1 second.

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
- Open3D `pcd.colors` can produce values outside [0,1] (e.g. from depth-to-color mapping), causing float32 overflow → inf → NaN embeddings → NaN cosine similarity. NaN silently propagated through topk() (NaN > any number in PyTorch) and corrupted fusion normalization.
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
- Tradeoff: FPFH computation and ICP are slightly slower with more points, but runtime is still under 1 second.

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
- Open3D `pcd.colors` can produce values outside [0,1] (e.g. from depth-to-color mapping), causing float32 overflow → inf → NaN embeddings → NaN cosine similarity. NaN silently propagated through topk() (NaN > any number in PyTorch) and corrupted fusion normalization.
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