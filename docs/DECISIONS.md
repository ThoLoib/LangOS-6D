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
