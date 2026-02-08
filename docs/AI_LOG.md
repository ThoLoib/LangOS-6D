# AI Log

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
