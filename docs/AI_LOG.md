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
