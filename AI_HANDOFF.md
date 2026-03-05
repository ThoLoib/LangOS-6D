# AI Handoff

**Project Goal**
- Reproduce OSCAR baseline experiments reliably.
- Use OSCAR as benchmark baseline for later thesis extensions with shape-aware retrieval.
- Keep baseline and experiment branches clearly separated.

**Current Status**
- main is the scaffold branch with AI docs.
- exp/oscar-repro: OSCAR baseline reproduction — pushed d3098bdd. All retrieval scripts configured and evaluated.
- exp/ulip2: next experiment branch — shape-aware retrieval extensions.
- oscar: clean upstream mirror of pullover00/OSCAR.
- All large data (models, renderings, eval scenes, descriptions) is local-only and gitignored.

**Data Layout (local, not committed)**
- eval/datasets/ycbv_gso/test/ — 12 BOP scenes with rgb, mask_visib, scene_gt.json, scene_gt_info.json, id_to_label.json (1051 entries)
- eval/datasets/mi3dor/image/test/ — MI3DOR test images (21 categories, 500 per category)
- object_database/ycbv_gso/ — 1051 3D models (21 YCBV + 1030 GSO) + descriptions_attributes.json
- object_database/MI3DOR/ — 3848 3D models + descriptions_attributes.json (only 10/21 categories)
- object_database/housecat6d/ — 194 3D models + descriptions_attributes.json
- object_database/descriptions_tessa/ — pre-generated descriptions from repo owner (ycbv_gso, MI3DOR, housecat6d, ycbv)
- object_images/ycbv_gso/ — 1050 rendered objects (8 views + bg + cam matrices each)
- object_images/MI3DOR/ — 3848 rendered objects
- object_images/housecat6d/ — 194 rendered objects

**How to Run + Test**
- Docker container startup:
  - docker compose build
  - docker compose run --rm -it oscar bash
- Full OSCAR pipeline (YCBV_GSO):
  - cd /app/object_retrieval
  - python retrieval_combi_eval.py
  - Result: results_topk_eval_ycbv_gso/accuracy_summary_topk_15.json
- Full OSCAR pipeline (MI3DOR):
  - python retrieval_mi3dor_eval.py
  - Result: results_mi3dor_f20/metrics_summary_topk_15.json
- DINOv2-only baseline (YCBV_GSO):
  - python i2i_bbox_dino.py or python txt_img_wacv2.py

**Baseline Results (exp/oscar-repro)**
- YCBV_GSO full pipeline: 75.95% top-1 accuracy (GT masks, threshold=0.37, topk=15)
  - Paper reports ~60% for YCBV_GSO; difference is GT masks vs GroundedSAM
- MI3DOR full pipeline: NN=77.95% (paper: 89.4%)
  - Gap caused by incomplete descriptions: only 10/21 categories have descriptions

**Key Constraints / Invariants**
- Keep oscar as clean upstream mirror.
- Keep reproduction changes on exp/oscar-repro.
- Do not commit datasets or generated assets.
- id_to_label.json must align with BOP object IDs used in annotations.

**Next 3 Tasks**
1. Begin shape-aware retrieval experiments on exp/ulip2.
2. Generate missing MI3DOR descriptions for remaining 11 categories (or obtain from repo owner).
3. Obtain HouseCat6D BOP test scenes for evaluation.

**Open Questions / Risks**
- MI3DOR descriptions incomplete (10/21 categories) — affects MI3DOR eval accuracy significantly.
- HouseCat6D eval scenes not available — cannot evaluate HouseCat6D retrieval yet.
- GSO extraction layout is fragile — must stay per-object folder to keep scripts working.
- Branch switches are safe for data since data is gitignored and untracked.
