# AI Handoff

**Project Goal**
- Reproduce OSCAR baseline experiments reliably.
- Use OSCAR as benchmark baseline for later thesis extensions with shape aware retrieval.
- Keep baseline and experiment branches clearly separated.

**Current Status**
- main is still the scaffold branch with docs commits ab5bf7c4 and 7a41373e.
- Reproduction work is happening on exp/oscar-repro.
- exp/oscar-repro includes gitignore to exclude large datasets and generated assets with commit b70f4063 pushed.
- Local YCB-V test data exists at eval/datasets/ycbv_gso/test.
- Local GSO data was downloaded and extracted to per object folders under home tholoi thesis datasets gso extracted_by_zip.
- Local runtime folders were prepared under object_database gso models_orig, object_images gso, object_images ycbv_gso.
- On main these data folders are local untracked files and are not committed.

**How to Run + Test**
- Baseline container startup from oscar Readme:
  - docker compose build
  - docker compose run --rm -it oscar bash
- YCB-V plus GSO retrieval script:
  - cd object_retrieval
  - python i2i_bbox_dino.py
- Script expects paths:
  - ../object_images/ycbv_gso
  - ../eval/datasets/ycbv_gso/test
  - ../eval/datasets/ycbv_gso/test/id_to_label.json

**Key Constraints / Invariants**
- Keep oscar as clean upstream mirror.
- Keep reproduction changes on exp/oscar-repro.
- Do not commit datasets or generated assets.
- id_to_label.json must align with BOP object IDs used in annotations.

**Next 3 Tasks**
1. Finalize eval/datasets/ycbv_gso/test/id_to_label.json and verify mapping against current reference folders.
2. Verify object_images/ycbv_gso contains both YCB-V and GSO references and run python object_retrieval/i2i_bbox_dino.py.
3. Commit only reproducibility scripts and config notes on exp/oscar-repro and never data.

**Open Questions / Risks**
- object_images/ycbv was missing earlier and source of YCB-V reference images still needs confirmation.
- GSO extraction layout is fragile and must stay per object to keep helper scripts working.
- Branch switches while data is local and untracked can cause confusion if paths change on another branch.
