# AI Handoff

**Project Goal**
- Build on OSCAR to enable language + shape based 3D object retrieval for 6D pose estimation.
- Evaluate whether shape embeddings from partial point clouds improve retrieval and downstream 6D pose estimation.
- Compare shape encoders (e.g., PointNet-style vs. ULIP-2) and fusion strategies for image/language/shape similarity.

**Current Status**
- main is a clean scaffold with README.md describing goals, research questions, and planned approach.
- Placeholder folders exist with .gitkeep: docs/, experiments/, esults/, scripts/, data/, 
otes/, src/, ssets/.
- Branching strategy documented in README.md: oscar as baseline, main as thesis workspace, exp/* for ablations.
- Uncommitted change stashed on oscar: docker-compose.yml adds GPU device reservation for the oscar service.
- No runnable code or experiment scripts in main yet.

**How to Run + Test**
- From oscar/Readme.md (baseline branch):
  - docker compose build
  - docker compose run --rm -it oscar bash
  - cd rendering
  - wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/blender.zip
  - unzip blender.zip
  - ./blender-3.4.1-linux-x64/blender -b -P rendering.py
  - python retrieval_combi_eval.py
  - python retrieval_combi_eval_mi3dor.py
- No tests documented in main.

**Key Constraints / Invariants**
- oscar branch should remain a clean mirror of the upstream OSCAR baseline.
- exp/* branches are reserved for ablation work and should be branched from oscar.
- data/ is intended for datasets and symlinks and should not be committed.

**Next 3 Tasks (Priority)**
1. Add a .gitignore that excludes datasets, large artifacts, and local outputs (align with data/ and esults/).
2. Create an experiment log template in experiments/ and document how to reproduce OSCAR results.
3. Add initial scaffolding or scripts in src/ and scripts/ to mirror the OSCAR pipeline steps.

**Open Questions / Risks**
- No verified run or evaluation procedure in main yet; reproduction steps are documented only on oscar.
- Dataset size and storage strategy for data/ and esults/ not defined.
- The docker-compose.yml GPU change is currently stashed and not committed.
