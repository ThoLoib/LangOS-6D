# CLAUDE.md

## Purpose

This file initializes Claude Code for work on the OSCAR thesis project. It should give a compact but actionable overview of the project, the current architecture, the main technical constraints, and the immediate problem to solve.

Claude's role is not to act as a generic assistant. Its job is to inspect the repository and runtime setup, propose technically sound changes, implement them when appropriate, and keep the documentation current.

## Project Overview

The project extends OSCAR for a master's thesis at TU Wien. The broad goal is to improve object retrieval and pose estimation in cluttered scenes by combining language, image, and 3D shape signals.

Recent work introduced a structured 8-step pipeline:

1. Object localization
2. Point cloud extraction
3. CLIP retrieval
4. DINOv2 re-ranking
5. ULIP-2 shape matching
6. Multi-signal fusion
7. Scale estimation
8. Pose estimation

The current system already supports end-to-end debugging and visualization and has been used for experiments on YCBV_GSO and MI3DOR.

## Primary Repositories and Runtime Context

Assume the working setup is based on WSL plus Docker.

- OSCAR repository: `~/thesis/OSCAR`
- FoundationPose repository: `~/thesis/FoundationPose`
- OSCAR runs in its own Docker container
- FoundationPose currently exists separately and should remain isolated from OSCAR's main dependency stack

Claude should prefer working from the WSL host shell with access to the repositories and `docker` commands, not from inside the OSCAR container only. The task is to reason about the full setup, not just a single Python environment.

## Source Documents

Use these files as the primary source of truth before making major changes:

- `README.md`
- `AI_HANDOFF.md`
- `docs/DECISIONS.md`
- `docs/AI_LOG.md`

If code and docs disagree, inspect the current code and then update the docs accordingly.

## Established Technical Decisions

These decisions are already documented and should be treated as the current baseline unless there is a strong reason to change them:

- Keep the OSCAR runtime environment and the FoundationPose runtime environment in separate containers.
- Do not force all OSCAR and FoundationPose dependencies into one Python environment.
- Run FoundationPose from Step 8 via HTTP call to the `foundationpose` compose service.
- Preserve cache directories and model state through Docker volumes where that reduces repeated setup costs.
- Keep ICP as the default fallback when FoundationPose is unavailable.

The main reason for the split is repeated dependency incompatibility across:

- `torch`
- `torchvision`
- `pytorch3d`
- `transformers`
- NumPy ABI-sensitive packages

## Current Technical Status

### Partial-to-Partial Point Cloud Matching (Step 5)

Step 5 now supports partial-view point clouds (`--ulip-partial-views`) as an alternative to full-mesh sampling. Preprocessing script: `rendering/generate_partial_pointclouds.py`. The partial PCs are stored as `.npz` files alongside the rendered images in `object_images/{dataset}/{object_id}/`. All three ULIP modes (pc, cross, both) work with partial views.

### FoundationPose

FoundationPose integration uses a **two-container HTTP architecture**:

- `pipeline/step8_pose_estimation.py` — calls FoundationPose via HTTP, falls back to ICP
- `pipeline/foundationpose_bridge.py` — HTTP client (httpx) that calls the FP service
- `pipeline/config.py` — `foundationpose_url` points to `http://foundationpose:5050`
- `FoundationPose/foundationpose_server.py` — Flask server inside the FP container
- `docker-compose.yml` — defines both `oscar` and `foundationpose` services

Operational pattern:
- `docker compose up -d foundationpose` starts the FP service
- `docker compose run --rm -it oscar bash` starts OSCAR
- `--pose_method foundationpose` triggers the HTTP path; ICP is the default fallback

Previous approaches that were tried and abandoned:
- Venv inside OSCAR container: failed due CUDA runtime vs devel mismatch
- Subprocess bridge to separate interpreter: same root cause
- These are documented in `docs/DECISIONS.md` and `docs/AI_LOG.md`

## Current Problem Claude Should Focus On

The pipeline is functional end-to-end. The thesis methodology specifies components that are not yet implemented in the codebase. The next steps are:

1. Align the codebase with the thesis methodology (see `docs/THESIS_ALIGNMENT_PLAN.md` when created)
2. Implement Sub-step B2: GeDi-based geometry re-ranking
3. Implement mask post-processing (largest connected component + dilation)
4. Add SHREC'18 ObjectNN+ evaluation (Stage 1 tuning)
5. Add majority voting fusion strategy
6. Add SigLIP and Uni3D encoder alternatives for ablations
7. Set up BOP-core evaluation (YCB-V, T-LESS, LM-O)

## Expected Approach

When working on this project, Claude should:

1. Read the source documents listed above before making changes.
2. Inspect Docker-related files (`docker-compose.yml`, Dockerfiles, startup scripts) to understand the runtime context.
3. Preserve the two-container separation between OSCAR and FoundationPose.
4. Keep the ICP fallback intact.
5. Document every meaningful design change in:
   - `AI_HANDOFF.md`
   - `docs/DECISIONS.md`
   - `docs/AI_LOG.md`

## Working Rules for Claude

Claude should optimize for reproducibility and low fragility.

- Do not recommend merging all dependencies into one environment unless there is strong evidence that the version constraints have been resolved.
- Prefer explicit paths, explicit interpreters, explicit volume mounts, and explicit service boundaries.
- Verify assumptions against the actual repository layout and scripts before proposing changes.
- Preserve fallback paths that keep the rest of the pipeline usable.
- Favor incremental changes that can be tested quickly.

## Immediate Goal

1. Implement the thesis methodology components not yet in the codebase (GeDi, mask refinement, full-database fusion).
2. Set up SHREC'18 ObjectNN+ evaluation for Stage 1 configuration tuning.
3. Run the full ablation grid (E1–E7, O1–O5) as defined in the thesis evaluation chapter.
4. Set up BOP-core pose evaluation (Stages 3a/3b).

## Definition of Success

This work is successful when:

- The codebase implements all components described in the thesis methodology chapter
- The ablation grid (E1–E7, O1–O5) can be run on SHREC'18, MI3DOR, and BOP-core datasets
- OSCAR can trigger FoundationPose reliably
- ICP fallback still works when FoundationPose is unavailable or fails
- The solution is understandable from the repo docs
- The rest of the OSCAR pipeline remains stable

## Documentation Discipline

After each meaningful debugging or integration step, update the project docs. Do not leave important runtime knowledge only in terminal history.
