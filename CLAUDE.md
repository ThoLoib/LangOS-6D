# CLAUDE.md

## Purpose

This file initializes Claude Code for work on the OSCAR thesis project. It should give a compact but actionable overview of the project, the current architecture, the main technical constraints, and the immediate problem to solve.

Claude's role is not to act as a generic assistant. Its job is to inspect the repository and runtime setup, propose technically sound changes, implement them when appropriate, and keep the documentation current.

## Project Overview

The project extends OSCAR for a master's thesis at TU Wien. The broad goal is to improve object retrieval and pose estimation in cluttered scenes by combining language, image, and 3D shape signals.

The system is organized as a structured 8-step pipeline (see `pipeline/step*.py`), plus an
out-of-band geometry re-ranking sub-step B2. It supports end-to-end debugging and visualization
and has been used for experiments on YCBV_GSO, MI3DOR and SHREC'18.

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

FoundationPose integration uses a **two-container HTTP architecture**: Step 8 calls the FP
service over HTTP via `pipeline/foundationpose_bridge.py` and falls back to ICP.

Operational pattern:
- `docker compose up -d foundationpose` starts the FP service
- `docker compose run --rm -it oscar bash` starts OSCAR
- `--pose_method foundationpose` triggers the HTTP path; ICP is the default fallback

Previous approaches that were tried and abandoned:
- Venv inside OSCAR container: failed due CUDA runtime vs devel mismatch
- Subprocess bridge to separate interpreter: same root cause
- These are documented in `docs/DECISIONS.md` and `docs/AI_LOG.md`

## Current Problem Claude Should Focus On

The methodology components are implemented (see `docs/THESIS_ALIGNMENT_PLAN.md`) and the Stage-1
ablation grid has been run on SHREC'18. For the live state of the evaluation — which cells are
done, what is still queued, and how to resume — read `AI_HANDOFF.md`, which is kept current.

## Expected Approach

When working on this project, Claude should:

1. Preserve the two-container separation between OSCAR and FoundationPose.
2. Keep the ICP fallback intact.
3. Document every meaningful design change in:
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

### Git

- **This machine is tessa's PC; its working branch is `tessa-pc` — push there.** Never push to
  `main`.
- Commit only when asked.

### Experiment scripts

- Experiment scripts are **flat and flag-based**: no subcommands, and no dataset downloading or
  management. The user provides the data; the script validates that it is present and reports
  what is missing. Do not add download or fetch logic.
- Do not `pip install pointnet2_ops` on one machine only. Both PCs must run the same pure-torch
  FPS path, or Uni3D/ULIP embeddings silently mismatch across machines — see
  `docs/LAPTOP_EMBEDDINGS_SETUP.md`.

## Documentation Discipline

After each meaningful debugging or integration step, update the project docs. Do not leave important runtime knowledge only in terminal history.
