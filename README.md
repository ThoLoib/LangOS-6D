# LangOS-6D

Language- and shape-aware 3D object retrieval for 6D pose estimation.

`main` is the documentation hub. Active implementation happens in experiment branches.

## Branch Overview

| Branch | Purpose | Status |
|---|---|---|
| `oscar` | Upstream OSCAR mirror | Stable |
| `main` | Documentation, decisions, handoff | Stable |
| `exp/oscar-repro` | OSCAR baseline reproduction and evaluation | Completed |
| `exp/ulip2` | Initial 8-step pipeline with ULIP point-cloud channel | Stable |
| `exp/ulip2-full` | Active ULIP full experiments (`pc`, `cross`, `both`) | Active |

## Latest Validated Highlights (`exp/ulip2-full`)

- ULIP mode switch in Step 5: `pc`, `cross`, `both`.
- Recursive CAD mesh discovery for nested YCBV/GSO layouts.
- On-disk CAD embedding cache for faster repeated runs.
- Fusion and debug flow keep `best_view_path` (image) separate from `cad_model_path` (mesh).
- Required dependencies aligned: `open-clip-torch`, `trimesh`.

## Working Rule

- Implement and test code changes in `exp/*` branches.
- Keep `main` as the canonical project overview and handoff branch.
