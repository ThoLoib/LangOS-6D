# LangOS-6D

Language- and shape-aware 3D object retrieval for 6D pose estimation.

## Project Scope

This thesis project builds on OSCAR (language and image based open-set CAD retrieval) and extends it with a shape-aware channel from RGB-D observations.

Goal:
- improve retrieval robustness when text and appearance alone are ambiguous,
- compare shape-only and cross-modal ULIP variants,
- measure impact on downstream scale and 6D pose estimation.

Core idea:
- extract a segmented point cloud from RGB-D,
- embed it with ULIP-2,
- fuse CLIP, DINOv2, and ULIP scores to rank CAD candidates.

## Branch Overview

| Branch | Purpose | Status |
|---|---|---|
| `oscar` | Upstream OSCAR mirror | Stable |
| `main` | Documentation, decisions, handoff | Stable |
| `exp/oscar-repro` | OSCAR baseline reproduction and evaluation | Completed |
| `exp/ulip2` | Initial 8-step pipeline with ULIP point-cloud channel | Stable |
| `exp/ulip2-full` | Active ULIP full experiments (`pc`, `cross`, `both`) | Active |

## Where To Find What

- High-level project and branch map: this file (`main`)
- Experiment details, commands, and ULIP implementation notes: `exp/ulip2-full` `README.md`
- Handoff summary: `AI_HANDOFF.md`
- Decision log: `docs/DECISIONS.md`
- Change log: `docs/AI_LOG.md`