# AI Handoff (main)

Updated: 2026-03-17

## Purpose

`main` is the documentation hub. Use it to understand branch responsibilities and current experiment status.

## Branch Map

| Branch | Responsibility | Notes |
|---|---|---|
| `oscar` | Clean upstream mirror | Do not modify |
| `main` | Docs, decisions, handoff | Keep readable and up to date |
| `exp/oscar-repro` | Baseline reproduction | Reference metrics and setup |
| `exp/ulip2` | Initial ULIP integration | 8-step pipeline operational |
| `exp/ulip2-full` | Active ULIP full work | Mode ablation and runtime fixes |

## Current Technical Status

- Baseline retrieval reproduced: YCBV_GSO top-1 75.95%.
- `exp/ulip2-full` adds:
  - ULIP mode switch (`pc`, `cross`, `both`),
  - recursive CAD mesh loading,
  - CAD embedding disk cache,
  - corrected mesh-path propagation into pose estimation,
  - debug support for ULIP mode comparisons.

## Workflow

1. Keep `main` docs-only.
2. Implement and validate in `exp/*` branches.
3. Sync validated outcomes back into `main` docs.
