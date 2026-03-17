# AI Log

## 2026-03-17 Main Documentation Synchronization

Goal
- Ensure `main` reflects the current branch landscape and validated technical status.

Changes
- Rewrote `README.md` as a concise branch overview.
- Updated `AI_HANDOFF.md` with branch responsibilities and workflow.
- Updated `docs/DECISIONS.md` to reflect current coordination and ULIP runtime decisions.
- Kept `main` documentation-only; implementation remains on `exp/*` branches.

## 2026-03-17 ULIP Full Stabilization (`exp/ulip2-full`)

Goal
- Stabilize ULIP full debug and evaluation workflow.

Outcomes
- Added ULIP retrieval modes (`pc`, `cross`, `both`).
- Fixed CAD discovery for nested layouts (recursive loading).
- Added CAD embedding disk cache.
- Fixed pose-estimation mesh path propagation.
- Added dependencies `open-clip-torch` and `trimesh`.
