# Decisions

## 2026-03-17 Keep `main` as documentation hub

Decision
- Maintain `main` as the canonical branch for documentation and handoff.

Rationale
- Reduces confusion while implementation is distributed over multiple `exp/*` branches.
- Improves onboarding and branch coordination.

Alternatives Considered
- Keep status only in experiment branches; rejected due poor visibility.

## 2026-03-17 Expose ULIP mode switch

Decision
- Support `pc`, `cross`, and `both` modes as runtime options.

Rationale
- Required for controlled thesis ablations between shape-only and cross-modal retrieval.

Alternatives Considered
- Keep only `pc`; rejected because comparisons become impossible.

## 2026-03-17 Recursive CAD mesh discovery

Decision
- Use recursive mesh lookup for CAD models.

Rationale
- Dataset layout is nested; non-recursive lookup misses most models.

Alternatives Considered
- Enforce flat per-object directories; rejected as too invasive.

## 2026-03-17 Separate image view path and CAD mesh path

Decision
- Keep DINO best-view image path separate from CAD mesh path in fusion outputs.

Rationale
- Prevents pose estimation from receiving image files as mesh inputs.

Alternatives Considered
- Always re-search mesh path in Step 8; retained only as fallback.
