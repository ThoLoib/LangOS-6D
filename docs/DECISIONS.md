# Decisions

## 2026-02-06: Reset main to a thesis scaffold

**Decision**
- Keep main as a clean thesis workspace scaffold rather than the OSCAR baseline code.

**Rationale**
- The README specifies that OSCAR is the benchmark baseline and that main is for thesis work, with ablation branches derived from oscar.

**Alternatives Considered**
- Unknown / not found in repository evidence.

## 2026-02-06: Branching strategy for baseline and ablations

**Decision**
- Use oscar as the baseline mirror and exp/* branches for ablations.

**Rationale**
- Documented in README.md to separate the pristine baseline from experimental modifications.

**Alternatives Considered**
- Unknown / not found in repository evidence.

## 2026-02-06: Enable GPU access in OSCAR Docker compose (stashed)

**Decision**
- Add a GPU device reservation to docker-compose.yml for the oscar service.

**Rationale**
- Needed for NVIDIA GPU access inside the OSCAR container.

**Alternatives Considered**
- Unknown / not found in repository evidence.
