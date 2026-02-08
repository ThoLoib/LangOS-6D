# Decisions

## 2026-02-06 reset main to scaffold

Decision
- Keep main as a clean thesis workspace scaffold rather than the OSCAR baseline code.

Rationale
- README defines OSCAR as benchmark baseline while thesis workflow and integration notes belong on main.

Alternatives Considered
- Unknown or not found in repository evidence.

## 2026-02-06 branch strategy

Decision
- Use oscar as baseline mirror and exp branches for ablations and reproduction work.

Rationale
- Separates pristine upstream baseline from experimental and thesis specific changes.

Alternatives Considered
- Unknown or not found in repository evidence.

## 2026-02-06 enable GPU access in compose

Decision
- Add GPU device reservation for the oscar service in compose during setup.

Rationale
- Required to access NVIDIA GPU inside container.

Alternatives Considered
- CPU only execution.

## 2026-02-08 exclude datasets from git tracking

Decision
- Add gitignore rules to stop tracking heavy local data and generated assets.

Rationale
- Large local dataset commits caused very slow push and upload size issues.
- Reproduction data should remain local runtime state, not repository history.

Alternatives Considered
- Keep tracking data in git, rejected due size and performance.
