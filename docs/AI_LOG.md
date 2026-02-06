# AI Log

## 2026-02-06

**Goal**
- Document the repository state after resetting main and creating the thesis scaffold.

**Changes (Chronological)**
- Reset main to a clean scaffold and removed OSCAR baseline files from this branch.
- Added README.md describing the thesis goal, research questions, approach, and branching strategy.
- Created placeholder directories with .gitkeep: docs/, experiments/, esults/, scripts/, data/, 
otes/, src/, ssets/.
- Recorded documentation files for handoff, log, and decisions.

**Files Touched (Why)**
- README.md to document project goals, research questions, and branch strategy.
- docs/.gitkeep to keep the docs/ folder in git.
- experiments/.gitkeep to keep the experiments/ folder in git.
- esults/.gitkeep to keep the esults/ folder in git.
- scripts/.gitkeep to keep the scripts/ folder in git.
- data/.gitkeep to keep the data/ folder in git.
- 
otes/.gitkeep to keep the 
otes/ folder in git.
- src/.gitkeep to keep the src/ folder in git.
- ssets/.gitkeep to keep the ssets/ folder in git.
- AI_HANDOFF.md to summarize the project status and next steps.
- docs/AI_LOG.md to log session work.
- docs/DECISIONS.md to record inferred decisions.

**Commands Run**
- Unknown / not found in repository evidence.

**Notable Decisions + Rationale**
- Keep main as a clean scaffold and use oscar for the baseline, enabling ablation branches to cleanly diverge from baseline.
- Use placeholder folders to define repo structure early.

**TODOs / Follow-ups**
- Add a .gitignore for datasets and experiment artifacts.
- Document reproduction steps for OSCAR in experiments/ or docs/.
- Add initial scripts or code structure to src/ and scripts/.

## 2026-02-06 (Update)

**Goal**
- Update AI docs after enabling GPU support in the OSCAR Docker compose configuration.

**Changes (Chronological)**
- Observed an uncommitted change on oscar that adds GPU device reservations to docker-compose.yml.
- Updated AI documentation to reflect the stashed compose change and baseline run commands from oscar/Readme.md.

**Files Touched (Why)**
- docker-compose.yml (stashed on oscar) to enable GPU access for the oscar service.
- AI_HANDOFF.md to reflect the stashed compose change and baseline commands.
- docs/AI_LOG.md to record this update.
- docs/DECISIONS.md to capture the GPU enablement decision.

**Commands Run**
- Unknown / not found in repository evidence.

**Notable Decisions + Rationale**
- Enable GPU support in Docker compose so the OSCAR container can access NVIDIA GPUs.

**TODOs / Follow-ups**
- Decide whether to apply and commit the docker-compose.yml GPU change on oscar or main.
