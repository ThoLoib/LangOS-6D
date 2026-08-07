# OSCAR+: Shape-Aware Open-Set CAD Retrieval

This branch (`exp/ulip2v2`) extends the original two-stage OSCAR baseline with a modular pipeline and integrates **ULIP-2 shape-aware retrieval** as a third scoring channel.

Baseline reproduced at **75.95% Top-1** on YCBV-GSO.
The pipeline adds scale estimation and 6D pose estimation on top of the retrieval result.

> **Status (2026-07-09):** End-to-End pipeline runs successfully on YCBV-GSO. Features: ULIP `pc`/`cross`/`both` modes, partial-to-partial matching, multi-view aggregation (softmax top-k, k=8, τ=0.5), SAM2.1 segmentation, depth gating, configurable SOR/ROR, FoundationPose HTTP integration with ICP fallback, optional scale gate with fast estimate, rotation evaluation for ULIP candidates. Debug visualization via `--debug-viz` with per-step ranking CSVs. Retrieval evaluation is shared across datasets via `object_retrieval/eval_common.py` with thin per-dataset wrappers; one run emits six explicit variants (`clip_only`, `dino_only_full`, `ulip_only_full`, `dino_only_clip_pruned`, `ulip_only_clip_pruned`, `clip_pruned_dino_ulip`) — see `AI_HANDOFF.md` for details.

## ULIP Modes (Step 5)

Step 5 supports three retrieval modes:

- `pc`: observed point cloud -> ULIP point encoder -> CAD point embeddings
- `cross`: query ROI image -> OpenCLIP image encoder -> CAD point embeddings
- `both`: weighted combination of `pc` and `cross` query embeddings

Additionally, `--ulip-partial-views` switches the reference side from full-mesh sampling to precomputed partial PCs per view (best-of-8-views scoring via softmax top-k aggregation, k=8, τ=0.5). This eliminates the domain mismatch between the partial observed PC and the full CAD model.

CLI flags:

```bash
--ulip_mode {pc,cross,both}
--ulip_image_weight 0.5
--ulip-partial-views          # use partial PCs per view instead of full mesh
```

## ULIP CAD Cache

Step 5 stores CAD embeddings in an on-disk cache inside the CAD/images directory.

- Full mesh mode: `.ulip_cache_<hash>.pt`
- Partial views mode: `.ulip_partial_cache_<hash>.pt`
- First run: computes all CAD embeddings (slow)
- Subsequent runs with same config+meshes: loads from cache (much faster)

## FoundationPose Integration

FoundationPose runs in a **separate Docker container** and is called via HTTP from OSCAR.

- FoundationPose repo on host: `~/thesis/FoundationPose`
- Docker image: `shingarey/foundationpose_custom_cuda121:latest`
- Compose service: `foundationpose` (exposes port 5050)
- OSCAR calls `http://foundationpose:5050/estimate_pose` from Step 8
- If FoundationPose is unavailable or fails, Step 8 falls back to ICP automatically

Architecture:
- OSCAR container: `tholoi/oscar-plus` (CUDA 12.2, Python 3.11)
- FP container: `shingarey/foundationpose_custom_cuda121` (CUDA 12.1, Python 3.8, pytorch3d, kaolin, nvdiffrast)
- Communication: HTTP over Docker compose network
- Shared data: OSCAR repo mounted read-only at `/oscar` in the FP container for CAD model access

Important:
- Do not force FoundationPose dependencies into OSCAR's main Python environment.
- The two-container split exists because of incompatible CUDA/torch/pytorch3d versions.

---

## Pipeline Overview

```
Natural language prompt + RGB-D image
          |
          v
+--------------------------------------------------------------+
| Step 1 | Object Localization  | GroundingDINO + SAM          |
| Step 2 | Point Cloud          | RGB-D -> 3D point cloud      |
| Step 3 | CLIP Retrieval       | Prompt/description matching  |
| Step 4 | DINOv2 Re-Ranking    | Visual feature comparison    |
|        |                      | (batch + disk cache)         |
| Step 5 | ULIP-2 Shape Match   | 3D geometry similarity       |
|        |                      | (partial views optional)     |
| Step 6 | Score Fusion         | CLIP * DINO * ULIP -> rank   |
|        |                      | (NaN-safe min-max norm.)     |
| Step 7 | Scale Estimation     | RANSAC+ICP coarse alignment  |
| Step 8 | Pose Estimation      | FoundationPose or ICP fallback |
+--------------------------------------------------------------+
          |
          v
  Best matching CAD model + 6D pose + scale factor
```

All pipeline code lives in `pipeline/`. Each step is a self-contained module with a single dataclass result.

### Pipeline File Reference

| File | Description |
|---|---|
| `__init__.py` | Package marker, exports `__version__` |
| `run_pipeline.py` | Main entry point (`OSCARPlusPipeline` class), CLI argument parsing, orchestrates Steps 1–8, per-step ranking CSV export |
| `config.py` | Central `PipelineConfig` dataclass with all tunable parameters |
| `step1_localization.py` | GroundingDINO + SAM2.1 object detection and segmentation |
| `step2_pointcloud.py` | RGB-D backprojection, depth gating, voxel downsampling, SOR/ROR filtering |
| `step3_clip_retrieval.py` | CLIP-based semantic candidate retrieval from text descriptions |
| `step4_dino_reranking.py` | DINOv2 visual re-ranking with multi-view aggregation (softmax top-k) and disk cache |
| `step5_shape_matching.py` | ULIP-2 shape matching (pc/cross/both modes, partial views, multi-view aggregation, optional ICP rotation eval) |
| `step6_fusion.py` | NaN-safe min-max score normalization and weighted CLIP/DINO/ULIP fusion |
| `step7_scale_estimation.py` | RANSAC + ICP coarse alignment for scale factor estimation, fast bbox fallback |
| `step8_pose_estimation.py` | 6D pose via FoundationPose (HTTP) or ICP fallback |
| `foundationpose_bridge.py` | HTTP client for the FoundationPose container (path translation, encoding, error handling) |
| `utils.py` | Shared helpers: camera intrinsics loading, image I/O, BOP format parsing |
| `debug_viz.py` | Debug visualization functions (`save_debug_step1`–`step7_8`), 3D projection, interactive point cloud HTML export, ULIP top-5 display (`--debug-viz` only) |

---

## Getting Started

### 1. Clone
```bash
git clone git@github.com:pullover00/OSCAR.git
cd OSCAR
git checkout exp/ulip2v2
```

### 2. ULIP-2 Checkpoint (for Step 5)
Clone the ULIP repo next to this one and place the checkpoint:
```bash
cd ..
git clone https://github.com/salesforce/ULIP.git
# Download checkpoint:
# ulip2_pointbert_10k.pt -> ULIP/checkpoints/ulip2_pointbert_10k.pt
```
The `docker-compose.yml` mounts `../ULIP` as `/ulip` inside the container.

### 3. Build and Run
```bash
docker compose build
docker compose run --rm -it oscar bash
```

### 3.1 FoundationPose Service

FoundationPose runs as a separate compose service. Start both services:

```bash
docker compose up -d foundationpose   # start FP service (waits for health check)
docker compose run --rm -it oscar bash # start OSCAR interactively
```

Verify the FP service is healthy:

```bash
curl http://localhost:5050/health
# -> {"status": "ok"}
```

The FP service uses the pre-built `shingarey/foundationpose_custom_cuda121` image which already
contains all compiled dependencies (pytorch3d, kaolin, nvdiffrast). No manual environment setup needed.

### 4. Persistence (models, embeddings, caches)

With the current compose mounts, the following data persists across container restarts/re-creation:

- Ollama model store: `/root/.ollama` (named volume `ollama_data`)
- HuggingFace cache: `/root/.cache/huggingface` (named volume `hf_cache`)
- Torch/OpenCLIP caches: `/root/.cache/torch`, `/root/.cache/clip` (named volumes)
- Project outputs and embedding caches (inside repo): persisted via `.:/app`
  - Example: `pipeline_output/`, `debug_output/`
  - Example: `.ulip_cache_*.pt` in `object_database/...`
  - Example: `.dino_cache_*.pt` in `object_images/...`

---

## Preprocessing a New Dataset

Preprocessing a CAD gallery for retrieval is two stages, run in order:

1. **Render + partial point clouds + descriptions** (`rendering/onboard_and_sync.sh` / `onboard_dataset.sh`)
2. **Embedding caches** — CLIP-text, DINOv2, SigLIP, ULIP-2 ×3, Uni3D (`tools/precompute_embeddings.py`)

Both stages are idempotent and resumable: rerunning skips whatever's already built and only computes what's missing or changed.

**Database layout** (what stage 1 produces, and what stage 2 reads):
```text
OSCAR/
+-- object_database/{dataset}/
|   +-- {object_id}/
|       +-- textured_simple.obj          <- CAD model (or meshes/model.obj)
|       +-- texture_map.png              <- texture (optional)
|   +-- descriptions_attributes.json     <- VLM captions, one entry per rendered view
+-- object_images/{dataset}/
    +-- {object_id}/
        +-- {obj_id}_0.png … _41.png            <- rendered views (42 icosphere angles)
        +-- {obj_id}_view0_CamMatrix.npy … _41   <- 3x4 camera matrices
        +-- {obj_id}_view0_partial.npz  … _41    <- per-view partial point clouds
```

### Stage 1 — Render, partial point clouds, descriptions

**Recommended: one command, from WSL/host** (Docker rendering + rclone sync to Drive):
```bash
bash rendering/onboard_and_sync.sh --dataset MI3DOR --remote gdrive:Masterthesis/OSCAR

# Only specific sub-steps (render | partial | describe):
bash rendering/onboard_and_sync.sh --dataset MI3DOR --remote gdrive:... --step render
# Skip descriptions for now, run them later separately:
bash rendering/onboard_and_sync.sh --dataset MI3DOR --remote gdrive:... --skip-describe
# See what would run without doing it:
bash rendering/onboard_and_sync.sh --dataset MI3DOR --remote gdrive:... --dry-run
```
Supported dataset names (see `rendering/onboard_dataset.sh` header): `ycbv_gso, ycbv, gso, MI3DOR, housecat6d, shrec18` (OBJ-based) and `tless, lmo, itodd` (BOP PLY-based, auto-prepared).

**Without Drive sync** (Docker-only — same rendering, no rclone):
```bash
bash rendering/onboard_dataset.sh --dataset MI3DOR
bash rendering/onboard_dataset.sh --dataset MI3DOR --step render     # one step
bash rendering/onboard_dataset.sh --dataset MI3DOR --step render --overwrite
```
Steps run, in order: `prepare` (BOP PLY → object_database layout, only for tless/lmo/itodd) → `render` (42-view Blender renders) → `partial` (per-view partial point clouds) → `describe` (VLM captions → `descriptions_attributes.json`).

**Manual / individual scripts**, if you need to run one step directly instead of via the orchestrator:

```bash
# Render (needs Blender 3.4+ at rendering/blender-*/blender):
cd rendering && ./blender-3.4.1-linux-x64/blender -b -P rendering.py
# (env vars: OBJECT_FOLDER, OBJECT_IMAGES, NUM_VIEWS — see onboard_dataset.sh for how it sets these)

# Partial point clouds (front-face-culled per-view point clouds, for ULIP-2 partial mode):
python3.11 rendering/generate_partial_pointclouds.py \
    --cad_dir object_database/MI3DOR/ \
    --images_dir object_images/MI3DOR/ \
    --num_points 10000

# Descriptions (VLM captions per rendered view):
python3 rendering/generate_descriptions.py \
    --images_dir object_images/MI3DOR/ \
    --output object_database/MI3DOR/descriptions_attributes.json
```

`generate_partial_pointclouds.py`: loads each CAD mesh via trimesh, normalizes to the same unit bounding box `rendering.py` uses, and for every view loads the stored camera matrix, samples 50k points on the mesh surface, keeps only front-facing points (face normal · view direction > 0), and resamples down to `--num_points`. Output: compressed `.npz` per view with keys `points` (N,3) and `colors` (N,3). ~1s/object.

### Stage 2 — Embedding caches

Once stage 1 has produced renders + partial point clouds + descriptions, build the gallery embeddings with **`tools/precompute_embeddings.py`** — a small, dataset-agnostic script (dataset choice is just paths, nothing in the code is dataset-specific):

```bash
docker compose run --rm oscar bash -lc \
  "python3 tools/precompute_embeddings.py \
     --dataset MI3DOR \
     --data-root eval/datasets/mi3dor/mi3dor_full \
     --images-dir object_images/MI3DOR \
     --desc-file object_database/MI3DOR/descriptions_attributes.json \
     --results-root object_retrieval/results_MI3DOR_stage1"

# List the 6 passes with a one-line description of each, no paths needed:
python3 tools/precompute_embeddings.py --list

# Check gallery readiness (CAD/render/description counts) without building anything:
python3 tools/precompute_embeddings.py --dataset MI3DOR --data-root ... --images-dir ... \
    --desc-file ... --results-root ... --dry-run

# Rebuild only specific passes:
python3 tools/precompute_embeddings.py --dataset MI3DOR ... --passes siglip,uni3d
```

| Pass | Builds | Cache file |
|---|---|---|
| `base` | CLIP-text (descriptions), DINOv2 (rendered views), ULIP-2 colored partial-view shape | `.clip_text_cache_*.pt`, `.dino_cache_*.pt`, `.ulip_partial_cache_*.pt` |
| `siglip` | SigLIP image embeddings (alternative to DINOv2) | `.siglip_cache_*.pt` |
| `ulip_fullmesh` | ULIP-2 colored, full CAD mesh (no partial views) | `<data_root>/cad/.ulip_cache_*.pt` |
| `ulip_pc_rgb` | ULIP-2 colored, partial-view, PC-mode query tag — reuses `base`'s cache (same config), near-instant | (same file as `base`) |
| `ulip_pc_xyz` | ULIP-2 XYZ-only (8k pts, no color), partial-view | `.ulip_partial_cache_*.pt` (different digest) |
| `uni3d` | Uni3D-g, partial-view point clouds | `.ulip_partial_cache_*.pt` (different digest, `encoder=uni3d` tag) |

Note the distinction between what each channel embeds: `clip`/`dino`/`siglip` embed the *rendered images*; `shape` (built by `base`, `ulip_fullmesh`, `ulip_pc_rgb`, `ulip_pc_xyz`, `uni3d`) embeds the object's *3D point cloud/geometry* — no images involved, even though the underlying model (ULIP-2) shares a joint embedding space with images and text.

All caches are content-fingerprinted (model config + source data — never absolute paths or mtimes), so they're safe to copy/rclone to another machine, as long as that machine reproduces query-side encoding with the identical checkpoints/patches. See `docs/LAPTOP_EMBEDDINGS_SETUP.md` for exactly what an eval machine needs (checkpoints, the Uni3D inference patch for portable FPS, `timm` version) to consume caches built here.

---

## Experiment 1 — Stage 1: SHREC'18 Retrieval Tuning

Runs the thesis ablation grid (E1, E2, E2b, E4, E6, E7, O1, O2, O4, O5 — 33 cells) on
SHREC'18 ObjectNN+, scores every cell with the **official** track metrics, and writes the
winning configuration to `best_config.json` for Stages 2–5. Selection rule: **highest nDCG,
tie-break mAP**.

One entry point, flag-driven, no subcommands:

```bash
docker compose run --rm oscar \
    python3 experiments/experiment1_shrec18_stage1.py --all --resume
```

The script **never downloads anything**. You provide the data; it validates what is present
and tells you exactly what is missing.

### What you must provide

| What | Where | How to get it |
|---|---|---|
| Raw SHREC'18 | `eval/datasets/shrec18/shrec18_full/` — `cad/*.obj` (3,308), `rgbd/*.ply` (2,101), `results/` | SHREC'18 ObjectNN+ track download |
| Official GT + scorer | `eval/shrec18_official/` | `git clone https://github.com/hkust-vgd/shrec18 eval/shrec18_official` (gitignored) |
| Rendered gallery | `object_images/shrec18/` — 42 views + 42 `_partial.npz` per CAD | "Preprocessing a New Dataset" above, `--dataset shrec18` |
| Descriptions | `object_database/shrec18/descriptions_attributes.json` | same, `describe` step |

The official clone matters: `rgbd.csv` and `cad.csv` carry the real category **and**
subcategory for all 2,101 queries and 3,308 CADs, and the track's own `metrics.py` is reused
unchanged — so the numbers are leaderboard-comparable rather than reconstructed. Graded
relevance is subcategory match = 2, category match = 1, and every metric is cut at
*f* = category size.

### Setup, end to end

```bash
# 1. Official GT + scorer (gitignored; the script errors out clearly without it)
git clone https://github.com/hkust-vgd/shrec18 eval/shrec18_official

# 2. Gallery preprocessing — renders, partial point clouds, descriptions
bash rendering/onboard_dataset.sh --dataset shrec18

# 3. Gallery embedding caches (all six passes).
#    This also validates the inputs first and reports anything missing,
#    and it writes no ablation results — so it doubles as the setup check.
docker compose run --rm oscar \
    python3 experiments/experiment1_shrec18_stage1.py --precompute

# 4. Run the grid
docker compose run --rm oscar \
    python3 experiments/experiment1_shrec18_stage1.py --all --resume
```

Steps 2 and 3 are the GPU-heavy part — rendering the gallery and encoding it with DINOv2 /
SigLIP / ULIP-2 / Uni3D. **A strong GPU is strongly advised for them.** Both are idempotent
and resumable, and the resulting caches are fingerprinted by content (relative path + size,
never absolute paths or mtimes), so they can be built once and copied elsewhere; `--precompute`
also writes `object_images/shrec18/precompute_manifest.json` recording what produced them.
Step 4 only needs to encode the queries and score the grid.

There is no dedicated "validate only" flag: input validation runs at the start of every real
invocation (and under `--precompute`), and missing renders, descriptions or CAD meshes are
reported as a list of what to provide. `--allow-partial-gallery` downgrades that to a warning.

**Query preprocessing is automatic.** On first run the script builds, caches under
`eval/datasets/shrec18/stage1/`, and reuses thereafter:

- `queries/<hash>.npz` — normalized points + colors (feeds the shape channels and all geometry)
- `queries/<hash>.png` — RGB crop (feeds the CLIP and DINOv2 channels)
- `gt/official_labels.json` — parsed from the official CSVs

Sanity-check the crops by adding `--viz-check 16` to a run — it writes
`eval/datasets/shrec18/stage1/viz_check.png` and then continues, so pair it with a small
selection rather than expecting it to exit on its own:

```bash
docker compose run --rm oscar python3 experiments/experiment1_shrec18_stage1.py \
    --ablations E1a_text_only --limit-queries 20 --viz-check 16
```

Delete `object_retrieval/results_shrec18_stage1/E1a_text_only/` afterwards — see the traps
below.

### Useful flags

```bash
--list                     # print the ablation registry (name, group, passes, question) and exit
--ablations E1,E4,O4_V8    # run groups or individual cells instead of --all
--resume                   # skip ablations/passes whose outputs already exist
--overwrite                # recompute even if results exist
--with-geometry            # add the 6 GeDi/Chamfer cells (needs the gedi service)
--limit-queries 20         # smoke test
--allow-partial-gallery    # run against an incomplete gallery
--precompute               # build the gallery reference caches only, then exit
--bench-gedi N             # time N GeDi+RANSAC fits, extrapolate full-DB cost, exit
--viz-check N              # contact sheet of N query crops
```

### How it runs: two tiers

Six expensive **channel-score passes** (`base`, `siglip`, `ulip_pc_fullmesh`, `ulip_pc_rgb`,
`ulip_pc_xyz`, `uni3d`) are computed once and cached as
`object_retrieval/results_shrec18_stage1/_cache/scores_<pass>.pt`, each holding full per-query
score vectors over all 3,308 CADs. The 33 grid cells are then **derived cheaply** from those
vectors — fusion weights and method, shortlisting, view-count swaps. `O4 V∈{8,16,32,42}` is
re-aggregated from the FPS-ordered view prefix, so changing the view budget costs no
re-rendering and no re-encoding.

Only `base` and `siglip` consume the query **images**; all four shape passes are pc-mode and
read the query `.npz` point clouds.

### Geometry cells (`--with-geometry`)

Adds `E2_fitness`, `E2_chamfer_unaligned`, `E2_chamfer_ransac`, `E2_chamfer_icp`, `O1c`, `O1d`,
which re-rank the top-5 fusion shortlist per query using GeDi descriptors + RANSAC and a
trimmed one-sided surface distance.

```bash
docker compose up -d gedi        # wait for the container to report healthy
docker compose run --rm oscar \
    python3 experiments/experiment1_shrec18_stage1.py --all --resume --with-geometry
```

These are by far the most expensive cells in the grid, and the cost is almost entirely GeDi +
RANSAC. The three aligned signals (`fitness`, `chamfer_ransac`, `chamfer_icp`) share **one**
registration, so the first aligned cell pays for it and the other two are cache hits.
Per-(query, CAD) scores are appended to `_cache/geometry_scores.jsonl` and are fully resumable:
interrupting costs at most the query in flight, so the run can be stopped and continued freely.

### Outputs

```text
object_retrieval/results_shrec18_stage1/
+-- <ablation>/metrics_summary.json     <- metrics + the full config that produced them
+-- <ablation>/results_per_query.json
+-- stage1_summary.csv                  <- every cell, one row
+-- stage1_summary.tex                  <- booktabs table, paste-ready
+-- best_config.json                    <- winner + frozen PipelineConfig for Stages 2-5
```

Aggregation reruns automatically at the end of every invocation, so the summary always
reflects every cell computed so far.

**Reading the metrics.** The seven columns come from the unmodified official
`eval/shrec18_official/metrics.py`, and they are not seven independent views. Every metric is
cut at `f` = the query's category size (~165 on average), which makes precision, recall, F1,
NNT1 and NNT2 identical to one another by construction; NNT2 in particular is inoperative
(`k2 = min(len(x), 2k)` collapses back to `f`). Only nDCG reads the graded relevance
(same sub-category = 2, same category = 1) — precision and AP binarise it. So the winner is
selected on **nDCG**, tie-broken by mAP.

This matters most for the geometry cells: re-ranking touches only the top 5, and permuting 5
items inside a 165-item prefix cannot change that prefix's membership, so precision is
*identical across every geometry variant* including the one that is worse than baseline. That
is the protocol working as specified, not a bug — but it means nDCG is the only official metric
that can tell the re-ranking arms apart. `results_per_query.json` carries enough per-query
detail (`top10` with labels, `first_relevant_rank`, `AP`, `nDCG`) to compute top-heavy
diagnostics like hit@k or MRR post hoc, without re-running anything.

### Traps worth knowing before you run

- **`--resume` does not notice changed query images.** The pass cache is validated on the
  gallery object list and on query **IDs** only — not on image content. Replacing the query
  crops under the same filenames and re-running gives you `[pass:base] loaded cache` and the
  **old** embeddings, with no warning. Delete `_cache/scores_base.pt` and
  `_cache/scores_siglip.pt` (and the per-ablation output dirs) whenever the crops change.
- **Smoke results poison a later full run.** Ablation directories are not tagged by query
  count, so `--limit-queries 20` followed by `--all --resume` keeps the n=20 numbers. Delete
  the ablation dirs between a smoke test and a real run.
- **Don't bypass the service entrypoint.** `docker compose run --rm --entrypoint bash oscar`
  breaks Open3D (`libgomp.so.1: cannot open shared object file`); the entrypoint sets up the
  environment.
- **If you reuse caches built on another machine, the encoder side must match exactly.** The
  run prints a provenance warning when the caches were built at a different commit. Only the
  encoder-side files matter (`pipeline/step3-5`, the encoder fields in `config.py`,
  `eval_common.build_pipeline`) — rendering and onboarding may differ. Uni3D in particular
  needs the pure-torch FPS patch applied wherever the caches were built *and* wherever queries
  are encoded, or its embeddings mismatch silently. `docs/LAPTOP_EMBEDDINGS_SETUP.md` lists the
  exact checkpoint and patch requirements.
- **Give WSL enough memory for the geometry cells.** The oscar container and the GeDi service
  share it, and the 50%-of-host default has been enough to get GeDi OOM-killed mid-run. See
  `AI_HANDOFF.md` for the `.wslconfig` settings and the rest of the operational notes.

---

## Running the Pipeline

### Debug mode (recommended for testing)
Saves 7 diagnostic PNG images to `debug_output/`:
```bash
# Via convenience script (YCBV-GSO defaults + FoundationPose):
./scripts/run_debug_pipeline_foundationpose.sh

# Or manually:
python3.11 -m pipeline.run_pipeline \
  --rgb eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
  --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
  --camera eval/datasets/ycbv_gso/test/000048/scene_camera.json \
  --prompt "I need the red mug" \
  --descriptions object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json \
  --reference_images object_images/ycbv_gso/ \
  --cad_models object_database/ycbv_gso/ \
  --ulip_repo /ulip \
  --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt \
  --pose_method foundationpose \
  --output debug_output \
  --debug-viz --until-step 8
```

If FoundationPose is unavailable or fails, the pipeline falls back to ICP automatically.

| Output File | Content |
|-------------|---------|
| `debug_01_localization.png` | Scene + mask overlay, cropped ROI, prompt analysis |
| `debug_02_pointcloud.png` | Depth (raw + masked), point cloud projections |
| `debug_02_pointcloud_3d.html` | Interactive 3D point cloud viewer |
| `debug_03_clip.png` | Query ROI vs. Top-5 CLIP candidates |
| `debug_04_dino.png` | Query vs. best DINOv2 match, ranking table |
| `debug_05_ulip.png` | 3D point cloud scatter, Top-3 ULIP-2 shape matches |
| `debug_06_fusion.png` | CLIP/DINO/ULIP/Fused score table + winner |
| `debug_07_scale_pose.png` | 3D wireframe overlay on scene, scale/pose info |

### Full pipeline (single image)
```bash
# Via convenience script:
./scripts/run_pipeline.sh

# Or manually:
python3.11 -m pipeline.run_pipeline \
    --rgb eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
    --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
    --prompt "mustard bottle" \
    --descriptions object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json \
    --reference_images object_images/ycbv_gso/ \
    --cad_models object_database/ycbv_gso/ \
    --camera eval/datasets/ycbv_gso/test/000048/scene_camera.json \
    --pose_method foundationpose
```

---

## Key Configuration (pipeline/config.py)

```python
voxel_size              = 0.002     # Point cloud downsampling (2mm, ~4000 pts)
depth_scale             = 10000.0   # BOP depth: 16-bit PNG, 0.1mm units
weight_clip             = 0.3       # Fusion weights
weight_dino             = 0.4
weight_ulip             = 0.3
ulip2_mode              = "cross"   # "pc" | "cross" | "both"
ulip2_use_partial_views = False     # True = partial PCs per view
ollama_model            = "gemma3:4b"  # LLM for prompt parsing
pose_method             = "icp"     # Pose estimation method

# Multi-view aggregation (Steps 4 & 5)
dino_view_aggregation   = "topk_softmax"
dino_view_topk          = 8         # Top views for aggregation
dino_view_temperature   = 0.5       # Softmax temperature
ulip_view_aggregation   = "topk_softmax"
ulip_view_topk          = 8
ulip_view_temperature   = 0.5

# Scale gate (optional, disabled by default)
scale_gate_enabled      = False
scale_icp_min_confidence = 0.15     # ICP confidence fallback threshold
```

---

## Legacy Evaluation (original OSCAR)

The original flat scripts are still available for reproducing the baseline:
```bash
# YCBV-GSO (baseline: 75.95% Top-1)
python retrieval_combi_eval.py

# MI3DOR
python retrieval_combi_eval_mi3dor.py
```

---

## Citation
```
@article{pulli2026oscar,
  title={OSCAR: Open-Set CAD Retrieval from a Language Prompt and a Single Image},
  author={Pulli, Tessa and Weibel, Jean-Baptiste and Hoenig, Peter and Hirschmanner, Matthias and Vincze, Markus and Holzinger, Andreas},
  journal={arXiv preprint arXiv:2601.07333},
  year={2026}
}
```