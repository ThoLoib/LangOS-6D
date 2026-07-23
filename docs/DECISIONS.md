# Decisions

## 2026-07-23 Stage-1: official SHREC'18 evaluation for all runs + two-PC precompute

Decision
- **Adopt the official SHREC'18 evaluation everywhere** (tuning ablations and final numbers), superseding the reconstructed category-only GT. The experiment loads the track's own `rgbd.csv`/`cad.csv` (real category+subcategory for all 2,101 queries / 3,308 CADs, cloned from `hkust-vgd/shrec18` into `eval/shrec18_official/`, gitignored) and scores with the track's unchanged `metrics.py` (graded relevance: subcategory=2, category=1). Metric set is now nDCG/precision/recall/F1/AP/NNT1/NNT2 at top-f; best config = highest (graded) nDCG, tie-break mAP. Runs on all 2,101 queries.
- **Split the compute across two PCs.** The gallery-generating PC precomputes the query-independent reference embeddings (DINO, SigLIP, ULIP partial RGB + XYZ-only, ULIP full-mesh, Uni3D) via `python experiments/experiment1_shrec18_stage1.py --precompute`; the eval PC does only query-side work, GeDi (fusion top-5), fusion, and metrics.
- **Cache fingerprints are now content-stable** (`step4._dir_fingerprint`, `step5._get_cache_path`, `_get_partial_cache_path`): hash of file **size + relative path** instead of mtime, so a cache built on one PC is reused on the other (mtimes change after copy/Drive; sizes are byte-stable).
- **Provenance manifest + verify-on-load**: `--precompute` writes `object_images/shrec18/precompute_manifest.json` (encoders, checkpoint, dims, code commit); the eval PC warns loudly if the shipped caches were built at a different commit, guarding against encoder-path divergence between the two repos.
- **Shape channel: encode all 42 partial views, aggregate the top-16** for the first experiment (`SHAPE_AGG_VIEWS=16` in the experiment script). Full-resolution embeddings stay in the cache as the reusable asset; only the retrieval-time pooling is trimmed.
- **Query crops rendered with Mesa/EGL** in a derived image `oscar-plus-egl` (base image lacks `libEGL`); `_offscreen_available()` auto-uses the GL mesh renderer when EGL is present, else the CPU point-splat.

Rationale
- The official GT+scorer are the only way OSCAR+ numbers are comparable to the published participants; using them for tuning too keeps one consistent metric. The two-PC split removes the ~13 h reference-encoding bottleneck from the eval PC by running it on the otherwise-idle gallery PC, where the data is already local.

Alternatives considered
- Reconstructed union-find GT (category-only, train split) — kept as a fallback (`build_gt`) but superseded now that the official CSVs are available.
- Capping the *encoding* at 16 views — rejected: throws away the full embeddings; instead encode 42, aggregate 16.
- Requiring identical Docker/repo on both PCs — relaxed to "identical encoder files + provenance check"; rendering/onboarding code may diverge freely.

## 2026-07-20 Stage-1 experiment runner: two-tier execution, reconstructed GT, train split only

Decision
- `experiments/experiment1_shrec18_stage1.py` is the single entry point for thesis Experiment 1 (Stage 1 ablation grid on SHREC'18 ObjectNN+). It is flag-based (no subcommands) and never downloads/manages data — the user provides raw SHREC'18, rendered views, and descriptions; the script only validates presence.
- Execution is split into expensive cached *channel-score passes* (one per encoder×reference combo: base, siglip, ulip_fullmesh, ulip_pc_rgb, ulip_pc_xyz, uni3d) and cheap *derivations* (fusion weights/method, scoping, view budget, geometry re-ranking) computed from the cached vectors. Derivations reuse `pipeline/step6_fusion.ScoreFusion` on synthetic result objects rather than re-implementing fusion.
- SHREC'18 category GT is reconstructed by union-find over the `results/` relevance lists (exactly 20 components; 1,452 train queries, 3,305/3,308 CADs). Stage 1 therefore tunes on the labeled train split only; the 3 unlisted CADs stay as distractors. An official label file with the same JSON schema can be dropped in as replacement.
- Geometry ablations (E2, O1c–e) are gated behind `--with-geometry`; GeDi-signal cells are skipped with a warning when the gedi service is unreachable. O1e approximates "GeDi inside fusion" on a text+view top-10 pool (full-database GeDi RANSAC is infeasible).
- E7/O5 force `ulip2_mode="pc"` in both arms so the point cloud (not the query image) is what varies; the BASE config keeps the repo default `cross` mode.

Rationale
- One pass over 42 cached views serves all O4 budgets (FPS prefix), all fusion/scope variants, and both E1/O2 cascade arms — the grid runs in roughly one encoder pass per encoder combo instead of one per grid cell.
- The dump ships without category labels; the co-occurrence graph over the organiser-provided training lists provably partitions into the 20 track categories (README guarantees list purity).

Alternatives considered
- Running `eval_common.run_evaluation` per grid cell — rejected: cannot express full-database 3-channel fusion, majority voting, pc-mode, or geometry re-ranking, and would re-encode per cell.
- Waiting for official GT for all 2,101 queries — rejected for now: blocks the experiment; train-split tuning is defensible for a tuning stage.

## 2026-07-17 View-count-independent caches for DINO/SigLIP and ULIP partial

Decision
- DINO/SigLIP and ULIP partial caches always store embeddings for ALL available views (42 for icosphere subdiv=1). The `num_views` parameter is applied as a runtime filter after cache load, not during cache generation. Cache filenames no longer include `num_views`.

Rationale
- Views are FPS-ordered: the first 8 of 42 are always identical images/point clouds regardless of when `num_views` is set. Caching per-view-count caused redundant multi-hour cache rebuilds during ablation O4 (V ∈ {8, 16, 42}) — 3 separate caches encoding the same data.
- Runtime trimming is O(N) dictionary/tensor slicing, negligible vs. encoding cost.

Alternatives considered
- Keep per-view-count caches — rejected: wastes disk and encoding time for identical data.
- Lazy per-view encoding with partial cache merge — rejected: added complexity for no benefit since all 42 views are precomputed anyway.

## 2026-07-17 infer_model_id: generic vs. specific filename heuristic

Decision
- `infer_model_id()` in `rendering.py` uses a whitelist of generic model filenames (`model.ply`, `model.obj`, `textured_simple.obj`, etc.) to decide the ID strategy. Generic names → parent directory is the object ID. Non-generic names → filename stem is the object ID.

Rationale
- Datasets have two layout patterns: (a) one directory per object with a standardized mesh name (BOP, YCB-V/GSO), and (b) flat or category directories with many uniquely-named mesh files (MI3DOR, SHREC'18, HouseCat6D). The old code always used parent directory, collapsing MI3DOR from 3848 to 21 objects, SHREC'18 from 3308 to 1.
- The generic name whitelist is small, stable, and covers all known dataset conventions.

Alternatives considered
- Count files per directory to detect shared dirs — rejected: breaks for YCB-V/GSO directories that have multiple format alternatives (textured_simple.obj + model.obj).
- Per-dataset config flag — rejected: adds manual configuration burden; the heuristic works automatically for all 7 datasets.

## 2026-07-17 Docker preprocessing + WSL rclone sync (two-process architecture)

Decision
- Preprocessing (Blender, partial PCs, LLaVA) runs inside Docker via `onboard_dataset.sh`. Rclone sync runs on the WSL host via `rclone_watch.sh`. The two processes run in parallel — Docker writes to mounted volume, WSL rclone reads from same path.

Rationale
- Docker container has GPU access and Python dependencies (numpy, trimesh, transformers) but no rclone. WSL has rclone but not the Python stack. Keeping them separate avoids installing rclone in Docker or Python deps on bare WSL.
- `rclone copy` (not `sync`) prevents deletion of previously-synced remote files when local files are cleaned up for disk space.

Alternatives considered
- Install rclone inside Docker — rejected: adds config complexity (OAuth tokens, service accounts) to the container.
- Run everything on WSL without Docker — rejected: no numpy/trimesh/transformers on bare WSL Python.

## 2026-04-23 OSCAR+ eval: CLIP-pruned DINO/ULIP derived by id-filter on single full pass

Decision
- `run_query` in `object_retrieval/eval_common.py` runs CLIP once, DINO once at full depth, and ULIP once at full depth. The CLIP-pruned DINO and ULIP variants are produced by `_filter_dino_result_by_ids` / `_filter_shape_result_by_ids` — pure id-intersection on the full rankings, preserving the original stage scores and order. No second CLIP-gated rerank pass.

Rationale
- `step4_dino_reranking.rerank()` computes per-object aggregated cosine similarity (`_aggregate_view_scores` over views) with no cross-candidate normalisation. Restricting the candidate pool only changes which rows survive the final `topk()` — the per-object score is identical.
- `step5_shape_matching.match()` computes per-object cosine similarity between the query embedding and each CAD embedding independently. Candidate gating only truncates the final top-k.
- Therefore the derived pruned ranking is mathematically equivalent to an explicit CLIP-gated rerank. The DINO filter also backfills `clip_score` from the CLIP score map so downstream fusion inputs match byte-for-byte.
- Saves exactly one DINO pass + one ULIP matmul per query vs. an explicit double-run design, and guarantees the full-set and CLIP-pruned variants share a common source of truth (no drift from re-running the encoder twice on the same input).

Alternatives considered
- Explicit double-run (one full + one CLIP-gated) — rejected on cost grounds; offers no additional information.
- Compute only the CLIP-pruned variant (drop the full variants) — rejected; the thesis needs the full-set numbers for comparison.
- Move the filter inside `ShapeMatcher.match()` — rejected; keeping the filter as an eval-module helper keeps the pipeline module's behaviour unchanged and non-eval callers unaffected.

Breakage risk
- If a future stage introduces cross-candidate normalisation (softmax across pool, z-score, rank-based features), the equivalence breaks and the helper must be revisited. Guarded by the docstring in `_filter_dino_result_by_ids`.

## 2026-04-23 OSCAR+ eval: auto-expand full-ranking depth from reference counts

Decision
- `run_evaluation` computes `dino_full_top_k = max(cfg.dino_top_k, len(dino_rer._ref_embeddings))` and `ulip_full_top_k = max(cfg.ulip2_top_k, len(shape_m._cad_embeddings))` once at startup. Both are threaded into `run_query`. `cfg.dino_top_k` / `cfg.ulip2_top_k` now only control the number of top candidates reported per query; they no longer gate the ranking depth the pipeline actually produces.

Rationale
- Deriving CLIP-pruned variants by id-filtering requires the full ranking to actually contain every CLIP candidate. With `dino_top_k=5` (the MI3DOR default) the full pass would silently drop most CLIP candidates, corrupting the derived pruned variant.
- Using the loaded reference count as the depth is the tightest upper bound that always covers the CLIP set, and is free — DINO and ULIP already compute similarities against every reference; only the final sort widens.
- A user-visible log line on startup reports whether auto-expansion kicked in and to what depth. The used depths are persisted in `metrics_summary_topk_K.json` under `config.dino_full_top_k_used` / `ulip_full_top_k_used` so results are reproducible without needing the source `EvalConfig`.

Alternatives considered
- Require the user to set `dino_top_k` / `ulip2_top_k` equal to the reference count — rejected; silent footgun if forgotten, and couples eval config to dataset size.
- Cap depth at a fixed large sentinel (e.g. 10⁶) — works, but hides the actual depth used and makes the summary less informative.

## 2026-04-23 OSCAR+ eval: exactly six unambiguous variant names, no runtime-config-dependent labels

Decision
- The summary's `variants` block contains exactly: `clip_only`, `dino_only_full`, `ulip_only_full`, `dino_only_clip_pruned`, `ulip_only_clip_pruned`, `clip_pruned_dino_ulip`. Primary variant = `clip_pruned_dino_ulip`. The earlier ambiguous keys (`dino_only`, `ulip_only`, `fusion_all`, `fusion_clip_ulip`) are removed; so are the `prune_dino_with_clip` / `prune_ulip_with_clip` config flags that selected between full and pruned at runtime.

Rationale
- The old scheme required a reader to consult the runtime config to know whether `dino_only` referred to the full-set or the CLIP-gated pool. That broke the JSON contract — identical filenames could contain different semantics across runs.
- Every run now emits both perspectives under explicit names, so downstream analysis scripts and thesis-text comparisons have a single stable schema.
- `fusion_all` fused DINO + ULIP inputs whose meanings depended on the config; replacing it with `clip_pruned_dino_ulip` (explicit DINO-pruned + ULIP-pruned) gives a well-defined fusion semantics. Dropping `fusion_clip_ulip` (which carried the same ambiguity on the ULIP side) keeps the variant set minimal and comparable.

Alternatives considered
- Keep `fusion_all` but rename it — rejected; the fused inputs were still config-dependent, so the renaming would hide the ambiguity rather than fix it.
- Keep the runtime-config toggle for flexibility — rejected; the cost of running both is negligible (one DINO rerank + one ULIP matmul), and the JSON-contract benefit is large.

## 2026-04-13 Step 7 scale: ICP confidence fallback to sorted-bbox estimate

Decision
- After RANSAC+ICP scale estimation in `estimate()`, if the computed confidence is below `config.scale_icp_min_confidence` (default 0.15), override the scale factor with `estimate_fast()` (rotation-invariant sorted-bbox). The ICP transformation T is still returned for use as coarse alignment initial pose in Step 8.

Rationale
- For heavily truncated partial views (objects cut off at image boundary), RANSAC+ICP produces degenerate alignments where axis ratios become wildly inconsistent (e.g. [3.0, 1.5, 0.5]). The Partial-Aware Scale logic picks the two highest ratios, yielding a scale factor far from ground truth (observed: 2.25× for scissors). Confidence = 0.00 in this case.
- `estimate_fast()` (sorted-bbox, no ICP) was already added for the scale gate and is reliable precisely because it does not depend on alignment quality.
- Keeping T from ICP even when scale is overridden preserves the coarse orientation estimate for Step 8. The scale and the alignment transform are independent outputs.

Alternatives Considered
- Use `estimate_fast()` for Step 7 entirely — rejected; for fully-visible objects RANSAC+ICP gives a better, axis-aware scale that accounts for the partial-view depth underestimation.
- Lower the `best_2` threshold to require a minimum ratio similarity — equivalent to the confidence check, but less interpretable.
- Make `scale_icp_min_confidence` un-configurable — rejected; the right threshold depends on object type and dataset.

## 2026-04-13 scale gate uses estimate_fast, not full ICP

Decision
- `_select_candidate_with_scale_gate()` now calls `estimate_fast()` (sorted-bbox) for gate decisions instead of `estimate()` (RANSAC+ICP). Step 7 still runs `estimate()` afterward for coarse alignment.

Rationale
- Using full RANSAC+ICP per candidate in the gate loop was the original design, but it was non-deterministic and slow. The same ICP degeneration that caused 2.25× scale for scissors also caused the gate to reject the correct candidate (conf=0.00 < threshold) on some runs but accept it on others. `estimate_fast()` is deterministic, cheap, and already available. The gate only needs a plausibility check, not a precise metric.

Alternatives Considered
- Keep full ICP in gate — rejected; non-deterministic behavior makes results session-dependent.
- Use `max_extent` fallback — less accurate than sorted-bbox for elongated objects.

## 2026-04-13 scale gate after fusion for metric candidate selection

Decision
- Add an optional post-fusion candidate selection step ("scale gate") that iterates over fused candidates in rank order and accepts the first whose `ScaleEstimator` result falls within a configurable scale range (`[scale_gate_min, scale_gate_max]`). Disabled by default (`scale_gate_enabled=False`).

Rationale
- ULIP-2 intentionally normalizes scale away (unit-sphere normalization before encoding). Objects with similar shape but different physical size therefore receive similar ULIP similarity scores. Scale estimation belongs to the metric RGB-D/CAD alignment stage (Step 7), not the embedding stage.
- The existing pipeline always used the top-1 fusion candidate unconditionally for Steps 7 and 8. The scale gate allows the pipeline to skip implausibly-sized candidates and use the next best match instead, without changing the ULIP scoring logic.
- Scale estimation via RANSAC+ICP is already available in `ScaleEstimator`; reusing it here avoids adding a new metric.

Alternatives Considered
- Inject a size feature into the ULIP embedding or fusion score — rejected; ULIP's scale invariance is intentional and scale estimation depends on the observed metric point cloud which is not available during ULIP scoring.
- Hard-code a fixed scale range — kept configurable (`scale_gate_min`, `scale_gate_max`) since valid ranges differ per dataset and object class.
- Enable by default — rejected; the gate assumes CAD meshes have consistent real-world scale. Datasets with arbitrarily-normalized CAD models would have all candidates rejected.

## 2026-04-13 scale gate reject policy

Decision
- Two reject policies: `fallback_best` (default) returns the top-1 fusion candidate when no candidate passes the scale check; `fail` skips Steps 7 and 8 entirely.

Rationale
- `fallback_best` preserves the previous behavior (always use top-1) as the safe fallback. The scale gate then acts as an upgrade path rather than a hard failure condition.
- `fail` is useful in evaluation contexts where an incorrect pose estimate is worse than no estimate.

Alternatives Considered
- Single hard-fail behavior — rejected; would break existing experiments when the gate finds no plausible candidate.
- Silent fallback without logging — rejected; the rejection log is essential for debugging and analysis.

## 2026-04-13 rotation evaluation via ICP on partial views, not random SO(3) augmentation

Decision
- Evaluate rotation sensitivity by running lightweight ICP between the observed partial PC and each Top-K candidate's best partial reference PC. Record `registration_fitness` and `registration_rmse` per candidate. Default weight 0.0 means debug-only (no ranking change).

Rationale
- The alternative (random SO(3) augmentation of the query PC before ULIP encoding) was rejected for several reasons:
  - Non-deterministic unless seeded; makes scores session-dependent.
  - Taking the max over random rotations can reward accidental alignments.
  - In `both` mode, the image embedding stays fixed while the PC embedding is rotated — conceptually inconsistent.
  - Expensive: requires multiple ULIP forward passes per query.
- ICP on known partial views is deterministic, cheap (per Top-K not all candidates), and directly answers the research question: does geometric alignment to the candidate's reference view improve match confidence?

Alternatives Considered
- Random SO(3) augmentation — rejected; see above. May be added later as an explicit ablation flag if needed.
- RANSAC+ICP with full mesh — more expensive and less targeted than partial-view ICP; full mesh ICP is already used in Step 7.
- FoundationPose for multi-candidate validation — too expensive for per-candidate evaluation; ICP is the right diagnostic tool here.

## 2026-04-13 DRY mesh-path resolution helper

Decision
- Extract the repeated image-path fallback logic (detect `.png/.jpg` in `cad_model_path`, fall back to recursive `_find_cad_mesh()`) into a single `_resolve_mesh_path_for_candidate()` method on `OSCARPlusPipeline`. Use it in Steps 7, 8, scale gate, and debug viz.

Rationale
- The same 5-line logic was duplicated in Step 7 and Step 8 in `run_pipeline.py`. A third copy would have appeared in the scale gate loop. A single helper reduces the chance of the copies diverging.

Alternatives Considered
- Keep duplication — rejected; the fallback logic evolves (e.g. supported extensions) and duplicated copies are a maintenance hazard.

## 2026-04-09 GT bbox_center compensation made optional

Decision
- Add `gt_bbox_center_compensation` config flag (default: `False`) and `--gt-bbox-compensation` CLI arg. When OFF, GT wireframe overlay uses the BOP pose directly without subtracting `R_gt @ bbox_center`.

Rationale
- The tuna_can mesh has bbox_center only 4.2mm from origin. The compensation (subtracting R_gt @ bbox_center from translation) was introducing visible error (~5px shift) rather than correcting it.
- BOP datasets vary: some meshes are well-centered, others are not. A one-size-fits-all compensation is incorrect.
- Default OFF matches the common case for BOP datasets where meshes are near-centered. Users working with off-center meshes can opt in.

Alternatives Considered
- Always-on compensation (previous behavior) — rejected; incorrect for near-centered meshes.
- Remove compensation entirely — rejected; some meshes genuinely need it.

## 2026-04-09 SAM2 model_type warning fix

Decision
- Load `Sam2Config` explicitly and override `model_type = "sam2"` before passing to `Sam2Model.from_pretrained()`.

Rationale
- `facebook/sam2.1-hiera-large` declares `model_type: "sam2_video"` in its HuggingFace config.json, while `Sam2Model` expects `"sam2"`. The architectures are compatible for image segmentation — the mismatch is purely metadata. Overriding the config avoids the warning without forking the model or patching transformers.

Alternatives Considered
- Suppress the warning via `warnings.filterwarnings` — rejected; hides potentially useful warnings from other sources.
- Wait for HuggingFace to fix the metadata — rejected; no control over upstream timeline.

## 2026-04-03 multi-view aggregation for Steps 4 and 5

Decision
- Replace hard-max (single best view) object scoring in Step 4 (DINOv2) and Step 5 (ULIP-2 partial views) with a configurable multi-view aggregation strategy. Default: softmax-weighted top-k views (`topk_softmax`, k=4, τ=0.1).
- Approach inspired by OPEN (Chu et al., TCSVT 2024, Eq. 2-3): softmax over query-to-view cosine similarities produces view weights; weighted sum of similarities becomes the object score.

Rationale
- Hard-max is brittle: a single noisy or lucky view determines the entire object score. Under occlusion or viewpoint mismatch, the best-matching view may still be poor, while multiple moderately-good views collectively provide stronger evidence.
- The OPEN paper shows that query-guided multi-view attention (softmax over similarities) improves retrieval accuracy by 6-11% in occluded scenarios. Our adaptation is inference-time only (no training changes), using raw cosine similarities as logits instead of learned attention.
- Top-k selection (k=4 out of typically 8 views) discards low-quality views that would dilute the score under mean aggregation, while still being more robust than hard-max.

Alternatives Considered
- Full softmax over all views: diluted by poor views when many are available. Rejected as default; available as `"softmax"` option.
- Mean aggregation: simple but equally weights all views including bad ones. Available as `"mean"`.
- Learned attention weights (full OPEN reproduction): requires training infrastructure and paired data. Out of scope for inference-time improvement.
- Keeping hard-max only: rejected; too sensitive to single-view noise. Still available as `"max"`.

## 2026-04-03 deterministic depth conversion and configurable point cloud filtering

Decision
- Remove the `if depth.max() > 100` heuristic from both `run_pipeline.py` and `step2_pointcloud.py`.
- Convert depth to float32 meters once in `run_pipeline.py` before `pipeline.run()`, preferring BOP `depth_scale` from `scene_camera.json` when available.
- Add 2D median-relative depth gating (`depth_gate_tolerance=0.3`) before backprojection.
- Make SOR/ROR parameters configurable in `PipelineConfig`. Reduce `depth_trunc` default to 2.0m.

Rationale
- The heuristic `if depth.max() > 100` could trigger in both `run_pipeline.py` and `step2_pointcloud.py`, risking double division. It was also input-order-dependent (results differed for 16-bit vs already-converted depth).
- BOP `depth_scale` (e.g. 0.1 for YCBV) is authoritative but was ignored in favor of `config.depth_scale` (10000.0). They happen to agree for YCBV (`raw × 0.1 / 1000 = raw / 10000`) but this is fragile for other datasets.
- Depth outliers within the mask (sensor noise, transparent surfaces, mask bleed) passed through to the point cloud unchecked, degrading shape matching and scale estimation.
- `depth_trunc=10.0m` was too permissive for tabletop scenes — passed far-plane noise.

Alternatives Considered
- Keep the heuristic with a guard against double application — rejected; brittle, hard to reason about.
- Always use `config.depth_scale` and ignore BOP metadata — rejected; fails silently when config doesn't match dataset convention.
- Z-score depth gating instead of median-relative — rejected; median-relative is more robust to non-Gaussian depth distributions in partially occluded objects.

## 2026-03-26 partial-to-partial point cloud matching in Step 5

Decision
- Add a preprocessing script (`rendering/generate_partial_pointclouds.py`) that generates partial point clouds per view using front-face culling.
- Add `ulip2_use_partial_views` config flag and `--ulip-partial-views` CLI flag.
- When enabled, Step 5 loads per-view partial PCs and scores using best-of-8-views cosine similarity instead of a single full-mesh embedding.

Rationale
- The original Step 5 compared a partial observed PC (single depth view, ~4k points) against full CAD model PCs (uniformly sampled from the entire mesh surface). This domain mismatch is a known weakness: features from occluded sides of the CAD model dilute the embedding, reducing discriminative power for shape matching.
- Partial-to-partial comparison aligns the reference representation with the query: both are single-view observations. The best-of-8-views scoring selects the reference view most similar to the observed viewpoint.

Alternatives Considered
- Raycasting (trimesh `intersects_location`): technically more accurate (handles self-occlusion) but orders of magnitude slower without an embree backend (~60h estimated vs ~10min for front-face culling on 1051 objects × 8 views). Rejected for practical reasons.
- Depth buffer rendering via trimesh scene: more complex setup, requires OpenGL context, limited benefit over front-face culling for the target object types.
- Keeping full-mesh-only: rejected; this is a known domain mismatch that the thesis aims to address.

## 2026-03-26 debug visualization as optional mode of the main pipeline

Decision
- Remove `pipeline/debug_steps.py` entirely.
- Extract all visualization functions into `pipeline/debug_viz.py`.
- Add `--debug-viz` and `--until-step` flags to `pipeline/run_pipeline.py`.
- Shell scripts call `pipeline.run_pipeline --debug-viz` instead of `pipeline.debug_steps`.

Rationale
- `debug_steps.py` (~1473 lines) contained a complete duplicate of the 8-step pipeline logic (`run_debug()`) alongside the visualization functions. Any change to the pipeline had to be mirrored in two places.
- Making debug a flag on the main pipeline eliminates the duplication and ensures debug mode always runs the same code path as production.

Bug fixes included
- `_find_cad_mesh()` was nested inside `save_debug_step7_8()` and unreachable from `run_debug()` at runtime (NameError when `cad_model_path` was a PNG). Now at module level in `debug_viz.py`.
- `detection_prompt` (undefined variable) replaced with `prompt_elements.detection_phrase` in step 1 visualization call.
- Mesh-path resolution (image-path → real mesh lookup) now runs unconditionally in steps 7+8, not only in debug mode.

Alternatives Considered
- Merge debug visualizations into the existing `visualization.py` — rejected; `debug_viz.py` is much richer (multi-panel PIL composites, trimesh wireframe projection, Plotly HTML) and would bloat the simple viz module with debug-only dependencies.
- Keep `debug_steps.py` as a thin CLI wrapper — initially implemented but removed in favor of shell scripts, since the wrapper added no logic beyond what the shell script already provides.

## 2026-03-24 BOP depth_scale convention — always use config divisor

Decision
- Always use `config.depth_scale` (default 10000.0) as the divisor when converting raw depth pixels to metres.
- Do not use the `depth_scale` field from `scene_camera.json`.

Rationale
- BOP `scene_camera.json` defines `depth_scale` as a **multiplier** (e.g. 0.1 for this dataset: raw × 0.1 = depth in mm).
- The pipeline divides raw depth by `config.depth_scale` (a **divisor** convention).
- Using the JSON value (0.1) as a divisor gave depths 100× too large, producing a translation error of ~855mm in the predicted pose.
- The config value (10000.0) is correct: it converts 16-bit PNG depth (0.1mm units) to metres.

Alternatives Considered
- Detect and adapt to the BOP convention at runtime — rejected; fragile, adds edge-case logic, and the config value is already correct for the target dataset.

## 2026-03-24 GT wireframe bbox-center compensation

Decision
- When projecting the GT wireframe overlay, subtract `R_gt @ bbox_center` from the GT translation vector before rendering.

Rationale
- BOP ground truth poses are annotated with models centered at the mesh bounding-box origin.
- The pipeline's OBJ files have a non-zero `bbox_center` offset (e.g. mug: ~8.3mm in X → ~7.8px shift at scene depth).
- Without this correction the GT wireframe is visibly misaligned even when the pose is geometrically correct.

Alternatives Considered
- Re-centre the OBJ meshes at the origin — too invasive, affects all downstream steps.
- Apply no correction and accept the visual offset — rejected; defeats the purpose of the overlay.

## 2026-03-20 two-container HTTP architecture for FoundationPose

Decision
- Run FoundationPose as a separate Docker compose service with a Flask HTTP API.
- OSCAR calls `http://foundationpose:5050/estimate_pose` from Step 8.
- Replace the subprocess bridge and venv-inside-OSCAR approach.

Rationale
- The OSCAR container (CUDA 12.2 runtime, Python 3.11) cannot compile pytorch3d, kaolin, or nvdiffrast which require a CUDA devel image.
- A virtual environment inside the OSCAR container cannot bridge this CUDA/ABI gap.
- HTTP over the Docker compose network gives full dependency isolation with zero shared Python state.
- The pre-built `shingarey/foundationpose_custom_cuda121` image already has all compiled dependencies.

Alternatives Considered
- Venv inside OSCAR container (previous approach): failed due CUDA runtime vs devel mismatch and Python 3.11 vs 3.8 ABI conflicts.
- Two-container with shared-volume CLI handoff (`docker compose exec`): viable but requires Docker socket in OSCAR container or host-side orchestration.
- Install CUDA devel toolkit in OSCAR image: bloats image by 10+ GB, fragile compilation chain, ongoing maintenance.
- HTTP API (chosen): simplest inter-container call, no Docker socket needed, JSON in/out, healthcheck support.

## 2026-03-19 persist model and embedding caches via compose volumes

Decision
- Persist Ollama data and model caches in docker compose with named volumes.

Rationale
- Prevent repeated model downloads and cache warmups across `docker compose run --rm` sessions.
- Keep runtime reproducible while reducing setup latency.

Alternatives Considered
- Keep cache only in ephemeral container filesystem; rejected due repeated startup cost.

## 2026-03-19 run FoundationPose in a separate Python environment (superseded)

> Superseded by 2026-03-20 two-container HTTP architecture.

Decision
- Execute FoundationPose from Step 8 via subprocess bridge using a configurable interpreter.

Rationale
- Single-env installation caused repeated dependency conflicts.
- Subprocess bridge allows one end-to-end pipeline call while preserving stability of both stacks.

Why superseded
- The venv approach could not work because the OSCAR container lacks CUDA devel headers needed to compile pytorch3d/kaolin/nvdiffrast. The two-container approach eliminates this class of problem entirely.

## 2026-03-18 staged FoundationPose switch

Decision
- Use a staged migration path for FoundationPose:
- first install FoundationPose and expose it via Docker volume,
- then keep Step 8 on ICP fallback until API integration is implemented and validated.

Rationale
- Reduces risk of breaking the current end-to-end pipeline while environment dependencies are prepared.
- Allows iterative verification (setup, weights, extension build, API wiring, evaluation).

Alternatives Considered
- Immediate hard switch from ICP to FoundationPose in Step 8; rejected due incomplete integration and higher regression risk.

## 2026-03-17 enable ULIP mode switch in debug and pipeline

Decision
- Expose ULIP retrieval mode as a runtime option (`pc`, `cross`, `both`) instead of hardcoding point-cloud-only behavior.

Rationale
- Needed for direct thesis ablation: shape-only vs full ULIP cross-modal retrieval on identical scenes.

Alternatives Considered
- Keep single `pc` mode only — rejected, prevents controlled comparison.

## 2026-03-17 recursive CAD mesh discovery for ycbv_gso

Decision
- Use recursive mesh lookup in CAD object folders and prefer known mesh filenames in `meshes/`.

Rationale
- ycbv_gso object layouts are nested; non-recursive lookup found only 21 models.
- Recursive lookup resolves 1051 models and stabilizes Step 5 coverage.

Alternatives Considered
- Enforce one flat file layout per object — rejected, too invasive for downloaded assets.

## 2026-03-17 cache ULIP CAD embeddings on disk

Decision
- Save/reload CAD embeddings in `.ulip_cache_<hash>.pt` keyed by model+config+mesh inventory.

Rationale
- Step 5 over 1000+ CAD models is the dominant runtime; repeated runs should not recompute unchanged embeddings.

Alternatives Considered
- In-memory cache only — rejected, not persistent across process/container restarts.

## 2026-03-17 separate image view paths from CAD mesh paths in fusion

Decision
- Keep DINO `best_view_path` separate from `cad_model_path` in fusion output.

Rationale
- Passing image paths as mesh paths caused Step 8 ICP to read `.png` as CAD mesh and fail.

Alternatives Considered
- Force Step 8 to ignore fusion path and always search filesystem — kept as fallback only.

## 2026-03-12 default pose_method to icp

Decision
- Changed `pose_method` default from `"foundationpose"` to `"icp"` in config.py.

Rationale
- FoundationPose is marked `NotImplementedError`. It always fell back to ICP anyway, but the fallback path did not forward `initial_pose` from Step 7's coarse alignment. Using ICP directly ensures the coarse alignment is used as the initial transform.

Alternatives Considered
- Implement FoundationPose wrapper — deferred, not critical for thesis prototype.
- Keep foundationpose default and fix the fallback — done as well, but direct ICP is cleaner.

## 2026-03-12 reduce voxel_size from 5mm to 2mm

Decision
- Changed `voxel_size` from `0.005` to `0.002` in config.py.

Rationale
- At 5mm, the observed point cloud had only ~810 points — too sparse for reliable ULIP-2 shape matching (expects 10,000 points). At 2mm, ~4,200 points are retained from a single depth view, providing much better surface coverage.

Alternatives Considered
- 0.001m (1mm, ~10k+ points): too dense, slower without significant quality gain.
- 0.003m (3mm, ~2-3k points): considered as middle ground, 2mm chosen for better ULIP coverage.

## 2026-03-12 DINOv2 batch encoding with disk cache

Decision
- Rewrote step4_dino_reranking.py with batch encoding (32 images/forward pass) and `.pt` disk cache.

Rationale
- Serial encoding of 9,459 reference images took ~45 minutes (1 forward pass per image). Batch encoding reduces this to ~5 minutes. Disk cache makes subsequent runs instant.
- Cache keyed by model name + fingerprint (hash of file count + newest modification time) to auto-invalidate when reference images change.

Alternatives Considered
- Pre-compute embeddings offline and store as a separate file — less flexible, manual step.
- Use FAISS index — overkill for ~10k vectors, simple cosine similarity is fast enough.

## 2026-03-12 NaN handling in ULIP and fusion

Decision
- Added explicit NaN detection and replacement throughout Step 5 and Step 6.

Rationale
- Open3D `pcd.colors` can produce values outside [0,1] (e.g. from depth-to-color mapping), causing float32 overflow -> inf -> NaN embeddings -> NaN cosine similarity. NaN silently propagated through topk() and corrupted fusion normalization.
- Fix: clip colors to [0,1], replace NaN similarities with -1.0, skip NaN in min-max normalization.

Alternatives Considered
- Discard objects with NaN entirely — too aggressive, could lose valid partial matches.
- Use nanmean/nanmin — less explicit, harder to debug.

## 2026-03-12 switch LLM to gemma3:4b

Decision
- Changed `ollama_model` from `"mistral-small3.1"` to `"gemma3:4b"`.

Rationale
- gemma3:4b fits in 6GB VRAM alongside the other models (GroundingDINO, SAM, CLIP, DINOv2, ULIP-2). Responds within 5-10 seconds for prompt parsing.
- mistral-small3.1 required more VRAM and was slower on the RTX 4050 Laptop GPU.

Alternatives Considered
- CPU-only inference for LLM — too slow (30+ seconds).
- Skip LLM entirely, use only heuristic parser — less robust for complex prompts.

## 2026-03-12 wireframe overlay via trimesh

Decision
- Installed trimesh in Docker for 3D wireframe overlay in debug visualization.

Rationale
- The debug image Step 7+8 previously showed a 2D thumbnail pasted onto the scene, which didn't convey pose orientation. Projecting CAD mesh edges using the estimated pose + camera intrinsics gives visual verification of alignment quality.

Alternatives Considered
- Use Open3D offscreen rendering — harder to integrate, requires display server.
- Use matplotlib 3D projection — less precise, no mesh topology awareness.

## 2026-03-04 use GT masks for retrieval eval

Decision
- Run retrieval_combi_eval.py with ground-truth segmentation masks from BOP data rather than GroundedSAM predictions.

Rationale
- Isolates retrieval accuracy from segmentation errors. Gives upper-bound performance.
- Paper's full pipeline uses GroundedSAM which adds segmentation noise. Our 75.95% vs paper's ~60% is consistent with this difference.

Alternatives Considered
- Run GroundedSAM for fair 1:1 comparison — possible future work but not primary focus.

## 2026-03-04 focus on full OSCAR pipeline only

Decision
- Skip running individual baselines (i2i_bbox_clip, i2i_seg_clip, etc.) that require ycbv_test_bop19 data. Focus on retrieval_combi_eval.py as the main evaluation script.

Rationale
- Most baseline scripts need ycbv_test_bop19 (21-object YCBV BOP test set) which is not downloaded.
- The full OSCAR pipeline is what the thesis aims to improve, not the individual baselines.

Alternatives Considered
- Download ycbv_test_bop19 and run all baselines — deferred, not critical for thesis progress.

## 2026-02-23 download only missing GSO models

Decision
- Create download_missing_gso.py that checks existing folders and downloads only absent models from Gazebo Fuel API.

Rationale
- Full re-download of 1030 models wastes bandwidth. Script checks folder existence and downloads only the ~722 missing ones.
- Fixed ZIP extraction: Fuel ZIPs have no top-level directory so must extract into named subfolder.

Alternatives Considered
- Re-download everything — rejected, too slow.
- Manual download — rejected, 722 models.

## 2026-02-19 rendering multi-dataset config

Decision
- Add a dataset config block at the top of rendering.py with use_folder_name flag rather than separate scripts per dataset.

Rationale
- YCBV uses textured.obj and GSO uses model.obj — both produce duplicate model_name when derived from filename. Using parent folder name (use_folder_name=True) avoids collisions.
- Single script with config section is easier to maintain than duplicating.

Alternatives Considered
- Separate rendering scripts per dataset — rejected to avoid duplication.
- Renaming model files inside each folder — too invasive on downloaded data.

## 2026-02-08 exclude datasets from git tracking

Decision
- Add gitignore rules to stop tracking heavy local data and generated assets.

Rationale
- Large local dataset commits caused very slow push and upload size issues.
- Reproduction data should remain local runtime state, not repository history.

Alternatives Considered
- Keep tracking data in git, rejected due size and performance.

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
