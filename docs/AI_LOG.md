# AI Log

## 2026-07-30 HPR occlusion param + upsample jitter; shrec18_v2; MI3DOR full-mesh ablation

Goal
- Fix the partial point-cloud occlusion leak, prepare a corrected SHREC'18 onboard, and add a partial-vs-full-mesh cross ablation for MI3DOR — without disturbing the already-onboarded MI3DOR data.

Changes
- **HPR leak found & fixed (configurable).** Ground-truth occlusion tests (angular z-buffer vs the mesh) on SHREC'18 samples showed the fixed Katz HPR `param=3.2` leaks ~2–11% occluded points (worst on open/concave shapes). Added `--hpr-param` (default 3.2) and `--jitter-std` (default 0.0) to `generate_partial_pointclouds.py`, threaded through `sample_visible_surface`/`process_object`. Jitter perturbs duplicated points on upsampling (sparse views → 10k) to avoid coincident-duplicate collapse in PointBERT FPS+kNN — parity with step5's query-side jitter, which the gallery never got.
- **SHREC'18 → 2.8 + jitter 0.001; everything else unchanged.** `rendering/onboard_dataset.sh` defaults `HPR_PARAM=3.2/JITTER_STD=0`; new `shrec18_v2` case sets `2.8/0.001`. Verified on samples: heavily-upsampled view went 703→10000 unique after jitter; dense views unchanged.
- **`shrec18_v2` full-onboard pipeline.** `oscar_queue_ctl/run_shrec18_v2.sh` (render+partials@2.8/jitter+descriptions → full embed set incl. `ulip_fullmesh` → sync → verify; keeps renders local for eval). Armed to auto-start after MI3DOR via `arm_shrec18_v2.sh` (`shrec18v2-arm` user unit). Existing `shrec18_fixed` renders predate the 2026-07-28 render fix, so a fresh slot is justified.
- **MI3DOR `ulip_fullmesh` ablation** added to its embed passes (partial-view vs full-mesh gallery, same ULIP-2 cross space; 3848/3848 mesh↔id match). Fixed `preprocess_galleries.sh`: `--mesh-glob ''` argparse rejection (per-dataset real/placeholder glob) and missing `**/.ulip_cache_*.pt` in the object_database cache sync.
- MI3DOR partials/embeddings left at param 3.2 (already onboarded; user decision).

## 2026-07-24 Merge tessa-pc: Uni3D-g (E7) + XYZ ULIP-2 (O5), cross-PC FPS portability

Goal
- Integrate the gallery PC's shape-encoder work and verify the eval PC can embed Uni3D/XYZ-ULIP queries into the same space as the shipped gallery caches.

Changes
- Merged `origin/tessa-pc` (ea57dffb) into `feat/stage1-official-eval-precompute` (merge `21df33db`): took tessa's `experiment1` (superset — already contained this branch's official-eval + precompute), unioned `config`/`step5` (content-stable fingerprints + real `Uni3DEncoder` coexist), kept this branch's docs + `/eval/shrec18_official/` gitignore line (tessa never touched them). Brought in `docs/uni3d_inference.patch` + `docs/LAPTOP_EMBEDDINGS_SETUP.md`.
- Verified: eval image `tholoi/oscar-plus` lacks `pointnet2_ops`/`knn_cuda`/`pytorch3d`/`einops` (has `timm 1.0.25`, `open_clip`). Upstream Uni3D `point_encoder.py` FPS hard-depends on `pointnet2_ops`; the patch's try/except → deterministic pure-torch FPS (seeded idx 0) means both PCs take the identical branch. See DECISIONS 2026-07-24.
- Unrelated in-flight WIP (onboard/bop/ycbv scripts) preserved in `stash@{0}`; onboard scripts left at merged HEAD (conflicted with tessa's onboard edits — user to reconcile).

Still needed on eval PC to run E7/O5
- Clone `baaivision/Uni3D`@`64e03c3` + `git apply docs/uni3d_inference.patch`, mount `-v ~/thesis/Uni3D:/uni3d`.
- Mirror checkpoints: `uni3d-g/model.pt` (2.03 GB, HF BAAI/Uni3D) and `ulip2_pointbert_8k_xyz.pt` (HF SFXX/ulip).
- Sync the `shrec18_fixed` gallery + precomputed `.pt` caches from Drive (gallery name must match — this PC currently has `shrec18`).

Goal
- Make Stage-1 numbers leaderboard-comparable and offload the expensive reference encoding to the gallery-generating PC.

Changes
- **Gallery downloaded in full** (3,308 models, 47 GB) from `gdrive:Masterthesis/OSCAR/object_images/shrec18`. Fixed throughput: the shared rclone client_id + default 10 req/s pacer capped it at ~0.2 MB/s; a private OAuth client_id + `--drive-pacer-min-sleep 10ms` took it to ~10 MB/s. Excluded `*_CamMatrix.npy` / `*_bg.png` (pose-only).
- **Official evaluation** (`experiments/experiment1_shrec18_stage1.py`): new `load_official_gt` (parses `eval/shrec18_official/rgbd.csv`+`cad.csv`) and `score_official` (replicates `evaluate.py`'s loop, reusing the unchanged official `metrics.py` — graded relevance, top-f). `run_ablation`/`aggregate`/`main` now report nDCG/precision/recall/F1/AP/NNT1/NNT2 and select by nDCG. Verified: official `metrics.py` runs under py3.11 (scored the dataset's `results/` lists → P=1.0, nDCG=1.0), and a synthetic `run_ablation` integration test passed.
- **Two-PC precompute**: `--precompute` mode + `run_pass(build_only=True)` build every gallery reference cache with no query scoring; `precompute_gallery` writes a provenance manifest; `verify_precompute_provenance` warns on commit mismatch at eval start.
- **Content-stable cache fingerprints** (`step4._dir_fingerprint`, `step5._get_cache_path`, `_get_partial_cache_path`): size+relpath instead of mtime, so caches survive cross-machine transfer.
- **Mesa/EGL**: committed `oscar-plus-egl` (base image + `libegl1 libgl1-mesa-dri ...`) so Open3D renders query meshes headlessly; `_offscreen_available()` gates GL-vs-splat; adaptive point-splat fallback improved.
- `SHAPE_AGG_VIEWS=16` (encode 42, aggregate 16); `prepare_queries` hardened to regenerate when a cached index doesn't cover the full query set.

Status
- Base reference pass encoding on the RTX 4050 (validated `run_pass` on real encoders). Next: gallery PC runs `--precompute`; eval PC pulls caches + runs the grid. Blocked only on the two-PC cache handoff.

## 2026-07-20 Experiment 1 script (Stage 1 SHREC'18 ablation grid) + two latent bug fixes

Goal
- Implement thesis Experiment 1: the Stage-1 retrieval-tuning ablation grid (E1, E2, E2b, E4, E6, E7, O1, O2, O4, O5) on SHREC'18 ObjectNN+, selecting the best OSCAR+ config by DCG (tie-break mAP).

Changes
- `experiments/experiment1_shrec18_stage1.py` (new, the only entry point): input validation → GT reconstruction (union-find over `results/`, 20 categories / 1,452 train queries — verified) → query preprocessing (PLY → RGB crop via OffscreenRenderer with numpy point-splat fallback + raw `.npz` point cloud) → cached channel-score passes → per-ablation derivations via the production `ScoreFusion` → PSB/SHREC metrics (NN, FT, ST, E@32, full-list DCG, R@1/5, mAP) → `stage1_summary.csv/.tex` + `best_config.json`. Resumable (`--resume`), geometry gated (`--with-geometry`), smoke-testable (`--limit-queries`, `--allow-partial-gallery`, `--viz-check`).
- `pipeline/step4_dino_reranking.py`: **bug fix** — view files were sorted lexicographically (`_0, _1, _10, _11, … _19, _2, …`), so `views[:N]` was not the FPS prefix promised by `config.num_views`; ablation O4 would have been silently wrong. Added numeric `_view_sort_key` in `load_reference_images` and a defensive re-sort in `_apply_view_limit` (covers stale caches).
- `pipeline/step5_shape_matching.py`: **bug fix** — `_get_partial_cache_path` did not include the encoder type, so a Uni3D run (E7) would have collided with the ULIP-2 partial cache. Encoder tag added for non-default encoders only (existing ULIP-2 caches keep their fingerprint).
- `object_retrieval/eval_common.py`: added `EvalConfig.pipeline_overrides` — arbitrary `PipelineConfig` field overrides applied in `build_pipeline` before components are constructed (unlocks `appearance_encoder`, `shape_encoder`, `ulip2_use_colors`, `num_views`, … for experiments without widening `EvalConfig`).

Verified
- GT union-find on the real dataset: exactly 20 components, 1,452 queries, 3,305/3,308 CADs (3 distractors), cache reload OK.
- Derivation tier self-test in the `tholoi/oscar-plus` container: single-channel ranking, min-max weighted fusion (hand-computed), Borda majority voting, O4 view-budget switch, CLIP-pruned scope + tail ordering, hand-computed PSB metrics, -inf sanitation — all passed.
- Not yet runnable end-to-end: `object_images/shrec18` + `object_database/shrec18/descriptions_attributes.json` are not on local disk (renders live on Google Drive, sync/onboarding incomplete); validation reports this and exits with instructions.
## 2026-07-24 Uni3D-g integration, ULIP-2 XYZ-only arm, CLIP-text cache, standalone precompute tool

Goal
- Add the remaining two Stage-1 ablation encoders (ULIP-2 XYZ-only O5, Uni3D-g E7) so all 6 embedding passes can run unattended on the gallery PC.
- Give the CLIP-text channel an on-disk cache like DINO/ULIP already have.
- Extract the embedding-precompute driver out of the SHREC'18-specific ablation script into a clean, dataset-agnostic tool someone unfamiliar with the codebase can run.

Changes
- `docker-compose.yml`: mounted `../Uni3D:/uni3d`; repointed `../ULIP` → `../ULIP_thesis` (the real clone with checkpoints — the old mount was an empty root-owned dir).
- `pipeline/config.py`: added Uni3D-g fields (`uni3d_model_name="uni3d-g"`, `pc_model=eva_giant_patch14_560`, `embed_dim=1024`, etc.).
- `pipeline/step5_shape_matching.py`: real `Uni3DEncoder` (`_load`/`encode`) — builds the model from the mounted repo + checkpoint with import isolation from ULIP's own `models` package (both expose a top-level `models` module; naive import would collide), encodes xyz+rgb (6-ch) via `normalize_pointcloud`. Cache key (`_get_partial_cache_path`) already included an `encoder=` tag for non-ULIP2 encoders, so Uni3D/XYZ-ULIP/colored-ULIP caches get distinct digests with no collisions.
- `experiments/experiment1_shrec18_stage1.py`: added `ULIP_CKPT_XYZ` and wired the `ulip_pc_xyz` pass to the released ULIP-2 8k-xyz PointBERT checkpoint (input_dim=3, 512-d SLIP ViT-B tower — distinct from the colored 10k/1280-d checkpoint).
- `pipeline/step3_clip_retrieval.py`: `CLIPRetriever` now caches description text embeddings to disk (`.clip_text_cache_<model>_<hash>.pt`, next to the description file). Fingerprint = CLIP model name + description texts (content, not path/mtime) — labels are intentionally excluded so an `id_to_label` remap doesn't invalidate the cache.
- `tools/precompute_embeddings.py` (new): standalone, dataset-agnostic version of the `--precompute` path from `experiment1_shrec18_stage1.py`. Same `PASS_DEFS`/`run_pass` logic, but as ~370 readable lines with no SHREC'18 ablation/evaluation code, real `--dataset`/`--data-root`/`--images-dir`/`--desc-file`/`--results-root` CLI args, `--list`/`--dry-run`/`--passes` subset selection, and a top-level tqdm progress bar. `validate_inputs()` was relaxed to only require `<data_root>/cad/` (the original required SHREC'18's raw `rgbd/`/`results/` query-GT folders too, which don't exist for other datasets and aren't needed to build gallery embeddings).
- `docs/LAPTOP_EMBEDDINGS_SETUP.md` (new) + `docs/uni3d_inference.patch` (new): what an eval/query-side machine needs to reproduce these embeddings — exact checkpoint filenames, the two Uni3D inference patches (optional `pointnet2_ops`/pure-torch FPS fallback, optional `losses` import) as a `git apply`-able patch against upstream `64e03c3`, `timm==1.0.25` pin, and the FPS-portability warning (CUDA vs. pure-torch FPS must match on both machines or E7 scores silently mismatch).
- `README.md`: replaced the outdated manual-Blender/`description_genertor` preprocessing section with the current `onboard_and_sync.sh`/`onboard_dataset.sh` workflow, and added a new "Precomputing Gallery Embeddings" section documenting `tools/precompute_embeddings.py` and the 6-pass table.

Bugs fixed
- ULIP-2 XYZ-only pass crashed (`RuntimeError`, `Conv1d(6,...)` fed 3-channel input) — the colored checkpoint has `input_dim=6`; fixed by switching to the XYZ-only checkpoint (`input_dim=3`) and its native 512-d embed dim (was defaulting to the colored arm's 1280-d, causing a `pc_projection` size mismatch).
- Uni3D import crashed on `pointnet2_ops` (hard CUDA-ext dependency, no fallback upstream) and `h5py` (pulled in via `models.uni3d` → `losses` → the training data stack) — patched both to be optional for inference-only use.

## 2026-07-24 (cont.) Query PC cache, generalized precompute tool, autonomous multi-dataset orchestrator

Goal
- Cache the expensive query-side point-cloud embeddings (previously re-encoded every run).
- Make `tools/precompute_embeddings.py` work for any dataset's CAD layout, not just shrec18's `cad/*.obj`.
- Set up autonomous gallery preprocessing for MI3DOR + ycbv + gso + housecat6d + tless + itodd (renders + partial PCs + descriptions + gallery embeddings, NO queries/ablations), triggered after shrec18_fixed fully completes.

Changes
- `experiments/experiment1_shrec18_stage1.py`: added `_pc_query_cache_path()` + `_load_or_build_pc_query_cache()` — the pc-mode passes (ulip_pc_rgb/xyz, uni3d) now cache query point-cloud embeddings under `eval/datasets/shrec18/stage1/query_pc_cache/` (content-fingerprinted by encoder config; was re-encoded ~1-2s/query every run, the single biggest ablation cost). The `cross`-mode image-query cache already existed; this closes the gap for the far more expensive pc branch.
- `tools/precompute_embeddings.py`: added `--mesh-glob` (per-dataset CAD glob; only the ulip_fullmesh pass reads meshes). `validate_inputs()` now derives the gallery from `rendered ∩ described` (meshes optional — a missing/partial mesh set only warns and only affects ulip_fullmesh). `--data-root` is now optional when `--mesh-glob` is given.
- `oscar_queue_ctl/preprocess_galleries.sh` (new, host-side): waits for shrec18_fixed's two completion flags (`embed_shrec18_fixed.ok` gallery + `query_caches_shrec18_fixed.ok`), verifies shrec18_fixed renders are on Drive and deletes them locally to free space, then for each of the six datasets runs onboard (render/partial/describe via `onboard_and_sync.sh`) → embed (5 mesh-free passes: base/siglip/ulip_pc_rgb/ulip_pc_xyz/uni3d) → sync caches → reconcile-verify → delete local → notify. HALT-ON-ERROR between every step; disk pre-flight (≥60 GB) before each dataset. Runs as systemd --user unit `oscar-preprocess-galleries` (linger on → survives logout).
- `oscar_queue_ctl/watch_ablation_run.sh` (new): waits on the query-cache-building run via Docker **container** status (NOT `kill -0` on the inner PID — that PID runs as root and `thomas` gets EPERM, which is indistinguishable from "gone" and fired the sync prematurely — a bug caught and fixed this session), then rclone-syncs `eval/datasets/shrec18/stage1` + `ulip_query_img_cache.pt` to Drive and touches the `query_caches_shrec18_fixed.ok` gate flag.

Bugs fixed
- `object_retrieval/eval_common.py`: `build_pipeline` unconditionally called `shape_m._load_model()` (builds a full ULIP-2 PointBERT, prints "training from scratch for pointbert.") even for `shape_encoder="uni3d"` passes, wasting a model load. Now skipped for uni3d (loaded lazily on first encode instead).
- Watcher premature-fire bug (root-PID `kill -0` EPERM) — see above; rewrote to poll `docker inspect .State.Running`.

## 2026-07-17 Onboarding pipeline, multi-dataset model ID fix, cache optimization

Goal
- Create an automated preprocessing pipeline for all thesis datasets (render, partial PCs, descriptions) that works across Docker (GPU) and WSL (rclone sync to Google Drive).
- Fix `infer_model_id()` which collapsed MI3DOR (3848→21), SHREC'18 (3308→1), and HouseCat6D to a handful of IDs.
- Make DINO/SigLIP and ULIP partial caches reusable across `num_views` ablation (O4).

Changes
- `rendering/rendering.py`: rewrote `infer_model_id()` — generic filenames (`model.ply`, `textured_simple.obj`, etc.) use parent dir; specific filenames use stem. Added `_GENERIC_MODEL_NAMES` set. Added PLY vertex color material (Vertex Color → Principled BSDF node chain) after `bpy.ops.import_mesh.ply()`.
- `rendering/onboard_dataset.sh`: removed all rclone logic (script runs inside Docker where rclone is unavailable). Added `MESH_GLOB` for SHREC'18. Cleaned up leftover `$RCLONE_REMOTE` references that caused `unbound variable` errors.
- `rendering/onboard_and_sync.sh` (new): WSL-side launcher — starts Docker container running `onboard_dataset.sh`, starts `rclone_watch.sh` in background, runs final sync, supports `--delete-after-sync`, `--skip-describe`, `--step`.
- `rendering/rclone_watch.sh` (new): background sync watcher for WSL — polls `object_images/` and `object_database/` directories, syncs to Google Drive every `--interval` seconds, auto-exits after 2 idle rounds.
- `pipeline/step4_dino_reranking.py`: cache path no longer includes `num_views` (uses `_vall_` suffix). Added `_apply_view_limit()` method — trims `_ref_embeddings` to first N views after cache load. Cache always encodes all available views. Encoding loop no longer filters by `max_views`.
- `pipeline/step5_shape_matching.py`: `_collect_partial_items()` no longer filters by `num_views`. Added `_apply_partial_view_limit()` — trims stacked per-object tensors after cache load/build. Applied on both cache-hit and cache-miss paths.

Bugs fixed
- BOP PLY models (LM-O, T-LESS, ITODD) rendered as grey blobs — Blender imported vertex colors but had no material to use them.
- `onboard_dataset.sh` crashed with `$RCLONE_REMOTE: unbound variable` inside Docker due to `set -u` and leftover rclone references.
- `onboard_and_sync.sh` used `rclone sync` which would delete previously-synced files from remote after local deletion. Changed to `rclone copy`.
- Old cache system created separate cache files for each `num_views` value, causing redundant multi-hour cache rebuilds during ablation O4.

Results
- LM-O end-to-end test: 8 objects × 42 views rendered (with vertex colors), 336 partial PCs generated (11s), descriptions generated, all synced to Google Drive.
- All 7 datasets verified: correct unique model ID counts match expected object counts.

## 2026-04-23 OSCAR+ evaluation suite: shared eval_common, per-dataset wrappers, MI3DOR partial PCs, single-pass DINO/ULIP

Goal
- Consolidate duplicated eval logic in `retrieval_mi3dor_eval_oscarplus.py` into a reusable module, add per-dataset wrappers for YCBV-GSO and HouseCat6D, make result variant names unambiguous, wire partial-view pointclouds through MI3DOR's non-standard CAD layout, and cut per-query runtime by running DINO and ULIP exactly once instead of twice.

Changes
- `object_retrieval/eval_common.py` (new, ~680 lines): `EvalConfig`, metric helpers, constant-memory accumulators (`make_accum` / `update_accum` / `finalize_accum`), `build_pipeline(cfg, cad_mesh_items=None)` with optional partial-view branch, `run_query` (single-pass full DINO + full ULIP + id-filter for CLIP-pruned variants), ULIP cache helpers, image crop helpers, `_make_per_query_record`, `_filter_dino_result_by_ids`, `_filter_shape_result_by_ids`, `run_evaluation` main loop.
- `object_retrieval/retrieval_mi3dor_eval_oscarplus.py` (rewrite, ~170 lines): thin MI3DOR wrapper — CONFIG block, `to_category_label`, description-coverage filter, `_collect_filtered_cad_mesh_items()` (restricts CAD meshes to categories with CLIP descriptions), category iteration factory.
- `object_retrieval/retrieval_ycbv_eval_oscarplus.py` (new, ~140 lines): BOP scene iteration with `bbox_visib` crop, grandparent-dir obj_id extraction for `<name>/meshes/model.obj`, identity `to_label_fn`.
- `object_retrieval/retrieval_housecat6d_eval_oscarplus.py` (new, ~148 lines): BOP scene iteration with `mask_visib` crop, excludes `bg/` + `collision/` CAD subdirs, identity `to_label_fn`.
- `object_retrieval/precompute_ulip_query_embeddings.py` (new): standalone ViT-bigG-14 batch encoder (float16, ~5 GB, fits 6 GB GPU), writes `ulip_query_cache_*.pt`. The eval scripts detect the cache and skip per-query image-encoder calls.
- `rendering/generate_partial_pointclouds.py` (+60/-6): new `--mesh-glob` CLI + `_build_mesh_map_from_glob()` helper for MI3DOR-style CAD layouts. `process_object()` accepts an explicit `mesh_path=` kwarg.
- `object_retrieval/retrieval_mi3dor_eval.py` (baseline): one-line fix — `bop_root` path updated to point at the image-test subtree.
- `.gitignore`: added `/object_retrieval/results_*/`, `/object_retrieval/top*_rankings_*.json`, `/debug_output/`, `/object_retrieval/ulip_query_cache_*.pt`.

Design evolution within the session
- Started with separate CLIP-gated and full DINO/ULIP passes per query (double-run). Refactored to a single full pass + id-intersection filter for the CLIP-pruned variants after confirming:
  - DINO: `_aggregate_view_scores` is per-object (topk_softmax over views); `sims = query_emb @ cand_tensor.T` has no cross-candidate normalisation.
  - ULIP: `match()` computes per-object cosine similarity; candidate gating only truncates the final top-k.
  - Therefore derived pruned rankings are mathematically equivalent to explicit CLIP-gated runs. `_filter_dino_result_by_ids` backfills `clip_score` from the CLIP score map so even that field matches byte-for-byte.
- `cfg.dino_top_k` / `cfg.ulip2_top_k` were too small (5/5) for id-filtering on a large reference set. Added auto-expansion in `run_evaluation`: `dino_full_top_k = max(cfg.dino_top_k, len(dino_rer._ref_embeddings))` and the analogous for ULIP. One-shot log line announces the depths used; depths are recorded in the summary JSON under `config.dino_full_top_k_used` / `ulip_full_top_k_used`.

Variant set
- Final summary `variants` block contains exactly these six keys (no config-dependent names remain): `clip_only`, `dino_only_full`, `ulip_only_full`, `dino_only_clip_pruned`, `ulip_only_clip_pruned`, `clip_pruned_dino_ulip`. Primary = `clip_pruned_dino_ulip`.
- Per-query records (`results_topk_K.json`): `category, filename, gt, pred, clip_candidates, dino_candidates_full, dino_candidates_clip_pruned, ulip_candidates_full, ulip_candidates_clip_pruned, matched_files, clip_pruned_dino_ulip_pred, clip_pruned_dino_ulip_top5`.

Results
- One run of `retrieval_mi3dor_eval_oscarplus.py` / `retrieval_ycbv_eval_oscarplus.py` / `retrieval_housecat6d_eval_oscarplus.py` now produces all six comparison perspectives. No scripted config toggles required.
- Runtime per query (vs. the double-run intermediate): saves one DINO rerank + one ULIP matmul per query. GPU peak memory unchanged.
- Partial-view ULIP for MI3DOR is a single config toggle (`ulip2_use_partial_views=True` in the `EvalConfig`); `.npz` files produced by the updated generator script are discovered automatically under `object_images/MI3DOR/<obj_id>/`.

## 2026-04-13 Scale gate reliability fixes, Step 7 ICP fallback, debug CSV + ULIP top-5

Goal
- Make the scale gate deterministic and reliable for partial/cut-off objects.
- Prevent Step 7's RANSAC+ICP from producing a degenerate scale factor (observed: 2.25× for cut-off scissors, confidence 0.00) that corrupts FoundationPose input.
- Add full ranking CSVs and upgrade ULIP debug viz from top-3 to top-5.

Changes
- `pipeline/step7_scale_estimation.py`:
  - `estimate_fast()` added (previous session): rotation-invariant sorted-bbox scale estimate; no ICP or point sampling; returns `(scale_factor, confidence)`. Used by scale gate.
  - `estimate()` now checks computed ICP confidence against `config.scale_icp_min_confidence` (default 0.15). When confidence is too low (degenerate alignment), scale factor is overridden with `estimate_fast()` result. ICP transformation T is still returned for coarse alignment in Step 8.
- `pipeline/config.py`:
  - Added `scale_icp_min_confidence: float = 0.15` under Schritt 7 section.
- `pipeline/run_pipeline.py`:
  - `_select_candidate_with_scale_gate()` rewritten to use `estimate_fast()` instead of the full `estimate()`. Now returns 4-tuple `(candidate, mesh_path, selected_rank, rejection_log)` — no `scale_result` returned, so Step 7 always runs its full RANSAC+ICP for coarse alignment.
  - Scale gate block updated: unpacks 4-tuple, sets `scale_gate_failed` flag, enriches `results["scale_gate"]` with `policy`, `selected_rank`, `fallback_used`, `candidates_checked`.
  - Steps 7 and 8 both guarded with `and not scale_gate_failed`. Warning logged when skipping.
  - `scale_gate_failed = False` initialized alongside other shared vars.
  - Added `import csv`.
  - New method `_write_ranking_csvs(results)`: writes `rankings_clip.csv`, `rankings_dino.csv`, `rankings_ulip.csv`, `rankings_fusion.csv`, and (when rejections exist) `rankings_scale_gate.csv` to `output_dir`. Called at the end of `_save_results()`.
- `pipeline/debug_viz.py`:
  - `save_debug_step5()`: top_n increased from 3 to 5, figure height 6→9, row spacing adjusted (0.30→0.175), title updated to "Top-5", score label now includes ICP `registration_fitness` when > 0.

Diagnosed during session
- FoundationPose CUDA error (`unknown error`) on first call after container idle: stale GPU context. Fix: `docker compose restart foundationpose`. Not a code issue.
- Scale 2.25 confidence 0.00 for scissors: RANSAC+ICP gave degenerate alignment on a heavily cut-off partial view. Ratios were spread across [~3.0, ~1.5, ~0.5]; best-2-mean = 2.25. Now caught by `scale_icp_min_confidence` fallback.

## 2026-04-13 Branch `exp/ulip2v2`: Scale gate + rotation variance evaluation

Goal
- Address two weaknesses of ULIP-2 shape matching: scale invariance (ULIP intentionally normalizes scale away, so top-1 fusion may be wrong size) and rotation sensitivity (ULIP is not guaranteed rotation-invariant in pc mode).
- Branch `exp/ulip2v2` created from `exp/ulip2-full` commit `d629a47a`.

Changes
- `pipeline/config.py`:
  - Added scale gate fields: `scale_gate_enabled` (False), `scale_gate_min` (0.8), `scale_gate_max` (1.2), `scale_gate_min_confidence` (0.0), `scale_gate_max_candidates` (5), `scale_gate_reject_policy` ("fallback_best").
  - Added rotation eval fields: `ulip2_rotation_eval` (False), `ulip2_rotation_eval_top_k` (5), `ulip2_rotation_eval_method` ("icp"), `ulip2_rotation_eval_weight` (0.0).
- `pipeline/run_pipeline.py`:
  - New helper `_resolve_mesh_path_for_candidate()`: single source of truth for image-path detection and `_find_cad_mesh()` fallback. Replaces duplicated code in Steps 7 and 8.
  - New helper `_select_candidate_with_scale_gate()`: iterates fused candidates in rank order, runs `ScaleEstimator.estimate()` on each, returns first that passes the scale check. Returns 5-tuple `(candidate, scale_result, mesh_path, selected_rank, rejection_log)`.
  - New scale gate block between Step 6 and Step 7: calls `_select_candidate_with_scale_gate`, sets `effective_best_model`, stores `results["scale_gate"]` with `selected_object_id`, `selected_rank`, `fallback_used`, `policy`, `candidates_checked`, `rejections`.
  - `scale_gate_failed` flag: set when `policy=fail` and no candidate passes; prevents Steps 7 and 8 from running with a rejected candidate.
  - Steps 7 and 8 now use `effective_best_model or results["fusion"].best_match` so scale-gate-selected candidate propagates through.
  - `_create_summary()` includes `scale_gate_selected` and `scale_gate_rejections`.
  - CLI flags added: `--scale-gate`, `--scale-gate-min`, `--scale-gate-max`, `--scale-gate-min-confidence`, `--scale-gate-max-candidates`, `--scale-gate-reject-policy`, `--ulip-rotation-eval`, `--ulip-rotation-eval-top-k`, `--ulip-rotation-eval-weight`.
  - All new flags wired into `PipelineConfig(...)` construction.
- `pipeline/step5_shape_matching.py`:
  - `ShapeCandidate` extended with `best_partial_pc_path: str = ""`, `registration_fitness: float = 0.0`, `registration_rmse: float = 0.0`.
  - `ShapeMatcher.__init__()`: added `_partial_view_paths: Dict[str, List[Tuple[int, str]]] = {}`.
  - `_load_cad_models_partial()`: stores discovered partial view paths in `_partial_view_paths` for later use by rotation eval.
  - `match()`: populates `best_partial_pc_path` on each `ShapeCandidate` by looking up `best_view_idx` in `_partial_view_paths`.
  - `_run_rotation_eval()`: runs ICP for top-K candidates, logs fitness/RMSE per candidate, optionally adjusts `shape_score` if `ulip2_rotation_eval_weight > 0` and re-sorts.
  - `_register_partial_pointclouds_icp()` (module-level): loads reference `.npz`, normalizes both PCs to unit sphere, voxel-downsamples, estimates normals, runs Open3D point-to-plane ICP (50 iterations). Returns `(fitness, rmse, 4×4 transform)`.
- `scripts/run_debug_pipeline_foundationpose.sh`:
  - Fixed trailing whitespace after `\` on several lines (shell was treating `\ ` as a literal argument, causing `unrecognized arguments: ` error).
  - Updated to include `--scale-gate`, `--scale-gate-min 0.8`, `--scale-gate-max 1.2`, `--ulip-rotation-eval`, `--ulip-rotation-eval-top-k 5`, `--ulip-rotation-eval-weight 0.1`.

Known limitation discovered during implementation
- `fusion_top_k=1` (config default) truncates `FusionResult.candidates` to 1 entry before the scale gate sees it. The scale gate loop therefore iterates over 1 candidate and the fallback is also candidates[0]. Fix planned: override `top_k` at the fusion call site to `max(fusion_top_k, scale_gate_max_candidates)` when scale gate is enabled.

Results
- Scale gate and rotation eval are off by default; no behavioral change for existing runs.
- With `--scale-gate`, the pipeline tries up to 5 fusion candidates before falling back to top-1.
- With `--ulip-rotation-eval` and `--ulip-partial-views`, ICP fitness/RMSE is logged per top-K candidate for diagnostic purposes.

## 2026-04-09 SAM2 warning fix, GT bbox compensation toggle, README file reference

Goal
- Fix spurious SAM2 model_type warning in Step 1.
- Make GT bbox_center compensation optional (was always-on, caused visible shift for near-centered meshes).
- Add pipeline file reference table to README.

Changes
- `pipeline/step1_localization.py`: Load `Sam2Config` explicitly, override `model_type = "sam2"` before `Sam2Model.from_pretrained()`. Suppresses warning from HuggingFace metadata mismatch (`sam2_video` in config.json vs expected `sam2`). Updated docstrings from "SAM" to "SAM2.1".
- `pipeline/debug_viz.py`: Made bbox_center compensation conditional on `cam.get("gt_bbox_center_compensation", False)` instead of always-on.
- `pipeline/run_pipeline.py`: Wires `config.gt_bbox_center_compensation` into camera dict. Added `--gt-bbox-compensation` CLI flag.
- `pipeline/config.py`: Added `gt_bbox_center_compensation: bool = False` in Debug section. Updated SAM section header to "SAM2.1".
- `README.md`: Added "Pipeline File Reference" table listing all 15 `pipeline/*.py` files with descriptions.

Results
- SAM2 warning no longer appears in pipeline output.
- GT wireframe overlay defaults to direct pose (no bbox adjustment), which is correct for near-centered meshes like tuna_can. Users can opt in with `--gt-bbox-compensation` for meshes with significant origin offset.

## 2026-04-03 Multi-view aggregation for Steps 4 and 5

Goal
- Replace brittle hard-max view scoring in Steps 4 (DINOv2) and 5 (ULIP-2 partial views) with a configurable, query-conditioned multi-view aggregation strategy. Inspired by OPEN (Chu et al., TCSVT 2024) Equations 2-3 (softmax attention over multi-view similarities).

Changes
- `pipeline/step4_dino_reranking.py`:
  - Added `_aggregate_view_scores()` function supporting `max`, `mean`, `softmax`, `topk_softmax` modes.
  - Replaced hard-max per-object aggregation with grouped view scores → configurable aggregation.
  - Default: `topk_softmax` with k=4, τ=0.1.
  - Best view path still tracked for debugging/visualization.
- `pipeline/step5_shape_matching.py`:
  - Added same `_aggregate_view_scores()` function.
  - Replaced `view_sims.max(dim=0)` in partial mode with configurable aggregation.
  - Default: `topk_softmax` with k=4, τ=0.1.
- `pipeline/config.py`: Added `dino_view_aggregation`, `dino_view_topk`, `dino_view_temperature`, `ulip_view_aggregation`, `ulip_view_topk`, `ulip_view_temperature`.

Results
- Object-level scores now incorporate signal from multiple good views, reducing sensitivity to single-view noise or viewpoint mismatch.
- Setting aggregation to `"max"` preserves previous behavior for A/B comparison.

## 2026-04-03 Step 2 point cloud quality improvements

Goal
- Fix fragile depth conversion (double-scaling risk, BOP `depth_scale` ignored) and add configurable depth filtering for cleaner point clouds.

Changes
- `pipeline/run_pipeline.py`: Depth conversion now prefers BOP `depth_scale` from `scene_camera.json` (raw × depth_scale / 1000 = meters), falls back to `config.depth_scale` (raw / config.depth_scale = meters). Removed `if depth.max() > 100` heuristic — conversion is deterministic, runs once before `pipeline.run()`.
- `pipeline/step2_pointcloud.py`:
  - Removed internal `if depth.max() > 100` heuristic (caller guarantees meters).
  - Added `_gate_depth()`: median-relative 2D depth gating before backprojection. Configurable via `depth_gate_enabled` and `depth_gate_tolerance`.
  - SOR/ROR now config-driven: `sor_nb_neighbors`, `sor_std_ratio`, `ror_enabled`, `ror_nb_points`, `ror_radius`.
  - Added logging at each filtering stage (mask stats, gating, backprojection, SOR, ROR, final bbox).
- `pipeline/config.py`: Added `depth_gate_enabled`, `depth_gate_tolerance`, `sor_nb_neighbors`, `sor_std_ratio`, `ror_enabled`, `ror_nb_points`, `ror_radius`. Changed `depth_trunc` default from 10.0 to 2.0m.

Results
- Depth conversion is now deterministic and BOP-correct for YCBV (`depth_scale=0.1`).
- Depth gating removes sensor noise / mask bleed outliers before they pollute the point cloud.
- depth_trunc=2.0m eliminates far-plane points in tabletop scenes.

## 2026-04-02 Pipeline audit fixes and SAM2.1 migration

Goal
- Apply fixes identified by the pipeline audit (`docs/PIPELINE_AUDIT.md`) and migrate SAM to SAM2.1.

Changes
- `pipeline/step2_pointcloud.py`: tightened statistical outlier removal `std_ratio` from 2.0 to 1.0. The previous value was too lenient, keeping noisy depth points that degraded point cloud quality.
- `pipeline/run_pipeline.py`:
  - Localization now uses `visual_query` (LLM-extracted object name) instead of `detection_phrase` for GroundingDINO. This passes a cleaner, attribute-enriched query to detection.
  - Removed `text_query=visual_query` from CLIP `retrieve()` call. Text-image fusion in CLIP is intentionally disabled pending proper tuning (see PIPELINE_AUDIT finding #4).
  - Fixed mesh path resolution: added null guard (`if not resolved_mesh`) to prevent crash when no valid mesh is found.
- `pipeline/step6_fusion.py`: renamed unused variable `raw` → `_` (cosmetic).
- `scripts/run_debug_pipeline_foundationpose.sh`: updated to scene 000049 ("tuna can"), added `--ulip_mode pc` and `--ulip-partial-views` flags.
- New `docs/PIPELINE_AUDIT.md`: comprehensive audit of all 8 pipeline steps with 20 ranked findings, parameter shortlist, and ablation recommendations.

## 2026-04-02 Migrate SAM → SAM2.1 in Step 1

Goal
- Replace SAM ViT-L (`facebook/sam-vit-large`) with SAM2.1 Hiera-L (`facebook/sam2.1-hiera-large`) for better mask quality (especially in cluttered scenes) and faster inference.

Changes
- `pipeline/config.py`: updated `sam_model` default to `facebook/sam2.1-hiera-large`, corrected SAM2 GitHub URL.
- `pipeline/step1_localization.py`:
  - Imports: `SamModel`/`SamProcessor` → `Sam2Model`/`Sam2Processor`.
  - `_load_model()`: uses SAM2 classes.
  - `_segment()`: added explicit `images=` kwarg to processor call; switched to `processor.post_process_masks(pred, orig)` (SAM2 API drops `reshaped_input_sizes` and the `.image_processor` indirection).
  - Updated header comments (SAM2 → SAM2.1, corrected GitHub URL).

Rationale
- SAM2.1 produces higher-quality masks, especially at ambiguous boundaries. The mask feeds into every downstream step (ROI crop, point cloud, pose estimation), so improvements compound. The change is API-compatible — output is still a `(H, W)` bool mask.

## 2026-03-29 Move load_object_descriptions into CLIPRetriever

Goal
- Align Step 3 with Step 4 pattern: data loading as class method instead of standalone utility function.

Changes
- Moved `load_object_descriptions()` from `pipeline/utils.py` into `CLIPRetriever._load_object_descriptions()` as a static method in `pipeline/step3_clip_retrieval.py`.
- Added `import json` to `step3_clip_retrieval.py`.
- Removed unused `List` import from `pipeline/utils.py`.

Rationale
- `load_object_descriptions` was only used by `CLIPRetriever.load_descriptions()`. Step 4's analogous `load_reference_images` is already a method on `DINOReranker`. This makes both steps consistent.

## 2026-03-26 Partial-to-partial point cloud matching for Step 5

Goal
- Replace the full-mesh CAD point cloud comparison in Step 5 with partial-view point clouds rendered from the same 8 viewpoints as the reference images. This eliminates the domain mismatch between the partial observed PC (single depth view) and the full CAD PC (uniformly sampled from entire surface).

Changes
- New `rendering/generate_partial_pointclouds.py`: standalone preprocessing script (no Blender needed). Uses trimesh to load and normalize CAD meshes, then samples visible surface points per camera viewpoint using front-face culling. Produces `{obj_id}_view{N}_partial.npz` files alongside existing PNGs and camera matrices.
- Modified `pipeline/config.py`: added `ulip2_use_partial_views: bool = False` config field.
- Modified `pipeline/step5_shape_matching.py`:
  - `ShapeCandidate` gains `best_view_idx: int = -1` field (index of best matching partial view).
  - `ShapeMatcher` gains `_partial_mode` flag and new methods: `_load_cad_models_partial()`, `_collect_partial_items()`, `_get_partial_cache_path()`, `_try_load_partial_cache()`, `_save_partial_cache()`.
  - `load_cad_models()` now has a dual path: if `ulip2_use_partial_views=True`, loads partial `.npz` files and encodes per-view embeddings `(num_views, embed_dim)` per object.
  - `match()` uses best-of-N-views scoring (max cosine similarity over 8 views) when in partial mode.
  - Separate cache file (`.ulip_partial_cache_<hash>.pt`) with `"partial": True` flag to avoid collisions with full-mesh cache.
  - Fallback: if no `.npz` files exist for an object, falls back to full mesh sampling with a logged warning.
- Modified `pipeline/debug_viz.py`:
  - New `_load_view_thumb()` helper to load a specific view image.
  - `save_debug_step5()` shows "Best View: N" in score labels and loads the matching view thumbnail instead of the first alphabetical image.
- Modified `pipeline/run_pipeline.py`: added `--ulip-partial-views` CLI flag, wired to config.

Design decisions
- Front-face culling was chosen over raycasting for performance: raycasting 262k rays/view with trimesh's rtree backend took ~2.6s per 5000 rays (estimated ~60h for full dataset), while front-face culling takes ~0.02s per view (~10 min for 1051 objects × 8 views).
- Front-face culling is an approximation (no self-occlusion handling) but works well for convex and mildly concave objects typical of the YCBV-GSO dataset.
- Blender camera coordinate convention (X right, Y up, -Z forward) differs from OpenCV (X right, Y down, +Z forward); camera matrix decomposition accounts for this when computing camera positions from stored RT matrices.
- Texture-based mesh visuals are converted to per-face ColorVisuals before sampling to extract vertex colors from textured OBJ files.

Preprocessing results (ycbv_gso)
- 1051 objects × 8 views = 8408 partial point clouds generated in ~10 minutes.
- Each `.npz` contains `points` (10000, 3) within [-0.5, 0.5] and `colors` (10000, 3) in [0, 1].
- Different views produce distinct partial PCs (verified via per-view centroid comparison).

## 2026-03-26 Debug visualization refactored into main pipeline

Goal
- Eliminate the duplicated pipeline logic in `debug_steps.py` by making debug visualization an optional mode of the normal pipeline.

Changes
- Deleted `pipeline/debug_steps.py` (~1473 lines, contained a full copy of the 8-step pipeline in `run_debug()`).
- New `pipeline/debug_viz.py` (~1070 lines): all visualization functions extracted from the old file. `_find_cad_mesh()` promoted to module level (was nested inside `save_debug_step7_8()`, causing a NameError at runtime).
- Modified `pipeline/run_pipeline.py`:
  - `OSCARPlusPipeline.__init__()`: new `debug_viz: bool = False` parameter.
  - `OSCARPlusPipeline.run()`: new `gt_data=None` parameter for GT wireframe overlay.
  - Debug-viz hooks (calls to `_dbv.save_debug_step*()`) added after each of the 8 steps, guarded by `if self.debug_viz`.
  - Mesh-path resolution added before step 7: detects image-paths (`.png/.jpg`) in `cad_model_path` and resolves via `_find_cad_mesh()`. Result shared with step 8.
  - GT pose matrix built from `gt_data` parameter (same logic as old `run_debug()` lines 1294-1312).
  - New CLI flags: `--debug-viz` (rich debug images), `--until-step N` (converted to `skip_steps`).
  - `main()`: loads GT data from `scene_gt.json` + `id_to_label.json` when `--debug-viz` and `--camera` are set.
  - Bug fix: `detection_prompt` (undefined) → `prompt_elements.detection_phrase` in step 1 viz call.
- Modified `scripts/run_debug_pipeline_foundationpose.sh`: calls `pipeline.run_pipeline --debug-viz` with full YCBV-GSO defaults.
- New `scripts/run_pipeline.sh`: convenience script for normal pipeline execution.

Behavioral changes vs. old `debug_steps.py`
1. CLIP retrieval now receives `text_query=visual_query` from prompt parsing (old code omitted it) — may produce slightly different rankings.
2. Prompt parsing uses `_extract_prompt_elements()` (Ollama + heuristic) instead of duplicated logic.
3. `_find_cad_mesh` bug fixed — was unreachable in old code due to nested scope.

Impact
- Single source of truth for pipeline logic (no more `run_debug()` copy).
- `git grep "def run_debug"` returns no results.
- Debug shell script remains compatible (same output files, same CLI flags via `"$@"`).

## 2026-03-24 GT overlay + intrinsics/depth fixes

Goal
- Add ground truth pose wireframe overlay to debug_07_scale_pose.png for visual pose validation

Changes
- pipeline/debug_steps.py: load scene_gt.json + id_to_label.json in run_debug(); build 4x4 GT pose matrix; draw magenta GT wireframe via second _project_cad_wireframe() call; compensate for mesh bbox_center offset (subtract R_gt @ bbox_center from GT translation before projection); add "Predicted"/"GT" legend to Panel A; add Δt/ΔR metrics to Panel C; Panel C height +90px when GT shown
- pipeline/debug_steps.py + run_pipeline.py: moved camera loading before depth conversion so real fx/fy/cx/cy reach generate(); depth_scale always taken from config (BOP JSON field uses multiplier convention incompatible with pipeline divisor convention)
- pipeline/step2_pointcloud.py: identified dead code — PinholeCameraIntrinsic object created but never used; depth_scale param in generate() never exercised from pipeline call sites

Key finding
- BOP scene_camera.json depth_scale=0.1 is a multiplier; pipeline divides by config.depth_scale=10000.0. Using the JSON value caused depths to be 100× too large, resulting in ~855mm translation error in predicted pose. Always use config value.
- GT wireframe shift (~8px) caused by mesh bbox_center offset from origin; compensated by adjusting GT translation by -R_gt @ bbox_center before projection.

## 2026-03-20 FoundationPose two-container HTTP integration

Goal
- Replace the broken venv/subprocess FoundationPose integration with a clean two-container HTTP architecture.

Changes
- New file: `FoundationPose/foundationpose_server.py`
  - Minimal Flask server with `/health` and `/estimate_pose` endpoints.
  - Runs inside the FP container's conda env (Python 3.8, torch 2.1.0+cu121, pytorch3d, kaolin, nvdiffrast).
  - Lazy-loads scorer, refiner, and GL context on first request.
  - Accepts base64-encoded numpy arrays + camera matrix + CAD path via JSON POST.
  - Returns 4x4 pose matrix + confidence as JSON.

- Rewritten: `pipeline/foundationpose_bridge.py`
  - Now an HTTP client using httpx (was: subprocess launcher).
  - Encodes RGB/depth/mask as base64 numpy blobs.
  - Auto-translates CAD paths from `/app/...` (OSCAR container) to `/oscar/...` (FP container).
  - Configurable timeout (120s read, 10s connect).

- Rewritten: `pipeline/step8_pose_estimation.py`
  - Removed `_run_foundationpose_local()` (local import path — never worked in OSCAR container).
  - Removed `_run_foundationpose_subprocess()` (subprocess path — broken due CUDA mismatch).
  - Removed `_estimate_megapose()` (was always NotImplementedError).
  - Single FoundationPose path now calls `foundationpose_bridge.call_foundationpose()`.
  - ICP fallback preserved and unchanged.

- Modified: `pipeline/config.py`
  - Replaced `foundationpose_python` (str) with `foundationpose_url` (str, default `http://foundationpose:5050`).
  - Removed `foundationpose_repo_path` (no longer needed — FP container manages its own repo).

- Modified: `pipeline/debug_steps.py`, `pipeline/run_pipeline.py`
  - Replaced `--foundationpose_python` and `--foundationpose_repo` CLI args with `--foundationpose_url`.

- Modified: `docker-compose.yml`
  - Added `foundationpose` service using `shingarey/foundationpose_custom_cuda121:latest`.
  - FP service mounts `../FoundationPose:/workspace` and `.:/oscar:ro`.
  - Entrypoint activates conda env and runs `foundationpose_server.py`.
  - Healthcheck on `/health` endpoint.
  - Removed `../FoundationPose:/foundationpose` volume mount from oscar service (no longer needed).

- Updated: `README.md`, `AI_HANDOFF.md`, `docs/DECISIONS.md`
  - Replaced venv setup instructions with two-container startup instructions.
  - Removed references to `/foundationpose/.venv/bin/python`.
  - Updated command examples.

Diagnosis that motivated this change
- OSCAR container: `nvidia/cuda:12.2.0-runtime-ubuntu22.04`, Python 3.11, no CUDA dev headers.
- FoundationPose needs: CUDA devel image, Python 3.8, torch 2.0/2.1+cu118/cu121, pytorch3d/kaolin/nvdiffrast (all require compilation).
- A venv inside the OSCAR container cannot bridge this gap: no nvcc, wrong Python ABI, wrong CUDA version.
- The pre-built `shingarey/foundationpose_custom_cuda121` image has everything pre-compiled.

Options evaluated
1. HTTP API between two containers (chosen) — simplest, no Docker socket, no shared Python.
2. Shared-volume CLI handoff via `docker compose exec` — viable but requires Docker socket in OSCAR container.
3. Fix the venv inside OSCAR — not viable due CUDA runtime vs devel mismatch.
4. Install CUDA devel in OSCAR image — bloats image, fragile compilation chain.

Removed items
- `foundationpose_python` config field and `--foundationpose_python` CLI arg (replaced by `foundationpose_url`).
- `foundationpose_repo_path` config field and `--foundationpose_repo` CLI arg (FP container manages its own repo).
- `_run_foundationpose_local()` in step8 (never worked in OSCAR container).
- `_run_foundationpose_subprocess()` in step8 (broken due CUDA mismatch).
- `_estimate_megapose()` in step8 (was always NotImplementedError).
- `../FoundationPose:/foundationpose` volume mount in oscar service.

Impact
- FoundationPose can now actually run from the OSCAR pipeline (was previously broken).
- ICP fallback remains intact and robust.
- `docker compose up -d foundationpose` + `docker compose run --rm -it oscar bash` is the new startup pattern.

Manual follow-up needed
- Delete obsolete 11 GB venv: `rm -rf ~/thesis/FoundationPose/.venv`
- Test end-to-end with `--pose_method foundationpose` to validate the HTTP path.

## 2026-03-19 FoundationPose integration and split-environment execution (superseded)

> Superseded by 2026-03-20 two-container HTTP architecture.
> The subprocess bridge and venv approach did not work due to CUDA/ABI incompatibilities.

Goal
- Run FoundationPose in Step 8 without breaking OSCAR runtime dependencies.

Changes
- Added subprocess execution path for FoundationPose in step8.
- Created `pipeline/foundationpose_bridge.py` as standalone subprocess script.
- Added `foundationpose_python` config field and CLI arg.
- Added persistent volumes for Ollama, Torch, and CLIP caches.

Why superseded
- The dedicated venv at `/foundationpose/.venv` (created inside OSCAR's CUDA 12.2 runtime container) could not compile pytorch3d, kaolin, or nvdiffrast due to missing CUDA dev headers and Python ABI mismatch.

## 2026-03-18 FoundationPose setup and compose update

Goal
- Prepare a reproducible local setup for FoundationPose and document current switch status.

Changes
- Host setup:
  - cloned `NVlabs/FoundationPose` to `~/thesis/FoundationPose`
  - installed Docker image `foundationpose:latest`
- OSCAR integration prep:
  - updated `docker-compose.yml` volumes with `../FoundationPose:/foundationpose` (superseded by 2026-03-20)
- Codebase check:
  - verified `pipeline/step8_pose_estimation.py` still uses a FoundationPose template path and falls back to ICP.

Impact
- FoundationPose assets are available locally.
- Runtime behavior of Step 8 was unchanged until 2026-03-20 HTTP integration.

## 2026-03-18 Step 1 localization cleanup

Goal
- Verify what changed in `pipeline/step1_localization.py` and document it.

Changes
- Confirmed a non-functional cleanup in Step 1:
  - removed one duplicated comment line in the module header.
- No runtime logic, model call, threshold, or output schema changed.

Impact
- Behavior unchanged.

## 2026-03-17 ULIP Full Mode, CAD Cache, and Pose Path Fixes

Goal
- Enable side-by-side experiments for ULIP `pc` vs ULIP `cross` (full cross-modal) in the debug pipeline.
- Fix slow Step 5 by caching CAD embeddings.
- Fix Step 8 failures caused by image paths being passed as CAD mesh paths.

Changes
- Modified `pipeline/step5_shape_matching.py`:
  - added ULIP cross-modal image encoding support (`open-clip-torch`)
  - recursive CAD mesh discovery (supports `meshes/model.obj` style layouts)
  - added CAD embedding disk cache (`.ulip_cache_<hash>.pt`)
  - stores cached CAD embeddings on CPU to reduce repeated GPU work
- Modified `pipeline/debug_steps.py`:
  - added CLI args `--ulip_mode` and `--ulip_image_weight`
  - forwards `query_image` to Step 5
  - robust CAD mesh path resolution before Step 7/8
- Modified `pipeline/step6_fusion.py`:
  - separated DINO `best_view_path` (image) from true `cad_model_path` (mesh)
  - prevents Step 8 from trying to load PNG as mesh
- Modified dependencies:
  - root `requirements.txt`: added `open-clip-torch`, `trimesh`

Results
- `open_clip` import error resolved.
- CAD loading count corrected from 21 to 1051 models for ycbv_gso.
- Step 8 no longer fails with `CAD-Mesh leer: ...png` due to wrong path propagation.
- Step 5 subsequent runs are faster due to cache reuse.

## 2026-03-12 Pipeline Debugging, ULIP NaN Fix, Batch Cache, ICP Alignment

Goal
- Fix all runtime bugs in the 8-step pipeline after initial end-to-end test.
- Improve DINOv2 speed (Step 4) from serial encoding to batch + disk cache.
- Fix ULIP-2 NaN scores (Step 5) caused by Open3D color overflow.
- Fix score fusion NaN propagation (Step 6).
- Fix ICP pose estimation not using coarse alignment from Step 7.
- Add 3D wireframe overlay for debug visualization (Step 7+8).

Changes

### Modified: pipeline/step4_dino_reranking.py (rewritten)
- Replaced serial 1-by-1 DINOv2 encoding with batch encoding (32 images/pass).
- Added `.pt` disk cache keyed by model name + fingerprint (file count + newest mtime).
- First run: ~5 min for 9,459 reference images. Subsequent runs: instant from cache.

### Modified: pipeline/step5_shape_matching.py
- Fixed overflow bug: `np.asarray(pcd.colors)` -> `np.clip(raw, 0.0, 1.0)`.
- Added NaN filtering in `match()`: replaced with -1.0 before `topk()`.

### Modified: pipeline/step6_fusion.py
- Made `_minmax()` NaN-safe: filters NaN values before computing min/max.

### Modified: pipeline/step7_scale_estimation.py (rewritten previously)
- Two-stage approach: RANSAC + ICP -> Partial-Aware Scale (2 best-visible axes).

### Modified: pipeline/step8_pose_estimation.py
- Added `initial_pose` parameter forwarding to ICP.
- ICP now uses coarse alignment from Step 7 as initial transform.

### Modified: pipeline/config.py
- Changed defaults: `pose_method` to `"icp"`, `voxel_size` to `0.002`, `ollama_model` to `"gemma3:4b"`.

### Modified: pipeline/debug_steps.py
- Added 3D wireframe overlay projection using trimesh.

Bugs Fixed
- NaN ULIP scores, NaN in topk rankings, NaN in fusion normalization.
- FoundationPose fallback not passing initial_pose.
- Wireframe projection and scaling issues.

Pipeline Test Results (scene 000048/000001, "i need the blue coffee can")
- Step 1: confidence 0.847
- Step 6: master_chef_can fused=0.8473
- Step 8: ICP fitness=0.9895, RMSE=0.007m

## 2026-03-05 ULIP-2 Pipeline Integration + Visualization

Goal
- Implement full 8-step shape-aware retrieval pipeline on branch exp/ulip2.
- Integrate real ULIP-2 point cloud encoder (PointBERT Colored, 10k points, 1280-dim).
- Add LLM-based prompt parsing via Ollama.
- Add visualization module for intermediate results.

Changes

### New: pipeline/ module (17 files)
- Created full pipeline package: config, orchestrator, 8 step modules, utils, visualization.
- ULIP-2 integration in step5 (loads ~400 MB point encoder + projection).
- LLM prompt parsing via Ollama with heuristic fallback.

### Modified: docker-compose.yml
- Added volume `../ULIP:/ulip` and GPU device reservation.

### Modified: requirements.txt (root)
- Added: `ollama`, `open3d`, `easydict`, `timm`, `pyyaml_env_tag`, `termcolor`.

### Patched: ULIP repo (separate repo)
- Made `knn_cuda` and `pointnet2_ops` optional with fallbacks.

Bugs Fixed
- KeyError in camera intrinsics, missing packages, PyTorch 2.6 weights_only change.

## 2026-03-04 Retrieval evaluation

Goal
- Run full OSCAR retrieval pipeline on YCBV_GSO and MI3DOR, compare to paper.

Results
- YCBV_GSO: 75.95% top-1 accuracy (paper ~60%, difference: GT masks vs GroundedSAM).
- MI3DOR: NN=77.95% (paper NN=89.4%, gap: descriptions only 10/21 categories).

## 2026-02-19 to 2026-02-23 Rendering and data pipeline

Goal
- Render all 3D models for YCBV_GSO, HouseCat6D, and MI3DOR datasets using Blender.

Rendering Results
- YCBV_GSO: 1050/1051 rendered.
- HouseCat6D: 194 real objects rendered.
- MI3DOR: 3848/3848 rendered.

## 2026-02-08 YCB-V and GSO repro setup

Goal
- Move reproduction work to exp/oscar-repro, set up local YCB-V plus GSO data layout.

Changes
- Prepared YCB-V test folder, downloaded GSO assets, fixed git tracking for large files.

## 2026-02-06 Repository scaffold and GPU setup

Goal
- Document the repository state after resetting main and creating the thesis scaffold.

Changes
- Reset main to a clean scaffold.
- Added README, placeholder directories, AI documentation files.
- Documented GPU support intent for Docker compose.
