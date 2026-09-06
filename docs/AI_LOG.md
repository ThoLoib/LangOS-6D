# AI Log

## 2026-09-04 The colour fix reversed the full-mesh finding — but only for retrieval, not for pose

Goal
- Re-measure the four full-mesh arms with the corrected texture sampler (2026-09-01 entry) and
  check the prediction recorded before the run.

The prediction held where it mattered
- Recorded beforehand: *GSO and YCB-V slightly better, T-LESS unchanged, overall finding stands.*

| pc, full-mesh | before | after | Δ | predicted |
|---|---|---|---|---|
| YCB-V | 0.635 | **0.740** | +0.105 | "slightly better" — direction right, magnitude badly underestimated |
| T-LESS | 0.157 | 0.159 | **+0.002** | "unchanged" ✓ |
| LM-O | 0.436 | 0.446 | +0.010 | — |

- The **controls reproduce exactly**: `3a_pc_v2` = 0.4636 and `3a_cross_v2` = 0.4818, identical to
  the pre-fix runs. The partial caches were untouched, so they had to — and they did. That
  validates the whole re-run rather than just asserting it.

The finding itself reversed
- `3a_cross_fullmesh_v2` = **0.5151**, up from 0.4639, and **above the frozen config (0.4818)**.
  It wins on R@1, R@5, R@10, MRR and the isolated shape channel. Full mesh is the better gallery
  representation in cross mode; in pc mode partial still wins, but the entire remaining gap now
  comes from T-LESS (YCB-V +0.069 and LM-O +0.046 go the other way).
- Conclusion drawn on 2026-09-01 — "partial beats full mesh" — was therefore **substantially an
  artefact of the colour defect**. What survives is narrower and better supported: partial pays off
  only where the query is itself partial *and* shape is the sole discriminating channel.

**But the retrieval win does not reach the pose.** `3b_cross_fullmesh` D_sym median **18.91 mm**
against **18.37 mm** for cross-partial — full mesh finds the right CAD more often and poses it
slightly worse. Per dataset: YCB-V 21.8 vs 23.6 (better), T-LESS 15.0 vs 13.6 and LM-O 34.0 vs
28.5 (worse). Both still beat the OSCAR baseline (21.73 mm) clearly.

- Mechanism, and it is the same one a third time: **retrieval and pose ask different things of a
  substitute**. Retrieval rewards the model that matches the object as a whole; FoundationPose
  registers against the *visible* surface, which rewards a representation matching the partial
  observation. Whatever fits the observation domain wins — not whatever is objectively more
  complete. Same structure as T-LESS in retrieval and as the geometry stage overall.

The isolated shape channel, available for the first time
- `arm_ranks` (added 2026-09-01) makes the per-channel target rank fall out of every run:

| arm | fused | shape alone | DINO alone | CLIP alone |
|---|---|---|---|---|
| pc, partial | 0.4636 | 0.0211 | 0.3739 | 0.2086 |
| pc, full mesh | 0.3878 | 0.0089 | 0.3739 | 0.2086 |
| cross, partial | 0.4818 | 0.1997 | 0.3739 | 0.2086 |
| cross, full mesh | 0.5151 | 0.2272 | 0.3739 | 0.2086 |

- DINO and CLIP are bit-identical across runs — they do not depend on the shape gallery, so they
  had to be. A cheap consistency check that came for free.
- **The pc-mode shape channel reaches 2.1 % R@1 on its own** against 1316 objects, while DINO
  reaches 37 %. In cross mode it jumps to 20 %, because there ULIP-2's *image* tower is doing the
  work — effectively a second appearance channel. The contrast with Stage 1 (isolated shape
  0.5353 nDCG) is not a contradiction: SHREC scores graded category relevance, BOP demands the
  exact instance.

Stage 1, same fix
- `E2b_fullmesh` **0.5935 / NN_sub 0.3598** — now the strongest arm without geometry, ahead of
  BASE (0.5868 / 0.3413). Isolated, partial still wins (0.5353 vs 0.4956).
- **Fused full mesh wins, isolated partial wins** — on SHREC and on BOP alike. The partial channel
  is more accurate alone but errs in ways that correlate with text and appearance; the complete
  mesh errs differently and therefore complements the fusion better.
- The first attempt produced `rendered: 0 -> scored gallery: 0`: the driver's default `images_dir`
  is `object_images/shrec18` (empty), the renders live in `shrec18_v2`, and the chain script had
  omitted `--images-dir`/`--desc-file`. The second attempt then wrote into the repo root because
  `--results-root` is relative and the container ran from `/app`; the arms were moved into the
  canonical folder afterwards, deliberately without the two-arm `best_config.json`.

A gap the tooling found immediately
- `tools/results_overview.py` generates `docs/RESULTS_OVERVIEW.md` from the actual result
  directories. On first run it surfaced three things held wrongly in my head: Stage 1 has **39**
  arms, not 44; the two new arms sat in the wrong directory; and **Stage 1 has no cross × full-mesh
  cell at all** — the very combination that wins on BOP. Pass and arms added, run queued.
- `tools/compare_arms_by_category.py` pairs two arms per GT category. First result (pc partial vs
  pc full mesh, nDCG): partial wins machine +0.39, keyboard +0.36, book +0.33; full mesh wins desk
  −0.28, light −0.23, table −0.21. Partial views for small detail-rich objects, complete meshes for
  large flat furniture.

Two self-inflicted monitoring errors, recorded because both are the same class as the silent
config defects
- I declared the pose chain dead from a `ps` filter that never included it — reading absence from a
  list it could not have appeared in as a crash.
- The first Monitor's `pgrep -f` searched for the three script names, which appear in **its own
  command line**; it always matched itself, so the abort branch was dead code. It delivered the 3b
  milestones correctly the whole time, which hid the defect. Replaced with a `ps -eo args` match
  anchored on `^bash scripts/…` plus a check for "script alive but no compute process".

## 2026-09-01 Textured meshes lost their colour in the full-mesh path — and the hypothesis it inspired was wrong

Goal
- Check whether the partial-vs-full-mesh gap in Stage 3 (−0.113 R@1 in pc-mode) is a configuration
  artefact, prompted by the size of the difference.

The defect
- `sample_pointcloud_from_mesh` read colour only from `mesh.visual.face_colors`. Textured meshes
  carry their colour in an image plus UV coordinates, where that attribute is `None`; the caller
  then substitutes `np.zeros_like(points)`, so the *coloured* ULIP-2 backbone received three dead
  channels. Measured per dataset (std of sampled RGB): SHREC'18 `None`, GSO `None`, YCB-V `None`,
  LM-O 0.04, while the corresponding partial clouds carry 0.18–0.28.
- Scope is the full-mesh path only, i.e. Stage-1 arm `E2b_fullmesh*`, the Stage-3 full-mesh arms,
  and **all of Stage 2**, which ran full-mesh throughout via the silent fallback. Every partial-view
  arm — including the headline configurations of Stages 1 and 3 — is unaffected.
- Separate from the code defect: **MI3DOR, T-LESS, ITODD and HouseCat6D carry no colour in the
  files at all**. MI3DOR returns a uniform (0.4, 0.4, 0.4) for every one of the 3848 meshes, which
  is trimesh's default, i.e. the files are colourless. No sampler fix can recover that; only the
  renders have colour. Stage 2's shape channel therefore ran colour-blind over the whole gallery —
  a plausible contributing reason for ULIP-2 being its weakest channel (FT 0.510 vs DINO 0.629),
  which had been attributed to cross-modal difficulty alone.

The hypothesis was wrong, and that is the more useful finding
- The obvious conclusion — "the full-mesh arms lost their colour, hence the gap" — does not
  survive contact with the data. Three independent counter-arguments:
  1. **The damage does not track the colour.** YCB-V loses real colour (0.21 → none) and drops
     0.036. T-LESS loses none (both sides are 0.0) and drops 0.193.
  2. **The query-mode asymmetry rules it out.** The gallery is point-cloud-encoded in *both* modes,
     so a gallery-only property must affect them equally. It affects pc six times more
     (−0.113 vs −0.018). That points at the *relation* between query and gallery, not the gallery.
  3. A discriminability probe I ran was **worthless and is recorded as such**: it compared a max
     over 42×42 view pairs (partial) against a single cosine (full mesh), which is structurally
     larger and measures nothing.
- What the gap actually is: query–gallery domain match. In pc-mode the query is a partial cloud and
  the partial gallery consists of exactly those, with 42 chances per object to match the viewpoint;
  a full mesh offers one embedding of a complete surface on the same unit sphere as the query
  patch. The per-dataset staggering follows from how much the shape channel must carry alone.
- Target-rank distribution on T-LESS (pc): partial median 2, full mesh median 8, **no shape channel
  median 24**. Full mesh lifts the ranking well above the no-shape baseline — it is informative and
  fails only at rank 1, which is precisely what Recall@1 measures.

A trap in the fix
- The cache fingerprint (`_get_cache_path`) hashes config flags and **file sizes**, not the sampled
  colour values, and `ulip2_use_colors` was already `True`. The hash is therefore unchanged by the
  fix: without moving the old caches aside, the re-run would have silently loaded the colourless
  embeddings and produced identical numbers. Four caches (gso, ycbv, housecat6d, shrec18) are
  stashed rather than deleted by `scripts/run_fullmesh_color_redo.sh`.

Changes
- `_sample_face_colors` resolves colour in three steps: `face_colors` → `to_color().vertex_colors`
  averaged over each sampled triangle → `None`. Verified after the fix: GSO 40/40 objects coloured,
  YCB-V 0.21–0.28, SHREC 41/60 (≈30 % of its CADs have unresolvable textures and stay colourless —
  a documented limitation of that arm).
- `eval_bop_pose` now records `arm_ranks`, the target's rank **per channel**. `run_query` already
  computes those arms in the same pass, so the isolated shape channel (`ulip_only_full`) now falls
  out of every run for free — previously it would have required separate runs with weights (0,0,1),
  and the partial-vs-full-mesh effect stayed hidden behind a fusion that absorbs 70 % of it.
- Prediction recorded before the re-run so the result cannot be fitted afterwards: **GSO and YCB-V
  should improve slightly, T-LESS should not move, and the overall finding should stand.** If that
  does not hold, something else is still unaccounted for.

## 2026-08-31 Full-mesh IDs collapsed on HouseCat6D — caught before the run, contained to Stage 3

Goal
- Audit the queued Stage-3 arms (`3a_fullmesh`, `3a_pc_fullmesh`) before launching them, after the Stage-2 silent full-mesh fallback showed that configuration errors in this pipeline do not announce themselves.

The defect
- `stage3_gallery._absorb_dataset` loaded full-mesh shape references via `load_cad_models(cad_dir=…)`, which resolves object ids through `_collect_mesh_items`. That helper derives the id from the **directory name**, which is correct only while every object owns a directory.
- HouseCat6D is grouped by category — `object_database/housecat6d/<category>/<object>.obj` — so all **199 objects collapsed onto 12 category ids** (`bottle`, `box`, `cup`, …). None of them matches a gallery id, since gallery ids are the render-directory names (`bottle-85_alcool`).
- Consequence had it run: **199 of 1316 gallery entries (15 %) without a shape embedding**, and 199 of 1257 (16 %) of the *proxy* gallery, which 3b/3c depend on. No exception, no warning — the arm would have completed and produced plausible numbers with a partially amputated shape channel.
- The code carried a comment demanding exactly this check ("VERIFY per dataset (n_absorbed logged) before trusting the numbers"). A comment is not a check; nobody had run it.

Blast radius — no other run was affected
- `load_cad_models` has exactly one caller in the evaluation code (`stage3_gallery.py:342`); the other two are in the interactive `run_pipeline.py`. Stage 1 and Stage 2 never reach `_collect_mesh_items`: both pass explicit `cad_mesh_items` built from the **file stem** (`eval_common.py:413`, `retrieval_mi3dor_eval_oscarplus.py:157`), which is the correct rule for their flat layouts.
- Verified empirically rather than by reading, by intersecting the cached embedding ids with the render directories:

| run | cache ids | gallery ids | matched |
|---|---|---|---|
| MI3DOR full-mesh (Stage 2, all of it) | 3848 | 3848 | **3848** |
| MI3DOR category-filtered | 1817 | — | **1817** |
| SHREC'18 (Stage 1, three caches) | 3308 | 3308 | **3308** |

- No completed Stage-3 run had used the broken branch either: the single earlier `3a_fullmesh` attempt (2026-08-28) died with `FileNotFoundError` from the unrelated wildcard-path bug before producing output. The defect therefore never reached a published number.

The fix
- `_FULLMESH_ID_MODE` states the id rule per dataset explicitly, and `load_cad_models` accepts a resolved `mesh_items` list instead of inferring one. Kept deliberately separate from `DATASET_LAYOUT["id_mode"]`, which feeds `build_pipeline`'s `cad_mesh_items` fallback and was not touched while the partial arms were mid-run.
- A **hard coverage gate** replaces the comment: below 95 % of gallery ids carrying an embedding, the assembly raises with the missing ids instead of continuing.
- Two independent confirmations before launch: 1316/1316 coverage against the render directories, and the derived ids reproduce the *precomputed* full-mesh caches exactly (ycbv 21, gso 1030, housecat6d 199, all `identisch=True`) — so the precompute path had always used the correct convention and only the Stage-3 runtime path deviated. Post-run both arms logged `Deckung 100.0%` on all six datasets.

Stage-3 results these arms produced
- Gallery representation × query modality (R@1):

| query | partial | full-mesh | Δ |
|---|---|---|---|
| cross | **0.4818** | 0.4639 | −0.018 |
| pc | **0.4636** | 0.3504 | −0.113 |

- Partial views win in both modes, and the margin is **six times larger** when the query is itself a partial cloud. That is the mechanism, not a coincidence: pc-vs-partial matches observation to observation, while a full mesh compares a partial query against a complete surface. Stage 1 (SHREC, pc-mode) showed the same sign at +0.0495 nDCG, so the finding transfers to BOP and strengthens.
- Geometry re-ranking, all four clean cells, at 98 % registration coverage:

| query | no geometry | distance | fitness |
|---|---|---|---|
| cross | **0.4818** | 0.4229 | 0.4278 |
| pc | **0.4636** | 0.3725 | 0.3820 |

- Geometry loses in every configuration on BOP, and loses harder in pc-mode — consistent with the shape channel already carrying the depth information the re-rank re-derives. **Fitness beats distance on BOP in both modes**, inverting Stage 1 (distance 0.6405 > fitness 0.6251): the registration distance is the better signal where it operates scale-invariantly, the pure overlap measure wins once real metric scale is in play.
- OSCAR baseline (E5, τ-pruned CLIP → DINO best-view, no shape): R@1 **0.3198** vs 0.4818 for the full fusion, **+0.162**.

## 2026-07-30 Geometry cells landed; re-ranking is invisible to five of the seven metrics

Goal
- Finish the geometry ablations on the duty cycle and check them against the built-in correctness test.

The correctness check passed exactly
- `E2_none` reproduces `E1c_full_fusion` **bit-identically** on all seven official metrics (nDCG `0.5878959504080713`, P `0.26864481953094016`, AP `0.16009754131143725`). It shares the entire geometry code path but applies no geometry, so this pins the re-ranking plumbing as side-effect-free and makes the other geometry numbers trustworthy.

Results (all on 2,101 official queries)

| cell | nDCG | P | AP |
|---|---|---|---|
| `O5_xyz_only` (grid best) | **0.6106** | 0.2850 | 0.1740 |
| `E2_chamfer_ransac` | 0.6028 | 0.2686 | 0.1611 |
| `E2_chamfer_icp` | 0.6027 | 0.2686 | 0.1612 |
| `E2_fitness` | 0.6010 | 0.2686 | 0.1610 |
| `E2_none` = BASE | 0.5879 | 0.2686 | 0.1601 |
| `E2_chamfer_unaligned` | 0.5850 | 0.2686 | 0.1597 |

- Geometry re-ranking buys **+0.015 nDCG** over BASE. The unaligned control lands *below* BASE, which is the expected sign: comparing clouds without registering them is worse than not re-ranking at all, so the gain really is attributable to the alignment and not to the shortlist being reshuffled.
- **ICP buys nothing** (0.6027 vs 0.6028, i.e. inside the ±0.005 noise floor) despite costing ~5.4 s/query on top of RANSAC. `d_icp < d_ransac` on individual pairs — the refinement is real — but the *ranking* of five candidates does not change. This is the argument for shipping RANSAC-only in the frozen config.
- The one-pass restructure paid off exactly as projected: once `E2_fitness` had run, `E2_chamfer_ransac` completed in **96 s** (pure cache reads) instead of a second ~23 h registration pass.

The finding that matters for the write-up: five metrics cannot see this at all
- P, R, F1, NNT1, NNT2 are **identical to four decimal places across all six E2 cells**, including the control that is worse than baseline. This is not a bug — it is forced by the official protocol. Every metric is cut at `f` = category size (~165 average), and geometry re-ranks only `GEOM_SHORTLIST = 5` items. Reordering 5 items inside a 165-item prefix cannot change *which* items are in that prefix, so any set-membership metric is provably blind to it.
- Reading `eval/shrec18_official/metrics.py` confirms the structure: P, R, F1, NNT1, NNT2 are all identical **by construction** at `k = f`; **NNT2 is inoperative** (`k2 = min(len(x), 2k)` collapses back to `f`); and only `dcg` reads the graded relevance (`total += x[i] * w`, subcat=2 / cat=1) — precision and AP binarise it.
- Consequence: **nDCG is the only metric in the official set that can rank the geometry arms**, which is what makes it the selection criterion rather than a stylistic preference.
- The official `dcg` has an off-by-one (`enumerate(x[1:])` while indexing `x[i]` — double-counts `x[0]`, drops the last element). **Left unpatched deliberately**: `dcg` and `idcg` share it, so the ratio stays well-defined, and patching would break comparability with every published SHREC'18 number.

Cross-machine replication
- Compared against the tessa-PC results on Drive: **every conclusion replicates**, max Δ 0.0096 nDCG. One systematic difference — the text channel is consistently weaker locally (P 0.1319 vs 0.1419 across all four CLIP-shortlisted cells), which traces to the query RGB crops, not to the shape path.
- That gap cannot be resolved by appealing to a specification: SHREC'18 is a **mesh-to-mesh** track and the raw distribution (uploaded to `gdrive:Masterthesis/OSCAR/raw_datasets/shrec18_full`, 33,156 objects / 25,110,058,881 bytes, verified identical) contains **no instruction for rendering query images**. The query RGB crop is entirely an OSCAR-side construct, so its viewpoint heuristic is a design decision to document, not a spec to comply with.

Duty cycle in production
- Nine cycles carried the run 329 → 2,101 aligned queries at ~180–215 queries/cycle, surviving eight consecutive idle transitions without a single skipped window after the `docker.exe` fix. The Resource Saver problem is closed.
- The watchdog's `failed=30` alarm was a **watchdog bug, not data corruption**: it counted raw records, and a failed pair stores `d_ransac: null`, which the `missing` check correctly treats as "not computed", so every new window legitimately retries it and appends another failed record. Deduplicated by `(qid, cad)` — the same rule the engine's cache uses — the true state is **5 failed pairs on 1 query** (`bf977dde…`, 3,501 points, 9.9th size percentile, planarity 0.334), with gedi healthy throughout. Fixed the *watchdog*, deliberately **not** the engine's retry: that retry is exactly what lets a transient outage self-heal.

Status at the stop
- 31 of 33 cells done. Cache: 22,358 records / 10,553 deduplicated pairs, last line complete valid JSON after `SIGKILL`.
- **Remaining: `O1c_gedi_post_fusion`** — it shortlists on text+view (`E1b`) instead of the BASE fusion, so 6,055 of its 10,505 pairs (57.6%) were never registered; 2,070 of 2,101 queries need ≥1 new fit ⇒ ~13 h compute. `O1d_shape_plus_gedi` is an alias of `E2_fitness` and materialises for free at aggregation.
- Resume is free (`--resume` + per-pair cache): `docker compose up -d gedi && bash _geom_duty.sh`.

## 2026-07-29 Battery duty cycle, and the Resource Saver trap it walked into

Goal
- Run the geometry experiment on a 2 h on / 30 min off cycle so it does not sit on the battery.

What broke
- The first two windows produced **zero work**. Both failed at `docker compose up -d gedi` with the Linux CLI missing — first `/usr/bin/docker: Input/output error` (23:21), then `No such file or directory` (23:55).
- Diagnosis needed both sides: PowerShell reported the engine healthy (`server=29.6.1`) while WSL had no `docker` at all, and `wsl -l -v` showed the **`docker-desktop` distro Stopped**. Cause: **Docker Desktop Resource Saver** stops the WSL engine a few minutes after the last container exits and tears the CLI bind-mount out of the Ubuntu distro. The duty cycle's idle phase creates that condition by design — the design meant to save power was disabling the tooling that starts the next window.
- A Docker Desktop restart fixed it at 23:23 and it broke again by 23:55, which is what proved restart is a workaround rather than a fix. Waking the engine (`docker.exe info` → 29.6.1) does **not** restore the Ubuntu mount, so engine health is the wrong thing to probe.

Fix
- Both scripts now call `/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe` (fixed path, works regardless of the integration; emits CRLF, so `tr -d '\r'` on every captured value) and operate on already-created containers via `docker start`/`stop` rather than `docker compose run` — see DECISIONS 2026-07-29 for why compose cannot be used through that binary.
- Verified end to end: gedi healthy 10 s after `docker start`, `stage1_geom` restarted with `/app` intact, logged `[resume] 16 ablations already computed`, reconnected to GeDi, and the pair count resumed climbing.

What went right
- The `!! gedi not healthy — skipping this window` guard, added speculatively hours earlier, was exercised for real twice within its first hour and behaved correctly: no run launched against a dead service, no `failed` records written, cache untouched at 329/2101 with `failed=0`. Without it these two windows would have repeated the 2026-07-27 cache poisoning instead of merely wasting time.
- Stopping mid-run cost nothing: the append-per-record cache and the tolerant loader meant the last line was complete valid JSON even after a SIGKILL (`exit 137`).

Cost
- ~1 h 45 min of wall clock across two dead windows. The ~24 h estimate assumes no further losses; each future skipped window costs a full 2.5 h cycle, and `_duty_watch.sh` now emits an explicit `window SKIPPED` line so it shows up in the tick history rather than as silent flatlining.

## 2026-07-28 (evening) Geometry run stabilised, then made 3× cheaper

Goal
- Establish whether the relaunched geometry run was actually stable, and get a defensible completion estimate.

Stability: the fixes hold, but the evidence is young
- `stage1_geom` ran ~1.5 h with `failed=0`, `RestartCount=0`, no OOM; gedi healthy throughout and self-recovered twice (`[geometry] GeDi is back — resuming.`) without losing work — the retry loop behaving as designed.
- Honest caveat recorded at the time: the previous run also looked healthy at this age and died 10 h in. What is different is *structural* (WSL 20 GB, restart policy, working healthcheck, 500k query cap, abort-on-unreachable), not observational.
- The `exit=0` GeDi dropout cause is still unknown. It is now survivable, not solved.

Instrumentation: fixed the watchdog, twice
- Two stale `_geom_watch.sh` monitors were still reporting the old aggregate `good=N (~N/5 queries)`. That number conflates the cheap `chamfer_unaligned` control (~0.5 s/query, no GeDi) with the expensive aligned signals, and it produced a bogus "427 queries done" reading. Stopped them; only the per-signal watchdog now runs.
- Per-signal truth at the time: `fitness=225  d_unaligned=350  d_ransac=0  d_icp=0`. Record count reconciled exactly: 375 restored + 350×5 unaligned + 45×5 new fitness.

Measured rate, three independent windows
- 40.2 s/query (10 min, 15 queries) · 34.6 s/query (30-min tick, 52 queries) · **34.8 s/query (83 min, 143 queries)**. Consistent, so the 500k point cap did fix the 40×-slowdown from the voxel-downsample attempt.

Root cause of the remaining cost, and the fix
- `pair_scores` keyed the cache per signal: `chamfer_ransac` needed `("fitness", "d_ransac")`, so a pair that already had `fitness` was still "missing" and **redid the whole 35 s GeDi + RANSAC** for a distance that costs ~0.5 s once the transform exists. Three aligned signals ⇒ three full registrations ⇒ ~60 h.
- Added `rerank(all_aligned=True)` (`pipeline/step_b2_geometry_reranking.py`) + a shared field set in `pair_scores`, so one registration yields `fitness`, `d_ransac`, `d_icp` together. See DECISIONS 2026-07-28 for the semantic consequence (fitness and d_ransac now share one RANSAC run).
- Smoke-tested before relaunching: all three fields written in 14.3 s for 2 pairs; second aligned signal read back in 0.00 s; `d_icp < d_ransac` in both pairs, confirming ICP is real and the fields are not aliases.
- Projection: **~26 h** for the six cells instead of ~60 h. The 225 fitness-only queries already cached are recomputed once (~2 h) because they lack `d_ransac`/`d_icp`.
- Post-switch measurement confirms it: **40.2 s/query** for all three aligned signals (10-min window), vs 34.8 s/q for `fitness` alone ⇒ ~23 h. A 30-min tick immediately after relaunch had suggested ~95 s/q, but that window included container startup (score-cache load + gedi warm-up); the conclusion drawn from it was wrong. Rate readings taken across a restart boundary are not comparable to steady-state ones.

Still to verify when the cells land
- `E2_none` must reproduce BASE exactly (nDCG 0.5879). That is the built-in correctness check on the whole geometry path; treat every geometry number as unverified until it passes.

## 2026-07-28 Stage-1 grid completed (27 cells); GeDi OOM poisoned the geometry run

Goal
- Run the full Stage-1 ablation grid on all 2,101 official queries, then fill the six geometry cells with `--with-geometry`.

Result: main grid (27 cells, ~6h40m)
- All six channel-score passes cached from the Drive-transferred gallery (DINO/SigLIP/ULIP-partial/ULIP-fullmesh caches all hit — `[init] ULIP CAD cache loaded (3308 models)`, no re-encoding). The content-stable fingerprints did their job across machines.
- **Best config: `O5_xyz_only`** — nDCG 0.6106, P 0.2850, mAP 0.1740; wins on every metric. BASE (`E1c_full_fusion`) 0.5879. Full table in `object_retrieval/results_shrec18_stage1/stage1_summary.csv`.
- Headline findings: dropping query colour helps (+0.023 nDCG over XYZ+RGB); the view budget saturates at **V16** (0.5858 vs V42 0.5879, and V32 0.5829 — non-monotonic, so the noise floor is ~±0.005); full-database fusion beats every shortlisted variant (cascade 0.4464, CLIP-pruned 0.4452, visual-first 0.5461), with all CLIP-shortlisted arms pinned at P=0.1319 = text-only, i.e. the shortlist itself is the ceiling; DINOv2 ≫ SigLIP (0.5879 vs 0.5093); weighted-sum > RRF (0.5879 vs 0.5668); ULIP-2 ≈ Uni3D (0.5879 vs 0.5843, inside noise).
- Caveat to carry into the thesis: `O5_xyz_only` also swaps the checkpoint (PointBERT/SLIP ViT-B, 512-d, 8192 pts) vs the colored arm's ViT-g/1280-d, so colour and encoder capacity change together — the "colour is a domain gap" reading is plausible but **not isolated** by this experiment.

Incident: GeDi OOM-killed, 10h of cached garbage
- `oscar-gedi-1` was **OOMKilled** (`exit=137`, `OOMKilled=true`) 50 min into the geometry run. WSL had only 15 GB (default 50% of the 31.6 GB host) shared between the oscar container (score tensors + 400-cloud LRU) and GeDi (model + 5,000×32 descriptors/cloud on GPU).
- Every subsequent RANSAC fit failed, and `_GeometryEngine._append_cache` wrote each failure as `{"failed": true}` — a **permanent** poison, since the B2 failure policy ranks failed candidates last. Final state: 3,280 records, 2,845 failed, only 87 queries usable.
- It ran unnoticed for 10h because the watchdog counted rows in `geometry_scores.jsonl` and checked `stage1_geom`'s status — both looked healthy. The apparent "speed-up" (26 s → 19 s/query) was the failure fast-path, not cache warming. Failure signals must be part of the filter, not just progress signals.

Repair (all applied 2026-07-28)
- `C:\Users\tholo\.wslconfig` (new): `memory=20GB`, `swap=8GB` (leaves ~11 GB for Windows + Docker Desktop). Verified 19 GB visible in WSL after `wsl --shutdown`.
- `docker-compose.yml` gedi: `restart: unless-stopped`; healthcheck switched from `curl` (**not installed in that image** — the check could never pass, so the container sat permanently "unhealthy" while serving fine) to a `python3 urllib` probe. Now reports `healthy` in ~15 s.
- `experiments/experiment1_shrec18_stage1.py`: new `_GeometryEngine._gedi_healthy()` (live probe, distinct from the cached `gedi_available`); `pair_scores` now **aborts with `SystemExit`** when fits fail *and* the service is unreachable, instead of caching bogus failures. Completed pairs stay cached and resumable.
- `_geom_purge_failed.sh` (new): drops `failed:true` records, backing the file up first. Kept 435 good pairs / 87 queries.
- `_geom_watch.sh` rewritten to report the good/failed split **and** gedi's container+health state, and to shout when either goes wrong.

Follow-up: the guard worked, and exposed a second fault
- The relaunched run aborted after ~90 s — correctly. `oscar-gedi-1` had exited **cleanly (`exit=0`, `OOMKilled=false`, no traceback)** ~1 min after startup, right after serving two `/health` probes and before any `compute_descriptors`. `restart: unless-stopped` revived it (`RestartCount=1`) and it was healthy again 60 s later, so by the time the watchdog looked, everything was green — the abort message was the only evidence.
- So GeDi has (at least) two failure modes: OOM under memory pressure, and this clean-exit-on-startup one, cause still unknown (the Flask dev server leaves no trace). Worth investigating separately; a production WSGI server would probably also log more.
- **Blanket abort was the wrong response** to a service that self-heals in ~90 s: it would end a 15 h run over a blip. `pair_scores` now retries — `_wait_for_gedi()` polls up to `GEDI_WAIT_S=300`s per attempt, `GEDI_RETRIES=4` — and aborts only if the service stays down. Failures are still never cached.

Status
- Geometry run relaunched 07:30 with the retry logic; watchdog reports the good/failed split plus gedi's state.
- Real rate is ~26 s/query (the pre-crash figure) → ~15 h per distinct shortlist. E2_* share the BASE top-5; `O1c` fuses text+view only, so it needs a second full pass. `best_config.json` may change if a geometry variant clears 0.6106.

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

## 2026-09-06 — Artefakt-Durchgang nach Review; zwei Fehler gefunden

**1. Stage-2-Kategorientabelle zeigte den falschen Arm.** Die Spalte „Fusion" in
`STAGE2_RESULTS.md` §5 enthielt `clip_pruned_dino_ulip` (den Kaskaden-Arm), nicht
`clip_dino_ulip_full` (die volle Fusion). Aggregiert fällt das nicht auf — 86.52 gegen 86.57 NN
— je Kategorie aber massiv: `vase` 0.476 (Kaskade) gegen 0.284 (Fusion), `camera` 0.908 gegen
0.974. Folge: die Aussage „feste Gewichtung schadet in **8** von 21 Kategorien" war falsch,
korrekt sind **9 von 21**, größter Verlust −0.132 (`vase`) statt −0.094 (`bookshelf`).
Nachgerechnet direkt aus `results_topk_15.json` über `eval_trace.arms[*].rel_positions`;
Aggregat gegen `metrics_summary_topk_15.json` verifiziert (86.5714 exakt).
Identifiziert wurde die Quelle durch Abgleich der publizierten Werte gegen alle
MI3DOR-Läufe × alle Arme (Fehler 0.0000 nur für `clip_pruned_dino_ulip`).

**2. Stage-4-Balken waren unsichtbar.** `.fill` ist ein `<span>` innerhalb `.track`; da `.track`
kein Flex-/Grid-Container ist, blieb `.fill` `display:inline` und ignorierte `width`/`height`.
`.track` selbst funktionierte, weil es Grid-Item von `.crow` ist (Blockifizierung).
Fix: `display:block` auf `.fill`. Vom Nutzer bemerkt, nicht von mir.

**3. Stage-1 §8 war unvollständig und teils veraltet.** Die handgepflegte `ARMS`-Liste hatte 36
statt 43 Einträge (u. a. fehlten `E2b_fullmesh_geo`, `E7_ulip2_cross*`) und ein falscher Wert
(`E2b_fullmesh_shape_only` 0.4858 statt 0.4956). Jetzt vollständig aus den
`metrics_summary.json` generiert.

**4. Zwei kleinere Inkonsistenzen** im Stage-1-Artefakt: ein überzähliges `</section>` schloss
§4 vorzeitig, und zwei Abschnitte trugen beide die Kennung C3.

### Methodisches
- „echtes CAD" als Begriff gestrichen (siehe AGREEMENTS) — Proxies sind auch CADs.
- „Schwaches Ranking" operativ definiert über die bedingte Top-1-Genauigkeit in der Shortlist.
  Die vorherige Formulierung war zirkulär und hielt der Rückfrage nicht stand, dass die
  aggregierten Retrieval-Werte beider Datensätze ähnlich sind.
- Stage-3-Tabelle: alle als „—" ausgewiesenen Werte existierten in den Ergebnisdateien
  (GT-Zeile normiert + je Datensatz, Geometrie-Zeile je Datensatz) und wurden nachgetragen.
