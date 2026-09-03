# OSCAR+ — Experiments 1–4 Implementation

*Thesis-input reference for the evaluation code. For each experiment it makes explicit:
**how the queries were produced**, **what was fed into the pipeline**, **which pipeline parts
were exercised**, and **what came out**. Pseudo-code mirrors the actual drivers; the metric and
config rationale live in `EVALUATION_STORY_AND_PLAN.md`, the pipeline itself in
`PIPELINE_IMPLEMENTATION.md`. Last updated 2026-09-01 — Experiment 4 vollstaendig (partial-Stufe, CLIP-Text, echtes Cache-Anhaengen, je ein Einstiegsskript).*

The three experiments map to the three evaluation stages and share one retrieval stack at an
**identical, audited configuration** (42 views · top-k-softmax k=5 · mean DINO pooling ·
weights 0.3/0.4/0.3 · τ=0.37 cascade parameter). The **only** intended per-experiment change is
the shape **mode** and the presence of a **pose** stage.

| Exp | Driver | Dataset | Query | Shape mode | Pose? |
|---|---|---|---|---|---|
| **1** | `experiments/experiment1_shrec18_stage1.py` | SHREC'18 (2101 q / 3308 CAD) | RGB-D scan | **pc** | no |
| **2** | `object_retrieval/retrieval_mi3dor_eval_oscarplus.py` | MI3DOR (~10.5k q / 3848 CAD) | monocular RGB | **cross** | no |
| **3** | `object_retrieval/eval_bop_pose.py` | BOP: YCB-V, T-LESS, LM-O (12,284 targets) | RGB-D crop | pc **and** cross | **yes** |

---

## 0. Shared execution model — Tier-1 / Tier-2

The retrieval grid is made affordable by separating **expensive per-channel scoring** from
**cheap fusion/geometry derivations**:

- **Tier-1 (score passes).** For each distinct encoder configuration ("pass") the per-query,
  per-gallery **channel score vectors** are computed once and cached (`scores_<pass>.pt`). A
  pass runs Steps 3–5 for the whole query set; the gallery side is already cached (§1.4 of the
  pipeline doc). Passes: `base` (CLIP+DINO), `siglip`, `ulip_pc_rgb` (shape pc), `ulip_cross_rgb`
  (shape cross), `ulip_pc_xyz`, `uni3d`, and the view-count variants `ulip_pc_rgb_v{8,16,32,42}`.
- **Tier-2 (derivations).** Fusion weights, fusion strategy, scope/pruning, and geometry are
  **cheap CPU derivations over the cached vectors** — the *production* `ScoreFusion` (Step 6) is
  reused, not re-implemented. This is why the weight sweep and the paired significance test cost
  almost nothing: the score vectors and per-query metrics already exist on disk.

```
# one pass, cached once:
scores[pass][q] = { "clip": (|G|,), "dino": {V: (|G|,)}, "shape": (|G|,) }   # Tier-1
# every ablation arm is then a Tier-2 read:
ranking = ScoreFusion(weights, scope, geometry).derive(scores[pass])         # Tier-2
```

---

## 1. Experiment 1 — SHREC'18 Stage-1 retrieval (pc-mode)

**Role.** Establish and characterise the method on clean RGB-D scans with category + subcategory
GT and an official leaderboard metric set [Nguyen et al. 2025]. This is the one stage where
design choices are made; all others transfer the audited config.

### 1.1 Query production
Each of the **2101 queries** is a real RGB-D scan of an object. Preprocessing produces, per query:
the RGB **ROI crop** (Step 1 mask), and — because depth is present — a **partial query point
cloud** (Step 2 back-projection). Query renders/PNGs are regenerated deterministically
(`tools/regen_query_pngs.py`). Because SHREC'18 CADs are **unitless**, the geometry path uses
`--diam-scale 1.0` (no mm→m conversion).

```
for scan in shrec18.queries:                        # 2101
    mask, bbox = segment(scan.rgb)                  # Step 1
    roi        = crop(scan.rgb, bbox, mask)
    query_pc   = backproject(scan.depth * mask, K)  # Step 2 (pc-mode: depth present)
    yield Query(id=scan.id, roi=roi, pc=query_pc, gt_cat=scan.cat, gt_subcat=scan.subcat)
```

### 1.2 What is fed / which parts are used
- **S_text** ← `roi`  (Step 3, CLIP)
- **S_view** ← `roi`  (Step 4, DINOv2, 42 views, top-5-softmax)
- **S_shape** ← `query_pc`  (Step 5, **pc-mode**, ULIP-2 partial views)
- **Fusion** (Step 6) → ranking; optionally **Geometry** (Sub-step B2, GeDi→RANSAC) over the
  partial `query_pc` vs the gallery CAD geometry.
- No pose (Steps 7–8 unused).

### 1.3 What comes out
Per arm: a full ranking per query, and two metric families —
- **Official leaderboard:** nDCG, mAP, precision/NN, FT/ST, E, DCG [Nguyen et al. 2025].
- **Depth / top-1:** `NN_sub` (subcategory nearest-neighbour = hit@1), `NN_cat`, `MRR`, hit@k —
  the pose-relevance family that bridges to Stage-3.
- Per-query records `results_per_query.json` (id, nDCG, AP, NN_sub, MRR, top-10) — the input to
  the **paired significance test** (`object_retrieval/paired_significance.py`).

### 1.4 The ablation grid (organised by pipeline block)
Every design ablation is run **isolated** (single channel, weights 1-hot) so the changed
variable is the only signal, and **in fusion** where that tells a different story.

```
Block A (channel design, isolated):
  A1 DINOv2 vs SigLIP        A2 view-count {8,16,32,42}   A3 ULIP-2 vs Uni3D
  A4 partial vs full-mesh    A5 pc vs cross query mode    A6 XYZ+RGB vs XYZ
  A7 shape view-count {8,16,32,42}   (A7f: does the isolated gain survive fusion?)
Block B (fusion):
  B1 weighted-sum vs RRF     B2 weight-sensitivity simplex   B3 channel contribution + OSCAR
  B4 scope: full-DB vs visual-first vs text-first(OSCAR τ-cascade)
Block C (geometry, on the fused ranking):
  C1 geometry signal (none/fitness/unaligned/RANSAC/+ICP)   C2 shortlist depth K {5,20,50}
  C3 shape-vs-geometry redundancy
```

Driver invocation (comma-separated arms, or `--all`; geometry needs `--with-geometry` + dGeDi):
```
python experiments/experiment1_shrec18_stage1.py --all --with-geometry \
       --data-root <shrec18> --images-dir <renders> --desc-file <descriptions.json> \
       --results-root results_shrec18_v2_stage1_42v_k5
# each arm writes: <arm>/metrics_summary.json, <arm>/results_per_query.json
# aggregate:       stage1_summary.csv, stage1_summary_depth.csv
```

---

## 2. Experiment 2 — MI3DOR Stage-2 transfer test (cross-mode)

**Role.** Take the audited config to a **monocular** dataset with **no depth** and measure
generalisation + the depth-free penalty. No design choices are made here.

### 2.1 Query production
Each of the **~10,500 queries** is a single RGB image (21 categories). There is **no depth**, so
**no query point cloud** — the shape channel must fall back to **cross-mode** (encode the image).
Query ULIP embeddings are encoded once and cached (`ulip_query_cache_mi3dor.pt`), shared across
shape-source modes.

```
for img in mi3dor.queries:                          # ~10.5k, monocular
    roi = crop(img)                                 # object-centric
    yield Query(id=img.id, roi=roi, gt_cat=category_of(img))   # NO point cloud
```

### 2.2 What is fed / which parts are used
- **S_text** ← `roi` (CLIP), **S_view** ← `roi` (DINOv2).
- **S_shape** ← `roi` via **ULIP-2's image tower** (Step 5, **cross-mode**) — the same aligned
  space, scored against the **full-mesh** CAD gallery — the MI3DOR `*_partial.npz` are absent on this
  machine, so the shape channel falls back to full-mesh (logged). In cross-mode that is also the
  *better* reference (see EVALUATION_STORY_AND_PLAN §4.1).
- **Fusion** (Step 6). **No geometry** (retrieval-only benchmark, no query clouds). No pose.

### 2.3 Seven arms in one pass
The driver derives all seven from a single scoring pass (`eval_common.run_evaluation`), covering
the full-DB fusion and the OSCAR-style CLIP-pruned cascade:
```
clip_only · dino_only · ulip_only
clip+dino+ulip (full-DB fusion, w=0.3/0.4/0.3)                 # OSCAR+
oscar_maxview / oscar_softmax / clip_pruned_dino_ulip          # OSCAR τ=0.37 cascade
```
Optional additions (this work): a **full-mesh** shape arm (`MI3DOR_MODES=fullmesh`, A4-transfer)
and a **cross-mode weight-sensitivity heatmap** (`mi3dor_weight_sweep.py`, Tier-2 re-fuse of
cached channels with a BASE self-check FT≈0.682). Uni3D is **not** applicable (pc-only, no cross).

### 2.4 What comes out
MI3DOR/SHREC-08 convention: **NN, FT, ST, nDCG@2R, mAP, ANMRR** (top-k=15), per arm, plus the
per-query records reused by the significance test.
```
MI3DOR_DINO_POOLING=mean MI3DOR_MODES=partial \
python retrieval_mi3dor_eval_oscarplus.py       # writes <mode>/results_topk_15.json
```

---

## 3. Experiment 3 — BOP Stage-3 retrieval → pose (pc and cross)

**Role.** The downstream question: is the retrieved CAD good enough to **pose** the object in a
cluttered real scene? Evaluated on `test_targets_bop19.json` for YCB-V, T-LESS, LM-O.

### 3.1 Query production
Each of the **12,284 targets** is an RGB-D crop. **GT visible bbox + mask** are used to crop, so
retrieval/pose are **isolated from segmentation error**. Depth gives both a pc-mode query cloud
and the input to FoundationPose. The retrieval gallery is a **union** across datasets with
**namespaced ids** (`<ds>/obj_*`), assembled from the precomputed per-dataset caches
(`stage3_gallery.assemble_gallery`):
- **3a:** `G_proxy ∪ G_target` — proxies + the exact target CADs present (retrieval can find self).
- **3b / 3c:** `G_proxy` only — exact targets removed; the top-1 is necessarily a **proxy**.
  `G_proxy = GSO(1030) ∪ HouseCat6D(199) ∪ ITODD(28)`.

```
for t in bop.test_targets(dataset):                 # 12,284
    roi      = crop_by_bbox(t.rgb, t.gt_bbox_visib, t.gt_mask)     # GT mask isolates seg
    query_pc = backproject(t.depth * t.gt_mask, t.K)               # pc-mode
    yield Target(key=t.key, roi=roi, pc=query_pc, K=t.K, gt_pose=t.pose, obj=t.obj_id)
```

### 3.2 What is fed / which parts are used, per mode
- **Retrieval (3a):** Steps 3–6 over the union gallery. **cross** (roi → ULIP image tower) or
  **pc** (query_pc → ULIP pc). Ranking = fused top-1; optional geometry (`--dgedi`).
- **Pose (gt / 3b / 3c):** the retrieved CAD + RGB-D + GT mask + K → **FoundationPose** (Step 8,
  ICP fallback). Proxies posed at **true metric size** (Step 7 unit conversion, no learned scale).

### 3.3 The four settings
```
3a  retrieval into G_proxy ∪ G_target   -> Recall@{1,5,10}, MRR (relevant = exact target)
gt  pose the object's OWN CAD           -> D_sym  (oracle upper bound on the pose stage)
3b  pose the top-1 PROXY (G_proxy)      -> D_sym  + Delta = D_sym - D_sym(gt)
3c  pose the next-best-non-self CAD     -> decomposes 3b: gallery-foreignness vs substitution loss
```
`D_sym` (symmetric surface discrepancy): sample N points (fixed seed) on the complete surfaces
of the GT-posed target and the estimated-posed retrieved model in the camera frame,
`D_sym = ½(D_T→P + D_P→T)`, in mm and /diameter, with F-score@{1%,5%}. `D_sym` is used (not
ADD/BOP-AR) because ADD compares two poses of the *same* model — undefined for a non-identical
proxy.

### 3.4 Experiment variants (this work)
```
--oscar-baseline : E5 — CLIP-τ prune + DINOv2 cascade, NO shape (ranked by oscar_maxview).
                   Answers "does the OSCAR+ shape channel's retrieval gain reach pose?"
--uni3d          : A3-transfer — Uni3D shape arm (pc-only).
--fullmesh       : A4-transfer — whole-mesh ULIP references (verify absorbed counts).
--dgedi          : geometry re-rank of the fused top-K (expected net-negative for pose).
--pc-query       : pc-mode shape (else cross-mode image query).
```
Invocation (per mode; 3c reuses a stored 3a ranking):
```
python eval_bop_pose.py --datasets all --mode 3a [--pc-query|--oscar-baseline|--uni3d|--fullmesh]
python eval_bop_pose.py --datasets all --mode gt        # oracle pose benchmark
python eval_bop_pose.py --datasets all --mode 3b --gt-records <gt-dir>
python eval_bop_pose.py --datasets all --mode 3c --from-3a <3a-dir>
# writes: <ds>_stage<mode>/{records.json,summary.json}, combined_stage<mode>.json
```

### 3.5 What comes out
- **3a:** Recall@{1,5,10}, MRR (± geometry), per-dataset.
- **gt/3b/3c:** D_sym (median/mean, /diameter), F@{1%,5%}, Delta; the 3c provenance split
  (real-CAD-of-another-object vs proxy) that decomposes the substitution cost.
- Stored raw poses → every pose metric reproducible offline.

---

## 4. Experiment 4 — onboarding and query latency

Experiments 1–3 answer *how well*. Experiment 4 answers *at what cost*: what it takes to make a
new object findable, and how long one query takes end to end. Both are wall-clock measurements
with the same gallery as Stage 3, so the numbers sit next to the accuracy numbers without
re-explaining the setup.

**Measurement harness** (`experiments/stage4_common.py`). CUDA kernels are asynchronous, so a
bare `perf_counter()` around an encoder call measures time-to-enqueue, not time-to-finish —
every measurement synchronises before and after. Statistics are **median + IQR + p95**, not
mean + stdev: latency distributions are right-skewed, a single outlier (swap, cache miss,
thermal throttling) moves the mean and not the median, and for an interactive system the bad
case is the relevant quantity. Repeated measurements of one step within one item are summed
before aggregation, so the statistics run over items and not over individual calls.

### 4.1 Experiment 4a — onboarding a CAD

**What is onboarded.** Base gallery = the **3b database** (`G_proxy` = gso 1030 + housecat6d 199
+ itodd 28 = 1257). The **59 target CADs** (ycbv 21 + tless 30 + lmo 8) are exactly the objects
that database is missing, so each is onboarded individually and the distribution over the 59
real CADs is the result. Vertex count, face count and file size are recorded per object so the
spread is explainable rather than just reported.

**Steps, each timed separately** — and within the encode steps, I/O is separated from
computation, otherwise `embed_dino` silently contains PNG decoding, which has nothing to do
with the encoder and scales differently on other hardware:

```
mesh        trimesh load → merge_vertices → fix_normals → bbox diameter
render      blender -b -P rendering/rendering.py   with RENDER_ONLY=<obj>, NUM_VIEWS=V
partial     generate_partial_pointclouds.py        (HPR per view)
describe    generate_descriptions.py               (LLaVA, one caption per view)
io_load_images / embed_dino    DINOv2 over the V renders
embed_clip                     CLIP text over the V captions (_encode_texts_batch)
io_load_clouds / embed_ulip    ULIP-2 over the V partial clouds
dgedi                          GeDi descriptors (optional; only if geometry is used)
cache_load / insert / save     the simulated append-only cache
```

SYNC and VERIFY (rclone to Drive, `PREPROCESSING.md` §1 steps 5–6) are deliberately out —
network time, not a property of the pipeline.

Three subtleties that cost measurements before they were caught, all on 2026-09-01:

- **`partial` needs the camera matrices.** `generate_partial_pointclouds.py` discovers its
  views from `<obj>_viewN_CamMatrix.npy`; without them it finds zero views and returns in
  milliseconds. It is also driven by `--images_dir`, resolving meshes through the
  `<cad_dir>/<obj_id>/` convention — **not** `--mesh-glob`, which is a standalone pattern
  keyed by *file stem* and would map every ycbv object onto `textured_simple`.
- **CLIP text goes through `_encode_texts_batch`**, the path `load_descriptions` itself
  uses. There is no public `encode_text`, and a per-string loop would measure something
  the pipeline never does — batching makes the step flat in V (4.6 ms at 16 as at 42).
- **Sub-process stages check their output, not their return code.** `render` counts the
  images produced (Blender exits 0 on Python errors), `partial` counts the clouds. Both
  had reported success while producing nothing.

Renders go to a **work directory**, never into `object_images/` — the experiment must not
overwrite the gallery it is measuring against.

**The cache-invalidation finding.** `_get_partial_cache_path` fingerprints the *entire*
inventory (one line per object per view). Adding one object changes the hash and invalidates
everything, so onboarding really costs **O(gallery)**, not O(1). Two numbers are therefore
reported:

- **incremental** — encode only the new object's views with models already loaded. This is
  measured directly, not simulated: it is exactly the work an append-only cache would do.
- **invalidation surcharge** (`--measure-invalidation`) — the full-gallery re-encode the current
  fingerprint forces on top.

Model load time is reported apart from both; it is a system start-up cost, not an onboarding cost.

### 4.2 Experiment 4b — query latency

**Query production.** N BOP instances drawn with a fixed seed. The language prompt is the
target object's **first stored description** — deliberately the same source the text channel
uses, because a hand-written prompt would make the segmentation better or worse than the system
could manage in operation and would mix a quality question into a latency measurement.

```
io_load        RGB + depth read and decode, K and depth_scale from scene_camera.json
segment        ObjectLocalizer.localize(rgb, prompt)   GroundingDINO box → SAM2.1 mask
pointcloud     backproject_masked(depth·scale, mask, K)
encode_query   ULIP-2 over the query cloud
clip / dino / ulip / fusion    run_query(...)          one pass, channels timed individually
geometry       GeDi + RANSAC over top-K              (only with --geometry, K=5)
pose           FoundationPose on the top-1 CAD       (unless --no-pose)
```

The per-channel timings come from wrapping `retrieve` / `rerank` / `match` / `fuse` on the
component objects **inside the experiment script**. `run_query` executes all channels in one
pass, and the modules are shared by all four experiments — a measurement run must not modify
them.

**Cold and warm are separated.** Loading GroundingDINO, SAM2.1, CLIP, DINOv2, ULIP-2 and
FoundationPose costs a multiple of one query; a figure that mixes both only reports how many
queries were averaged.

### 4.3 The 16-vs-42-view cost/benefit

Stage 1 already measured the quality side, and it is flat past 16 views:

| views | nDCG (Stage-1 O4) | vs V42 |
|---|---|---|
| 8 | 0.5714 | −0.0154 |
| 16 | **0.5820** | −0.0048 |
| 32 | 0.5800 | −0.0068 |
| 42 | 0.5868 | — |

V32 lands *below* V16, so beyond 16 views there is no reliable gain. Both scripts accept a
view-count list (`--num-views 16,42` / `--views 16,42`) and print a cost/benefit table with the
measured cost next to this quality column. On the query side the sweep is nearly free: gallery
embeddings are always cached for all 42 views and `_apply_view_limit()` only filters — nothing
is re-encoded. Steps that do not depend on view count (`mesh`) are excluded from the comparison,
since their difference would be pure measurement noise.

### 4.4 One entry point per side

```bash
bash scripts/stage4_onboarding.sh          # all 59 CADs, 16 and 42 views
bash scripts/stage4_query.sh               # ycbv, 50 queries, 16 and 42 views
```

Both print one table: a row per step, view counts as columns, the total, a cost/benefit
row against the Stage-1 quality, and cold start reported separately. The onboarding
wrapper splits across two environments — Blender lives outside the compose mount and runs
on the host, encoders and LLaVA need the container — and merges the results.

Do not measure with `-n 4`: after two warm-ups only two scored samples remain, and a
median of two flips on a single outlier. From `-n 50` the IQR stays under 10 ms.

### 4.5 What comes out

```
results_stage4/onboarding.json      per_step stats by view count, per-object records with
                                    vertex/face counts, model_load_once_s, invalidation
results_stage4/query_latency.json   per_step stats by view count, cold_start_s, per-query
                                    records with prompt and top-1
```

Both payloads carry `provenance` (GPU model, VRAM, torch version, git commit) — a latency figure
without the machine it was produced on is not citable.

---

## 5. Validation instrumentation (cross-experiment)

- **Paired significance test** — `object_retrieval/paired_significance.py`: pairs
  `results_per_query.json` by query id and reports, per comparison, the mean Δ and the
  **per-query win split** on nDCG and hit@1. The split is what separates a broad advantage
  from one carried by a few outliers — the mean alone cannot. Runs on the Stage-1 folder
  (and cross-folder for the config-change delta); `paired_significance_stage3.py` does the
  same for BOP, paired on the instance key.
- **BASE self-checks** — every re-derivation validates a known anchor before emitting results
  (Stage-1 `A7 V16==0.5256`; MI3DOR sweep `FT@(0.3,0.4,0.3)≈0.682`), guarding against the
  silent-misconfiguration bug class (wrong cache, degenerate channel, mis-set weight).

---

## References
See `PIPELINE_IMPLEMENTATION.md` §References. Experiment-specific: SHREC'18 / SHREC 2025 track
[Nguyen et al. 2025 `nguyenSHREC2025Retrieval2025`]; BOP toolkit and the YCB-V / T-LESS / LM-O
datasets (BOP challenge); MI3DOR monocular image-based 3D object retrieval benchmark; OSCAR
cascade baseline [Pulli et al. 2025 `pulliOSCAROpenSetCAD2025`]; FoundationPose [Wen et al. 2024].
