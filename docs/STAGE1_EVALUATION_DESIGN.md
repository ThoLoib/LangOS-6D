# Stage-1 Evaluation Design (SHREC'18 ObjectNN+)

Status: agreed 2026-07-31. Supersedes the metric/re-ranking parts of
`docs/THESIS_ALIGNMENT_PLAN.md`. Implemented in
`experiments/experiment1_shrec18_stage1.py`.

This document records *why* Stage 1 is evaluated the way it is. The short
version: the official SHREC'18 scalar metrics are mathematically incapable of
seeing a top-K re-ranking stage, so the geometry ablations need a second,
depth-matched metric table to be measurable at all.

---

## 1. The problem: official metrics cannot see the geometry stage

`eval/shrec18_official/evaluate.py` reads a submitted list with
`cutoff = 1000`, then cuts every scalar metric at `f = freqs[category]` — the
number of CADs in the query's category, ~165 on average. Inside
`metrics.py`, all but two metrics are a `np.count_nonzero` over that
fixed-length prefix:

```python
def precision(x):    return np.count_nonzero(x) / len(x)
def recall(x, freq): return np.count_nonzero(x) / freq
def nnt1(x, k):      return np.count_nonzero(x[:k]) / k
def nnt2(x, k):      k2 = min(len(x), k * 2); return np.count_nonzero(x[:k2]) / k2
```

Geometry re-ranking is a **permutation of the first K entries**, K << f. A
permutation cannot change how many relevant items fall inside the first 165.
Therefore `precision`, `recall`, `f1score`, `nnt1` and `nnt2` are
**algebraically invariant** to the entire B2 stage — not approximately, exactly.

This was confirmed empirically before it was derived: six E2 arms reported
`precision = 0.26864481953094016` to the last digit, including
`chamfer_unaligned`, which is a deliberately *worse* control.

Only `dcg` and `average_precision` are position-sensitive. `dcg` also carries a
known off-by-one (`total = x[0]`, then the loop adds `x[i]` again at `i = 0`,
double-counting rank 1 and dropping the last element). It is used **unpatched**
— see the 2026-07-30 entry in `DECISIONS.md`; comparability with the published
leaderboard beats correctness, and `dcg`/`idcg` share the defect so the ratio
stays monotone.

How much leverage a top-K re-rank has over nDCG, under the official weights at
f = 165 (total weight mass 31.03):

| depth | cumulative DCG mass |
|------:|--------------------:|
|     1 |  6.4 % |
|     5 | 12.7 % |
|    10 | 17.9 % |
|    20 | 25.9 % |
|    50 | 44.8 % |
|   100 | 70.7 % |

At the old `GEOM_SHORTLIST = 5` the geometry stage was competing for 12.7 % of
the DCG, and only the fraction of that it actually reordered.

**Consequence:** report the invariance as a finding, not as a null result. The
thesis states that the official scalar metrics are order-insensitive within the
evaluated prefix and are reported for comparability only.

---

## 2. Two tables

### Table A — official SHREC'18

`score_official()`, using `eval/shrec18_official/metrics.py` verbatim, graded
relevance (subcategory = 2, category = 1), cut at `f`, cutoff 1000. Every arm
appears here. Geometry arms tie on P/R/F1/NNT1/NNT2 **by construction**; the
caption says so. nDCG remains the model-selection metric, tie-break mAP.

### Table B — depth-matched retrieval quality

Computed at the geometry depth K, for **every** arm (it is post-hoc arithmetic
over cached score vectors, so it costs nothing for the non-geometric cells):

| metric | definition |
|---|---|
| `NN_cat`  | rank-1 item shares the query's category (rel ≥ 1) |
| `NN_sub`  | rank-1 item shares category **and** subcategory (rel = 2) |
| `hit@K`   | at least one relevant item in the top K, per grade |
| `MRR@K`   | reciprocal rank of the first relevant item within K |
| `mAP@K`   | average precision recomputed on the K-list, not the f-prefix |
| `nDCG@K`  | **corrected** DCG (no off-by-one), normalized |

`NN_sub` is the **headline** for the geometry arms. Language largely solves
category assignment; the open question is whether geometry picks the right
*variant* within it. That is also the harder target, so it has headroom the
category-level numbers do not.

Note the two nDCGs are deliberately different quantities: Table A's is the
official buggy one over the f-prefix, Table B's is corrected and cut at K.
They are never placed in the same column.

---

## 3. Choosing K

K is not a taste parameter. Before any geometry run, the script reports
base-fusion **hit-rate@K** for K ∈ {1, 5, 10, 20, 50, 100} at both relevance
grades. That curve is a hard ceiling: geometry re-ranks, it never inserts, so
no geometry arm can exceed base hit-rate@K on any top-1 metric.

**Selection rule:** the smallest K whose hit-rate@K is within **2 percentage
points** of hit-rate@100 (`choose_geom_k`, `HITRATE_TOL`). Concrete,
reproducible, and reported in the thesis alongside the curve. The curve is read
at the **sub-category** grade, since `NN_sub` is the headline the K is meant to
give headroom to.

> **Measured 2026-07-31 — the rule degenerates on this data.** The curve does
> not flatten anywhere below K = 100:
>
> | K | hit_cat@K | hit_sub@K | registrations | RANSAC time |
> |---:|---:|---:|---:|---:|
> | 1 | 0.4217 | 0.3503 | 2,101 | 0.25 h |
> | 5 | 0.6540 | 0.5526 | 10,505 | 1.3 h |
> | 10 | 0.7549 | 0.6497 | 21,010 | 2.5 h |
> | 20 | 0.8439 | 0.7463 | 42,020 | 5.0 h |
> | 50 | 0.9386 | 0.8639 | 105,050 | 12.5 h |
> | 100 | 0.9767 | 0.9229 | 210,100 | 25.1 h |
>
> Because it is still climbing at the deepest measured K, "within 2 points of
> the deepest K" simply returns that deepest K. The rule presupposes
> saturation, and the presupposition is false here — so it must not be quoted
> as if it selected K on evidence.
>
> **K is therefore a stated compute-budget decision**, reported next to the
> curve rather than derived from it. `--bench-rerank N` measures the real
> per-pair cost on the target machine and prints the K → wall-clock and
> marginal-return tables; run it before committing.
>
> Measured 2026-08-01 (CPU-only dev box, warm descriptor cache, 16 pairs over
> 4 queries spanning 6k–375k points; per-pair cost varied only ±11% across
> that range, so the projection is stable):
>
> | | s/pair | K=20 | K=50 | K=100 |
> |---|---:|---:|---:|---:|
> | ICP on | 3.29 | 38 h | 96 h | 192 h |
> | **ICP off** | **0.93** | **11 h** | **27 h** | **54 h** |
>
> **ICP is 68 % of the per-pair budget** — measured, not modelled; two earlier
> estimates of this (0.43 s/pair from RANSAC alone, then 2.9 s/pair from
> component arithmetic) were both wrong, which is why the benchmark exists.
>
> Marginal `hit_sub` per hour with ICP off: 10→20 = 1.79 pts/h,
> 20→50 = 0.73, 50→100 = 0.22.
>
> Note `O1c_gedi_post_fusion` shortlists on text+view rather than BASE fusion,
> so it needs its **own** full set of registrations and none of the shared
> cache helps it — it doubles whichever figure above is chosen.

The curve costs nothing to produce: `score_depth_matched` emits `hit_cat@N` /
`hit_sub@N` for every arm, so running BASE yields it as a by-product.
`--hit-rate-curve` prints it and the K it implies.

K is exposed as `--geom-k` (previously the hardcoded `GEOM_SHORTLIST = 5`, now
only the fallback when no BASE summary exists). Omitting the flag derives K
from the curve automatically. It is fixed once; **no K sweep is reported.**
`aggregate()` warns if Table B ever mixes rows computed at different K, since
the `@K` columns would not be comparable across them.

Two invariants the implementation must hold:

1. The shortlist entering B2 is **K distinct objects**. Fusion can surface
   several views of one object; they are deduplicated to object level first.
2. Positions K+1 … 1000 keep base-fusion order. The submitted list is always
   padded to the full ranking — never truncated to K, which would make
   `precision(x)` divide by K and produce a meaningless inflated number.

---

## 4. Geometry configuration

**Registration target: full CAD.** The query is registered against the full
sampled CAD cloud, not against the winning partial view. This matches how a
retrieved CAD is used downstream (as a complete model for pose estimation) and
avoids the 42-clouds-per-object cost.

**Scale is ignored at Stage 1.** SHREC'18 queries are metric SceneNN crops;
ShapeNetSem CADs use arbitrary model units. Both clouds are normalized to the
unit sphere. Metric scale is recovered downstream at pose estimation from
depth, so the retrieved CAD is a **shape proxy whose metric scale is
estimated**, never read from the file. Stage 3–5 BOP metrics (VSD/MSSD/MSPD,
ADD/ADD-S) operate in millimetres on the *posed* object and are unaffected by
this choice. Practical corollary: Stage-1 thresholds are expressed
**relatively**, Stage 3+ thresholds **absolutely** — which makes the stage
boundary explicit instead of smuggling an absolute constant across it.

**Inlier threshold.** `step_b2_geometry_reranking.py:467` uses
`max_correspondence_distance = voxel_size * 1.5`, the Open3D global-registration
convention (Zhou, Park & Koltun, *Open3D*, 2018). With `GEOM_VOXEL = 0.02` on
unit-sphere clouds (diameter 2) that is 0.03, i.e. **1.5 % of object diameter**,
with the voxel itself at 1 % of diameter.

The threshold is *derived from point-cloud resolution*, not chosen as a
percentage. This matters for defensibility: the registration literature anchors
inlier distances to sensor/scene resolution, not to object diameter — 3DMatch
uses τ₁ = 0.10 m on ~3 m fragments (Zeng et al., CVPR 2017), KITTI 0.30 m on
~80 m scans — so those ratios (~3 % and ~0.4 %) are not transferable constants.
The GeDi paper itself (Poiesi & Boscaini, TPAMI 2022, Fig. 2) treats the inlier
distance as a swept evaluation axis rather than a fixed value. The reported
%-of-diameter is stated as a derived consequence, not as the setting.

Sanity evidence at this setting: **0 registration failures in 190 pairs**, mean
fitness 0.340 — a well-spread, non-saturated distribution. A looser threshold
pushes mean fitness toward 0.6+, where it compresses and stops discriminating.

**ICP is off by default.** ICP adds a refinement degree of freedom that can
partly launder a wrong retrieval into a plausible fit — the exact confusion to
avoid in a *retrieval* evaluation. `E2_chamfer_icp` is retained as an explicit
arm, because that arm is the evidence for the default rather than a competitor
to it.

**`E2_scalegate` is dropped.** The legacy scale gate used an arbitrary 1000×
factor corresponding to no thesis config; it was removed deliberately (see
`step_b2_geometry_reranking.py` lines 86/404) and is not resurrected.

### Combining fitness and trimmed Chamfer

`E2_both` needs one ranking from two incommensurable numbers: RANSAC `fitness`
∈ [0, 1] (higher better) and `d_trim` ∈ [0, ~0.2] with a long tail (lower
better). Adding them raw lets the wider-spread signal silently win — on
representative data, `fitness + (−d_trim)` reproduces the fitness-only ranking
exactly, so the "fusion" would be no fusion at all.

**Rule: mean rank (Borda count), ties averaged** (Aslam & Montague, SIGIR 2001).
Rank the K candidates by fitness, rank by −d_trim, order by the mean of the two
ranks. Scale-free, **no free parameter**, robust to the Chamfer tail.

RRF (Cormack et al., SIGIR 2009) was considered and rejected *as the geometry
rule*: its `k = 60` was calibrated on TREC lists of thousands of documents, and
at K = 20 the weights span only 1/61 … 1/80 — a 1.3× spread — so it collapses
into mean rank anyway. Citing a constant that is provably inert would be hollow.
(`E6_rrf` is unrelated: that arm fuses the three *channels*, a different
question.) A z-normalized weighted sum was rejected because the weight would
have to be tuned, and tuning it on the evaluation queries is a tuned-on-test
problem. **The fusion rule is fixed, not ablated.**

### Shortlist source

All geometry arms re-rank the **BASE full-fusion shortlist** (`E1c_full_fusion`).
Geometry is thereby isolated as a re-ranking stage over a fixed input, and each
(query, CAD) registration is computed exactly once and shared across arms.
Geometry is *not* crossed with the encoder arms (E4, E7).

### The full-DB asymmetry

CLIP / DINOv2 / ULIP-2 score all 3,308 CADs per query — cached embeddings, a
matrix multiply, milliseconds. GeDi + RANSAC costs **0.430 s per pair**
(measured, `results_gedi_large_19x10`). Full-database geometry would be
2,101 × 3,308 ≈ **6.95 M registrations ≈ 830 h ≈ 35 days**, per cell.

So geometry can only ever exist on a shortlist. This is a hard constraint, not
a design preference, and the thesis states it. `--bench-gedi` measures the
per-fit cost that backs the claim.

Consequently the arm formerly specified as "S_GeDi replaces S_shape inside the
fusion score" (O1e) is **not** implementable as written. What is implemented
instead, under an honest name, is the shortlist-level question: within the top
K, does the base fusion score still carry information once geometry is
available?

- `E2_both` — discard the base score inside the shortlist; order by geometry
  alone (Borda of fitness and d_ransac).
- `O1e` — keep the base fusion rank as a **third** Borda signal alongside them.

Both use the *same cached registrations*, so `O1e` costs zero extra compute.
Note the framing this inverts: `E2_both` is the aggressive arm (a CAD ranked
first by all three channels gets no credit if RANSAC disagrees), and `O1e` is
the conventional cascade that keeps the earlier-stage score.

---

## 5. OSCAR pruning (O2)

Pulli et al. (2025) §3.2 prune by **threshold, not top-k**:

> form a candidate set 𝒮′ by selecting all objects with
> sim_text(sᵢ) ≥ τ_text (τ_text = **0.37**). If 𝒮′ is empty, fall back to the
> top-k text candidates.

The candidate set is therefore **variable-sized**, and top-k is only the
fallback. Three discrepancies were found in the repo:

1. `pipeline/config.py:74` sets `clip_threshold = 0.25`, not 0.37, with a stale
   comment claiming the mechanism is *"noch nicht implementiert"* — it is
   implemented, at `step3_clip_retrieval.py:323-338`.
2. `threshold` defaults to `None` in `retrieve()`, so the shipped pipeline runs
   **pure top-k = 20**; the paper's mechanism is dormant.
3. `step3_clip_retrieval.py:337` caps the threshold result at `top_k` anyway.
   The paper describes no such cap.

**Decision: reproduce the paper's mechanism at τ = 0.37** (not the config
default of 0.25, not top-20).

Reporting, because a variable-length list breaks `precision(x)`, which divides
by the *submitted* list length and so inflates on short lists:

- `O2_clip_cascade_padded` — 𝒮′ first, then the remaining gallery in
  base-fusion order to fill the list. Ordinary row in Table A, comparable to
  every other arm.
- `O2_clip_cascade_faithful` — submit 𝒮′ as-is. **Not a table row.** Reported
  as a short paragraph: median |𝒮′|, IQR, fallback rate, and NN/MRR.

τ = 0.37 was calibrated on MI3DOR/YCB-V caption-similarity distributions, so
the |𝒮′| statistics are also the check on whether it transfers to SHREC'18 at
all. If it selects ~3 or ~3,000 objects, the arm measures a mistuned constant
rather than the method, and the write-up must say so.

The existing top-20 cascade arms (`E1_oscar_cascade`, `E1d_clip_pruned`) are
kept — they are the fallback mechanism, and the contrast between them and the
threshold arm is itself informative.

**BASE is never pruned.** It stays full-database. Pruning BASE to ~165 would
hand P/R/F1 entirely to CLIP and make every downstream arm unmeasurable.

---

## 6. Query set and ground truth

All **2,101 official queries**, using the track's own `rgbd.csv` / `cad.csv`
(`eval/shrec18_official/`, `load_official_gt`) — real category *and*
subcategory labels for every query and all 3,308 CADs, the same GT the
published participants were scored against.

This supersedes the earlier plan's 1,452 reconstructed train queries. The
union-find reconstruction in `build_gt()` is retained only as a fallback when
the official kit is absent. There is no train/test split limitation to declare.

Statistical treatment for geometry arms, which differ from BASE only by a
K-element permutation and therefore leave most queries untouched:

- **paired** per-query deltas vs BASE on NN_sub / MRR@K / nDCG@K;
- **N_changed** — how many queries the re-rank actually moved, reported
  alongside the mean, because it is the effective sample size;
- bootstrap CI over paired deltas (or Wilcoxon signed-rank on the changed
  subset); per-category win/loss/tie counts.

Unpaired means would be swamped by the unchanged majority and show nothing.

---

## 7. Runtime budget (tessa-pc)

Measured per-cloud and per-pair costs from `results_gedi_large_19x10`:

```
cad_descriptor_time_s    mean 3.446
query_descriptor_time_s  mean 3.345
ransac_time_s            mean 0.430     <- the only per-pair cost
total_time_s             mean 4.277
```

Descriptors are ~90 % of the naive per-pair cost and are **per cloud**, so
precomputing them removes them entirely and makes K a nearly free knob:

| stage | work | time |
|---|---|---|
| descriptor precompute (one-time) | 3,308 CADs + 2,101 queries = 5,409 clouds | ~5.2 h |
| geometry eval, K = 20 | 42,020 registrations | ~5.0 h |
| *(K = 50 for reference)* | 105,050 | ~12.6 h |

Implemented as `--precompute-gedi`, backed by `PipelineConfig.gedi_cache_dir`
(default `None`, i.e. the previous uncached behaviour) and
`GeDiDescriptorModule.compute_and_cache`. Both the query cloud and the CAD
clouds go through it (`step_b2._cached_gedi`); previously the CAD descriptors
were recomputed on every pair (`step_b2:455`), which is what made a deep
shortlist unaffordable.

**Cross-machine constraint.** Descriptors are precomputed on `tessa-pc`;
development and smoke tests happen elsewhere. Entries are therefore validated
against a **SHA-1 of the input point array** rather than against recorded
settings. Hashing the cloud is deliberate: descriptors depend on
normalization, sampling mode, point count and cloud construction, and
enumerating those in metadata makes every new knob a silent-staleness bug in
waiting. The hash covers all of them at once and is exact.

This is not defensive padding — Uni3D/ULIP embeddings already mismatch
silently across the two machines when `pointnet2_ops` is installed on only one
(`docs/LAPTOP_EMBEDDINGS_SETUP.md`). The failure mode is a wrong number, not a
crash.

Three further properties, each from an observed failure in this project:
writes are atomic (`.tmp.npz` + `os.replace`) so a duty-cycle kill cannot leave
a truncated entry; **empty results are never cached**, so an unreachable GeDi
service cannot poison the cache the way it poisoned 2,845 pair scores on
2026-07-27; and a stale entry is reported as *stale* rather than *corrupt*,
because on the eval PC that distinction points at preprocessing drift between
machines rather than at a damaged file.

Partial population and resume are supported, so local smoke tests cannot
poison the production cache.

---

## 8. Summary of settled choices

| item | choice |
|---|---|
| Queries | all 2,101, official GT |
| Table A | official scorer verbatim, cutoff 1000, cut at f |
| Table B | NN_cat, NN_sub, hit@K, MRR@K, mAP@K, corrected nDCG@K — every arm |
| Headline (geometry) | NN_sub |
| K | from hit-rate curve; smallest K within 2 pts of hit-rate@100 |
| K sweep | no |
| Descriptor | GeDi only (no dGeDi) |
| Registration target | full CAD, scale ignored |
| Normalization | unit sphere |
| Inlier threshold | 1.5 × voxel_size = 1.5 % of diameter |
| ICP | off by default; `E2_chamfer_icp` retained as the arm that justifies it |
| Geometry arms | `E2_none`, `E2_fitness`, `E2_chamfer_unaligned`, `E2_chamfer_ransac`, `E2_chamfer_icp`, `E2_both`, `O1c`, `O1d`, `O1e` |
| `E2_scalegate` | dropped |
| Signal fusion | mean rank (Borda), ties averaged; fixed, not ablated |
| Shortlist | BASE full fusion, object-deduplicated; tail keeps base order |
| O2 threshold | τ = 0.37, paper-faithful; padded row + faithful paragraph |
| BASE scope | full database, never pruned |
| E2b | retained for the ULIP-2 shape channel only |

## Measured results (2026-07-31, all 2,101 queries, K = 5)

First run of the new code. K = 5 was used because the existing per-pair cache
was built at that depth; these are **not** the final numbers, but they validate
the design and are non-regressive — every Table A value reproduces the
2026-07-30 run exactly.

| arm | nDCG (A) | P (A) | NN_cat (B) | **NN_sub (B)** | MRR (B) |
|---|---:|---:|---:|---:|---:|
| `E1c_full_fusion` (BASE) | 0.5879 | 0.2686 | 0.4217 | 0.3503 | 0.5068 |
| `E2_fitness` | 0.6010 | 0.2686 | 0.4802 | 0.4065 | 0.5464 |
| `E2_chamfer_ransac` | 0.6028 | 0.2686 | 0.4836 | 0.4131 | 0.5492 |
| **`E2_both` (Borda)** | **0.6033** | 0.2686 | **0.4869** | **0.4141** | **0.5510** |
| `O1e_gedi_with_base` | 0.6014 | 0.2686 | 0.4802 | 0.4041 | 0.5472 |
| `O2_clip_threshold` | 0.4437 | 0.1319 | 0.3018 | 0.2285 | 0.3881 |

Four things this establishes:

1. **Table B does the job it was added for.** Geometry moves nDCG by +0.015 —
   easy to dismiss — but `NN_sub` by **+0.064 (0.3503 → 0.4141, +18 % relative)**.
   Same re-ranking, same queries; only the metric's ability to resolve it differs.
2. **The invariance claim is now measured, not just derived.** `P = 0.2686` is
   identical across all four full-fusion arms, and `hit_sub@5/@10/@20` are
   identical to BASE (0.5526 / 0.6497 / 0.7463) while `hit_sub@1` moves — a
   top-5 permutation changes rank 1 and cannot change top-5 membership.
3. **`E2_both` is the best geometry arm on both tables**, so the Borda
   combination earns its place; and **`O1e` < `E2_both`**, i.e. retaining the
   base fusion rank *hurts* once geometry is available. That is the opposite of
   the conventional-cascade intuition and is a clean, reportable answer.
4. **`NN_cat` is only 0.4217 at BASE** — the assumption that the language
   channel largely solves category assignment is **wrong** on SHREC'18. Both
   grades have headroom, so `NN_cat` is worth reporting on its own, not just as
   the easy contrast to `NN_sub`.

### OSCAR's τ does not transfer — and a calibrated τ that does

The per-query **maximum** CLIP similarity on SHREC'18 runs
min 0.2656 / p05 0.2949 / median 0.3289 / **max 0.4094**. OSCAR's
τ_text = 0.37 therefore sits above the 96.9th percentile of that
distribution: it clears on only **3.1 %** of queries.

| arm | τ | fallback | median &#124;𝒮′&#124; | IQR | nDCG | NN_sub |
|---|---:|---:|---:|---|---:|---:|
| `O2_clip_threshold` (paper) | 0.37 | 96.9 % | 20 | [20, 20] | 0.4437 | 0.2285 |
| `E1d_clip_pruned` (top-20) | — | — | 20 | — | 0.4452 | 0.2356 |
| `O2_clip_threshold_cal` | **0.2949** | 4.9 % | **176** | [29, 661] | **0.5078** | 0.2732 |
| `E1c_full_fusion` (no pruning) | — | — | 3308 | — | **0.5879** | 0.3503 |

Two separable conclusions, which is exactly why both arms exist:

1. **The constant does not transfer.** At τ = 0.37 the threshold mechanism
   essentially never runs, and the arm reproduces the top-20 cascade to within
   0.0015 nDCG. This is a negative-transfer result about a hyperparameter
   fitted to MI3DOR/YCB-V caption similarities, not a result about OSCAR's
   method.
2. **The mechanism itself does work — better than top-k, worse than no
   pruning.** Calibrated to this dataset (τ = 0.2949), |𝒮′| becomes genuinely
   variable (1 to 2,852, median 176) and nDCG rises to 0.5078, clearly beating
   every fixed top-k arm. It still loses to full-database fusion (0.5879),
   consistent with the standing finding that any shortlisting costs accuracy
   here.

τ_cal is fixed by a **coverage** rule — the 5th percentile of the per-query max
similarity, i.e. the highest threshold leaving 95 % of queries with a non-empty
set (`calibrate_tau`, `CLIP_TAU_FALLBACK_TARGET`). Calibrating on coverage
rather than on a target |𝒮′| keeps it from being a restatement of top-k, and
not calibrating on nDCG/NN keeps it off the evaluation metric.

### Reading the diagnostics

The diagnostic that catches a dead threshold is the **fallback rate**, not
median |𝒮′| — the median looks perfectly healthy (20) precisely when the
threshold is admitting nothing, because 20 *is* the fallback size.
`aggregate()` warns above a 25 % fallback rate.

### Reproducibility defect found and fixed

`sample_points_uniformly()` draws from Open3D's **global** RNG and (as of
0.19) takes no `seed` argument, so every run sampled a different cloud from the
same CAD mesh. Two consequences: geometry scores were not reproducible
run-to-run, and no CAD descriptor could ever be cached, since a re-sampled
cloud has a different fingerprint by construction.

Fixed by seeding `o3d.utility.random.seed()` **per object id** — not once
globally, which would make results depend on processing order — in both
`experiment1_shrec18_stage1.py` and `pipeline/step_b2_geometry_reranking.py`.
The seed is masked into the non-negative int32 range; Open3D's binding rejects
larger values, which silently failed for about half of all ids before the mask.

Measured effect on the descriptor cache: **1.94 s/cloud cold → 0.04 s/cloud
warm (48×)**, 12/12 clouds cached, 0 failures.

## Where each piece lives

| concern | code |
|---|---|
| Table A | `score_official()` → `metrics_summary.json["metrics"]` → `stage1_summary.{csv,tex}` |
| Table B | `score_depth_matched()` → `["metrics_depth"]` → `stage1_summary_depth.{csv,tex}` |
| graded relevance | `_graded_relevance()` (subcat = 2, cat = 1) |
| K selection | `load_hitrate_curve()`, `choose_geom_k()`, `--hit-rate-curve`, `--geom-k` |
| Borda fusion | `_average_ranks()`, `COMBINED_SIGNALS`, `apply_geometry()` |
| OSCAR threshold | `CLIP_TAU_TEXT`, `scope="clip_threshold"` in `derive_ranking()`; `["shortlist_stats"]` |
| calibrated τ | `calibrate_tau()`, `spec_tau()`, `CLIP_TAU_FALLBACK_TARGET`, `scope="clip_threshold_cal"` |
| descriptor cache | `PipelineConfig.gedi_cache_dir`, `GeDiDescriptorModule.compute_and_cache()`, `step_b2._cached_gedi()`, `--precompute-gedi` |
| deterministic sampling | `_cad_sample_seed()`, `o3d.utility.random.seed()` in both CAD loaders |
| cost measurement | `bench_rerank()` / `--bench-rerank N`; `--bench-gedi` answers the *different* question of full-DB feasibility |
| ICP switch | `PipelineConfig.geometry_skip_icp`, `--no-icp` (drops `E2_chamfer_icp` from the selection) |
| per-query records | `results_per_query.json` — carries `NN_cat`, `NN_sub`, `MRR`, `nDCG_K` so paired statistics need no re-run |

## References

- Pham et al., *SHREC'18: RGB-D Object-to-CAD Retrieval*, 2018 — benchmark,
  official metrics.
- Pulli et al., *OSCAR: Open-Set CAD Retrieval from a Language Prompt and a
  Single Image*, 2025 — §3.2 threshold pruning (τ_text = 0.37), K = 8 onboarding
  viewpoints.
- Poiesi & Boscaini, *Learning general and distinctive 3D local deep descriptors
  for point cloud registration*, TPAMI 2022 — GeDi; inlier distance as a swept axis.
- Zhou, Park & Koltun, *Open3D*, 2018 — `1.5 × voxel_size` correspondence convention.
- Zeng et al., *3DMatch*, CVPR 2017 — τ₁ = 0.10 m FMR protocol.
- Aslam & Montague, *Models for Metasearch*, SIGIR 2001 — Borda count.
- Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion…*, SIGIR 2009 — RRF
  (used for `E6_rrf` channel fusion; rejected for geometry-signal fusion).
- Shilane et al., *The Princeton Shape Benchmark*, 2004 — NN/FT/ST/E-measure.
