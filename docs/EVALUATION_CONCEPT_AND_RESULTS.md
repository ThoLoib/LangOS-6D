# OSCAR+ Evaluation — Concept, Results, Changes, and Open Items

**Status snapshot:** 2026-08-25. This document explains the full three-stage
evaluation of OSCAR+ *as actually implemented in the code*, the results obtained
so far (including superseded intermediate numbers), every change made during the
evaluation campaign and why, the conclusions those results support today, and
what is still open. It is deliberately exhaustive so it can seed the thesis
`evaluation` chapter and serve as the authoritative internal record.

Scope note: the numbers below are the current on-disk results. Three jobs were
still running when this was written and are marked **[running]**: the Stage-1
dGeDi geometry K-sweep, the fair SigLIP re-run (chained after it), and the
fusion-weight sensitivity sweep.

---

## 0. The one-paragraph version

OSCAR+ is evaluated in three stages that form a logical chain: **(1)** tune and
ablate the retrieval pipeline on a clean category-retrieval benchmark
(SHREC'18), freeze the winning configuration; **(2)** test whether that frozen
configuration transfers to a harder, monocular cross-dataset retrieval task
(MI3DOR); **(3)** measure whether the retrieved CAD is actually *useful for
downstream 6-DoF pose estimation* on real cluttered RGB-D scenes (BOP: YCB-V,
T-LESS, LM-O), including the realistic case where the exact CAD is unavailable
and only a *proxy* can be retrieved. The design deliberately isolates the
variable of interest at each step (ground-truth masks/boxes remove segmentation
noise in Stage 3; paired galleries isolate the cost of proxy substitution). The
headline findings so far: fusion beats every single channel; image-query
("cross") retrieval is best except on texture-less objects; geometric
re-ranking *helps category retrieval but is net-negative for pose*; exact-CAD
pose is excellent (~1.7 mm median) while a retrieved proxy costs ~16 mm median,
about half of which is "the proxy pool is too foreign" and half "substituting
any non-self CAD is intrinsically lossy."

---

## 1. Design philosophy and shared infrastructure

### 1.1 The three-stage arc

| Stage | Dataset(s) | Task | Question answered |
|---|---|---|---|
| **1** | SHREC'18 ObjectNN+ | Category/subcategory retrieval (2,101 RGB-D queries vs 3,308 CADs) | Which pipeline configuration retrieves best? (tune + ablate + **freeze**) |
| **2** | MI3DOR | Monocular image→CAD retrieval (10,500 queries) | Does the frozen Stage-1 config transfer to a different, image-only task? |
| **3** | BOP: YCB-V, T-LESS, LM-O | Retrieval **+ 6-DoF pose** on cluttered RGB-D (12,284 instances) | Is the retrieved CAD good enough to *pose* the object — even when it's only a proxy? |

The premise is "tune once, freeze, transfer." Stage 1 is the tuning stage;
Stages 2 and 3 reuse the frozen configuration unchanged so that any drop in
quality is attributable to task/dataset difficulty, not re-tuning. (This premise
is partly challenged by the results themselves — see §7.3.)

### 1.2 The pipeline being evaluated

OSCAR+ fuses three complementary channels into one ranking:

- **S_text** — CLIP text–image similarity (language grounding of the query
  against per-CAD text descriptions).
- **S_view** — DINOv2 multi-view appearance similarity (42 rendered views,
  top-k-softmax view aggregation).
- **S_shape** — ULIP-2 3D shape similarity. In Stage 1 this runs in **pc-mode**
  (the query point cloud is encoded, since SHREC queries are real RGB-D scans);
  in Stage 2/3 it can also run **cross-mode** (the query *image* is encoded by
  ULIP's image branch, since MI3DOR has no depth).

Fusion is a weighted sum with min–max normalisation, weights **(w_text, w_view,
w_shape) = (0.3, 0.4, 0.3)** (`ScoreFusion`, `pipeline/step6_fusion.py`). An
optional **geometry re-ranking** sub-step re-orders the fused top-K by 3D
registration quality (see §2 and §4.2).

### 1.3 Tier-1 / Tier-2 execution model (why the ablation grid is affordable)

The Stage-1 driver (`experiments/experiment1_shrec18_stage1.py`) splits work:

- **Tier 1 — channel-score passes** (expensive, GPU): each pass computes, per
  query, a full score vector over the gallery for one (encoder × reference)
  combination and caches it under `<results>/_cache/scores_<pass>.pt`. The
  appearance channel stores one vector per view budget, so the "number of views"
  ablation needs no extra encoding.
- **Tier 2 — derivations** (cheap, CPU): fusion weights/method, candidate
  scoping, and geometry re-ranking are post-processing of the cached vectors.
  Crucially, **fusion is not re-implemented** — the cached vectors are wrapped in
  the pipeline's own dataclasses and fused by the production `ScoreFusion`, so
  every ablation exercises the real fusion code.

This is why, e.g., the fusion-weight sensitivity sweep (§3.4) and the paired
significance test (§8) cost essentially nothing: the expensive vectors already
exist on disk, and per-query metrics are persisted for post-hoc analysis.

### 1.4 Determinism policy

All explicit RNGs are seeded (`PYTHONHASHSEED=0`, numpy/torch seeds, seeded
surface sampling). Two sources are **not** bit-reproducible and are documented
rather than hidden: FoundationPose's GPU pose-hypothesis sampling, and open3d
RANSAC on builds that ignore the seed. The mitigation is to **store the raw
estimated poses** in every `records.json`, making all pose metrics reproducible
from the stored poses even though a re-run of the estimator would differ
slightly.

### 1.5 Gallery preprocessing

Point clouds for the shape channel are partial rendered views (SAM-6D style)
generated with hidden-point-removal; the global preprocessing default was
unified to **HPR radius 2.8 / jitter 0.001** across datasets during this campaign
(§6). Legacy galleries built with older parameters are pinned to their original
values to keep already-computed caches valid.

---

## 2. Stage 1 — SHREC'18 retrieval tuning (the ablation grid)

### 2.1 Concept

Stage 1 runs the full retrieval-side ablation grid on SHREC'18 ObjectNN+ and
selects the best configuration by **graded nDCG, tie-break mAP**, scored with the
track's own official metric code (leaderboard-comparable). The winner is frozen
for Stages 2/3/5. Relevance is category/subcategory-graded (the ObjectNN+
protocol), i.e. this is a *category* retrieval task.

### 2.2 The ablation grid (32 arms)

| Group | Axis | Arms |
|---|---|---|
| **E1** | Channel set | text-only, view-only, shape-only, text+view (OSCAR-equivalent), **full fusion (BASE)**, CLIP-pruned |
| **E2** | Local geometry re-ranking | none / RANSAC fitness / unaligned trimmed distance (diagnostic control) / RANSAC-aligned / RANSAC+ICP / fitness+distance Borda |
| **E2b** | Shape reference | partial rendered views (BASE) vs full mesh |
| **E4** | Appearance encoder | DINOv2 (BASE) vs SigLIP |
| **E6** | Fusion strategy | weighted sum (BASE) vs reciprocal rank fusion |
| **E7** | Shape encoder | ULIP-2 (BASE) vs Uni3D |
| **O1** | Is S_shape redundant given geometry? | 4–5 configs of shape vs GeDi placement |
| **O2** | Scope/ordering | full-DB fusion (BASE) vs CLIP-cascade vs threshold pruning vs visual-first |
| **O4** | Number of views V | 8/16/32/42 |
| **O5** | Query cloud composition | XYZ+RGB vs XYZ-only |

Two features signal methodological care and are worth preserving in the write-up:

- **E2_chamfer_unaligned is an explicit "not-a-method" control** — it measures
  the trimmed distance *without* alignment, existing only to prove the geometry
  gain comes from evaluating the distance *after* RANSAC alignment.
- The **ICP arm is retained as the evidence for keeping ICP off** ("ICP adds a
  refinement DOF that can partly launder a wrong retrieval into a plausible fit
  — the confusion to avoid in a *retrieval* evaluation"; measured +0.0001 nDCG
  for ~5.4 s/query).
- **O1's original "S_GeDi replaces S_shape inside the fusion" is documented as
  infeasible** (it needs a GeDi score for every gallery entry: 2,101 × 3,308 ≈
  6.95M RANSAC fits ≈ 830 h per cell) and honestly reframed to a shortlist-level
  question rather than faked.

### 2.3 Metrics

Graded **nDCG** (primary), **mAP**, precision/recall/F1, NN-T1/T2, plus a
"Table B" depth-matched cut at the geometry shortlist depth K so geometry arms
can be compared on equal footing. Per-query nDCG/AP are persisted to
`results_per_query.json` for paired post-hoc statistics.

### 2.4 Full results

The complete grid (nDCG, mAP) for the original **CLS** run and the **mean** re-run.
Arms that are exact aliases of BASE (E2_none, E2b_partial, E4_dinov2, E6_weighted,
E7_ulip2_pc, O1b, O2_full_database, O4_V42, O5_xyzrgb — all = E1c) are omitted.
Geometry arms in the mean run are **[running]** (dGeDi K-sweep in progress; the
20-query smoke registered 100/100, so they should recover the CLS ~0.64 level).

**E1 — channel set**

| Arm | nDCG CLS | nDCG mean | mAP CLS |
|---|---|---|---|
| E1a text-only (CLIP) | 0.4218 | 0.4218 | 0.047 |
| E1 view-only (DINOv2) | 0.5574 | 0.5506 | 0.163 |
| E1 shape-only (ULIP-2) | 0.5256 | 0.5256 | 0.138 |
| E1b text+view (OSCAR-equiv.) | 0.5622 | 0.5519 | 0.146 |
| E1_oscar_cascade (faithful OSCAR) | 0.4596 | 0.4561 | 0.050 |
| **E1c full fusion (BASE)** | **0.5970** | **0.5889** | 0.173 |
| E1d clip-pruned | 0.4580 | 0.4565 | 0.050 |

**E2 — local geometry re-ranking** (mean = [running])

| Arm | nDCG CLS | mAP CLS |
|---|---|---|
| E2_chamfer_unaligned (control) | 0.5734 | 0.158 |
| E2_fitness | 0.6358 | 0.175 |
| E2_chamfer_ransac | 0.6375 | 0.176 |
| **E2_both (geometry winner)** | **0.6428** | **0.177** |

**E2b — shape reference: full mesh vs partial views**

| Reference | nDCG CLS | nDCG mean | mAP CLS |
|---|---|---|---|
| E2b_partial (= BASE) | 0.5970 | 0.5889 | 0.173 |
| E2b_fullmesh | 0.5985 | 0.5897 | 0.177 |

E2b is a **fusion-mode** ablation: only the shape *reference* changes (partial
views → full mesh), everything else held fixed, and the full fused pipeline is
scored. Here full mesh wins by +0.0015 nDCG (CLS) / +0.0008 (mean) — negligible.
**Stage 2 runs the same fusion-mode ablation and there *partial* wins** (fused
mAP 0.671 vs 0.657, §3.4). So at the ablation's own (fusion) level the two stages
**disagree** — full mesh in Stage 1, partial in Stage 2 — both by small margins.
**Net: the shape-reference source barely affects fused retrieval.**

(A *separate* diagnostic, not the E2b ablation itself: the Stage-2 ULIP-**only**
arm shows full mesh is the better shape descriptor *in isolation*, 0.518 vs 0.451
— but that isolated advantage does not survive fusion.) BASE uses partial views
for deployment realism (a real scan is partial, SAM-6D style).

**E4 / E6 / E7 — encoder, fusion, shape-encoder**

| Arm | nDCG CLS | nDCG mean | mAP CLS |
|---|---|---|---|
| E4_siglip (vs DINOv2) | 0.5245 | 0.5245 | 0.101 |
| E6_rrf (vs weighted sum) | 0.5792 | 0.5731 | 0.140 |
| **E7_uni3d (vs ULIP-2 BASE)** | **0.6005** | **0.5917** | 0.180 |

**O1 — is S_shape redundant once geometry exists?** (mean = [running])

| Arm | nDCG CLS | mAP CLS |
|---|---|---|
| O1a no geometry, no shape (= E1b) | 0.5622 | 0.146 |
| O1c GeDi post-fusion, no shape | 0.6068 | 0.145 |
| O1d shape in fusion + GeDi | 0.6358 | 0.175 |
| O1e geometry + base rank (Borda) | 0.6352 | 0.180 |

**O2 — scope / ordering · O4 — views · O5 — query cloud**

| Arm | nDCG CLS | nDCG mean | mAP CLS |
|---|---|---|---|
| O2_clip_cascade (CLIP top-20) | 0.4580 | 0.4565 | 0.050 |
| O2_clip_threshold (τ=0.37) | 0.4572 | 0.4559 | 0.050 |
| O2_clip_threshold_cal (τ fitted) | 0.5254 | 0.5189 | 0.091 |
| O2_visual_first | 0.5639 | 0.5560 | 0.163 |
| O4_V8 | 0.5804 | 0.5736 | 0.153 |
| O4_V16 | 0.5928 | 0.5833 | 0.172 |
| O4_V32 | 0.5895 | 0.5811 | 0.168 |
| O4_V42 (BASE) | 0.5970 | 0.5889 | 0.173 |
| O5_xyzrgb (BASE) | 0.5970 | 0.5889 | 0.173 |
| O5_xyz_only | 0.5954 | 0.5854 | 0.177 |

### 2.5 What Stage 1 tells us

- **Fusion clearly beats any single channel** (0.597 vs best single 0.557).
  DINO (view) is the strongest single channel; text alone is weak on SHREC.
- **Geometry re-ranking helps category retrieval materially** (0.597 → 0.643,
  CLS). The **unaligned control (0.573) is *below* BASE** while the aligned arms
  jump to ~0.64 — proving the gain comes from evaluating the distance *after*
  RANSAC alignment, exactly as the control was designed to show. This is the
  opposite of geometry's effect on BOP pose (§4) — a central, thesis-worthy nuance.
- **Uni3D is the best shape encoder** — E7_uni3d beats ULIP-2 BASE in both runs
  (0.6005 vs 0.5970 CLS; 0.5917 vs 0.5889 mean), and is the auto-picked winner of
  the mean no-geometry run. (Caveat: this is a fusion-backbone swap, not a solo
  encoder comparison — §8.3.)
- **The OSCAR CLIP-cascade *hurts* on SHREC.** Full-database fusion (0.597) far
  exceeds every CLIP-pruned variant: cascade/threshold collapse to ~0.457 because
  τ=0.37 (Pulli et al.'s value) does not transfer — it admits far too few
  candidates. Even re-fitting τ to SHREC (threshold_cal, 0.525) stays below
  full-DB. **Conclusion: OSCAR's language-first pruning is a liability on this
  dataset; simultaneous full-DB fusion is better.**
- **O1 (geometry vs shape redundancy):** geometry contributes *more* than the
  shape channel — GeDi-post-fusion without shape (0.607) already beats
  shape-in-fusion (0.597), and shape+geometry together (0.636) is best. So shape
  and geometry are complementary, not redundant.
- **O4 (views):** more views help with diminishing returns — V8 0.580 < V16 0.593
  < V42 0.597 (V32 slightly dips, 0.590); 42 views is the peak.
- **O5 (query cloud):** RGB in the query cloud adds almost nothing (XYZ+RGB 0.5970
  vs XYZ-only 0.5954, +0.0016) — geometry, not colour, drives the shape channel.
- **Mean pooling is marginally *worse* on SHREC than CLS** (−0.008 on fusion),
  adopted anyway for cross-stage consistency (§6.2).

---

## 3. Stage 2 — MI3DOR monocular retrieval

### 3.1 Concept

Apply the frozen pipeline to MI3DOR (10,500 monocular image queries → CAD
gallery), in **cross-mode** (image-query shape channel, since there is no depth).
Report the standard 3D shape-retrieval metric suite. Arms cover single channels,
the full fusion, a CLIP-pruned fusion, and OSCAR-style cascade variants.

### 3.2 Metrics

NN (nearest-neighbour top-1 accuracy), FT (first-tier), ST (second-tier),
F1, nDCG@2R, mAP, ANMRR (lower is better) — the SHREC-family retrieval metrics.

### 3.3 Results and the weights-bug correction

The originally-shipped fused result silently ran with fusion weights
**(0, 0.5, 0.5)** — i.e. **CLIP switched off**. Corrected to (0.3, 0.4, 0.3) and
re-run (`..._tau037_dinomean_fixedw`), all 7 arms:

| Arm | NN | FT | ST | nDCG@2R | mAP | ANMRR↓ |
|---|---|---|---|---|---|---|
| clip_only | 68.0 | 0.575 | 0.755 | 0.720 | 0.580 | 0.339 |
| dino_only_full | 83.0 | 0.629 | 0.753 | 0.751 | 0.647 | 0.297 |
| **ulip_only_full** ⚠ | **0.0** | 0.000 | 0.000 | 0.000 | 0.000 | 1.006 |
| **clip_dino_ulip_full (full-DB)** | 85.2 | **0.674** | **0.816** | **0.805** | **0.699** | **0.246** |
| oscar_maxview (cascade) | 84.9 | 0.575 | 0.755 | 0.733 | 0.592 | 0.337 |
| oscar_softmax (cascade) | 85.0 | 0.575 | 0.755 | 0.734 | 0.592 | 0.337 |
| clip_pruned_dino_ulip (cascade) | 85.0 | 0.575 | 0.755 | 0.734 | 0.592 | 0.337 |

> ⚠ **CRITICAL — this Stage-2 run is compromised (regression, 2026-08-24).**
> `ulip_only_full = 0.0` is not "ULIP is weak" — the ULIP channel returned an
> **empty ranking for every query** (`ulip_candidates_full: []`). Root cause: the
> MI3DOR **CAD meshes were not on the machine** (`object_database/MI3DOR/model/test/`
> absent → 0 CAD meshes), and the ULIP shape gallery is keyed off that mesh list,
> so it had nothing to match against. The 829 MB partial-ULIP cache exists on disk
> but could not be associated with gallery objects without the mesh list. The
> autonomous re-run set `MI3DOR_MODES=partial` to work around the missing CADs, but
> that flag does not remove the dependency. **Consequence: ULIP contributed nothing
> to the fused arm either — the "0.699" full fusion is really CLIP+DINO.** The old
> runs (CADs present) had working ULIP (ulip_only mAP 0.451) but CLIP-off. So
> **the correct CLIP+DINO+ULIP 3-way fusion has never actually run on MI3DOR.**
> Fix: restore the 4.38 GiB CAD meshes from Drive (both gallery caches are already
> local, so no re-encoding) and re-run. Until then, treat the Stage-2 numbers as
> CLIP+DINO, not 3-way.

**What still holds structurally** (independent of the ULIP bug): the weights fix
improves the graded metrics (adding CLIP: mAP 0.671→0.699 in the old-vs-new
comparison), though it lowers top-1 NN by ~1.8; and **full-database fusion beats
the OSCAR cascade** — the three cascade arms (oscar_maxview/softmax,
clip_pruned) cap at mAP ≈ 0.592 because they can only re-rank the ~20-item CLIP
shortlist, whereas full-DB fusion reaches 0.699. High NN (~85) but low FT/mAP is
the cascade's signature: good top-1, poor deeper ranking. (Both conclusions will
need re-confirming once ULIP is restored.)

### 3.4 Intra-Stage-2 ablations (pooling + shape source)

Two further ablations live in the MI3DOR runs (from the earlier weights-bug runs,
but clean at the channel level since they don't involve the CLIP weight):

**DINO pooling CLS → mean** (`dino_only_full`, partial) — the empirical basis for
freezing mean pooling:

| Pooling | NN | FT | mAP |
|---|---|---|---|
| CLS (`_tau037`) | 78.0 | 0.587 | 0.597 |
| **mean (`_dinomean`)** | **83.0** | **0.629** | **0.647** |

Mean pooling lifts MI3DOR DINO by +5.0 NN / +0.050 mAP — a much larger gain than
its −0.008 nDCG *cost* on SHREC (§2), which is why the frozen config uses mean.

**Shape reference — fullmesh vs partial views** (the E2b analog, `_dinomean` run,
CADs present so ULIP worked):

| Shape source | ulip_only NN | ulip_only mAP | fused mAP |
|---|---|---|---|
| **fullmesh** | **78.1** | **0.518** | 0.657 |
| partial views | 68.1 | 0.451 | 0.671 |

Both shape sources were run through the **full fusion** pipeline (only the shape
reference changed — the fusion-mode ablation). In fusion **partial wins** (mAP
0.671 vs 0.657, NN/FT/nDCG@2R agreeing) — the *opposite* of Stage-1 E2b, where
full mesh won by +0.0015. So the fusion-level shape-source ablation does **not**
favor full mesh consistently across stages; both effects are small. The isolated
`ulip_only` arm *does* favor full mesh (0.518 vs 0.451) — i.e. full mesh is a
better raw shape descriptor — but that advantage does not survive fusion.
(Caveat: these Stage-2 fusion numbers are DINO+ULIP — the CLIP-off bug — so the
true 3-way fusion comparison awaits the CAD re-run.)

---

## 4. Stage 3 — BOP proxy-CAD pose

### 4.1 Concept (the paired design)

Stage 3 asks the real downstream question: **is the retrieved CAD good enough to
pose the object?** It uses BOP test targets (`test_targets_bop19.json`) with
**ground-truth visible bbox + mask**, so retrieval and pose are isolated from
segmentation error. Four modes, all on the same queries:

| Mode | Gallery | What it measures |
|---|---|---|
| **3a** | G_proxy ∪ all target CADs (1,316) | Retrieval only — is the exact CAD ranked highly among proxies? (Recall@k, MRR) |
| **gt** | — (uses the object's own CAD) | Exact-CAD FoundationPose benchmark → `D_posed_gt` (the pose upper bound) |
| **3b** | G_proxy only (targets removed) | Pose the top-1 *proxy* → `D_posed` + paired `Delta = D_posed − D_posed_gt` |
| **3c** | reuse 3a ranking, exclude only the exact self | Pose the best available *non-self* CAD → decomposition diagnostic (§4.6) |

`G_proxy = GSO ∪ HouseCat6D ∪ ITODD`. Query modes: **pc** (partial cloud, ULIP
PointBERT) vs **cross** (RGB crop, ULIP image encoder). Geometry axis: with/without
dGeDi re-rank (repo config, top-5).

### 4.2 Why D_sym and not ADD/ADD-S/BOP-AR for proxies

ADD, ADD-S and BOP-AR (VSD/MSSD/MSPD) compare two poses of the **same** model in
a shared object coordinate frame — they are undefined for a non-identical proxy.
So proxy pose quality is measured by **D_sym**, the symmetric complete-surface
discrepancy: place the GT-posed target and the estimated-posed proxy in the
camera frame, uniformly sample both surfaces (N=10,000, fixed seed), and compute
`D_sym = ½(D_T→P + D_P→T)` in mm and normalised by target diameter, plus an
**F-score** at 1% and 5% of the diameter. Critically, the **gt** benchmark is
scored with the *same* D_sym so that `Delta = D_posed − D_posed_gt` is a clean
paired subtraction on one scale (this is why gt does not use BOP-AR in the
implementation — see §4.7 for the concept-vs-code deviation).

`D_sym` semantics differ by mode: for **gt** (same CAD) it is *pure pose error*;
for **3b/3c** (different CAD) it is *pose error + shape mismatch, entangled*.
`Delta` partially isolates the substitution cost.

### 4.3 Retrieval results (3a, 12,284 instances)

| Variant | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| pc | 0.464 | 0.726 | 0.808 | 0.584 |
| pc + geo | 0.413 | 0.726 | 0.808 | 0.547 |
| **cross (best)** | **0.482** | **0.733** | **0.812** | **0.597** |
| cross + geo | 0.458 | 0.733 | 0.812 | 0.579 |

**Per-dataset R@1 (all four variants):**

| Variant | YCB-V | T-LESS | LM-O |
|---|---|---|---|
| pc | 0.671 | 0.350 | 0.400 |
| pc + geo | 0.534 | 0.336 | 0.426 |
| cross | 0.732 | 0.332 | 0.464 |
| cross + geo | 0.602 | 0.360 | 0.504 |

**Cross wins overall and on YCB-V + LM-O; pc wins texture-less T-LESS.** The
per-dataset view makes the geometry story precise: geometry re-ranking is
**net-negative overall** but **dataset-split** — it *helps* weak-semantics cases
(LM-O: 0.400→0.426 pc, 0.464→0.504 cross; T-LESS-cross 0.332→0.360) and *hurts*
strong-semantics YCB-V badly (0.732→0.602 cross), which dominates the mean. R@5/R@10
are unchanged by geometry — it only reorders within the top-5 shortlist.

### 4.4 Exact-CAD pose benchmark (gt)

| | D_sym mean | D_sym median | F@1% | F@5% |
|---|---|---|---|---|
| Combined | 4.87 mm | **1.72 mm** | 0.341 | **0.944** |
| YCB-V | 2.83 | 2.03 | 0.483 | 0.981 |
| T-LESS | 6.04 | 1.41 | 0.282 | 0.937 |
| LM-O | 5.30 | 3.48 | 0.214 | 0.875 |

Exact-CAD FoundationPose is highly accurate (median 1.4–3.5 mm), confirming the
pose harness is sound — the proxy, not the pose estimator, is the bottleneck.

### 4.5 Proxy pose (3b)

| 3b — cross (no geo) | D_sym mean | D_sym median | F@5% | Delta median |
|---|---|---|---|---|
| Combined | 33.6 mm | **18.4 mm** | 0.302 | **15.8 mm** |
| YCB-V | 31.8 | 23.6 | 0.359 | 20.8 |
| T-LESS | 32.3 | 13.6 | 0.283 | 11.6 |
| LM-O | 45.0 | 28.5 | 0.223 | 24.6 |

Geometry (dGeDi) applied to 3b makes it **worse** overall (median 18.4 → 28.8 mm)
— consistent with the retrieval finding that geometry is net-negative in BOP.
**Headline number: the substitution cost `Delta` is 15.8 mm median** — the tax
for posing with a stand-in instead of the true CAD (vs 1.7 mm exact).

### 4.6 The 3c decomposition (next-best-non-GT)

3c answers the diagnostic question *"is 3b's error because the proxy pool is too
foreign, or because substituting any non-self CAD is intrinsically lossy?"* It
reuses the stored 3a ranking (which contains the exact CAD **plus** all other
real CADs and proxies) and poses the highest-ranked candidate that is **not** the
exact target — the best available stand-in from a richer pool. Retrieval is free
(reused); only pose re-runs.

| | D_sym median | D_sym mean | Delta median | F@5% |
|---|---|---|---|---|
| **3c** (best non-self) | **15.3 mm** | 28.0 | 12.4 | 0.389 |
| 3b (proxy-only) | 18.4 | 33.6 | 15.8 | 0.302 |
| gt (exact CAD) | 1.7 | 4.9 | — | 0.944 |

**Provenance split (the key result):**

| Next-best substitute is… | n | D_sym median |
|---|---|---|
| a **real target CAD** of another object | 6,742 | **10.35 mm** |
| a **G_proxy** item | 5,542 | 20.10 mm |

**Conclusion:** the proxy-substitution penalty splits **roughly in half**:
- **Gallery foreignness ≈ half the cost** — a real, in-distribution CAD of a
  *different* object poses ~2× better than a proxy (10 vs 20 mm). A richer,
  better-matched gallery would materially help.
- **Substitution is intrinsically lossy for the other half** — even the best
  real-CAD stand-in (10.35 mm) is still ~6× worse than the object's own CAD
  (1.7 mm). No gallery fixes that residual.

Per-dataset 3c medians track the story: T-LESS 8.75 mm (many near-identical CADs
available), LM-O 24.25 mm (distinctive objects → runner-up is usually a foreign
proxy).

### 4.7 Concept-vs-code deviation (to reconcile)

The written Stage-3 concept (`STAGE3_EVALUATION_CONCEPT.md`) specifies **BOP-AR
(VSD/MSSD/MSPD)** for the 3a oracle and conditional-exact pose. The
implementation **descoped BOP-AR** and used the D_sym "gt" benchmark instead, so
that gt and 3b/3c live on one comparable scale for the `Delta` pairing. Raw poses
are stored, so BOP-AR for the gt/oracle case is **derivable without re-running**
FoundationPose. Recommendation: either update the concept doc, or derive the gt
BOP-AR table from stored poses so an examiner does not catch the mismatch. (User
decision 2026-08-25: deriving BOP-AR is **not** required for now.)

---

## 5. What was changed during the evaluation, and why (chronological)

| # | Change | Why | Commit |
|---|---|---|---|
| 1 | **Stage-2 fusion weights** 0/0.5/0.5 → 0.3/0.4/0.3 | `EvalConfig` default silently switched CLIP off; the shipped "full fusion" was DINO+ULIP only | 1f660fde |
| 2 | **Preprocessing HPR unified** to 2.8 / 0.001 | Image and point-cloud galleries were built with inconsistent parameters | d59c5603 |
| 3 | **Stable SHA-256 resample seed** (partial PC) | Python `hash()` salting made partial-cloud resampling non-deterministic across processes | 3513aefa |
| 4 | **dGeDi backend for Stage-1 geometry** (scale-invariant unit-diameter) | Stage 3 uses dGeDi; Stage 1 used legacy in-process GeDi → geometry axis not comparable across stages | b9f548de, cb007a3e |
| 5 | **dGeDi SHREC scale fix** (`--diam-scale 1.0`) | `compute_diameters` applied a mm→m ÷1000 to SHREC's unitless CADs → 1000× scale error → 0 RANSAC | cb007a3e |
| 6 | **dGeDi health-gate fix** (probe dGeDi, not GeDi) | `gedi_available()` always probed the GeDi service (5060); with only dGeDi (5061) up it skipped the geometry ablation entirely → silent fallback to mean-only | ced8a59e |
| 7 | **SigLIP MAP-head pooling fix** | SigLIP was pooled via `last_hidden_state[:,0]` (a nonexistent CLS token → degenerate patch-0 embedding), making the E4 DINOv2-vs-SigLIP comparison unfair to SigLIP | f50a844f |
| 8 | **CLS → mean DINO pooling** | Consistency with MI3DOR (where mean was proven better) and Stage 3 | b9f548de |
| 9 | **Stage-3 "3c" next-best-non-GT diagnostic** | Decompose the 3b substitution error into gallery-foreignness vs intrinsic-substitution-loss | b49f0d94 |
| 10 | **Fusion-weight sensitivity sweep** | Justify the fixed (0.3/0.4/0.3) weights as a sensitivity analysis (not a selection) | 3600ccaa+ **[running]** |

### 5.1 The silent-misconfiguration pattern (a meta-observation)

Three of the changes above (#1 Stage-2 weights, #6 dGeDi health gate, #7 SigLIP
pooling) were **silent misconfigurations** that each would have put a wrong
number into a thesis table if unaudited. Two of the three were caught only
because a value looked "off," not by a systematic check. This is the single
biggest execution risk in the campaign and motivates a **per-arm sanity-check
pass** (verify each channel is live, non-degenerate, and weighted as declared)
before finalising any table.

---

## 6. Notable methodology decisions worth stating in the thesis

### 6.1 Isolation of the variable of interest
Stage 3 uses GT masks/boxes so retrieval/pose are not contaminated by
segmentation; the paired 3a/3b galleries differ *only* by target-CAD
availability, making `Delta` a clean causal measure of substitution cost.

### 6.2 CLS vs mean pooling is dataset-dependent
CLS was slightly better on SHREC (0.597 vs 0.589); mean was better on MI3DOR.
Mean was frozen for cross-stage consistency. This is direct evidence that the
"tune once, freeze, transfer" premise does not hold perfectly (§7.3).

### 6.3 The SigLIP comparison was unfair (now fixed, re-run pending)
`E4_siglip = 0.5245` was **identical across the CLS and mean runs** — a tell that
SigLIP's score vector was never re-pooled and that it never used SigLIP's native
attention-pooling head. The fixed number **[running]** will show whether SigLIP
was genuinely weaker than DINOv2 or merely handicapped.

---

## 7. Conclusions supported *today*

1. **Multimodal fusion is justified.** In every stage the fused configuration
   beats each single channel; DINO carries the most single-channel signal but
   CLIP and shape add real value in fusion (Stage 2 mAP 0.647→0.699 by adding
   CLIP correctly).
2. **Image-query ("cross") retrieval is the best default**, except on
   texture-less objects (T-LESS), where the point-cloud query wins — an
   intuitive and defensible appearance-vs-geometry trade-off.
3. **Geometric re-ranking is stage-specific: it helps category retrieval
   (SHREC) but is net-negative for BOP retrieval *and* pose.** Its value is
   confined to weak-semantics cases and does not translate into better pose.
   Recommendation the results support: geometry belongs in the *pose* stage, not
   as a retrieval re-ranker.
4. **Exact-CAD pose is excellent; the proxy is the bottleneck.** Median exact
   1.7 mm vs proxy 18.4 mm; the substitution tax is ~15.8 mm median.
5. **The proxy penalty decomposes ~50/50** into "gallery too foreign" (fixable
   with a richer/better-matched gallery) and "substitution is intrinsically
   lossy" (not fixable by retrieval). This is the most novel single result and
   directly answers the "why is 3b so much worse than gt?" question.
6. **The frozen config does not transfer perfectly** — pooling and geometry both
   flip sign between datasets/tasks — so downstream claims must be framed as
   "SHREC-optimal, transferred," not "globally optimal."
7. **OSCAR's language-first CLIP pruning is a liability on SHREC** — every
   CLIP-cascade/threshold arm collapses to ~0.457 vs full-DB fusion 0.597,
   because τ=0.37 does not transfer; even a re-fitted τ stays below full-DB. In
   Stage 2 the cascade caps mAP at ~0.592 vs full-DB 0.699. **Simultaneous
   full-database fusion beats the cascade in both stages.**
8. **Shape and geometry are complementary, not redundant** (Stage-1 O1): geometry
   contributes more than the shape channel, and the two together are best (0.636).
   Colour in the query cloud adds essentially nothing (O5, +0.0016); more views
   help with diminishing returns, peaking at 42 (O4). **Uni3D marginally
   out-performs ULIP-2** as the shape backbone (E7).

> **Caveat on Stage 2:** conclusions 1 and 7 rest partly on the MI3DOR run, whose
> ULIP channel was empty (§3). The *direction* (fusion > single, full-DB > cascade)
> is robust, but the absolute Stage-2 numbers are CLIP+DINO, not 3-way, and must be
> re-confirmed after the CAD-mesh re-run.

---

## 8. Open items

### 8.1 Running now
- **Stage-1 dGeDi geometry K-sweep** [running] — K=50 geometry pass ~74–80%
  done; then K=20/K=5 derived from the cached pairs. Fills the mean-run geometry
  cells (E2_*, O1c/d/e) → `results_shrec18_v2_stage1_mean_dgedi_k{50,20,5}`.
- **Fair SigLIP re-run** [running, chained after dGeDi] — recomputes only the
  SigLIP channel via the MAP head → `..._mean_siglipfix`; reports fixed E4_siglip
  vs DINOv2 0.5889.
- **Fusion-weight sensitivity sweep** [running] — 66-point simplex on the cached
  mean-DINO vectors → `weight_sweep.csv`; reports the nDCG span around BASE.

### 8.2 Recommended but not yet done
- **Re-run Stage 2 with the MI3DOR CAD meshes present** (HIGH priority): the
  meshes (4.38 GiB / 3,869 objects) are on Drive; both gallery caches (mean-DINO,
  ULIP-partial) are already local, so no re-encoding — just download + a
  query-inference pass. This produces the first correct CLIP+DINO+ULIP MI3DOR
  fusion. Until then the Stage-2 numbers are CLIP+DINO only (§3).
- **Paired significance test** (~1–2 h; per-query nDCG already persisted): report
  bootstrap 95% CI of the paired delta for adjacent arms (e.g. E7_uni3d 0.5917
  vs E1c 0.5889, Δ ≈ 0.003 is almost certainly n.s.) so marginal "winners" are
  not over-claimed. **Not yet implemented.**
- **Per-arm sanity-check pass** — automated verification that each channel is
  live/non-degenerate and weighted as declared (would have caught #1/#6/#7
  mechanically).
- **ULIP-only = 0.0 in Stage 2** — investigate or explain the cross-mode ULIP
  standalone collapse.
- **gt BOP-AR** — derivable from stored poses if the concept-vs-code deviation
  (§4.7) is to be reconciled by table rather than by doc edit. (Deprioritised.)
- **step5 SHA-256 seed fix** — the runtime query-path resample seed
  (`step5_shape_matching.py`) still uses Python `hash()`; apply the same SHA-256
  fix as the preprocessing path (deferred, code-correctness only).
- **Stage-1 ULIP-2 cross arm** — warranted now that cross > pc in Stage 3;
  conditional add to the Stage-1 grid.

### 8.3 Framing caveats to carry into the thesis text
- Fusion weights (0.3/0.4/0.3) were **never swept as a selection** — justify via
  the sensitivity sweep, or state provenance.
- **Single query dataset per stage** (Stage 1 = SHREC, Stage 2 = MI3DOR) limits
  the strength of the tuning-transfer claim.
- **D_sym entangles pose and shape error** for proxies — state plainly that a 3b
  D_sym is not a pose-accuracy number in the same sense as the gt one.
- **E7 "Uni3D vs ULIP-2"** is a fusion-backbone swap, not a solo-encoder
  comparison — the label slightly over-claims.

---

## 9. File map (where each result lives)

| Result | Path |
|---|---|
| Stage-1 CLS grid | `object_retrieval/results_shrec18_v2_stage1/` |
| Stage-1 mean (no geo) | `object_retrieval/results_shrec18_v2_stage1_mean_mean_only/` |
| Stage-1 mean+dGeDi | `object_retrieval/results_shrec18_v2_stage1_mean_dgedi_k{50,20,5}/` **[running]** |
| Stage-1 SigLIP fix | `object_retrieval/results_shrec18_v2_stage1_mean_siglipfix/` **[running]** |
| Stage-1 weight sweep | `object_retrieval/results_shrec18_v2_stage1_mean_mean_only/weight_sweep.csv` **[running]** |
| Stage-2 fixed weights | `object_retrieval/results_mi3dor_oscarplus_v2_tau037_dinomean_fixedw/` |
| Stage-3 (3a/gt/3b/3c) | `object_retrieval/results_bop_stage3_v2/{3a_*,gt,3b_*,3c_cross}/` |

Drivers: `experiments/experiment1_shrec18_stage1.py` (Stage 1),
`object_retrieval/retrieval_mi3dor_eval_oscarplus.py` (Stage 2),
`object_retrieval/eval_bop_pose.py` (Stage 3, `--mode 3a|gt|3b|3c`).
Prior Stage-3 write-up: `docs/STAGE3_RESULTS_SUMMARY.md`.

**Superseded / older result dirs (do NOT report as current):**

| Dir | Status |
|---|---|
| `results_bop_stage3{,_full,_ulippc,_ulipcross}` | pre-v2 Stage-3 runs (old concept) → superseded by `results_bop_stage3_v2` |
| `results_mi3dor_oscarplus_f20_3` (Drive) | older MI3DOR (CLS, f20, 10,031 queries, weights bug) |
| `results_mi3dor_oscarplus_v2_tau037{,_dinomean}` | weights-bug MI3DOR runs; kept for the pooling + fullmesh/partial ablations (§3.4) |
| `results_{ycbv,tless,lmo,gso,housecat6d,itodd}_stage1` | **not experiments** — Stage-3 gallery embedding `_cache/` only |
| `results_stage1_singlerun`, `results_bop_stage3_v2_smoke` | smoke / single-run scratch |

**Complete-coverage note:** the Stage-1 grid (§2.4) contains every non-alias arm
of the 32-cell grid; Stage 2 (§3.3–3.4) covers all 7 fusion arms plus the pooling
and shape-source ablations; Stage 3 (§4) covers all four 3a variants, gt, both 3b
variants, and 3c. The only unreported numbers are the Stage-1 mean-run **geometry**
cells (E2_*, O1c/d/e — dGeDi [running]) and the **weight-sweep span** ([running]).
