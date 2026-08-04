# Stage-1 SHREC'18 Retrieval Experiment — Implementation Walkthrough

This document explains *how* the Stage-1 experiment is implemented, in
pseudo-code. It complements two things:

- **`docs/STAGE1_EVALUATION_DESIGN.md`** — *why* the metrics and depth-K
  design are what they are.
- **`experiments/stage1_reproduce.py`** — a flat, config-first driver you can
  read in five minutes and edit to run a new ablation.

The authoritative code is **`experiments/experiment1_shrec18_stage1.py`**
(~3.2k lines). The pseudo-code below mirrors its logic without the CLI,
benchmarks, precompute paths and edge-case handling.

---

## 1. The problem

Given SHREC'18 ObjectNN+:

- **Gallery:** 3,308 ShapeNetSem CAD models, each pre-rendered into 42
  FPS-ordered views + per-view partial point clouds.
- **Queries:** 2,101 real RGB-D scans (colored meshes → a crop + a point
  cloud).
- **Ground truth:** official `rgbd.csv`/`cad.csv` give each query and CAD a
  (category, subcategory). A retrieved CAD is *relevant* iff its category
  matches the query's.

**Goal:** score the whole gallery for every query with the OSCAR+ multi-channel
pipeline, run an ablation grid over its design choices, and pick the winning
configuration by **nDCG (tie-break mAP)**. The winner is frozen for later
stages.

## 2. The three channels

Every arm is some combination of three per-query score vectors over the
gallery (each is a length-3308 vector of similarities):

| Channel | Symbol | Signal | Encoder |
|--------|--------|--------|---------|
| Text/semantic | `S_text` | query crop vs CAD view *descriptions* | CLIP |
| Appearance | `S_view` | query crop vs CAD *rendered views* | DINOv2 (or SigLIP) |
| Shape | `S_shape` | query *point cloud* vs CAD point clouds | ULIP-2 (or Uni3D) |

BASE fuses all three with weighted sum, weights `(0.30, 0.40, 0.30)`.

## 3. The key architectural idea: two tiers

The encoder work is expensive; the design choices we ablate are mostly cheap
re-combinations of the *same* score vectors. So the experiment is split:

```
TIER 1  (expensive, cached once per encoder×reference combination)
    for each needed "pass":                 # base / siglip / ulip_pc_rgb / ...
        encode gallery references (cached on disk by build_pipeline)
        for each query:
            score_vector[gallery] = similarity(query, references)
        save store[pass][query] = score_vector      ->  _cache/scores_<pass>.pt

TIER 2  (cheap, derived from cached vectors — no encoders touched)
    for each ablation:
        ranking = derive_ranking(spec, cached vectors)   # fusion + scope
        if spec.geometry:
            ranking = apply_geometry(spec, ranking)       # re-rank top-K
        metrics = score(ranking, ground_truth)
```

Consequences that make the grid affordable:

- **View count V (ablation O4) is free.** All 42 views are encoded once; V just
  changes how many are aggregated at derivation time.
- **Fusion / scope / threshold / RRF are free** — post-processing of cached
  vectors, run through the *production* `ScoreFusion` so the ablation exercises
  real pipeline code, not a re-implementation.
- **Aliases cost nothing.** Many grid cells are numerically identical to BASE
  (e.g. "DINOv2 appearance" *is* BASE); they are cross-referenced, not re-run.

## 4. Tier 1 — a channel score pass (pseudo-code)

```
function run_pass(pass_key):
    if cached store exists and covers this gallery + queries:
        return it

    cfg = EvalConfig for this pass          # encoder, reference type, mode
    (clip, dino, shape) = build_pipeline(cfg)   # loads + caches ref embeddings

    # pre-stack all reference embeddings into one matrix per channel
    dino_refs  = stack(dino._ref_embeddings)     # (Σ views, D)
    shape_refs = stack(shape._cad_embeddings)

    for each query q:
        rec = {}
        if channel needs CLIP:
            rec.clip = clip.retrieve(q.crop)     over full gallery
        if channel needs DINO:
            sims = q.crop_embedding · dino_refs.T
            rec.dino = { V: aggregate_views(sims, top-V) for V in {8,16,32,42} }
        if channel needs shape:
            q_emb = pc-mode: encode(q.point_cloud)   # cross-mode: encode(q.crop)
            sims  = q_emb · shape_refs.T
            rec.shape = aggregate_views(sims)        # over partial views, or max
        store[q] = rec
    save store to _cache/scores_<pass>.pt
    return store
```

View aggregation (`_aggregate_groups`) reduces a CAD's per-view similarities to
one number via **top-k soft-max** (the "soft-k-max": DINO top-5 τ=0.5; ULIP
partial top-8 τ=0.5).

## 5. Tier 2 — deriving one ranking (pseudo-code)

```
function derive_ranking(spec, query):
    # 1. gather this arm's active channel vectors from the caches
    vecs = { ch: store[pass][query][ch]  for ch in spec.channels }

    # 2. scope: optionally prune to a candidate pool (ablation O2)
    if spec.scope is a CLIP/DINO shortlist:
        order = argsort(pruning_channel, desc)
        if threshold scope:  keep = { i : sim[i] >= tau }   # top-k fallback if empty
        else:                keep = top-20
        pool, tail = order[:keep], order[keep:]     # tail keeps its 1st-stage order
    else:
        pool, tail = whole gallery, []

    # 3. fuse the pool
    if exactly one channel active:                  # single-channel shortcut
        ranked = sort(pool by that channel)
    else:
        results = wrap vecs into CLIP/DINO/Shape result objects
        ranked  = ScoreFusion.fuse(results, method=spec.fusion)   # production code
    return ranked + tail                            # ALWAYS full-length list
```

Two deliberate rules:

- **The returned ranking is always the full gallery** (pool, then tail). The
  official `precision(x)` divides by the submitted list length, so submitting a
  short shortlist would inflate precision while recall collapsed.
- **Threshold pruning keeps a variable-sized set** (`sim_text ≥ τ`), matching
  OSCAR's actual filter; top-20 is only the fallback when nothing clears τ. Per
  query we record `|S'|` and whether it fell back, reported as a paragraph, not
  a table row.

## 6. Tier 2 — geometry re-ranking (Sub-step B2, pseudo-code)

Only the top-`GEOM_K` of the fused ranking is touched; the tail keeps its
fusion order.

```
function apply_geometry(spec, ranking):
    top = ranking[:GEOM_K]
    for each cad in top:
        if (query, cad) not in pair_cache:
            corr   = GeDi_descriptors(query_pc, cad_pc)      # HTTP to gedi service
            T, fit = RANSAC(corr)                             # rigid alignment
            d      = trimmed_surface_distance(query_pc, cad_pc after T)
            (with ICP: refine T first — OFF by default)
            pair_cache[(query,cad)] = { fitness: fit, d_ransac: d, ... }

    key(cad) = case spec.geometry:
        'fitness'          ->  fitness                    (higher better)
        'chamfer_ransac'   -> -d_ransac                   (closer better)
        'both_borda'       ->  mean rank of (fitness, -d_ransac)   # scale-free
        'both_borda_base'  ->  mean rank of (fitness, -d_ransac, base-fusion rank)
    reordered_top = sort(top by key, desc)
    return reordered_top + ranking[GEOM_K:]
```

Why it matters:

- The **pair cache is per-(query, cad) and depth-independent**, so K=50 reuses
  every K=20 pair and only computes ranks 21–50. It is resumable (a crash keeps
  all completed fits) and never caches a fit made while the GeDi service was
  down (that would poison the ranking permanently).
- **`both_borda` combines fitness and distance by mean rank, not a weighted
  sum** — fitness spans ~0.23 while D_trim spans ~0.09, so a raw sum would just
  reproduce the fitness ranking. Borda is scale-free and parameter-free.
- Geometry can **re-order** the shortlist but never **insert** into it, so the
  base-fusion hit-rate@K is a hard ceiling on every top-1 metric. That is why K
  is read off the hit-rate curve (or set explicitly: we used 20 and 50), not
  guessed.

## 7. Metrics — two tables

Two scorings run on the same full ranking per query:

- **Table A — official, depth-independent** (`score_official`): calls the
  SHREC'18 track's *unmodified* `metrics.py` on the top-`f` results
  (`f` = category size). Yields nDCG / precision / recall / F1 / AP / NNT1 /
  NNT2. Leaderboard-comparable; **invariant to any permutation of the first
  K < f entries**, so it cannot see B2 geometry re-ranking.

- **Table B — depth-matched** (`score_depth_matched`): everything cut at the
  geometry depth K, where re-ranking *is* visible. Reports NN_cat / **NN_sub**
  (sub-category, the geometry question) / MRR / mAP@K / a *corrected* nDCG@K,
  plus `hit_cat@N` / `hit_sub@N` for N ∈ {1,5,10,20,50,100} (which yields the
  hit-rate curve used to choose K).

```
for each query:
    ranking   = derive_ranking(spec, query)
    if geometry: ranking = apply_geometry(spec, ranking)
    A[query]  = score_official(ranking, gt)          # depth-independent
    B[query]  = score_depth_matched(ranking, gt, K)  # cut at K
metrics   = mean over queries of A
depth     = mean over queries of B
write metrics_summary.json + results_per_query.json
```

## 8. Top-level flow (what `main` / `stage1_reproduce.py` does)

```
Phase 0  load gallery ids, official GT, query index
Phase 1  run every needed tier-1 pass (cached)              # expensive, once
         calibrate CLIP τ if any arm needs it
Phase 2  for each ablation: derive_ranking -> [geometry] -> score  (cheap)
Phase 3  fill in aliases; aggregate -> stage1_summary.csv/.tex,
         best_config.json; select winner by nDCG (tie mAP)
```

## 9. Outputs

```
object_retrieval/results_shrec18_v2_stage1/
    <ablation>/metrics_summary.json    Table A + Table B metrics + full config
    <ablation>/results_per_query.json  per-query top-10, ranks, deltas
    stage1_summary.csv / .tex          aggregate across all arms (thesis table)
    best_config.json                   the frozen winner
    _cache/scores_<pass>.pt            tier-1 score vectors (reused across runs)
    _cache/geometry_scores.jsonl       per-(query,cad) B2 pair cache (resumable)
    k20/  k50/                         archived summaries per geometry depth
```

## 10. What we actually ran

- Gallery: **`shrec18_v2`** (the render-fixed gallery: weld + recomputed
  normals, Standard material, camera lights).
- Two geometry depths: **K=20** (`hit_sub` ceiling 0.746) and **K=50**
  (ceiling 0.864). Table A is identical across depths; only the geometry arms
  and Table B move.
- **Best arm: `E2_both` (fitness + distance, mean-rank Borda), nDCG 0.6428 at
  K=50** (vs 0.6292 at K=20; BASE 0.5970).
- Notable negative-transfer finding: CLIP threshold pruning with OSCAR's
  τ=0.37 admits nothing on ~97% of SHREC'18 queries (untextured CAD), so it
  falls back to top-k and *hurts* relative to full-database fusion — reported
  explicitly rather than hidden.

---

*Faithful reproduction:* `experiments/stage1_reproduce.py` imports the real
encoders, fusion, geometry and metrics from
`experiment1_shrec18_stage1.py` — it re-expresses control flow and config, not
the algorithms, so its numbers are identical to the canonical script's.
