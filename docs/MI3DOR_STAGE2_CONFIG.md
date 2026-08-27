# MI3DOR (Stage-2) Retrieval — Full Config Reference

Every configuration value used for the MI3DOR Stage-2 evaluation, what was
**pinned** (and why) vs **changed** (and why). Sources:
`object_retrieval/retrieval_mi3dor_eval_oscarplus.py` (driver + CONFIG block),
`object_retrieval/eval_common.py` (`EvalConfig`, `build_pipeline`,
`run_evaluation`, `RANKING_KEYS`), `pipeline/config.py` (`PipelineConfig`
defaults).

Run date: **2026-08-05**. Results: `object_retrieval/results_mi3dor_oscarplus_f20_3/{fullmesh,partial}/`
(local + Drive `gdrive:Masterthesis/OSCAR/...`).

---

## 0. TL;DR

The **only intentional change from the frozen SHREC'18 Stage-1 winner** is
`ulip_view_topk: 8 → 5` (equalise the ULIP/shape view aggregation to DINO's).
**KORREKTUR 2026-08-27:** die Umstellung wurde erst mit `ea84ffb8` am **2026-08-07 11:17**
wirksam (bei `c2e21202`, 08-04, stand noch **8**). Der hier dokumentierte Lauf `f20_3` vom
**2026-08-05 lief daher bei `ulip_view_topk = 8`**, nicht bei 5. Erst die Läufe ab dem
07.08. (`_dinomean`) nutzen k=5. Everything else is the pinned SHREC-winner /
OSCAR-baseline config. Geometry re-ranking is **not run** on MI3DOR (retrieval-
only benchmark, no query point clouds). The driver runs **both** shape modes
(`fullmesh` + `partial`) and reports **six** ranking arms.

> ⚠️ Caveat for reporting: `ulip_view_topk=5` is **un-ablated on MI3DOR** — it
> is *not* the value the SHREC winner was measured at (that was 8). Note the
> equalised top-5 whenever these numbers are cited.

---

## 1. Data paths (fixed to this machine)

| Param | Value |
|---|---|
| `ref_dir` | `../object_images/MI3DOR` — gallery (3,848 instances × 42 views + cached embeddings) |
| `bop_root` | `../eval/datasets/mi3dor/image/test` — query images (21 categories, ~10.5k) |
| `desc_file` | `../object_database/MI3DOR/descriptions_attributes.json` — CLIP text descriptions (3,848 entries) |
| `cad_mesh_glob` | `../object_database/MI3DOR/model/test/*/*.obj` — full meshes (fullmesh mode + ULIP cache key) |
| `result_folder` | `results_mi3dor_oscarplus_f20_3` → written as `/{fullmesh,partial}/` |
| `ulip_query_cache_path` | `ulip_query_cache_mi3dor.pt` — per-query ULIP image embeddings (computed once, shared by both modes) |

Paths are relative to the `object_retrieval/` working directory (the driver is
run with `cd /app/object_retrieval`).

## 2. Encoders — FIXED (standard OSCAR+ stack, identical to SHREC'18)

| Channel | Encoder | Notes |
|---|---|---|
| Text | CLIP `ViT-B/32` | `clip_model_name`; scores query crop vs CAD view *descriptions* |
| Appearance | DINOv2 `facebook/dinov2-base` | `appearance_encoder="dinov2"`. SigLIP `google/siglip-base-patch16-224` is configured but **not exercised** (it is the E4 alternative). |
| Shape | ULIP-2 `pointbert_colored`, 10,000 pts, **xyzrgb** (`ulip2_use_colors=True`), embed 1280 | `shape_encoder="ulip2"` (Uni3D is the E7 alternative, unused). Query **images** → ULIP **cross-mode** image encoder (`ulip2_mode="cross"`); the log's `ulip_fallback_cross_count 10500` confirms all queries used it. |

## 3. View aggregation

| Param | Value | Fixed/Changed | Why |
|---|---|---|---|
| `num_views` | **42** | FIXED | SHREC O4 sweep peaked at 42 (V8 .580 < V16 .593 < V42 .597) |
| `dino_view_aggregation` | `topk_softmax` | FIXED | OPEN/CNOS-style pooling |
| `dino_view_topk` | **5** | FIXED | CNOS default k_v=5 (thesis Table 4.1) |
| `dino_view_temperature` | 0.5 | FIXED | CNOS default |
| `ulip_view_aggregation` | `topk_softmax` | FIXED | match DINO method |
| **`ulip_view_topk`** | **8 → 5** | **CHANGED** | **user request: equalise ULIP's top-k to DINO's.** Was 8 (pipeline default); now 5 so both view channels pool identically |
| `ulip_view_temperature` | 0.5 | FIXED | already matched DINO |

> Note: the `SHAPE_AGG_VIEWS=16` cap is a **Stage-1-experiment-script artefact
> only**. Production `step5._apply_partial_view_limit` trims partial views to
> `num_views`, so MI3DOR pools **all 42** partial clouds (both DINO and ULIP at
> 42).

## 4. Retrieval / fusion — FIXED (BASE config)

| Param | Value | Why |
|---|---|---|
| `fusion_method` | `weighted_sum` | OSCAR+ baseline (RRF is the E6 ablation) |
| `weight_clip / dino / ulip` | **0.30 / 0.40 / 0.30** | BASE fusion weights (SHREC E1c) |
| `clip_top_k` | **20** | CLIP-prune shortlist depth; the `*_clip_pruned` arms intersect with it |
| `dino_top_k`, `ulip2_top_k` | 9999 (auto-expanded to gallery size) | the `*_full` arms score the whole gallery (no prune) |
| `fusion_top_k` | 1 | final candidate count for the pipeline's own top-1 |
| `TOP_F` | 20 | precision / recall / F1 computed on top-20 |
| `topk` | [15] | reported retrieval depth (`results_topk_15.json`) |

**Six ranking arms reported** (`eval_common.RANKING_KEYS`):
`clip_only`, `dino_only_full`, `ulip_only_full` (cross-mode),
`dino_only_clip_pruned` (= OSCAR), `ulip_only_clip_pruned`,
**`clip_pruned_dino_ulip`** (= full fusion, the **primary** arm).

## 5. Deliberately OFF

| What | State | Why |
|---|---|---|
| **Geometry re-ranking** (GeDi/RANSAC, Sub-step B2) | **not run** | `geometry_reranking_enabled=True` in `PipelineConfig` defaults, but this eval path (`eval_common.run_evaluation`) never calls step_b2. MI3DOR is image-query retrieval only — no query point clouds / GeDi. Ran with `docker compose --no-deps`, so the gedi service never started. Stage-2 = retrieval arms only. |
| **Shape-source** | **both run** (not a single fixed value) | the driver loops `fullmesh` (`ulip2_use_partial_views=False`) **and** `partial` (True) — mirrors SHREC E2b. That is why there are two result sets. |

## 6. Data quirk handled at runtime

`_quarantine_foreign_views(ref_dir)` renames stray 3-digit gallery views
(`_NNN.png`, e.g. `_002.png`) → `*.foreign` so only the 42 real views
(`_0.png`..`_41.png`) load. Idempotent, non-destructive. Without it, `_002`
parses as index 2 and displaces the real `_41` from the top-42.

## 7. Results — CORRECTED (2026-08-07 re-run, τ=0.37 threshold, 7 arms, top-k=15)

Run: `results_mi3dor_oscarplus_v2_tau037/{fullmesh,partial}/`, n = **10,500**
queries/mode, metrics = Pulli's `retrieval_mi3dor_eval.py` scorer verbatim
(see `docs/RETRIEVAL_METRICS_REFERENCE.md`). CLIP shortlist S' = threshold
τ_text ≥ 0.37 with top-20 fallback; median |S'| = 20, fallback on 10,174/10,500
(97%) — τ=0.37 is above our image↔text CLIP scale, as documented.

**Seven arms** (fullmesh; *partial* shown only where it differs — CLIP & DINO are
shape-mode-independent, so `clip_only`/`dino_only_full` are identical across modes):

| arm | scope | NN | FT | ST | nDCG@2R | mAP | ANMRR↓ |
|---|---|---|---|---|---|---|---|
| clip_only | full DB | 67.95 | 0.575 | 0.755 | 0.720 | 0.580 | 0.339 |
| dino_only_full | full DB | 78.01 | 0.587 | 0.700 | 0.701 | 0.597 | 0.344 |
| ulip_only_full | full DB | 78.10 / 68.11 | 0.510 / 0.453 | 0.649 / 0.607 | 0.652 / 0.598 | 0.518 / 0.452 | 0.409 / 0.467 |
| **clip_dino_ulip_full** | full DB, 3-way | 83.42 / 84.11 | **0.620** | 0.745 / 0.755 | 0.746 / 0.752 | 0.635 / 0.640 | 0.304 / 0.300 |
| oscar_maxview (Pulli OSCAR) | τ=0.37 cascade | 84.79 | 0.575 | 0.755 | 0.7333 | 0.592 | 0.337 |
| oscar_softmax | τ=0.37 cascade | 84.51 | 0.575 | 0.755 | 0.7335 | 0.592 | 0.337 |
| **clip_pruned_dino_ulip** | τ=0.37 cascade | **85.93 / 85.52** | 0.575 | 0.755 | 0.734 | 0.592 | 0.337 |

Reading:
- **Cascade arms (oscar_*, clip_pruned_dino_ulip)** are full-gallery cascades
  (DINO/ULIP-reranked head + CLIP-ordered tail), so their deep-recall metrics
  equal `clip_only`'s (FT 0.575 / ST 0.755) — recall is inherited from CLIP — while
  the re-ranked head lifts **NN from 67.95 → 85.93**. Report **NN** for these.
- **`clip_dino_ulip_full`** (full-DB 3-way fusion) is the best deep-metric arm
  (FT 0.620, ANMRR 0.300–0.304).
- **`oscar_maxview` vs `oscar_softmax`** differ only marginally (NN 84.79 vs 84.51;
  nDCG 0.7333 vs 0.7335) — the view-aggregation ablation is real but small here.
- fullmesh ≈ partial on the shared arms; partial slightly wins the 3-way fusion
  (NN 84.11, ANMRR 0.300), fullmesh wins `ulip_only_full`.

### vs. published OSCAR (paper: NN 89.4, FT 0.708, ST 0.850, DCG 0.844, ANMRR 0.205)

With **corrected** metrics (CLS pooling) we land **below** OSCAR: best NN ≈ **85.9**
(cascade) and best FT ≈ **0.620** (3-way fusion) vs OSCAR's 89.4 / 0.708. **The earlier
"beats OSCAR on every metric" result (FT 0.855) was entirely the metric bug** —
see the history note below. Mean pooling (§7.1) then closes part of the remaining
gap; the rest is attributed to the **10-category confound** (Pulli evaluates on 10
MI3DOR categories, we use all 21 → far fewer distractors) and description/preprocessing
differences.

### 7.1 DINO pooling ablation — CLS vs mean (2026-08-07, full n=10,500/mode)

Run: `results_mi3dor_oscarplus_v2_tau037_dinomean/{fullmesh,partial}/`. The **only**
change from §7 is the DINOv2 pooling: our CNOS-style **CLS token** vs Pulli's
**mean-patch-token** (`last_hidden_state.mean(dim=1)`, `dino_pooling="mean"`). The
gallery DINO cache is keyed by pooling so the two never collide. Verified clean
ablation: `clip_only` and `ulip_only_full` are **exactly unchanged** (Δ=0) — only the
DINO channel moves; `dino_only_full` is identical across shape modes (DINO is
shape-mode-independent).

| arm | metric | CLS | mean | Δ |
|---|---|---|---|---|
| dino_only_full (both modes) | NN | 78.01 | **83.03** | +5.0 |
| | FT | 0.587 | **0.629** | +0.042 |
| | ANMRR↓ | 0.344 | 0.297 | −0.046 |
| clip_dino_ulip_full — fullmesh | NN / FT / ANMRR | 83.42 / 0.620 / 0.304 | 85.17 / **0.639** / 0.283 | +1.8 / +0.020 |
| clip_dino_ulip_full — **partial** | NN / FT / ANMRR | 84.11 / 0.620 / 0.300 | **87.05 / 0.648 / 0.270** | +2.9 / +0.028 |
| cascades (oscar_*, clip_pruned) | NN | 84.5–85.9 | 84.9–86.2 (±0.5) | flat FT/ST (CLIP-inherited) |

**Mean pooling is the better choice on MI3DOR** and is now the **MI3DOR default**
(`retrieval_mi3dor_eval_oscarplus.py`: `MI3DOR_DINO_POOLING` defaults to `mean`).
Best config = **partial + 3-way fusion: NN 87.05 / FT 0.648 / ST 0.786 / ANMRR 0.270**.
Note: with mean-pooled DINO, **partial now beats fullmesh** on the fusion arm (was
~tied under CLS) — the stronger DINO makes the partial ULIP shape signal more
complementary. **Scope:** this default is MI3DOR-only; the global
`PipelineConfig.dino_pooling` stays `cls`, so SHREC and other benchmarks are
unaffected until pooling is separately ablated there.

### OLD (inflated, buggy) results — DO NOT CITE, kept for the record

| Metric | fullmesh | partial | Paper OSCAR |
|---|---|---|---|
| NN | 90.32 | 89.80 | 89.4 |
| FT | 0.855 | 0.846 | 0.708 |
| ST | 0.955 | 0.955 | 0.850 |
| nDCG@2R | 0.926 | 0.923 | 0.844 (DCG) |
| mAP | 0.899 | 0.893 | — |
| ANMRR ↓ | 0.130 | 0.137 | 0.205 |

These came from two bugs (tier metrics normalised by the ≤20 pruned-survivor count
instead of true |C|; shortlist-miss queries dropped so n=10,031≠10,500), both fixed
2026-08-06. See memory `mi3dor-shrec-metric-bugfix`.

## 8. Reproducing a single run

Two options:

1. **Full standard set** (both modes, 6 arms, as run):
   edit the CONFIG block at the top of
   `object_retrieval/retrieval_mi3dor_eval_oscarplus.py`, then
   `docker compose run --rm --no-deps oscar bash -lc \
   "cd /app/object_retrieval && python3 -u retrieval_mi3dor_eval_oscarplus.py"`.
   *(Container has no `python` — use `python3 -u`.)*

2. **One specific config** (single shape mode, config-first — analogous to
   `experiments/stage1_reproduce.py` for SHREC): use
   `experiments/mi3dor_reproduce.py`. All knobs are a flat CONFIG block at the
   top; it talks to the production `build_pipeline` / `run_evaluation`
   directly and writes `results_topk_<K>.json` + `metrics_summary_topk_<K>.json`
   to `OUTPUT_DIR`.
