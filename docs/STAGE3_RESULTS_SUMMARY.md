# Stage 3 — Proxy-CAD Pose Evaluation: Results Summary

**Status:** complete (run `results_bop_stage3_v2`, finished 2026-08-20). All results synced to
`gdrive:Masterthesis/OSCAR/object_retrieval/results_bop_stage3_v2`.

Stage 3 tests the core OSCAR+ hypothesis for downstream pose: **does the retrieved CAD support
6-DoF pose estimation, and how much accuracy is lost when the retrieved model is a *proxy* rather
than the object's own CAD?** Pose is estimated with **FoundationPose**; retrieval and pose are
isolated from segmentation by using the BOP ground-truth visible bounding box + mask.

## Setup at a glance

| | |
|---|---|
| **Datasets** | YCB-V, T-LESS, LM-O (`test_targets_bop19`) — **12,284 instances** (ycbv 4123, tless 6716, lmo 1445) |
| **Modes** | `3a` retrieval only · `gt` exact-CAD FP benchmark → `D_posed_gt` · `3b` proxy pose → `D_posed` + `Delta` |
| **Gallery** | `3a`: G_proxy (GSO ∪ HouseCat6D ∪ ITODD) ∪ all target CADs = **1316**;  `3b`: G_proxy only (targets removed → top-1 is always a proxy) |
| **Query modes** | `pc` = partial point cloud (ULIP PointBERT) · `cross` = RGB crop (ULIP image encoder) |
| **Fusion** | CLIP + DINOv2 + ULIP-2, weights **0.3 / 0.4 / 0.3**, DINO mean pooling, 42 views |
| **Geometry** | dGeDi re-rank, repo config (6000 keypoints / 10k RANSAC / +ICP), top-5 shortlist |
| **Pose metric** | `D_sym` = symmetric mean surface distance (mm and /diameter) + F-score @1% and @5% of diameter |
| **Substitution cost** | `Delta = D_posed − D_posed_gt`, paired per instance on `(dataset, scene, image, obj, gt_idx)` |

---

## 1. Retrieval (3a)

Recall@k / MRR over all 12,284 instances. The single relevant item is the exact target CAD
(present in the 3a gallery). Four variants: query mode {pc, cross} × geometry {off, dGeDi}.

| Variant | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| pc | 0.464 | 0.726 | 0.808 | 0.584 |
| pc + geo | 0.413 | 0.726 | 0.808 | 0.547 |
| **cross ✅ (best)** | **0.482** | **0.733** | **0.812** | **0.597** |
| cross + geo | 0.458 | 0.733 | 0.812 | 0.579 |

**Per-dataset R@1** (pre-geometry → post-geometry where geometry applies):

| Variant | YCB-V | T-LESS | LM-O |
|---|---|---|---|
| pc | 0.671 | 0.350 | 0.400 |
| pc + geo | 0.534 | 0.336 | 0.426 |
| cross | 0.732 | 0.332 | 0.464 |
| cross + geo | 0.602 | 0.360 | 0.504 |

**Findings**
- **Cross query wins overall** (R@1 0.482 vs 0.464), winning YCB-V (+0.061) and LM-O (+0.064).
  **pc wins T-LESS** (0.350 vs 0.332) — the texture-less objects favour the point-cloud query
  over the appearance-based image query.
- **Geometry re-ranking is net-negative for retrieval**, but the effect is dataset-dependent:
  it **helps where semantics are weak** (LM-O both modes: 0.400→0.426, 0.464→0.504; T-LESS-cross:
  0.332→0.360) and **hurts where semantics are already strong** (YCB-V cross: 0.732→0.602). The
  YCB-V drop dominates the instance-weighted mean. R@5/R@10 are unchanged by geometry — it only
  reorders within the top-5 shortlist, so it can move a correct top-1 down but never change set
  membership.

---

## 2. Exact-CAD pose benchmark (`D_posed_gt`)

FoundationPose with each object's **own** CAD (BOP `models_eval`), same GT pose target. This is
the upper bound and the reference for the substitution cost. `n = 12,284`, 0 failures, coverage 1.0.

| | D_sym mean | D_sym median | /diam mean | /diam median | F@1% | F@5% |
|---|---|---|---|---|---|---|
| **Combined** | 4.87 mm | **1.72 mm** | 0.054 | 0.015 | 0.341 | **0.944** |
| YCB-V | 2.83 | 2.03 | — | — | 0.483 | 0.981 |
| T-LESS | 6.04 | 1.41 | — | — | 0.282 | 0.937 |
| LM-O | 5.30 | 3.48 | — | — | 0.214 | 0.875 |

Exact-CAD FoundationPose is **highly accurate** (median 1.4–3.5 mm, F@5% 0.88–0.98), confirming the
benchmark and pose harness are sound.

---

## 3. Proxy pose (3b): `D_posed` + substitution cost `Delta`

Gallery is proxies only, so the top-1 is never the exact CAD. Two runs: the best retrieval config
(**cross**) with and without dGeDi geometry. Both paired against the same `D_posed_gt`.

### 3b — cross (no geometry)

| | D_sym mean | D_sym median | F@5% | Delta median |
|---|---|---|---|---|
| **Combined** | 33.6 mm | **18.4 mm** | **0.302** | **15.8 mm** |
| YCB-V | 31.8 | 23.6 | 0.359 | 20.8 |
| T-LESS | 32.3 | 13.6 | 0.283 | 11.6 |
| LM-O | 45.0 | 28.5 | 0.223 | 24.6 |

### 3b — cross + geometry (dGeDi)

| | D_sym mean | D_sym median | F@5% | Delta median |
|---|---|---|---|---|
| **Combined** | 44.0 mm | 28.8 mm | 0.254 | 26.1 mm |
| YCB-V | 33.8 | **21.6** | 0.411 | **18.9** |
| T-LESS | 46.3 | 29.1 | 0.176 | 26.5 |
| LM-O | 62.3 | 45.9 | 0.168 | 41.4 |

**Findings**
- The proxy imposes a **large, quantified pose penalty**: combined proxy median 18.4 mm vs exact-CAD
  1.7 mm, F@5% 0.302 vs 0.944. The substitution cost `Delta` is **15.8 mm median** (11.6–24.6 mm
  per dataset) — the tax for posing with a stand-in CAD instead of the true one.
- **Geometry hurts proxy pose overall** (combined median 18.4 → 28.8 mm, F@5% 0.302 → 0.254). It
  **helps only YCB-V** (median 23.6 → 21.6, Delta 20.8 → 18.9, F@5% 0.359 → 0.411) and **hurts
  T-LESS and LM-O** substantially.
- **Notable cross-mode nuance:** in *retrieval* (3a) geometry *helped* LM-O and T-LESS-cross, yet in
  *pose* (3b) it *hurts* them. The reason is that 3b's gallery is proxies-only — a *different*
  retrieval task than 3a — so a proxy the geometry re-rank judges "more geometrically similar" is
  not necessarily a better CAD for FoundationPose to refine against.

---

## 4. Key takeaways

1. **Cross (image-query) is the best retrieval configuration** overall; the exception is the
   texture-less T-LESS, where the point-cloud query is better.
2. **Geometric re-ranking (dGeDi) does not help in the BOP setting** — net-negative for both
   retrieval R@1 and proxy pose — despite helping SHREC'18's category retrieval (Stage 1). Its value
   is confined to weak-semantics cases (LM-O, texture-less T-LESS) in *retrieval*, and even there it
   does not translate into better *pose*. Geometry belongs in the pose stage, not retrieval re-ranking.
3. **Exact-CAD pose is excellent; the proxy is the bottleneck.** The 15.8 mm median substitution
   cost is the headline number quantifying "how much does using a retrieved proxy cost you."
4. Best headline configuration: **cross query, no geometry** (used for the reported 3b).

---

## 5. Metric definitions

- **Recall@k / MRR** — standard retrieval metrics; the one relevant item is the exact target CAD
  (in the 3a gallery only).
- **D_sym** — symmetric mean point-to-surface distance between the GT-posed target and the
  estimated-posed CAD, sampled from N=10,000 uniform surface points per mesh (sampled-Chamfer
  approximation; DiffCD, Härenstam-Nielsen et al., ECCV 2024). Reported in mm and normalised by
  target diameter.
- **F-score @τ** — with τ = frac × target-diameter (frac ∈ {0.01, 0.05}); precision = fraction of
  proxy points within τ of the target, recall = fraction of target points within τ of the proxy,
  F = 2PR/(P+R) (Knapitsch et al., Tanks and Temples, 2017).
- **Delta = D_posed − D_posed_gt** — paired per instance; the pose accuracy lost by substituting a
  retrieved proxy for the object's own CAD.

## 6. Provenance & reproducibility

- **Best-variant selection** (3a → 3b) is automatic by combined Recall@1 → **cross**.
- **Deterministic:** RNG seeding (`PYTHONHASHSEED=0`, seeded surface sampling), fused ranking, dGeDi
  Borda re-rank. **Not bit-reproducible:** FoundationPose GPU hypothesis sampling and open3d RANSAC —
  so the raw per-instance R/t poses are **stored** in every `records.json`, making all pose metrics
  reproducible from the stored poses. See `object_retrieval/STAGE3_DETERMINISM.md`.
- **Units:** FoundationPose works in metres; BOP meshes scaled 0.001 (mm→m) and returned t×1000.
- **Result files:** `object_retrieval/results_bop_stage3_v2/{3a_pc,3a_pc_geo,3a_cross,3a_cross_geo,
  gt,3b_cross,3b_cross_geo}/` — each with `combined_*.json`, per-dataset dirs, and `records.json`
  (raw poses + retrieved shortlist top-10).
