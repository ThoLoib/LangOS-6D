# Stage-3 source-audit response

**Responds to:** `TESSA_PC_STAGE3_SOURCE_AUDIT.md` (read-only static audit of
branch `tessa-pc` @ `42451b4`, 2026-08-12).
**Fixes commit:** `1430348b` (branch `tessa-pc`).
**Scope gated:** the ULIP-pc **full-fusion A-vs-B** rerun
(`results_bop_stage3_ulippc` vs `results_bop_stage3_ulippc_dgedi`, both 3a+3b,
`--pose`).

Each finding below is marked:
**APPLIED** (fixed in `1430348b`) · **DEFERRED** (real, but not fixed now — with
reason) · **N/A-RUN** (real, but not on this run's code path) · **CORRECT**
(verified fine by the audit). Findings were re-verified against the live code
before disposition — the audit was static/other-PC and a few needed checking.

---

## P0 — pre-final findings

| # | Finding | Disposition | What / why |
|---|---|---|---|
| P0.1 | Effective weights `0/0.5/0.5`, not `0.3/0.4/0.3` | **APPLIED** | Verified live. Set explicit `weight_clip=0.3, weight_dino=0.4, weight_ulip=0.3` in `stage3_gallery._base_cfg`. Forces the A/B rerun. |
| P0.2 | VSD taus `× diameter` **and** `normalized_by_diameter=True` (diameter twice) | **APPLIED** | Pass `_VSD_TAUS` dimensionless in `stage3_metrics.pose_errors`. Smoke: VSD 0.68–0.90 (was invalid). |
| P0.3 | Gallery partials in object frame, query in camera frame → pc-query not SO(3)-invariant | **DEFERRED (caveat)** | The only fixes are oracle-only (GT-pose canonicalize) or heavy (regen all partials). Documented as the likely reason **ULIP-pc 0.503 < ULIP-image 0.645**; not a shippable fix. Affects A and B equally, so the dGeDi delta is unaffected. |
| P0.4 | dGeDi query ÷ 2-pass mesh-vertex diameter, gallery ÷ max-pairwise FPS diameter | **APPLIED** | `compute_diameters.py` now reproduces the gallery's own divisor exactly (seeded 10k → FPS(6000) → `_diameter`, → metres). `diameters.json` regenerated; no descriptor regen needed. |
| P0.5 | `geo_applied=true` on any non-empty dict, even all-`ok=false` | **APPLIED** | `geo_applied` now requires ≥1 successful registration; record `dgedi_n_requested/n_ok` + per-dataset/combined `dgedi_coverage`. |
| P0.6 | Fusion clips negative cosines to 0; missing channel also 0 | **DEFERRED** | Edits shared `ScoreFusion` (Stage-1/MI3DOR use it) → risks frozen results. Consistent across A/B; in full scope every object is scored in every channel, so only tail ranking is affected. Do as a separate, Stage-1-re-verified change. |
| P0.7 | D_sym drops failed poses, averages survivors | **APPLIED** | `summarize_dsym` reports `n_attempted/n_failed/coverage` alongside the conditional mean. |
| P0.8 | Internal `bop_ar()` pools instances; official BOP is object-balanced | **APPLIED (enabler)** | Store raw `oracle_R/t` + `top1_R/t` in records; new `build_bop_csv.py` → BOP result CSVs → official `bop_toolkit/eval_bop19_pose.py` post-hoc (also gives correct VSD for the headline number). |

Extra (not in audit, from live runtime): **FoundationPose CUDA-OOM** — `gedi` +
`dgedi` + eval + FP on one 24 GB GPU starved FP (oracle AR 0.079 vs known-good
0.823). **APPLIED (ops):** launcher stops `gedi`, keeps `dgedi` down during Run A,
recreates it before Run B.

---

## P1 — preprocessing / representation

| # | Finding | Disposition | Note |
|---|---|---|---|
| P1.1 | Stage-3 query cloud ≠ production (no gating/SOR/voxel) | **DEFERRED** | Stage-3 is an oracle-raw track by design; document, don't unify now. |
| P1.2 | Production median depth-gate includes invalid values | **N/A-RUN** | `step2` production path; Stage-3 uses `query_cloud.backproject_masked`. Production hardening. |
| P1.3 | Point sampling / RANSAC not reproducible | **DEFERRED** | Noted; A/B is deterministic enough for the comparison. See §5.4. |
| P1.4 | Sparse cloud (<64) silently switches pc→image | **APPLIED** | `pc_query_fallback` flag per record + summary count. |
| P1.5 | DINO "mean" includes CLS token | **DEFERRED** | Minor, consistent gallery/query; fixing needs a DINO cache re-encode. |
| P1.6 | ROI crop differs (gallery/Stage-3/production) | **DEFERRED** | Separate oracle-mask track; document. |
| P1.7 | Colored-PC domains asymmetric (CAD vs sensor color) | **DEFERRED** | Future XYZ/RGB/normalized ablation. |
| P1.8 | Proxy scale not estimated before D_sym | **DEFERRED (labelled)** | D_sym is intentionally an *unscaled-proxy* discrepancy; name it as such. |

---

## §5 — dGeDi ops / provenance

| # | Finding | Disposition | Note |
|---|---|---|---|
| 5.1 | Descriptor cache lacks provenance validation | **PARTIAL** | P0.5 adds coverage/counts; full provenance block deferred. |
| 5.2 | Gallery seed uses mesh basename (collisions on `model.obj`) | **DEFERRED** | Affects reproducibility, not this run's correctness. |
| 5.3 | Running service keeps stale caches in memory | **APPLIED (ops)** | Launcher `--force-recreate dgedi` before Run B to reload `diameters.json`. |
| 5.4 | dGeDi ranking stochastic (unseeded keypoints/RANSAC) | **DEFERRED** | Single-seed for this run; seed sweep is a method-selection concern. |
| 5.5 | Exact diameter `_diameter` ~864 MB peak | **ACKNOWLEDGED** | Kept exact to match the gallery; one-time CPU cost. |
| 5.6 | 512-kp / 5k-iter RANSAC is an OSCAR approximation | **DOCUMENTED** | Labelled as approximation, not exact upstream reproduction. |

---

## §6 — production pipeline · **N/A-RUN**

6.1–6.10 (geometry-rerank top-1, unreadable-CAD ranking, metre-vs-native geometry,
ICP inverse convention, scale-after-alignment, B2/scale-gate identity, ambiguous
depth API, CLIP top-k on caption rows, unused text query, legacy GeDi config) all
live in `run_pipeline.py` / `step7` / `step8` / production `step_b2`. The Stage-3
eval driver uses `eval_bop_pose.py` + FoundationPose and does **not** exercise
these paths. **Deferred to production hardening** (real, but out of scope here).

## §7 — alternative encoders · **N/A-RUN**

We run **ULIP-pc**. 7.1 (Uni3D query embedding ignored — explains the earlier
Uni3D-pc 0.24 collapse), 7.2 (Uni3D cache fingerprint), 7.3 (SigLIP pooling),
7.4 (ULIP fail-open load) matter only for `--uni3d`/SigLIP runs. **Deferred**;
7.1/7.4 flagged for whenever Uni3D is revisited.

## §8 — verified correct · **no action**

Depth mm→m via `depth_scale`; OpenCV back-projection; RGB/depth/mask/K aligned;
`--pc-query` genuinely encodes the cloud; L2-normalized embeddings; dGeDi
transform direction + row pairing; exact-target pose uses `models_eval` +
symmetry; FoundationPose depth-in-metres + BOP mm→m mapping.

---

## Net

The applied set (P0.1, P0.2, P0.4, P0.5, P0.7, P0.8 + ops) yields a
correctly-labelled full-fusion A-vs-B run with trustworthy retrieval, valid
pose/BOP-AR, honest dGeDi coverage, and failure-aware D_sym — without heavy
cache regeneration or destabilising shared code. Deferred items are either
consistent across A/B (so the dGeDi delta is unaffected) or off this run's code
path.
