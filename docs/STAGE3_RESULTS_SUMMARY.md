# Stage 3 — Proxy-CAD Pose Evaluation: Results Summary

**Status:** complete. Final run chain finished **2026-08-30 21:41**; all 16 result directories
synced to `gdrive:Masterthesis/OSCAR/object_retrieval/results_bop_stage3_v2`.
Artifact: <https://claude.ai/code/artifact/280da703-95e6-461e-832a-1fdd92c24ab8>

Stage 3 tests the core OSCAR+ hypothesis for downstream pose: **does the retrieved CAD support
6-DoF pose estimation, and how much accuracy is lost when the retrieved model is a *proxy* rather
than the object's own CAD?** Pose is estimated with **FoundationPose**; retrieval and pose are
isolated from segmentation by using the BOP ground-truth visible bounding box + mask.

> **Supersedes the 2026-08-24 version of this file.** The geometry numbers there came from a single
> combined-signal run (`3a_cross_geo` = 0.458) predating the signal separation. Distance and fitness
> are now measured as separate arms (0.423 / 0.428), and the per-dataset conclusions are weaker than
> that run suggested — see §3.

## Setup at a glance

| | |
|---|---|
| **Datasets** | YCB-V, T-LESS, LM-O (`test_targets_bop19`) — **12,284 instances** (ycbv 4123, tless 6716, lmo 1445) |
| **Modes** | `3a` retrieval only · `gt` exact-CAD FP benchmark → `D_posed_gt` · `3b` proxy pose → `D_posed` + `Delta` · `3c` decomposition of `Delta` |
| **Gallery** | `3a`: G_proxy (GSO ∪ HouseCat6D ∪ ITODD) ∪ all target CADs = **1316**;  `3b`: G_proxy only = **1257** (targets removed → top-1 is always a proxy) |
| **Query modes** | `pc` = partial point cloud (ULIP PointBERT) · `cross` = RGB crop (ULIP image encoder) |
| **Gallery representation** | partial views (42 per CAD, base) · full mesh (A4 ablation) |
| **Fusion** | CLIP + DINOv2 + ULIP-2, weights **0.3 / 0.4 / 0.3**, DINO mean pooling, 42 views, top-k-softmax k=5 τ=0.5 — identical to Stage 1 and Stage 2 |
| **Geometry** | dGeDi re-rank, repo config (6000 keypoints / 10k RANSAC / +ICP), top-5 shortlist, signal ∈ {distance, fitness} |
| **Pose metric** | `D_sym` = symmetric mean surface distance (mm and /diameter) + F-score @1% and @5% of diameter |
| **Substitution cost** | `Delta = D_posed − D_posed_gt`, paired per instance on `(dataset, scene, image, obj, gt_idx)` |

---

## 1. Retrieval (3a)

Recall@k / MRR over all 12,284 instances. The single relevant item is the exact target CAD
(present in the 3a gallery).

| Arm | R@1 | R@5 | R@10 | MRR | YCB-V | T-LESS | LM-O |
|---|---|---|---|---|---|---|---|
| **cross, partial ✅ (frozen config)** | **0.4818** | 0.7330 | 0.8120 | 0.5971 | **0.732** | 0.332 | 0.464 |
| pc, partial | 0.4636 | 0.7258 | 0.8080 | 0.5844 | 0.671 | 0.350 | 0.400 |
| cross, full mesh | 0.4639 | **0.7680** | **0.8353** | **0.6021** | 0.566 | **0.396** | **0.490** |
| pc, full mesh | 0.3504 | 0.5793 | 0.6955 | 0.4592 | 0.635 | 0.157 | 0.436 |
| OSCAR baseline (E5, no shape) | 0.3198 | 0.4923 | 0.5418 | 0.4043 | 0.498 | 0.214 | 0.304 |

The three right-hand columns are per-dataset R@1. The OSCAR baseline is the faithful cascade:
CLIP-text threshold τ=0.37 shortlist → DINOv2 best-view re-rank, **no shape channel**.

**Findings**

- **The shape channel is worth +0.162 R@1** over the OSCAR cascade (0.4818 vs 0.3198). The margin
  holds on all three datasets and is relatively largest on T-LESS (0.214 → 0.332), i.e. exactly
  where texture and language carry least information.
- **Cross query beats pc overall** (0.4818 vs 0.4636), winning YCB-V (+0.061) and LM-O (+0.064);
  **pc wins T-LESS** (0.350 vs 0.332) — texture-less objects favour the point-cloud query over the
  appearance-based image query.

---

## 2. Gallery representation — partial views vs full mesh

The 2×2 of gallery representation × query modality, R@1:

| Query | partial | full mesh | Δ |
|---|---|---|---|
| cross | **0.4818** | 0.4639 | −0.018 |
| pc | **0.4636** | 0.3504 | −0.113 |

> **Provisional.** The four arms in this section are being re-measured
> (2026-09-01); the numbers may shift. The per-dataset structure and the mechanism below are
> established independently of that and are not expected to change.

**The aggregate is not the result — report the per-dataset split.** R@1 per dataset:

| query | dataset | partial | full mesh | Δ | share of instances |
|---|---|---|---|---|---|
| pc | YCB-V | 0.671 | 0.635 | −0.036 | 34 % |
| pc | **T-LESS** | 0.350 | **0.157** | **−0.193** | 55 % |
| pc | LM-O | 0.400 | 0.436 | **+0.036** | 12 % |
| cross | YCB-V | 0.732 | 0.566 | −0.166 | 34 % |
| cross | T-LESS | 0.332 | 0.396 | +0.064 | 55 % |
| cross | LM-O | 0.464 | 0.490 | +0.026 | 12 % |

T-LESS alone accounts for **93 %** of the pc-mode aggregate drop; on LM-O full mesh is *better* in
both modes, and in cross mode it also wins T-LESS, R@5, R@10 and MRR. Quoting "partial beats full
mesh by 0.113" is therefore misleading — it is in substance a statement about T-LESS.

**The mechanism is query–gallery domain match, not surface coverage.** The decisive evidence is
the asymmetry between the two query modes above: the gallery is point-cloud-encoded in *both*, so
any property of the gallery alone would affect them equally. It affects pc **six times** more. In
pc mode the query is a partial cloud and the partial gallery consists of exactly such clouds —
same domain — and each gallery object gets 42 chances to match the query's viewpoint, aggregated
by top-k-softmax over the best 5. A full mesh offers one embedding of a complete surface,
normalised onto the same unit sphere as the query patch. In cross mode the query passes through
the image tower, so neither representation is domain-matched and the gap collapses to 0.018.

The per-dataset staggering follows from how much the shape channel has to carry: on YCB-V text and
appearance absorb a domain-shifted shape channel, on texture-less T-LESS nothing does.

**Full mesh is informative but mis-calibrated at rank 1.** Target-rank distribution on T-LESS
(pc mode):

| arm | R@1 | median rank | > rank 50 |
|---|---|---|---|
| partial | 0.350 | 2 | 6 % |
| full mesh | 0.157 | 8 | 14 % |
| no shape channel (E5) | 0.214 | 24 | 28 % |

Full mesh still lifts the median rank far above the no-shape baseline (8 vs 24) — the channel is
not uninformative. It fails specifically at the top of the list, which is what Recall@1 measures.

Stage 1 (SHREC'18, pc-mode throughout) shows the same sign at +0.0495 nDCG in the isolated shape
channel, so the direction transfers; the BOP runs additionally record the per-channel target rank,
making the isolated comparison available there too.

Both full-mesh arms logged 100.0 % gallery coverage on all six datasets, verified per dataset by
the coverage gate added on 2026-08-31 (see `DECISIONS.md`).

---

## 3. Geometric re-ranking

dGeDi descriptors + RANSAC over the top-5, with the two signals measured separately:
**distance** = registration distance after alignment, **fitness** = inlier-radius overlap fraction.

| Query · signal | R@1 | Δ R@1 | MRR | R@5 | R@10 | coverage | YCB-V | T-LESS | LM-O |
|---|---|---|---|---|---|---|---|---|---|
| cross, none | **0.4818** | — | **0.5971** | 0.7330 | 0.8120 | — | 0.732 | 0.332 | 0.464 |
| cross, distance | 0.4229 | −0.059 | 0.5576 | 0.7330 | 0.8120 | 98 % | 0.542 | 0.338 | 0.480 |
| cross, fitness | 0.4278 | −0.054 | 0.5569 | 0.7330 | 0.8120 | 98 % | 0.558 | 0.337 | 0.480 |
| pc, none | **0.4636** | — | **0.5844** | 0.7258 | 0.8080 | — | 0.671 | 0.350 | 0.400 |
| pc, distance | 0.3725 | −0.091 | 0.5215 | 0.7258 | 0.8080 | 98 % | 0.477 | 0.305 | 0.387 |
| pc, fitness | 0.3820 | −0.082 | 0.5249 | 0.7258 | 0.8080 | 98 % | 0.490 | 0.314 | 0.390 |

**R@5 and R@10 are identical across every row.** Re-ranking only reorders within the top-5, so it
cannot change set membership at K≥5. Same structural blindness as Stage 1, where five of the seven
official metrics could not see the geometry stage at all.

**Findings**

- **Geometry loses in all four clean cells**, and loses harder in pc-mode (−0.09 vs −0.06) —
  consistent with the shape channel already carrying the depth information the re-rank re-derives.
- **This is not an implementation fault.** At 98 % registration coverage the geometry demonstrably
  fires, and it is informative in absolute terms: it puts the correct CAD at rank 1 in **58 %** of
  cases against 20 % chance with five candidates. But the fusion score it *replaces* achieves
  **66 %** within the same shortlist. Re-ranking is displacement, not addition — it pays only when
  it beats the score it overwrites. On SHREC'18 it does (47 % vs 34 %), on BOP it does not.
- **Fitness beats distance on BOP in both modes**, inverting Stage 1 (distance 0.6405 > fitness
  0.6251 nDCG). SHREC'18 is scale-invariant (`diameters.json` all 1.0), so the alignment distance is
  meaningful there; on BOP with true metric scale (0.024–1.54 m) the pure overlap measure wins.
- The per-dataset pattern survives but is **much weaker than the 2026-08-24 legacy run suggested**:
  geometry still helps slightly where semantics are weak (LM-O +0.016, T-LESS +0.005 for both
  signals) and hurts badly where they are strong (YCB-V −0.19). The legacy combined-signal run
  reported LM-O +0.040; that magnitude does not reproduce.

---

## 4. Exact-CAD pose benchmark (`D_posed_gt`)

FoundationPose with each object's **own** CAD (BOP `models_eval`), same GT pose target. This is the
upper bound and the reference for the substitution cost. `n = 12,284`, 0 failures, coverage 1.0.

| | D_sym mean | D_sym median | /diam mean | /diam median | F@1% | F@5% |
|---|---|---|---|---|---|---|
| **Combined** | 4.87 mm | **1.72 mm** | 0.054 | 0.015 | 0.341 | **0.944** |
| YCB-V | 2.83 | 2.03 | — | — | 0.483 | 0.981 |
| T-LESS | 6.04 | 1.41 | — | — | 0.282 | 0.937 |
| LM-O | 5.30 | 3.48 | — | — | 0.214 | 0.875 |

Exact-CAD FoundationPose is **highly accurate** (median 1.4–3.5 mm, F@5% 0.88–0.98), confirming the
benchmark and pose harness are sound.

---

## 5. Proxy pose (3b): `D_posed` + substitution cost `Delta`

Gallery is proxies only (1257), so the top-1 is never the exact CAD. Three arms, all paired against
the same `D_posed_gt`.

| Arm | D_sym median | /diam median | Delta median | coverage | YCB-V | T-LESS | LM-O |
|---|---|---|---|---|---|---|---|
| **OSCAR+ cross ✅** | **18.37 mm** | 0.139 | **15.79 mm** | 1.000 | 23.59 | **13.57** | 28.54 |
| OSCAR baseline (E5) | 21.73 mm | 0.173 | 18.86 mm | 0.9996 | 24.38 | 18.43 | 31.22 |
| OSCAR+ cross + geometry | 28.79 mm | 0.253 | 26.07 mm | 1.000 | — | — | — |

Right-hand columns are per-dataset `D_sym` medians in mm. The OSCAR baseline had 5 failures out of
12,284.

**Findings**

- **The retrieval margin carries through to pose.** OSCAR+ beats the OSCAR baseline by **3.4 mm**
  median (18.37 vs 21.73 mm) and does so on all three datasets. The +0.162 R@1 is therefore not an
  isolated retrieval number — it converts into a usable pose improvement.
- The proxy imposes a **large, quantified penalty**: 18.4 mm median vs 1.7 mm with the exact CAD,
  F@5% 0.302 vs 0.944. `Delta` = **15.8 mm median** is the headline substitution cost.
- **Geometry hurts pose even more than retrieval**: +57 % error against the same arm without
  re-ranking (18.37 → 28.79 mm). Combined with §3, the geometry axis is of no benefit anywhere in
  the BOP chain.

---

## 6. Decomposition (3c) — where the 15 mm come from

Only the exact GT model is removed from the gallery; everything else stays. This separates the case
where the best remaining substitute is a real CAD of a *different* object from the case where it
comes from the external proxy gallery.

| Case | n | share | D_sym median | D_sym mean |
|---|---|---|---|---|
| All | 12,284 | 100 % | 15.34 mm | 28.05 mm |
| Substitute is a real CAD (different object) | 6,742 | 54.9 % | **10.35 mm** | 16.86 mm |
| Substitute from the proxy gallery | 5,542 | 45.1 % | 20.10 mm | 41.67 mm |
| — YCB-V | 4,123 | — | 19.11 mm | 28.59 mm |
| — T-LESS | 6,716 | — | **8.75 mm** | 24.91 mm |
| — LM-O | 1,445 | — | 24.25 mm | 41.15 mm |

`n_target_was_top1 = 5,918`; `n_same_dataset = 6,216`.

**Findings**

- **Roughly half the substitution error is addressable.** A real CAD of another object costs
  10.35 mm; a proxy-gallery model costs 20.10 mm — nearly double. Gallery composition is therefore
  its own lever: part of the error disappears with better-matched proxies, the rest is intrinsic
  substitution loss.
- **T-LESS has the best pose median (8.75 mm) despite the worst retrieval (R@1 0.332).** Not a
  contradiction: the objects are geometrically so similar that a *wrongly* chosen CAD still supports
  a good pose. Retrieval accuracy and pose usefulness are not the same quantity — a point worth
  making explicitly in the thesis.

---

## 7. Key takeaways

1. **The shape channel is the contribution**: +0.162 R@1 over the faithful OSCAR cascade, and
   −3.4 mm pose error. Both directions of the claim are now measured.
2. **Best configuration: cross query, partial-view gallery, no geometry.** It is also the fastest —
   the stage that was switched off was the most expensive one.
3. **Geometric re-ranking does not help in the BOP setting**, for retrieval or for pose, despite
   helping SHREC'18. The criterion is general: geometry pays only where it beats the score it
   replaces. It belongs in the pose stage, not as blanket retrieval re-ranking.
4. **Partial views beat full mesh in aggregate but not per dataset.** T-LESS carries 93 % of the
   difference; on LM-O full mesh is better in both modes, and in cross mode it also wins T-LESS
   and the depth metrics. The cause is query–gallery domain match, evidenced by the pc-vs-cross
   asymmetry (−0.113 vs −0.018), not surface coverage. Report the split, not the aggregate.
5. **Exact-CAD pose is excellent; the proxy is the bottleneck.** 15.8 mm median substitution cost,
   of which roughly half is attributable to the proxy gallery rather than to substitution as such.

---

## 8. Metric definitions

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

## 9. Provenance & reproducibility

- **Gate:** the run chain promotes an arm to 3b/3c only if its 3a R@1 beats the frozen cross config
  (0.482). No arm did, so no additional pose runs were triggered — `3b_oscar` was run
  unconditionally as the baseline comparison.
- **Deterministic:** RNG seeding (`PYTHONHASHSEED=0`, seeded surface sampling), fused ranking, dGeDi
  re-rank. **Not bit-reproducible:** FoundationPose GPU hypothesis sampling and open3d RANSAC — so
  the raw per-instance R/t poses are **stored** in every `records.json`, making all pose metrics
  reproducible from the stored poses. See `object_retrieval/STAGE3_DETERMINISM.md`.
- **Units:** FoundationPose works in metres; BOP meshes scaled 0.001 (mm→m) and returned t×1000.
- **Result directories** under `object_retrieval/results_bop_stage3_v2/`:
  `3a_cross`, `3a_pc`, `3a_fullmesh`, `3a_pc_fullmesh`, `3a_oscar`,
  `3a_cross_geo{,_distance,_fitness,_borda}`, `3a_pc_geo{,_distance,_fitness}`,
  `gt`, `3b_cross`, `3b_cross_geo`, `3b_oscar`, `3c_cross` — each with `combined_*.json`,
  per-dataset dirs, and `records.json` (raw poses + retrieved shortlist top-10).
- **Superseded arms:** `3a_cross_geo` and `3a_pc_geo` are the pre-separation combined-signal runs;
  `3a_cross_geo_borda` fell through with 0 % coverage (arm deliberately killed) and equals
  `3a_cross`. Use the `_distance` / `_fitness` arms for the geometry conclusion.
