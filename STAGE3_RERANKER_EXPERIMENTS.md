# Stage-3 retrieval re-ranking — experiment log

**Date:** 2026-08-13 · **Branch:** `tessa-pc` · **Machine:** tessa

**Question:** Can a *geometric* cue re-rank the fused CLIP+DINO+ULIP top-K and
improve Stage-3 CAD **retrieval** (choosing the correct instance CAD from a
partial RGB-D observation), beyond the appearance/language/shape fusion?

**Config all experiments ran under** (after the audit fixes, commit `1430348b`):
full fusion **CLIP 0.3 / DINO 0.4 / ULIP-pc 0.3**, ULIP-2 **pc-query**, gallery =
G_proxy ∪ target CADs (1316). Query = GT-visible-mask back-projection (metric
depth). Metric: **Recall@1 / Recall@5** on the target CAD.

**Global caveats (apply to every number below):**
- Single-view, **GT mask**, clean-ish depth (oracle-isolated retrieval).
- ycbv scale/geometry tests use N≈100–150; render-and-compare uses **N=40**
  (SE ≈ ±0.08 — individual deltas are noisy; read the *patterns*).
- These probe *retrieval* only (no pose metrics unless stated).

---

## 0. Baseline (fused, no re-rank)

| dataset | fused R@1 | fused R@5 | object diam (median) |
|---|---|---|---|
| ycbv | 0.60–0.64 | 0.96–0.97 | 0.174 m |
| tless | 0.46–0.47 | ~0.67 | 0.092 m |
| lmo | 0.33 | ~0.58 | 0.159 m |

(ycbv baseline varies 0.60–0.64 across subsets; tless/lmo are harder.)

---

## 1. dGeDi geometric re-rank (both_borda = RANSAC fitness + trimmed chamfer)

### 1a. Re-rank the fused top-K by the geometry score — **HURTS**
ycbv, N=100, full fusion:

| variant | R@1 (K=20) | R@1 (K=5) |
|---|---|---|
| fused | 0.600 | 0.600 |
| +dGeDi 512 kp / 5k iter | 0.310 | 0.560 |
| +dGeDi 6000 kp / 10k iter | 0.400 | 0.470 |

- Re-ranking **demotes correct top-1s**; at K=20 it even pushes them out of top-5
  (R@5 0.96 → 0.52).
- **Keypoint density is not monotone:** at K=20 dense (6000 kp) > sparse (512 kp);
  at K=5 the order *reverses* (512 kp better) — geometry is an anti-signal, so a
  *weaker* signal does *less* damage.
- **Latency:** 512 kp ≈ 55 ms/pair (~1.1 s/query at K=20); 6000 kp ≈ 318 ms/pair
  (~6.4 s/query at K=20). The dGeDi reference (`demo.py`) uses 6000 kp / 10k iter
  **+ ICP**; our 512 kp is a deliberate speed approximation.
- **Why:** Stage-1 was *category* retrieval (geometry groups look-alikes → helps
  nDCG); Stage-3 is *instance* retrieval (one exact CAD → geometry can't pick the
  exact instance among shape-similar ones → it only adds noise).

### 1b. dGeDi as a conservative **GATE** (reject ~zero-overlap, keep fused order) — **NEUTRAL**
N=150, fitness threshold sweep:

| dataset | fused | thresh 0.05 | thresh 0.08 | thresh 0.20 |
|---|---|---|---|---|
| ycbv | 0.640 | 0.633 | 0.587 | 0.360 |
| tless | 0.460 | 0.460 | 0.480 | 0.327 |
| lmo | 0.327 | 0.340 | 0.340 | 0.320 |

- "Gate not reorder" **stops the harm** (neutral at low thresholds), unlike 1a.
- But it barely helps (+0.01–0.02 on tless/lmo, within noise). **Why:** the
  candidates appearance confuses are *shape-similar* → they have *good* fitness →
  the gate can't reject them. Geometry fires where it doesn't matter.

---

## 2. Metric-scale gate (`estimate_fast` naive size check)

Idea: the shape encoder is scale-invariant *by design* (ULIP/dGeDi normalize),
so absolute metric size — from RGB-D — is a complementary instance cue. Gate:
demote candidates the observation is physically **too big** for
(`sf = median(2-largest obs / cad) > 1 + tol`).

### 2a. Naive reorder vs gate (ycbv, N=150)

| variant | R@1 |
|---|---|
| fused | 0.613 |
| reorder by \|sf−1\| (pure) | 0.133 (catastrophic) |
| Borda(fused, scale) | 0.460 |
| **gate (raw bbox, tol 0.35)** | **0.747** |

→ *Gate*, not reorder, is the right framing (same lesson as 1b).

### 2b. Robust observed extent (ycbv, N=150, K=20) — **BIG ycbv gain**

| obs extent | best tol | R@1 | R@5 |
|---|---|---|---|
| raw (max−min) | 0.40 | 0.767 | 0.967 |
| **percentile (p2–p98)** | **0.10** | **0.813** | 0.973 |
| SOR | 0.20 | 0.807 | 0.967 |

- Percentile removes the depth-outlier tail → tight tol is safe → **+0.18 R@1**
  (0.633 → 0.813), R@5 untouched.
- **K-insensitive:** R@1 = 0.813 identical for K = 5, 10, 15, 20.

### 2c. Generalization to tless/lmo — **FAILS (conditional)**
At ycbv's optimal tol 0.10:

| dataset | fused | +scale gate | Δ |
|---|---|---|---|
| ycbv | 0.633 | 0.813 | +0.18 ✓ |
| tless | 0.467 | 0.333 | −0.13 ✗ |
| lmo | 0.333 | 0.280 | −0.05 ✗ |

- **No single tol helps all three.** Adaptive *absolute* margin (5–50 mm) also
  fails: lmo is hurt at *every* margin.
- **Size switch** (apply only when observed size > threshold) rescues **tless →
  neutral** (R@5 even rises) but **not lmo** — because lmo objects are the *same
  size as ycbv* (0.159 vs 0.174 m); lmo's failure is **occlusion, not size**.
- **Diagnosis:** true-target `sf` (pct extent) — ycbv median 0.91 (17% wrongly
  gated), **tless median 1.09 (47% wrongly gated)**, lmo 0.96 (25%). tless =
  small objects, so residual depth noise (~8–10 mm) is a large *fraction* of size
  → extent over-estimated; lmo = occlusion → extent unreliable. Neither is a
  removable-outlier problem, so no extent *statistic* (mean/percentile/SOR) fixes
  it.
- A reliability switch on GT `visib_fract` would fix lmo — but that's **oracle,
  not deployable**.

**Verdict:** scale is a *real* instance cue but **conditional** — helps large,
well-observed, size-diverse objects (ycbv); neutral-to-negative on small (tless)
or occluded (lmo). Not a robust universal re-ranker.

---

## 3. VLAD / pooled dGeDi global signature (retrieval, not registration)

**Not run to completion — deemed infeasible for the pipeline a priori:**
- **Redundant** with ULIP-pc (both global shape descriptors; ULIP is trained).
- **Scale-inconsistent**: `/features` self-normalizes each cloud by its own
  diameter → a partial query is blown to unit scale vs full-object gallery; a
  global signature can't co-scale per candidate (unlike the RANSAC path).
- To be good it would need **training** (PointNetVLAD/NetVLAD-style); the
  unsupervised-codebook version is a weak proxy.
- Same shape-only instance-discrimination ceiling.

→ Belongs in the thesis as an *ablation note*, not a component.

---

## 4. Render-and-compare (verify: does the posed CAD explain the depth?)

Re-rank the fused top-K by the **symmetric trimmed chamfer** between each
candidate CAD **posed** and the observed cloud. ycbv, N=40, K=5.

| pose under the chamfer | R@1 | R@5 | latency |
|---|---|---|---|
| fused (no geometry) | 0.525 | 0.900 | — |
| **FP pose, `refine_iter=5`** | **0.600** | 0.900 | ~13 s/query |
| FP pose, `refine_iter=1` | 0.450 | 0.900 | ~13 s/query |
| FPFH+RANSAC+ICP (fast, minimal ICP) | 0.450 | 0.900 | **0.18 s/query** |
| FPFH strong (3× RANSAC + multi-scale ICP, 200 iter) | 0.425 | 0.900 | 0.60 s/query |
| re-rank by FP *confidence* | 0.000 | 0.900 | — |

- **Only the accurate (full-refinement) FP pose helps** (+0.075, but N=40 → ~1 SE,
  noisy). Every *fast* pose (FPFH or coarse-FP) collapses to 0.450 — *below* fused.
- **`refine_iter=1` is a double negative:** it did **not** speed FP up (still
  ~13 s/query — FP's cost is the coarse hypothesis stage, not refinement) **and**
  it dropped quality to 0.450. So the exposed knob is neither a speed nor a
  quality lever.
- FP's **confidence** is *not* a valid cross-CAD signal (worse than random).
- **Iterating harder does *not* help** — the strong own-pose (multi-start RANSAC
  + multi-scale ICP, 200 iter) got *worse* (0.425) and 3× slower. The wall is the
  **coarse registration** on partial↔full, not the refinement: ICP only polishes
  a good init, and multi-start "keep best fitness" can even pick a wrong-but-
  confident pose. Classical FPFH/RANSAC cannot reach FP quality by iterating.
- **Why:** the render-compare signal is only as good as the pose; partial→full
  registration can't be *both* fast and accurate with a classical pipeline, and
  the gain needs full-refinement FP (~13 s/query at K=5 — not a re-ranker). The
  only cheap route to FP-quality pose is a **learned** partial-registration net
  (GeoTransformer/Predator) — a new model, not worth it for a noisy +0.075.

---

## 5. Cross-cutting conclusions

1. **No geometric cue is both quick and generalizing for Stage-3 retrieval.**
   - descriptor **re-rank** → hurts (look-alikes match well)
   - descriptor **gate** → neutral (can't reject shape-similar)
   - **VLAD signature** → redundant with ULIP + scale-inconsistent
   - **scale gate** → helps ycbv only (occlusion/size-twins break it, not
     deployably fixable)
   - **render-and-compare** → helps only with a full accurate pose (~13 s/query,
     not cheaply speedable)

2. **The mechanism is consistent:** the residual confusions are between objects
   that are geometrically *similar* (shape and often size). There is *no
   distinctive geometry to observe* → that is exactly where **appearance**
   (texture/text) is the discriminator, which the fusion already uses.

3. **Geometry's payoff is in the *pose* stage, not retrieval** — an accurate
   render-and-compare is precisely what FoundationPose does internally on the
   selected CAD. Re-ranking top-K by pose costs K× the pose you already run on
   the winner, for a small, noisy gain.

**Pipeline recommendation:** retrieval = **fused CLIP+DINO+ULIP** (appearance
gives occlusion-robustness via best-view aggregation); geometry/dGeDi = **pose
initialization/verification** downstream. Do **not** add a geometric retrieval
re-ranker.

**Open directions (for robustness, not quick wins):** multi-view accumulation
(active perception → complete cloud → geometry becomes reliable); a learned
partial-registration net (GeoTransformer/Predator) or template-matching coarse
pose (GigaPose) if a fast+accurate pose is ever needed; an *observation-derived*
reliability estimate (valid-depth fraction, completeness) to gate scale
deployably.

---

## 6. Threats to validity
- Small N for render-and-compare (N=40, SE ≈ ±0.08) — patterns are clear but
  point estimates are noisy; the scale-gate ycbv result (N=150) is firmer.
- GT mask + single view — real segmentation and multi-view would shift absolute
  numbers (and multi-view specifically would *help* the geometry cues).
- Latencies are warm-service, single-object estimates from our own logs
  (descriptor ~50–80 ms, FPFH pose ~35 ms, FP ~2.6 s/candidate).
