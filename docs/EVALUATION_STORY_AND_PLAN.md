# OSCAR+ Evaluation — Story, Results & Plan

*Working reference for the thesis evaluation. Captures: which dataset does what and
why, the shape-channel mode per stage, which metric/config is reported where, every
result we have so far, and the runs still outstanding. Numbers are mean-pooling, τ=0.37,
at the **audited config (42 views · top-k-softmax k=5)** unless stated. Companion docs:
the pipeline & preprocessing implementation in `PIPELINE_IMPLEMENTATION.md`, and the
three experiments in `EXPERIMENTS_IMPLEMENTATION.md`. Last updated 2026-08-26.*

> **Config correction (2026-08-26).** A comparability audit found Stage-1's shape channel ran
> at **16 gallery views + top-8** aggregation, while DINOv2 (all stages) and Stages 2–3's shape
> ran at **42 views + top-5**. Stage 1 was the sole outlier; it is being re-run at **42v + k=5**
> into `results_shrec18_v2_stage1_42v_k5`, with a paired significance test attaching a 95% CI to
> every delta. Values tagged *(16v/k8)* below are pre-correction and pending refresh; confirmed
> 42v/k5 values are given where available (e.g. fused BASE 0.5889 → **0.5868**; isolated shape
> 0.5256 → **0.5353**). All three stages now share one shape config.

---

## 0. The evaluation in one paragraph

Three stages, three questions, deliberately ordered. **Stage 1 (SHREC'18)** *establishes
and characterizes* the retrieval method on clean RGB-D scans that carry depth — so the
shape channel runs in **pc-mode** (encode the query point cloud). It fixes a single
reference **BASE** config and ablates every design axis around it. **Stage 2 (MI3DOR)**
takes the method to a monocular dataset with **no depth**, forcing the shape channel into
**cross-mode** (encode the query image) — a generalization test that also exposes the cost
of losing depth. **Stage 3 (BOP: YCB-V, T-LESS, LM-O)** asks the downstream question: does
better retrieval actually yield a CAD that can be **posed**? We do **not** carry a single
frozen "winner" forward: Stages 2–3 re-report the relevant configurations and their
**spread** (channel set, shape mode, geometry on/off), because the Stage-1-best choice does
not transfer uniformly — mode, geometry, and pooling all flip sign downstream (§7).

---

## 1. The three-stage story — dataset, role, mode, why

| Stage | Dataset(s) | Role | Query → gallery | Shape mode | Why this dataset / mode |
|---|---|---|---|---|---|
| **1** | SHREC'18 (2101 q / 3308 CAD) | **Establish + characterize** the method | RGB-D scan → CAD | **pc-mode** (has depth) | Clean depth + category **and** subcategory GT + an official leaderboard metric set. The one place we're allowed to make design choices. |
| **2** | MI3DOR (~10.5k q / 3848 CAD, 21 cat) | **Transfer test** of the frozen method | single **RGB image** → CAD | **cross-mode** (no depth) | Monocular, so no query point cloud exists — the shape channel *must* fall back to encoding the image. Tests generalization and quantifies the depth-free penalty. |
| **3** | YCB-V · T-LESS · LM-O (BOP) | **Retrieval → pose** | RGB-D crop → `G_proxy` (∪ `G_target`) | pc **and** cross | Real 6-DoF pose GT in cluttered scenes. Answers "does a good retrieval give a poseable model?", which retrieval metrics alone cannot. |

**Why a fixed BASE, not a frozen winner.** The BASE fixes one reference configuration so
every ablation varies exactly one axis against it — the design choices are *characterized*,
not silently re-tuned per stage. But the BASE is a reference point, not a locked deployment
config: Stages 2–3 report the **spread** of configurations (channel set, shape mode,
geometry on/off), because the Stage-1-optimal choice does not transfer uniformly (§7). The
weight sweep is likewise a **robustness** result, never a selection procedure.

**Gallery in Stage 3.** `G_proxy = GSO(1030) ∪ HouseCat6D(199) ∪ ITODD(28)`; setting **3a**
adds the exact target CADs `G_target,d`, setting **3b/3c** do not (proxy-only).

---

## 2. What we report where — metrics & config

Two metric *families* matter, and we had been quoting only the first:

- **Ranking quality** — `nDCG`, `mAP`. "Is the whole ranked list good?"
- **Top-1 / pose-relevance** — `NN_sub` (subcategory nearest-neighbour = `hit@1`), `MRR`.
  "Is the *first* returned model the right one?" — which is what a pose stage consumes.

The distinction is not cosmetic: geometry re-ranking gains **+0.05 nDCG** but **+0.13 hit@1**
(§5, Block C). Reporting only nDCG understates the contribution that matters most downstream.

| Stage · experiment | Headline metric(s) | Also report | Rationale |
|---|---|---|---|
| S1 retrieval (A/B/C) | **nDCG** | mAP, **NN_sub/hit@1**, MRR | ranking + top-1; top-1 bridges to pose |
| S1 leaderboard framing | SHREC official NN/FT/ST/E/DCG | — | comparability with the track |
| S2 MI3DOR | **NN, FT** | mAP, ANMRR | MI3DOR / SHREC-08 convention |
| S3-3a retrieval | **Recall@1/5/10, MRR** | pre- vs post-geometry | instance retrieval into the pose gallery |
| S3-3b/3c pose | **D_sym** (mm & /diam), **F-score@1%/5%** | Δ = D_sym − D_sym(GT) | surface fidelity of the *posed* proxy |
| S3-gt (oracle) | D_sym | — | sanity floor (exact CAD, exact pose) |

**The BASE reference config** — every ablation names the one axis it varies against this:

| Component | Value | Notes |
|---|---|---|
| **S_text** | CLIP image vs per-**view** text descriptions | 42 descriptions per CAD (one per rendered view); object score = **max** over them (best-matching view) |
| **S_view** | DINOv2-base, **42 rendered views** | each view = **mean**-pooled patch tokens; 42 per-view sims → **top-k-softmax k=5**, τ=0.5 |
| **S_shape** | ULIP-2 (coloured, 1280-d) | pc-mode on SHREC (query point cloud); gallery = partial views, **all 42** (`SHAPE_AGG_VIEWS = 42`) → top-k-softmax **k=5**, τ=0.5 — equalised to DINOv2 and Stages 2–3 (CNOS `k_v=5`, thesis Table 4.1). |
| **Fusion** | weighted sum, min–max norm, **w = (0.3, 0.4, 0.3)** | (text, view, shape) |
| **Scope** | **full database** | no CLIP pruning; τ = 0.37 is the *OSCAR-cascade* parameter (B4), **not** applied in BASE |
| **Geometry** | off | geometry arms re-rank the top-**K = 50** shortlist (GeDi→RANSAC) |
| **Determinism / pooling** | `PYTHONHASHSEED = 0`, mean pooling | mean chosen for cross-stage consistency (CLS variant archived) |

Per-stage the **only** change is shape mode: **pc** (S1, S3-pc) vs **cross** (S2, S3-cross).

*"mean" vs "top-k-softmax" are two different poolings, don't conflate them:* "mean" is DINOv2's
**within-view** patch-token pooling (one view → one vector); the **view aggregation** (per-view
sims → object score) is **top-k-softmax k=5** for both view channels (OPEN-style
query-conditioned attention, CNOS k_v=5). S_text is the exception — it uses **max** over its 42
per-view descriptions (OSCAR's original text mechanism, deliberately not softmax-aggregated).

### 2.1 Config comparability across the three stages

The audit's conclusion — the shape stack is now identical everywhere, and Stage 1 was the only
arm ever out of spec:

| axis | Stage 1 (was → now) | Stage 2 | Stage 3 |
|---|---|---|---|
| shape views | **16 → 42** | 42 | 42 |
| shape top-k | **8 → 5** | 5 | 5 |
| DINO views / top-k | 42 / 5 | 42 / 5 | 42 / 5 |
| DINO pooling | mean | mean | mean |
| weights | 0.3/0.4/0.3 | 0.3/0.4/0.3 | 0.3/0.4/0.3 |
| shape mode | pc *(intended)* | cross *(intended)* | pc & cross *(intended)* |

**Inherent (documented, not bugs):** the XYZ-vs-RGB colour ablation (A6) is confounded — the
xyz arm swaps the whole ULIP tower (ViT-B/512-d/8k pts vs ViT-g/1280-d/10k pts), since no ViT-g
xyz checkpoint exists; Uni3D (A3) is 1024-d vs ULIP-2's 1280-d (intrinsic to the encoder) and is
**pc-only** (no cross-mode); full-mesh (A4) has one embedding per CAD so its aggregation is trivially "max".

---

## 3. Stage 1 — SHREC'18 retrieval

Organised by **pipeline block**, not by E/O label. Each experiment carries a badge:
**◆ primary** (literature-grounded claim) · **◇ exploratory** (thesis-specific design
question). Every design ablation is run **isolated** (single channel, no fusion) so the
changed variable is the only signal — the fused numbers are reported alongside where they
tell a different story. nDCG shown; top-1 (`hit@1`) added where it matters.

### Block A — Channel design (each channel alone)

| # | Badge | Experiment | Why | Result (isolated nDCG) |
|---|---|---|---|---|
| **A1** | ◆ | Visual backbone: DINOv2 vs SigLIP | Which appearance encoder? | **DINOv2 0.5506** > SigLIP 0.5165 (fair MAP-pooled; the old 0.5245 used a degenerate patch-0 token). DINOv2 wins isolated *and* fused (0.5889 vs 0.5667). |
| **A2** | ◇ | Visual view count 8/16/32/42 | How many renders? | **isolated: V8 0.5302 · V16 0.5481 · V32 0.5426 · V42 0.5506** — more views help but ~flat past 16 (V16 is 99% of V42); V42 best, small V32 dip (noise). Mirrors the fused O4 trend (V8 0.5736 → V42 0.5889). |
| **A3** | ◆ | Shape backbone: ULIP-2 vs Uni3D (pc) | Which 3D encoder? | **tied isolated** (ULIP-2 0.5256 ≈ Uni3D 0.5277). But Uni3D wins *fused* on every metric (nDCG 0.5917>0.5889, P 0.288>0.281, AP 0.173>0.167). Encoder choice flips with fusion. |
| **A4** | ◆ | Shape reference: partial view vs full mesh | Match a rendered partial view or the whole mesh? | **partial 0.5256 ≫ full-mesh 0.4858** isolated. *In fusion they tie* (0.5889 vs 0.5897) — fusion had masked a real +0.04 shape-channel gap. |
| **A5** | ◇ | Shape **query mode: pc vs cross** | Encode the query point cloud or its image? | **pc 0.5256 ≫ cross 0.4673** (+0.058). This is the **bridge to Stage 2**: it quantifies exactly what the depth-free (cross) setting costs. |
| **A6** | ◇ | Query colours: XYZ+RGB vs XYZ-only | Do point colours help the shape channel? | **tie** (XYZ+RGB 0.5256 ≈ XYZ-only 0.5316; colours if anything hurt slightly). Within noise. |
| **A7** | ◇ | Shape view count (ULIP partial) | How many ULIP gallery views to pool? | **isolated: V8 0.5128 · V16 0.5256 · V32 0.5295 · V42 0.5389** — **monotone, more views keep helping** (unlike appearance, which plateaus). **But in fusion it vanishes:** arm `A7f` (BASE fusion, shape @42) = **0.5885 vs BASE 0.5889 = −0.0004**, so the isolated +0.013 is fully masked. BASE pools 16 → confirmed a sound default, not a compromise. Ran via a force-loaded partial-gallery cache from Drive (`.ulip_partial_cache_*.pt`, `SHREC_FORCE_PARTIAL_CACHE`); V16 reproduces E1_shape_only 0.5256 = validated. |

### Block B — Fusion

Order: **configure the combiner first, then show the payoff.**

| # | Badge | Experiment | Why | Result |
|---|---|---|---|---|
| **B1** | ◆ | Fusion strategy: weighted-sum vs RRF | Which combiner? | **weighted-sum 0.5889** > RRF 0.5731. RRF is standard (Cormack k=60) but its constant is calibrated for TREC-length lists; reported as a negative result, not tuned. |
| **B2** | ◇ | **Weight sensitivity (heatmap)** | Is (0.3,0.4,0.3) fragile? | **Robust.** pc optimum (0.2,0.4,0.4) = 0.5916 vs BASE 0.5889 (**+0.003, noise**) → no tuning needed. Cross-mode heatmap (Stage-2 bridge): optimum shifts to (0.3,**0.6**,0.1) = 0.5567 and BASE (0.5453) drops **below view-only (0.5506)** — the pc-tuned weights don't transfer, shape must be down-weighted without depth. |
| **B3** | ◆ | Channel contribution + OSCAR baseline | Does fusing modalities help, and does each channel add value? | Single → full: text 0.4218 · view 0.5506 · shape 0.5256; text+view (**= OSCAR's channels**) 0.5519; **full fusion 0.5889**. **Adding S_shape to OSCAR's text+view: +0.037** — the core OSCAR+ claim. OSCAR text-first cascade baseline = 0.4561, so full fusion beats OSCAR by **+0.133**. |
| **B4** | ◇ | Scope: which channel prunes? | Prune the gallery before fusion, by which channel? | full-DB **0.5889** (score all 3308) > **visual-first** 0.5560 (DINOv2 prunes to top-20, then fuse) > **text-first/OSCAR** 0.4565 (CLIP-text prunes first). **OSCAR's τ=0.37 does not transfer**: it prunes to empty on 98.3% of queries (→ top-k fallback). A SHREC-calibrated τ=0.29 recovers to 0.5189, still below full-DB. |

### Block C — Geometry re-ranking (on the best fusion)

| # | Badge | Experiment | Why | Result |
|---|---|---|---|---|
| **C1** | ◆ | Geometry signal | Does alignment-aware local geometry help? | none 0.5889 → fitness 0.6226 → unaligned (diagnostic) 0.6225 → **GeDi+RANSAC 0.6406** → +ICP 0.6406. **The real win is top-1**: `hit@1` **0.340 → 0.471 (+38% rel)**, `MRR` 0.504 → 0.640 — ~2.5× the nDCG gain, and it's the pose-relevant metric. |
| **C2** | ◇ | Geometry shortlist depth K (50/20/5) | How deep to re-rank? | **Deeper is better:** GeDi+RANSAC = **K50 0.6406 · K20 0.6287 · K5 0.6041** — a real +0.037 gradient, not flat. Re-ranking more candidates keeps helping (the correct model is often past rank 5). K=50 is the BASE geometry depth. |
| **C3** | ◇ | Shape vs geometry redundancy | Is S_shape redundant once GeDi re-ranks? | Complementary; all geometry re-ranks the fusion top-K shortlist. text+view 0.5519 · +shape-in-fusion 0.5889 · +GeDi-rerank-on-the-text+view-shortlist (no shape) 0.5917 · **+both (shape + GeDi) 0.6226** · GeDi⊕base Borda 0.6301. Shape and GeDi each add ~+0.04 and stack — neither redundant. |

*Methodology note — pooling.* CLS→**mean** was chosen for Stage-1/2/3 consistency; the
archived CLS run's best geometry arm was 0.6428 vs mean 0.6362 (≈ −0.007). Stated as a
decision, not an ablation.

---

## 4. Stage 2 — MI3DOR transfer test

BASE config carried over, **cross-mode** (no depth). Metrics follow the MI3DOR/SHREC-08
convention (NN, FT, mAP, ANMRR), top-k = 15, partial-view gallery, 3848 CADs / 21 categories.

| Arm | NN | FT | mAP | ANMRR |
|---|---|---|---|---|
| CLIP only | 67.95 | 0.575 | 0.580 | 0.339 |
| DINOv2 only | 83.03 | 0.629 | 0.647 | 0.297 |
| ULIP-2 only | 78.10 | 0.510 | 0.518 | 0.409 |
| **CLIP+DINO+ULIP (full)** | **86.57** | **0.682** | **0.705** | **0.238** |
| CLIP-pruned DINO+ULIP (primary) | 86.52 | 0.575 | 0.593 | 0.337 |

**What it shows.** The full 3-way fusion is the best arm (FT 0.682 / mAP 0.705 / NN 86.6),
and it clears DINOv2-alone — but **only modestly**, and ULIP-2 is the weakest single channel
(FT 0.510). This is exactly the cross-mode weight heatmap (B2) made real: without depth the
shape channel is weak, so at the frozen shape weight (0.3) it barely earns its place.
*(Historical note: an earlier run reported ULIP=0.0 — the CAD meshes were absent, so the
shape gallery was empty; fixed by restoring the meshes, this `ulipfix` run is the first true
3-way MI3DOR result.)*

---

## 5. Stage 3 — BOP pose (YCB-V · T-LESS · LM-O)

Settings: **3a** retrieval into `G_proxy ∪ G_target` (exact CAD present) · **3b** proxy-only
pose + D_sym · **3c** next-best-non-GT diagnostic · **gt** oracle (exact CAD, exact pose).

**3a — retrieval (Recall@1 / MRR):**

| Config | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| cross (no geo) | **0.482** | 0.733 | 0.812 | 0.597 |
| pc (no geo) | 0.464 | 0.726 | 0.808 | 0.584 |
| cross + geometry | 0.458 | — | — | 0.579 |
| pc + geometry | 0.413 | — | — | 0.547 |

Per-dataset (cross, no-geo) R@1: **YCB-V 0.732 · LM-O 0.464 · T-LESS 0.332** (T-LESS is the
hard, texture-less case).

**3b / 3c / gt — pose (D_sym, mm):**

| Setting | D_sym median | D_sym mean | F@5% | note |
|---|---|---|---|---|
| **gt** (oracle exact CAD) | **1.72** | 4.87 | — | sanity floor — pose stage is near-perfect with the right model |
| **3b** proxy pose | 18.37 | 33.6 | 0.313 | Δ vs oracle ≈ **+16.6 mm** = the cost of a proxy |
| 3b + geometry | 28.79 | 44.0 | 0.319 | geometry **hurts** |
| **3c** next-best-non-GT | 15.34 | 28.1 | 0.410 | real-CAD-of-another-object 10.35 vs proxy 20.10 → ~half the error is gallery foreignness, half is substitution loss |

**Two significant cross-stage findings:**
1. **Geometry re-ranking that helps on SHREC *hurts* on BOP** — 3a R@1 0.482→0.458 (cross),
   0.464→0.413 (pc); 3b D_sym 18.4→28.8 mm. Clean scans (SHREC) reward alignment; cluttered
   partial BOP depth against *proxy* geometry misleads it. → **geometry should be OFF in the
   pose pipeline**, and the thesis can state this with evidence.
2. **cross ≥ pc in the pose setting** (3a R@1 0.482 > 0.464) — the opposite of SHREC —
   because BOP query depth/masks are noisy, so the image is the more reliable shape cue.

**Planned Stage-3 additions (implemented + queued 2026-08-26; see `EXPERIMENTS_IMPLEMENTATION.md` §3.4):**
- **E5 — OSCAR vs OSCAR+ → pose.** The OSCAR baseline is OSCAR's *actual* mechanism — CLIP-τ=0.37
  threshold prune → DINOv2 best-view cascade, **no shape** (`--oscar-baseline`, ranked by
  `oscar_maxview`) — run for 3a (retrieval) and 3b (pose), diffed against OSCAR+ (0.482 / 18.37 mm).
  Closes *"does the shape channel's retrieval gain reach pose?"*.
- **A3/A4 transfer to pose.** Uni3D (`--uni3d`, pc-only) and full-mesh (`--fullmesh`) shape arms:
  **3a on all three**; **3b/3c only for whichever out-retrieves ULIP-2 cross** (0.482) — no point
  posing an arm that doesn't even retrieve better.
- **3c decomposition** is added for any arm that beats cross, to compare its
  foreignness-vs-substitution split (BASE: real-CAD 10.35 vs proxy 20.10 mm) against the winner's.
- **Geometry stays OFF** in the pose pipeline (finding 1 above). E3 (a ROCA-style alignability
  metric) is approximated by 3a ± geometry — optional.

---

## 6. Runs still outstanding (by stage)

*A four-job pipeline is running end-to-end (2026-08-26): Stage-1 42v/k5 re-run → Stage-2
(full-mesh + heatmap) → Stage-3 (E5/Uni3D/full-mesh, gated), plus the significance test after
Stage-1. `[~]` = implemented and queued; `[ ]` = still open.*

### Stage 1
- [x] **A2 / A7** isolated view sweeps (done): A2 42 best, flat past 16; A7 monotone, more views help.
- [~] **42v + k=5 re-run** — Stage-1 shape corrected to 42 views + top-5 (config-comparability audit,
  §2.1); full grid + geometry re-running into `results_shrec18_v2_stage1_42v_k5`. Confirmed so far:
  BASE 0.5889→**0.5868**, isolated shape 0.5256→**0.5353**, Uni3D fused 0.5913, XYZ-only 0.5880.
- [~] **Paired significance test** — implemented (`object_retrieval/paired_significance.py`), queued
  after the re-run; 95% bootstrap CI + Wilcoxon p per delta (nDCG + hit@1), incl. the config-change
  delta paired cross-folder. Decides which near-ties (A3, A6, B2, V32-vs-V42) are real.
- [ ] Depth/top-1 metric family re-report across the whole grid (extract from `stage1_summary_depth.csv`).

### Stage 2 (MI3DOR)
- [~] **Full-mesh transfer arm** (`MI3DOR_MODES=fullmesh`, A4) + **cross-mode weight heatmap**
  (`mi3dor_weight_sweep.py`, Tier-2 re-fuse, BASE self-check FT≈0.682) — implemented, queued after Stage 1.
- [ ] Reporting stance (BASE + cross-heatmap explanation vs down-weighted-shape appendix); NN/FT/ANMRR↔nDCG mapping note.

### Stage 3 (pose)
- [~] **E5 (OSCAR cascade → pose) + A3/A4 transfer** — `--oscar-baseline`/`--uni3d`/`--fullmesh` + 3c
  implemented; queued after Stage 2 with the gated selection (§5): 3a on all, 3b/3c only for runs
  that beat ULIP-2 cross.
- [ ] Confirm geometry-OFF for the pose pipeline as one documented decision; E3 alignability (optional).

---

## 7. Master table — every experiment, why, result

| ID | Stage | Experiment | Why we ran it | Result (headline) | Status |
|---|---|---|---|---|---|
| A1 | 1 | DINOv2 vs SigLIP | pick appearance encoder | DINOv2 0.5506 > 0.5165 | ✅ |
| A2 | 1 | visual #views | pick render count | isolated V42 0.5506 best, flat past 16 | ✅ |
| A3 | 1 | ULIP-2 vs Uni3D | pick shape encoder | tied isolated; Uni3D wins fused | ✅ |
| A4 | 1 | partial vs full mesh | pick shape reference | partial 0.5256 ≫ 0.4858 | ✅ |
| A5 | 1 | pc vs cross query | quantify depth-free cost | pc 0.5256 ≫ cross 0.4673 | ✅ |
| A6 | 1 | XYZ+RGB vs XYZ | do colours help | tie (noise) | ✅ |
| A7 | 1 | ULIP #views (isolated) | pick shape view count | monotone; V42 0.5389 > V16 0.5256 (more helps) | ✅ |
| A7f | 1 | ULIP #views (in fusion) | does the isolated gain survive fusion | **no** — shape@42 0.5885 ≈ BASE@16 0.5889 (−0.0004) | ✅ |
| B1 | 1 | weighted vs RRF | pick combiner | weighted 0.5889 > 0.5731 | ✅ |
| B2 | 1 | weight heatmap | is BASE robust | robust (+0.003 to optimum) | ✅ |
| B3 | 1 | channel contribution | does fusion / shape help | +shape +0.037; vs OSCAR +0.133 | ✅ |
| B4 | 1 | scope / cascade | prune, and how | full-DB best; τ=0.37 fails to transfer | ✅ |
| C1 | 1 | geometry signal | does local geometry help | 0.5889→0.6406; **hit@1 +38%** | ✅ |
| C2 | 1 | geometry depth K | how deep to re-rank | deeper better: K50 0.6406 > K20 0.6287 > K5 0.6041 | ✅ |
| C3 | 1 | shape vs GeDi redundancy | is shape redundant | complementary | ✅ |
| — | 2 | MI3DOR transfer | generalization, no depth | full fusion FT 0.682 / NN 86.6; shape weak | ✅ |
| 3a | 3 | pose-gallery retrieval | does retrieval find the CAD | R@1 0.482 (cross); geometry hurts | ✅ |
| 3b | 3 | proxy pose D_sym | surface fidelity of posed proxy | 18.4 mm median (oracle 1.7) | ✅ |
| 3c | 3 | next-best diagnostic | decompose the proxy error | 15.3 mm; foreignness vs substitution | ✅ |
| gt | 3 | oracle exact CAD | sanity floor | 1.72 mm median | ✅ |
| E5 | 3 | OSCAR vs OSCAR+ → pose | does retrieval gain reach pose | — | ⏳ run |

---

*Legend: ✅ have · ⏳ outstanding. All Stage-1 nDCG are mean-pooling, n=2101. Result
directories: S1 `results_shrec18_v2_stage1_mean_{mean_only,isolated,dgedi_k50,siglipfix,
wsweep_cross}`; S2 `results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix`; S3
`results_bop_stage3_v2/{3a_*,3b_*,3c_*,gt}`.*
