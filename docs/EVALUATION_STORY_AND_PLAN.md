# OSCAR+ Evaluation — Story, Results & Plan

*Working reference for the thesis evaluation. Captures: which dataset does what and
why, the shape-channel mode per stage, which metric/config is reported where, every
result we have so far, and the runs still outstanding. Numbers are mean-pooling, τ=0.37,
at the **audited config (42 views · top-k-softmax k=5)** unless stated. Companion docs:
the pipeline & preprocessing implementation in `PIPELINE_IMPLEMENTATION.md`, and the
experiments in `EXPERIMENTS_IMPLEMENTATION.md`. Last updated 2026-08-31 — Stages 1–3 are
complete, Stage 4 (latency) is implemented and awaiting its full run.*

> **Config correction — COMPLETE (2026-08-27).** A comparability audit found Stage-1's shape
> channel ran at **16 gallery views + top-8**, while DINOv2 (all stages) and Stages 2–3's shape
> ran at **42 views + top-5**. Stage 1 was the sole outlier and has been **fully re-run at
> 42v + k=5** → `results_shrec18_v2_stage1_42v_k5` (38 arms incl. geometry at K=50; on Drive).
> Headline: BASE **0.5868** · isolated shape **0.5353** · geometry winner **0.6405** ·
> hit@1 0.340→0.471. A paired significance test (§3.1) shows the **config change itself is a
> wash** (p=0.60) — it bought comparability, not different numbers. All three stages now share
> one shape config (§2.1).

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
not transfer uniformly — mode, geometry, and pooling all flip sign downstream (§8).

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
geometry on/off), because the Stage-1-optimal choice does not transfer uniformly (§8). The
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
| shape views | **16 → 42** | *n/a (full-mesh)* ¹ | 42 |
| shape top-k | **8 → 5** | *n/a (full-mesh)* ¹ | 5 |
| DINO views / top-k | 42 / 5 | 42 / 5 | 42 / 5 |
| DINO pooling | mean | mean | mean |
| weights | 0.3/0.4/0.3 | 0.3/0.4/0.3 | 0.3/0.4/0.3 |
| shape mode | pc *(intended)* | cross *(intended)* | pc & cross *(intended)* |

¹ **Stage 2 fährt eine Full-Mesh-Shape-Referenz** (ein Embedding pro CAD) — die MI3DOR
`*_partial.npz` fehlen, der Lauf fällt still auf Full-Mesh zurück (§4). View-Zahl und
View-Aggregation sind dort also gegenstandslos. Im cross-Modus ist Full-Mesh ohnehin die
*bessere* Referenz (§4.1), aber es war eine Nebenwirkung, keine Entscheidung. Die
DINO-/CLIP-Kanäle laufen unverändert bei 42 Views / k=5.

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
| **A3** | ◆ | Shape backbone: ULIP-2 vs Uni3D (pc) | Which 3D encoder? | **Tie — in both settings** (significance-tested, §3.1). Isolated nDCG 0.5353 vs 0.5337 (n.s.); fused 0.5868 vs 0.5913 looks like a Uni3D win but is **outlier-driven** (per-query wins 1009:1027, median 0, Wilcoxon p=0.54) → not a real effect. On **hit@1 ULIP-2 is significantly better** isolated (+0.018, p=0.038). **Keep ULIP-2** (it also has the cross-mode tower Uni3D lacks). |
| **A4** | ◆ | Shape reference: partial view vs full mesh | Match a rendered partial view or the whole mesh? | **partial 0.5256 ≫ full-mesh 0.4858** isolated. *In fusion they tie* (0.5889 vs 0.5897) — fusion had masked a real +0.04 shape-channel gap. |
| **A5** | ◇ | Shape **query mode: pc vs cross** | Encode the query point cloud or its image? | **pc 0.5256 ≫ cross 0.4673** (+0.058). This is the **bridge to Stage 2**: it quantifies exactly what the depth-free (cross) setting costs. |
| **A6** | ◇ | Query colours: XYZ+RGB vs XYZ-only | Do point colours help the shape channel? | **Colour slightly *hurts*, consistently** (significance-tested): isolated 0.5353 (RGB) vs 0.5422 (XYZ-only); XYZ-only wins **1152 of 1999** non-tied queries (Wilcoxon p=0.0007) — small but systematic, and **REAL on hit@1** (−0.032, p=0.0008). ⚠ *Confounded:* the xyz arm also swaps the ULIP tower (ViT-B/512-d/8k pts vs ViT-g/1280-d/10k), so this is not a clean colour-only ablation (§2.1). |
| **A7** | ◇ | Shape view count (ULIP partial) | How many ULIP gallery views to pool? | **isolated: V8 0.5128 · V16 0.5256 · V32 0.5295 · V42 0.5389** — **monotone, more views keep helping** (unlike appearance, which plateaus). **But in fusion it vanishes:** arm `A7f` (BASE fusion, shape @42) = **0.5885 vs BASE 0.5889 = −0.0004**, so the isolated +0.013 is fully masked. BASE pools 16 → confirmed a sound default, not a compromise. Ran via a force-loaded partial-gallery cache from Drive (`.ulip_partial_cache_*.pt`, `SHREC_FORCE_PARTIAL_CACHE`); V16 reproduces E1_shape_only 0.5256 = validated. |

### Block B — Fusion

Order: **configure the combiner first, then show the payoff.**

| # | Badge | Experiment | Why | Result |
|---|---|---|---|---|
| **B1** | ◆ | Fusion strategy: weighted-sum vs RRF | Which combiner? | **weighted-sum 0.5889** > RRF 0.5731. RRF is standard (Cormack k=60) but its constant is calibrated for TREC-length lists; reported as a negative result, not tuned. |
| **B2** | ◇ | **Weight sensitivity (heatmap)** | Is (0.3,0.4,0.3) fragile? | **Robust in beiden Modi.** pc (SHREC): Optimum (0.2,0.4,0.4) = 0.5916 vs BASE 0.5889 (**+0.003**). cross (**echte MI3DOR-Heatmap**, 231 Punkte, Selbstcheck FT@BASE 0.6851 ≈ erwartet 0.682): Optimum **(0.45,0.35,0.20)** FT 0.6902 vs BASE 0.6851 (**+0.005**). Beide Male ist kein Tuning nötig. **Ohne Tiefe gehört Shape herunter** (0.20 statt 0.30) — das Gewicht wandert aber **zu Text** (0.45), nicht zu View. ⚠️ *Korrigiert 2026-08-27:* zuvor stand hier (0.3,0.6,0.1) aus der **SHREC-cross-Heatmap als Stellvertreter** — die sagte die Richtung falsch voraus (view-lastig statt text-lastig). Die SHREC-cross-Heatmap taugt nicht als MI3DOR-Proxy. |
| **B3** | ◆ | Channel contribution + OSCAR baseline | Does fusing modalities help, and does each channel add value? | Single → full: text 0.4218 · view 0.5506 · shape 0.5256; text+view (**= OSCAR's channels**) 0.5519; **full fusion 0.5889**. **Adding S_shape to OSCAR's text+view: +0.037** — the core OSCAR+ claim. OSCAR text-first cascade baseline = 0.4561, so full fusion beats OSCAR by **+0.133**. |
| **B4** | ◇ | Scope: which channel prunes? | Prune the gallery before fusion, by which channel? | full-DB **0.5889** (score all 3308) > **visual-first** 0.5560 (DINOv2 prunes to top-20, then fuse) > **text-first/OSCAR** 0.4565 (CLIP-text prunes first). **OSCAR's τ=0.37 does not transfer**: it prunes to empty on 98.3% of queries (→ top-k fallback). A SHREC-calibrated τ=0.29 recovers to 0.5189, still below full-DB. |

### Block C — Geometry re-ranking (on the best fusion)

| # | Badge | Experiment | Why | Result |
|---|---|---|---|---|
| **C1** | ◆ | Geometry signal | Does alignment-aware local geometry help? | none 0.5889 → fitness 0.6226 → unaligned (diagnostic) 0.6225 → **GeDi+RANSAC 0.6406** → +ICP 0.6406. **The real win is top-1**: `hit@1` **0.340 → 0.471 (+38% rel)**, `MRR` 0.504 → 0.640 — ~2.5× the nDCG gain, and it's the pose-relevant metric. |
| **C2** | ◇ | Geometry shortlist depth K (50/20/5) | How deep to re-rank? | **Deeper is better** (42v/k5): GeDi+RANSAC **K50 0.6405 · K20 0.6279 · K5 0.6022** (+0.038 nDCG, **+0.046 hit@1** over the range), konsistent über *alle* Geometrie-Arme. K=50 ist die BASE-Tiefe. K=20/5 aus dem K=50-Cache abgeleitet. |
| **C3** | ◇ | Shape vs geometry redundancy | Is S_shape redundant once GeDi re-ranks? | Complementary; all geometry re-ranks the fusion top-K shortlist. text+view 0.5519 · +shape-in-fusion 0.5889 · +GeDi-rerank-on-the-text+view-shortlist (no shape) 0.5917 · **+both (shape + GeDi) 0.6226** · GeDi⊕base Borda 0.6301. Shape and GeDi each add ~+0.04 and stack — neither redundant. |

### 3.1 Paired significance — which deltas are real?

`object_retrieval/paired_significance.py` pairs the per-query records by query id
(n = 2101) and reports, per comparison: the mean Δ with a **95 % bootstrap CI (10 k
resamples)**, the **Wilcoxon signed-rank p**, and the **per-query win split**. Run on
nDCG and hit@1; results in `paired_significance_{nDCG,NN_sub}.csv`.

**The two tests answer different questions, and here that matters.** The CI tests the
*mean* difference; Wilcoxon tests whether one arm wins *consistently*. With heavy-tailed
per-query deltas a handful of large swings can move the mean while the win/loss split is
~50/50 — so **for near-ties the sign-consistency verdict is authoritative**: a "win"
carried by a few outliers is not a design argument.

| Comparison (nDCG) | Δ | CI | Wilcoxon p | wins | verdict |
|---|---|---|---|---|---|
| geometry: none vs GeDi+RANSAC | −0.0537 | excl. 0 | 0.0000 | 599:1264 | **REAL** (geometry helps) |
| partial vs full-mesh (isolated) | +0.0495 | excl. 0 | 0.0000 | 1015:974 | **REAL** (partial wins by larger margins) |
| DINOv2 vs SigLIP (isolated) | +0.0341 | excl. 0 | 0.0000 | 1213:811 | **REAL** |
| weighted-sum vs RRF (fused) | +0.0124 | excl. 0 | 0.0000 | 1320:718 | **REAL** |
| XYZ+RGB vs XYZ-only (isolated) | −0.0068 | incl. 0 | 0.0007 | 847:1152 | **consistent** — colour slightly hurts |
| ULIP-2 vs Uni3D (fused) | −0.0045 | excl. 0 | 0.54 | 1009:1027 | **outlier-driven → tie** |
| ULIP-2 vs Uni3D (isolated) | +0.0017 | incl. 0 | 0.11 | — | **tie** |
| config 16v/k8 → 42v/k5 (fused) | +0.0021 | excl. 0 | 0.60 | 938:1064 | **outlier-driven → wash** |

On **hit@1** every REAL row above holds and gets larger (geometry −0.130, SigLIP +0.070,
full-mesh +0.049, RRF +0.024), and two more become REAL: ULIP-2 > Uni3D isolated (+0.018,
p=0.038) and XYZ-only > XYZ+RGB (−0.032, p=0.0008).

**Two prior claims are corrected by this:** "Uni3D wins fused" is **not** a real effect
(A3 is a tie in both settings), and the **16v/k8 → 42v/k5 config correction is a wash** —
i.e. fixing the cross-stage comparability did not move the results, it only made the
stages comparable. Both are reassuring rather than disruptive.

*Methodology note — pooling.* CLS→**mean** was chosen for Stage-1/2/3 consistency; the
archived CLS run's best geometry arm was 0.6428 vs mean 0.6362 (≈ −0.007). Stated as a
decision, not an ablation.

---

## 4. Stage 2 — MI3DOR transfer test

BASE config carried over, **cross-mode** (no depth). Metrics follow the MI3DOR/SHREC-08
convention (NN, FT, mAP, ANMRR), top-k = 15, **full-mesh shape gallery**, 3848 CADs / 21 cat.

> **Korrektur 2026-08-27.** Diese Tabelle war als *partial-view gallery* beschriftet — falsch.
> Es existieren **keine** `*_partial.npz` für MI3DOR auf der Maschine; der Lauf fiel still auf
> Full-Mesh zurück (Logbeleg: `[init] WARNING: no partial PCs found … Falling back to full-mesh
> encoding.`). Beide Unterordner (`partial/`, `fullmesh/`) des `ulipfix`-Laufs tragen daher
> `ulip2_use_partial_views=False` und sind bitidentisch. **Stage 2 ist eine Full-Mesh-Stufe** —
> was im cross-Modus ohnehin die bessere Wahl ist (siehe A4-Transfer unten).

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

### 4.1 A4-Transfer — partial vs. full-mesh im cross-Modus

Aus `results_mi3dor_oscarplus_v2_tau037_dinomean/{partial,fullmesh}` (07./08.08.), dem
einzigen MI3DOR-Lauf, in dem `ulip2_use_partial_views=True` **tatsächlich griff**. Config
verifiziert identisch zur heutigen (42 Views, `ulip_view_topk=5`, mean-Pooling, cross,
τ=0.37, TOP_F=20, gleicher Checkpoint, n=10500); einziger Unterschied sind die Fusions-
gewichte `(0, 0.5, 0.5)` — **für die isolierten Arme wirkungslos**, da sie kein gewichtetes
Ranking bilden. Die fusionierten Arme jenes Laufs werden deshalb **nicht** zitiert.

| ULIP-2 isoliert | partial | **full-mesh** | Δ |
|---|---|---|---|
| NN | 68.11 | **78.10** | **+9.99** |
| FT | 0.453 | **0.510** | +0.057 |
| ST | 0.607 | **0.649** | +0.042 |
| nDCG@2R | 0.598 | **0.652** | +0.054 |
| mAP | 0.451 | **0.518** | +0.067 |
| ANMRR ↓ | 0.467 | **0.409** | besser |

**Full-mesh gewinnt auf MI3DOR auf jeder Metrik — exakt umgekehrt zu SHREC** (dort partial
+0.0495 nDCG isoliert, A4). Das ist kein Widerspruch, sondern ein verwertbarer Befund:

> **Die Referenz muss zur Natur der Query passen.** SHREC fragt mit einer *partiellen
> Punktwolke* (pc-Modus) → eine partielle Referenz ist geometrisch vergleichbar. MI3DOR fragt
> mit einem *Bild* (cross-Modus, ULIPs Image-Tower) → das Bild zeigt das **vollständige**
> Objekt, also passt die Full-Mesh-Referenz besser.

Damit ist der Full-Mesh-Fallback der Stage-2-Läufe (§4) inhaltlich **kein Schaden**: im
cross-Modus ist Full-Mesh die überlegene Referenz. Er war nur nicht als Entscheidung
dokumentiert.

*Ebenfalls berichtenswert:* τ=0.37 prunt auf MI3DOR bei **96,9 %** der Queries auf leer
(`fallback 10174/10500`) → die Kaskaden-Arme sind faktisch „CLIP-Top-20 → DINO", nicht
Schwellwert-Pruning. Konsistent mit SHREC (98,3 %, B4).

---

## 5. Stage 3 — BOP pose (YCB-V · T-LESS · LM-O)

Settings: **3a** retrieval into `G_proxy ∪ G_target` (exact CAD present) · **3b** proxy-only
pose + D_sym · **3c** next-best-non-GT diagnostic · **gt** oracle (exact CAD, exact pose).

**Status: complete** (chain finished 2026-08-30 21:41). Full write-up with per-dataset splits:
`docs/STAGE3_RESULTS_SUMMARY.md`.

**3a — retrieval (Recall@1 / MRR), all arms:**

| Arm | R@1 | R@5 | R@10 | MRR | YCB-V | T-LESS | LM-O |
|---|---|---|---|---|---|---|---|
| **cross, partial (frozen)** | **0.482** | 0.733 | 0.812 | 0.597 | **0.732** | 0.332 | 0.464 |
| pc, partial | 0.464 | 0.726 | 0.808 | 0.584 | 0.671 | 0.350 | 0.400 |
| cross, full mesh | 0.464 | **0.768** | **0.835** | **0.602** | 0.566 | **0.396** | **0.490** |
| pc, full mesh | 0.350 | 0.579 | 0.696 | 0.459 | 0.635 | 0.157 | 0.436 |
| OSCAR baseline (E5, no shape) | 0.320 | 0.492 | 0.542 | 0.404 | 0.498 | 0.214 | 0.304 |
| cross + geo (distance / fitness) | 0.423 / 0.428 | 0.733 | 0.812 | 0.558 / 0.557 | — | — | — |
| pc + geo (distance / fitness) | 0.373 / 0.382 | 0.726 | 0.808 | 0.522 / 0.525 | — | — | — |

**3b / 3c / gt — pose (D_sym, mm):**

| Setting | D_sym median | D_sym mean | Delta median | note |
|---|---|---|---|---|
| **gt** (oracle exact CAD) | **1.72** | 4.87 | — | sanity floor — the pose stage is near-perfect with the right model |
| **3b** OSCAR+ proxy | 18.37 | 33.6 | **15.79** | the cost of a proxy |
| 3b OSCAR baseline | 21.73 | 36.0 | 18.86 | +3.4 mm worse than OSCAR+ |
| 3b + geometry | 28.79 | 44.0 | 26.07 | geometry **hurts**, +57 % |
| **3c** next-best-non-GT | 15.34 | 28.1 | — | real CAD of another object 10.35 vs proxy gallery 20.10 → ~half the error is gallery foreignness, half is substitution loss |

**Four cross-stage findings:**
1. **The shape channel is the contribution, in both directions.** +0.162 R@1 over the faithful
   OSCAR cascade (0.482 vs 0.320) **and** −3.4 mm pose error (18.37 vs 21.73 mm). The retrieval
   gain reaches pose — that was the open question E5 was built to answer.
2. **Geometry re-ranking that helps on SHREC hurts on BOP** — all four clean cells lose, and pose
   loses hardest (+57 %). Not an implementation fault: at 98 % coverage geometry puts the right CAD
   first in 58 % of cases (chance 20 %), but the fusion score it *replaces* achieves 66 %.
   Re-ranking is displacement, so it pays only where it beats the incumbent score. → **geometry OFF
   in the pose pipeline**, stated with a mechanism rather than an observation.
3. **Fitness beats distance on BOP, inverting Stage 1.** SHREC is scale-invariant, so alignment
   distance is meaningful there; with true metric scale the pure overlap measure wins.
4. **cross ≥ pc in the pose setting** (0.482 > 0.464) — the opposite of SHREC — because BOP query
   depth/masks are noisy, so the image is the more reliable shape cue. The partial-vs-full-mesh
   advantage is likewise **YCB-V's alone**: full mesh wins T-LESS and LM-O and wins R@5/R@10/MRR
   overall. Report the split, not the aggregate.

---

## 6. Stage 4 — onboarding and query latency

Stages 1–3 answer *how well*. Stage 4 answers *at what cost* — the practical question a reader
asks once the accuracy case is made, and the one that decides whether the system is deployable.

**4a — onboarding.** A user has a CAD file and wants the object findable. Base gallery is the
**3b database** (G_proxy = 1257); each of the **59 target CADs** is onboarded individually and the
distribution over those 59 cases is reported. Steps measured separately: mesh preparation ·
Blender render (42 icosphere views, FPS-ordered) · partial clouds (HPR) · LLaVA descriptions ·
DINOv2 / CLIP-text / ULIP-2 encoding · dGeDi descriptors (optional) · cache write. Model load time
is reported apart — it is a system start-up cost, not an onboarding cost.

**4b — query.** Language prompt → GroundingDINO box → SAM2.1 mask → back-projected cloud →
CLIP / DINOv2 / ULIP-2 → fusion → geometry at K=5 (optional) → FoundationPose. Cold and warm are
separated: loading the five models costs a multiple of one query, so a figure mixing both only
reports how many queries were averaged.

**The open design question this exposes.** The cache key is a fingerprint over the *whole*
inventory (`_get_partial_cache_path`: one line per object per view). One new object changes the
hash and invalidates everything — onboarding really costs **O(gallery)**, not O(1). The incremental
cost is measured directly (encode only the new object's views with models already loaded, which is
the work an append-only cache would do); `--measure-invalidation` measures the surcharge the
current fingerprint forces on top. Both numbers are reported side by side.

**16 vs 42 views.** Stage 1 measures the quality side and it is flat past 16 — V8 0.5714 ·
**V16 0.5820** · V32 0.5800 · **V42 0.5868** nDCG, with V32 *below* V16. If onboarding scales
linearly with view count, 42 views cost 2.6× for 0.005 nDCG. Both scripts take a view-count list
and print a cost-benefit table with the Stage-1 quality next to the measured cost.

Scripts: `experiments/experiment4_onboarding.py`, `experiments/experiment4_query_latency.py`,
shared timing in `experiments/stage4_common.py`. Both are flag-based CLIs; see
`EXPERIMENTS_IMPLEMENTATION.md` §4.

---

## 7. Runs still outstanding (by stage)

*Stages 1–3 are complete as of 2026-08-30; nothing is queued. `[~]` = implemented and queued;
`[ ]` = still open.*

### Stage 1
- [x] **A2 / A7** isolated view sweeps (done): A2 42 best, flat past 16; A7 monotone, more views help.
- [x] **42v + k=5 re-run** — **DONE 2026-08-27**, 38 arms in `results_shrec18_v2_stage1_42v_k5`
  (+ Drive). BASE 0.5868 · isolated shape 0.5353 · geometry winner (GeDi+RANSAC, K=50) **0.6405**
  · hit@1 0.340→0.471. The config change itself is a wash (§3.1).
- [x] **Paired significance test** — **DONE 2026-08-27** (§3.1): 95% bootstrap CI + Wilcoxon +
  per-query win split on nDCG and hit@1. Corrected A3 (tie, not a Uni3D win) and A6 (colour
  consistently hurts); the config correction is a wash. `paired_significance_{nDCG,NN_sub}.csv`.
- [ ] Depth/top-1 metric family re-report across the whole grid (extract from `stage1_summary_depth.csv`).

### Stage 2 (MI3DOR)
- [x] **A4-Transfer (partial vs full-mesh)** — **erledigt ohne Neulauf** (§4.1): aus dem
  `_dinomean`-Lauf, dem einzigen mit tatsächlich aktivem `partial`. Full-mesh gewinnt im
  cross-Modus auf jeder Metrik (NN +9.99). Der 2026-08-27 gefahrene `fullmesh`-Arm war
  **redundant** (stiller Full-Mesh-Fallback → bitidentisch zum bestehenden Lauf).
- [x] **Cross-mode weight heatmap** (`mi3dor_weight_sweep.py`) — **DONE**: optimum at
  (0.45 text, 0.35 view, 0.20 shape), BASE only +0.005 FT away. Without depth, shape belongs
  down-weighted and the weight moves to text.
- [x] **OSCAR-Legacy-Baseline V=8** (`scripts/run_mi3dor_oscar_legacy.sh`) — **DONE**,
  `results_mi3dor_oscar_legacy_v8/fullmesh/`.
- [ ] Reporting stance (BASE + cross-heatmap explanation vs down-weighted-shape appendix); NN/FT/ANMRR↔nDCG mapping note.

### Stage 3 (pose)
- [x] **E5 (OSCAR cascade → pose)** — **DONE**: 3a R@1 0.320, 3b D_sym 21.73 mm. OSCAR+ wins both.
- [x] **A4 transfer (full mesh)** — **DONE** in both query modes (0.464 cross / 0.350 pc). Uni3D
  was dropped: Stage 1 showed ULIP-2 ≥ Uni3D and Uni3D has no cross-modal branch.
- [x] **Geometry, signals separated** — **DONE**: distance and fitness as own arms in both modes.
- [x] **Gate** — no arm beat the frozen cross config (0.482), so no additional 3b/3c fired.
- [x] **Geometry-OFF confirmed** as a documented decision, with the displacement mechanism (§5).

### Stage 4 (latency)
- [~] **4a onboarding / 4b query latency** — scripts implemented, smoke-tested on the light path
  (`--stages mesh`). The heavy path (Blender, LLaVA, encoders, FoundationPose) is not yet run.
- [ ] Run both at full scale and report the 16-vs-42-view cost-benefit table.

---

## 8. Master table — every experiment, why, result

| ID | Stage | Experiment | Why we ran it | Result (headline) | Status |
|---|---|---|---|---|---|
| A1 | 1 | DINOv2 vs SigLIP | pick appearance encoder | DINOv2 0.5506 > 0.5165 | ✅ |
| A2 | 1 | visual #views | pick render count | isolated V42 0.5506 best, flat past 16 | ✅ |
| A3 | 1 | ULIP-2 vs Uni3D | pick shape encoder | **tie in both** (fused 'win' outlier-driven, p=0.54); ULIP-2 better on hit@1 | ✅ |
| A4 | 1 | partial vs full mesh | pick shape reference | partial 0.5256 ≫ 0.4858 | ✅ |
| A5 | 1 | pc vs cross query | quantify depth-free cost | pc 0.5256 ≫ cross 0.4673 | ✅ |
| A6 | 1 | XYZ+RGB vs XYZ | do colours help | colour slightly **hurts**, consistently (p=0.0007) ⚠ confounded | ✅ |
| A7 | 1 | ULIP #views (isolated) | pick shape view count | monotone; V42 0.5389 > V16 0.5256 (more helps) | ✅ |
| A7f | 1 | ULIP #views (in fusion) | does the isolated gain survive fusion | **no** — shape@42 0.5885 ≈ BASE@16 0.5889 (−0.0004) | ✅ |
| B1 | 1 | weighted vs RRF | pick combiner | weighted 0.5889 > 0.5731 | ✅ |
| B2 | 1 | weight heatmap | is BASE robust | robust (+0.003 to optimum) | ✅ |
| B3 | 1 | channel contribution | does fusion / shape help | +shape +0.037; vs OSCAR +0.133 | ✅ |
| B4 | 1 | scope / cascade | prune, and how | full-DB best; τ=0.37 fails to transfer | ✅ |
| C1 | 1 | geometry signal | does local geometry help | 0.5889→0.6406; **hit@1 +38%** | ✅ |
| C2 | 1 | geometry depth K | how deep to re-rank | deeper better: K50 0.6406 > K20 0.6287 > K5 0.6041 | ✅ |
| C3 | 1 | shape vs GeDi redundancy | is shape redundant | complementary | ✅ |
| — | 2 | MI3DOR transfer | generalization, no depth | full fusion FT 0.682 / NN 86.6 (**full-mesh**); shape weak | ✅ |
| A4-t | 2 | partial vs full-mesh (cross) | transferiert die SHREC-Wahl? | **nein — full-mesh gewinnt** (NN 78.1 vs 68.1); Referenz muss zur Query passen | ✅ |
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
