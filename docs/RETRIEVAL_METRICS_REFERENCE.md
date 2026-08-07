# Retrieval Metrics Reference (MI3DOR + SHREC'18)

Exact definitions of every retrieval metric we report, matched to the canonical
scorers, for thesis cross-checking.

- **MI3DOR** metrics = **Tessa Pulli's** original scorer
  (`retrieval_mi3dor_eval.py`, github.com/pullover00/OSCAR), reproduced verbatim
  in `object_retrieval/eval_common.py` (`update_accum` / `compute_anmrr` /
  `average_precision_from_binary` / `dcg_at_k`).
- **SHREC'18** metrics = the **official** SHREC'18 scorer
  (`eval/shrec18_official/metrics.py` + `evaluate.py`), called from
  `experiments/experiment1_shrec18_stage1.py::score_official`.
- ACM MI3DOR paper (10.1145/3548690) is paywalled (couldn't fetch); Pulli's code
  is the operative reproduction target and implements the standard
  Princeton/MPEG-7 shape-retrieval metrics.

---

## MI3DOR (per query)

Let the query's category have **C** objects in the gallery (C = |class|, the
"num_rel"). The system returns a **full ranking of all N = 3848 gallery
objects**. Relevance is **binary category match**: `rels[i] = 1` if the object at
rank `i` (1-indexed) has the query's category, else 0. `rels.sum() = C` because
the ranking is the whole gallery. Queries whose category is absent (C = 0) are
skipped.

| Metric | Definition (as computed) | Notes |
|---|---|---|
| **NN** | `1` if `rels[1] = 1` else `0`; report `100 · mean` | top-1 category accuracy |
| **FT** (First Tier) | `(#rel in top-C) / C` = `Σ rels[1..C] / C` | recall@C |
| **ST** (Second Tier) | `(#rel in top-2C) / C` = `Σ rels[1..2C] / C` | recall@2C, **÷C** (Pulli) → max = 1 |
| **F1@20** | `2PR/(P+R)`, `P = Σrels[1..20]/20`, `R = Σrels[1..20]/C` | fixed depth TOP_F = 20 |
| **nDCG@2R** | `DCG@2C / IDCG@2C`, `DCG = Σ_{i=1..2C} rels[i]/log₂(i+1)`, `IDCG = Σ_{i=1..C} 1/log₂(i+1)` | binary gains |
| **mAP** | `AP = (1/C) · Σ_{r: rels[r]=1} (#rel up to r)/r`; mean over queries | ÷C |
| **ANMRR** ↓ | window `K = 2C`; `AVR = (1/C)·[ Σ_{rel, rank≤K} rank + (K+1)·(#rel beyond K) ]`; `NMRR = (AVR − (C+1)/2) / (K − (C+1)/2)`; mean | lower = better; miss penalty `K+1` |

**ANMRR denominator (thesis footnote).** Pulli uses `denom = K − (C+1)/2`. The
textbook MPEG-7 form is `denom = (K+1) − (C+1)/2` so the all-miss case
(`AVR = K+1`) maps to `NMRR = 1`; Pulli's omits the `+1`, letting NMRR slightly
exceed 1 in the worst case. Difference for MI3DOR's `C = 31..250` (`K = 62..500`):
**2.1% at C=31, 0.7% at C=100, 0.3% at C=250**. We match **Pulli** for direct
comparability to the published OSCAR numbers; flip the one line in
`compute_anmrr` for the strictly-normalised [0,1] variant.

---

## SHREC'18 (per query)

**Graded** relevance `g` over the ranking: `2` if (category, sub-category) match,
`1` if only category, `0` otherwise. `f = C` = class size from `cad.csv`. The
official scorer truncates to the top-f prefix first: `x = g[:C]` (evaluate.py
L91), then every metric is computed on `x`.

| Metric | Definition (as computed on `x = g[:C]`) | Notes |
|---|---|---|
| **precision** | `nonzero(x) / len(x)` = `nonzero(g[:C]) / C` | on top-C prefix |
| **recall** | `nonzero(x) / C` | |
| **F1** | `2PR/(P+R)` | |
| **NNT1** (First Tier) | `nonzero(x[:C]) / C` | |
| **NNT2** (Second Tier) | `nonzero(x[:2C]) / 2C` | **÷2C** (SHREC convention) |
| **AP** | `(1/C) · Σ_{k: x[k]≠0} (tp/(k+1))`, `tp` = cumulative hits | |
| **nDCG** | `dcg(x) / idcg(x)`, **graded** gains (sub-category = 2) | official `dcg` has a known index quirk, reproduced verbatim for leaderboard fidelity |

Because everything is evaluated on the length-C prefix, `precision = recall =
F1 = NNT1 = NNT2` collapse to `nonzero(g[:C])/C` for a single query; only nDCG and
AP differ. This is the official protocol as implemented, not a bug in our code.

**Stage-1 winner selection:** highest graded **nDCG**, `mAP` as tie-breaker
(→ `E2_both`, nDCG = 0.6428). Unaffected by the MI3DOR metric work.

---

## Key conventions that differ between the two benchmarks

| | MI3DOR (Pulli) | SHREC'18 (official) |
|---|---|---|
| relevance | binary (category) | graded (2 sub / 1 cat / 0) |
| ST / second tier denom | **÷C** | **÷2C** (NNT2) |
| eval depth | full gallery ranking | top-C prefix `g[:C]` |
| ANMRR | yes (`K−(C+1)/2`) | not reported |
| selection metric | — (benchmark report) | nDCG, mAP tiebreak |

These are intentional: each dataset is scored with **its own** canonical scorer.
