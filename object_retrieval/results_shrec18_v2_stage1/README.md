# SHREC'18 Stage-1 evaluation — shrec18_v2 (render-fixed) gallery

Official metrics (**Table A**: nDCG / precision / recall / F1 / AP / NNT1 /
NNT2) are **depth-independent** and identical across the runs below. The runs
differ only in the geometry re-ranking shortlist depth K, which changes the
geometry arms (E2_*, O1_*) and the depth-matched **Table B** (@K columns).

## Layout

- **Top level** (`stage1_summary*.{csv,tex}`, `best_config.json`) holds the
  Stage-1 summaries. The per-arm result folders that mirrored **K=50** were
  removed on 2026-08-06 as byte-identical duplicates of `k50/` — use
  `k50/<arm>/` for the primary/deeper result (and `k20/<arm>/` for K=20).
- **`k20/`** — the corrected first run: all arms depth-matched at **K=20**
  (`hit_sub` ceiling 0.746). Same Table A as top level; different geometry arms
  and Table B.
- **`k50/`** — deeper shortlist at **K=50** (`hit_sub` ceiling 0.864).
- Each arm folder holds `metrics_summary.json` (Table A + Table B + full config)
  and `results_per_query.json` (per-query top-10, ranks, deltas).
- The GeDi descriptor + per-pair geometry caches live under `_cache/` (on
  Google Drive only — not committed to git; regenerable and large).

## Run history

- **First run (2026-08-02)** — geometry at K=20. Table B originally mixed
  geometry depths [5, 20] (a bug: non-geometry arms were depth-scored at the
  K=5 fallback before K was set). Table A was correct. Superseded by the
  corrected, uniform-K=20 archive in `k20/`.
- **K=50 (2026-08-02)** — deeper shortlist; geometry improves.

## Headline result

Best arm at K=50: **E2_both** (fitness + trimmed distance combined by mean-rank
Borda), **nDCG 0.6428** (vs 0.6292 at K=20; BASE full fusion 0.5970).

Negative-transfer finding: `O2_clip_threshold` with OSCAR's tau=0.37 admits
nothing on ~98% of SHREC'18 queries (untextured CAD), so it falls back to top-k
and does *not* exercise the threshold mechanism — reported as negative transfer,
not as OSCAR's pruning. `O2_clip_threshold_cal` re-fits tau to this dataset
(tau≈0.29, |S'| median 289, 5% fallback).

See `docs/STAGE1_IMPLEMENTATION.md` for the implementation walkthrough and
`experiments/stage1_reproduce.py` for the config-first driver.
