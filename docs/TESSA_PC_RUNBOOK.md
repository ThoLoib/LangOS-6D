# Runbook — Stage-1 full run on tessa-pc

Operational instructions for executing the Stage-1 SHREC'18 grid on the
gallery/compute PC. **Why** anything is done this way:
`docs/STAGE1_EVALUATION_DESIGN.md`. This file is only *what to type, what to
expect, and what to do when it breaks.*

Total: **~10–35 h** depending on the depth K chosen in step 2, most of it
unattended.

---

## 0. Preconditions

| | check | fix if wrong |
|---|---|---|
| code | `git log --oneline -1` includes the Stage-1 metrics commit | `git fetch origin && git merge origin/feat/stage1-official-eval-precompute` |
| data | `ls eval/datasets/shrec18/shrec18_full/cad/*.obj \| wc -l` → **3308** | re-stage the dataset; the script will not download anything |
| GT | `wc -l eval/shrec18_official/{rgbd,cad}.csv` → **2101 / 3308** | `git clone https://github.com/hkust-vgd/shrec18 eval/shrec18_official` |
| gallery | `ls object_images/shrec18 \| wc -l` → **3308** | step 1 reports exactly what is missing |
| disk | **≥ 5 GB** free for `object_retrieval/results_shrec18_stage1/_cache/` | descriptors alone are 3.26 GB |
| WSL RAM | `C:\Users\tholo\.wslconfig` has `memory=20GB, swap=8GB` | the 15 GB default OOM-kills GeDi mid-run; needs `wsl --shutdown` |

```bash
docker compose up -d gedi          # healthy in ~15 s
docker compose run --rm -it oscar bash    # everything below runs in here
```

> If `docker` has vanished from the WSL distro (`command not found`), Docker
> Desktop's Resource Saver stopped the engine. Use the Windows binary
> `/mnt/c/Program\ Files/Docker/Docker/resources/bin/docker.exe` and pipe every
> captured value through `tr -d '\r'`. Operate on **existing** containers with
> `docker start` / `docker stop` — `docker compose` through the Windows binary
> resolves the relative bind mounts against Windows paths and breaks.

---

## 1. Precompute GeDi descriptors (one-time) — and validate on the way in

Input validation runs first inside this command, so there is no separate
cheap check to do. Watch the first two lines and **Ctrl-C immediately** if
they are wrong:

```
[validate] CADs: 3308  rendered: 3308  described: 3308  -> scored gallery: 3308
[precompute-gedi] 3308 CADs + 2101 queries = 5409 clouds -> .../gedi_descriptors
```

A shortfall is printed explicitly. **Do not continue with an incomplete
gallery** — it silently changes every number downstream.

> Avoid `--limit-queries N` as a "quick check": it changes the channel-score
> cache filename (`scores_base_n5.pt`), so instead of reading the cached
> passes it tries to recompute them, which needs the GPU and the full model
> stack. Limiting is for genuine smoke tests, not for validation.

```bash
python experiments/experiment1_shrec18_stage1.py --precompute-gedi
```

- 5,409 clouds (3,308 CADs + 2,101 queries), **~3–5 h**, **3.26 GB**
- Resumable and idempotent — re-run after any interruption, cached clouds are
  skipped instantly
- Progress prints every 100 clouds with a live s/cloud rate

This is what makes the shortlist depth affordable: descriptors are a per-cloud
cost, the registration that consumes them is per-pair. Without it the geometry
run is ~5× longer.

Expect `N cached, 0 failed`. A handful of failures is tolerable (those clouds
fall back to on-the-fly computation); **many** failures means the GeDi service
is unhealthy — see §7.

---

## 2. Measure throughput, then choose K

```bash
python experiments/experiment1_shrec18_stage1.py --bench-rerank 16 --no-icp
```

Prints measured s/pair on **this** machine plus a K → wall-clock table and the
marginal return per hour. Reference figures from the dev box (CPU-only,
0.93 s/pair):

| K | hit_sub ceiling | wall clock | marginal |
|---:|---:|---:|---:|
| 10 | 0.6497 | 5.4 h | 3.59 pts/h |
| **20** | **0.7463** | **10.8 h** | 1.79 pts/h |
| **50** | **0.8639** | **27.0 h** | 0.73 pts/h |
| 100 | 0.9229 | 54.0 h | 0.22 pts/h |

**Pick K from the measured table, not from the one above.** If this machine
reports materially worse than ~0.93 s/pair, take K=20 rather than starting a
run that will not finish.

`--no-icp` matters: ICP is **68 %** of the per-pair cost (3.29 → 0.93 s/pair)
for a measured 0.0001 nDCG effect at K=5. Keep it off for the full run.

---

## 3. Non-geometry grid

```bash
python experiments/experiment1_shrec18_stage1.py --all --overwrite
```

**`--overwrite` is required**, not optional: every existing summary predates
the depth-matched metrics and has no `metrics_depth` key. This is cheap
(~1 h) because it re-scores from the cached channel passes — it does not
re-encode anything.

### Verify before continuing

| check | expected |
|---|---|
| `E1c_full_fusion` nDCG | `0.5878959504080713` |
| `E1c_full_fusion` P | `0.26864481953094016` |
| `NN_cat` / `NN_sub` | `0.4217` / `0.3503` |
| `O2_clip_threshold_cal` τ | `≈ 0.2949`, fallback ≈ 4.9 % |
| `O2_clip_threshold` fallback | `96.9 %` + a WARNING — **this is expected**, it is the negative-transfer result |

**If Table A does not reproduce exactly, stop and report.** Those values come
from the 2026-07-30 run and the new code is meant to be non-regressive on them.

---

## 4. Hit-rate ceiling

```bash
python experiments/experiment1_shrec18_stage1.py --hit-rate-curve
```

Expected `hit_sub@K`: 0.3503 / 0.5526 / 0.6497 / 0.7463 / 0.8639 / 0.9229 for
K = 1/5/10/20/50/100.

The curve does not flatten, so the tool's suggested `--geom-k` will just be the
deepest measured K. **Ignore that suggestion** — use the K chosen in step 2.
The curve goes in the thesis next to K as the justification.

---

## 5. Purge the pair-score cache

```bash
mv object_retrieval/results_shrec18_stage1/_cache/geometry_scores.jsonl \
   object_retrieval/results_shrec18_stage1/_cache/geometry_scores.jsonl.bak_random_clouds
```

Those pairs were computed against **randomly sampled** CAD clouds — CAD
sampling was unseeded before this commit, so a re-run cannot reproduce them.
Mixing them with newly computed pairs would put two different cloud
realizations of the same CAD inside one ranking.

Renamed, not deleted: it remains the evidence for the K=5 ICP result.

---

## 6. Geometry run

```bash
python experiments/experiment1_shrec18_stage1.py \
    --all --resume --with-geometry --no-icp --geom-k <K>
```

Launch **detached** for anything over a couple of hours — a killed foreground
client takes the log plumbing with it while the container keeps running:

```bash
docker compose run -d --name stage1_geom oscar \
    python3 -u experiments/experiment1_shrec18_stage1.py \
    --all --resume --with-geometry --no-icp --geom-k <K>
```

Monitor by watching the pair cache grow; per-query progress is not printed and
`docker logs` is buried in harmless Open3D warnings (`Too few correspondences`,
`Read geometry::Image failed …jpg` — missing CAD textures, irrelevant to
point-cloud geometry):

```bash
wc -l object_retrieval/results_shrec18_stage1/_cache/geometry_scores.jsonl
```

Target line count ≈ `2101 × K` for the BASE-shortlist arms. `E2_chamfer_icp`
is auto-dropped under `--no-icp` (that arm *is* the ICP measurement) — this is
correct, not an error.

> **`O1c_gedi_post_fusion` doubles the cost.** It shortlists on text+view
> instead of BASE fusion, so none of the shared registrations apply to it. If
> the budget is tight, run everything else first and decide on O1c afterwards;
> `--resume` makes that safe.

---

## 7. Failure modes

**GeDi dies.** Two observed modes: OOM under memory pressure (fixed by the
`.wslconfig` RAM setting), and a clean `exit=0` about a minute after startup
with no traceback. `restart: unless-stopped` recovers both in ~90 s, which is
why the run waits (`GEDI_WAIT_S=300` × `GEDI_RETRIES=4`) rather than aborting.

```bash
docker inspect -f '{{.RestartCount}}' oscar-gedi-1
```

A dead service **cannot** poison the caches: failed registrations are not
written, and empty descriptor results are never cached. If the run does abort,
restart GeDi and re-run with `--resume`; completed pairs are kept.

**Counting failures.** `failed` counts read straight off the jsonl **over-count**
— a failed pair stores `d_ransac: null` and is legitimately retried, appending
another record. Dedupe by `(qid, cad)` before believing any failure number.

**Descriptor cache reports STALE.** Means the input cloud's fingerprint changed
— a preprocessing difference, not a damaged file. Expected once after this
commit (sampling became deterministic). If it recurs on later runs, the two
machines are building clouds differently; investigate before trusting results.

**Duty cycling / battery.** Everything is resumable at pair granularity.
Stopping and restarting the container is equivalent to continuing.

---

## 8. Deliverables

In `object_retrieval/results_shrec18_stage1/`:

| file | contents |
|---|---|
| `stage1_summary.csv` / `.tex` | **Table A** — official metrics, all arms |
| `stage1_summary_depth.csv` / `.tex` | **Table B** — depth-matched, all arms |
| `best_config.json` | argmax nDCG (tie-break mAP), frozen for Stages 2–5 |
| `<arm>/metrics_summary.json` | per-arm metrics + full config |
| `<arm>/results_per_query.json` | per-query records incl. NN_cat/NN_sub/MRR/nDCG_K |

The per-query files are what the paired statistics (deltas vs BASE, N_changed,
bootstrap CI) are computed from — no re-run needed for those.

### Final checks

- **`E2_none` must equal `E1c_full_fusion` bit-for-bit** on all seven Table A
  metrics. It shares the entire code path but applies no geometry, so a
  mismatch localises a bug immediately.
- `aggregate` must not warn about mixed geometry depths. If it does, some arm
  ran at a different K and its `@K` columns are not comparable — re-run it.
- Geometry arms will **not** match the K=5 reference numbers in
  `STAGE1_EVALUATION_DESIGN.md` §"Measured results". Those were measured at
  depth 5; a deeper K legitimately changes them. Only the **Table A** values
  for non-geometry arms are regression checks.

### Report back

s/pair and K chosen, wall clock, `E2_none` vs BASE equality, the Table A/B
CSVs, and any arm that was skipped and why.

---

## 9. Do not

- **Do not** `pip install pointnet2_ops` on one machine only — both PCs must
  run the same pure-torch FPS path or Uni3D/ULIP embeddings silently mismatch
  (`docs/LAPTOP_EMBEDDINGS_SETUP.md`).
- **Do not** patch `eval/shrec18_official/metrics.py`. The `dcg` off-by-one is
  known and deliberately kept; patching it makes every number incomparable to
  the published leaderboard.
- **Do not** force-push a shared branch. Merge, don't rewrite.
- **Do not** run the grid at mixed K and merge the results.
