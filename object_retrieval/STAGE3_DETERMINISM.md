# Stage-3 determinism

Per `STAGE3_EVALUATION_CONCEPT.md` §Reproducibility, every result stores the
software revision, gallery manifest, query list, config, and failure/coverage
counts. This note records exactly what is and is not bit-reproducible.

## Deterministic (seeded / fixed)

- **RNG seeding** — `eval_bop_pose._seed_everything(seed)` sets `PYTHONHASHSEED`,
  `random`, `numpy`, and `torch` (CPU + CUDA) seeds at process start. The run
  scripts also export `PYTHONHASHSEED=0` so namespaced-id hashing and any
  set/dict ordering that leaks into iteration is stable across processes.
- **Surface sampling (D_sym / F-score)** — `sample_surface_mm` reseeds the global
  numpy RNG to `DSYM_SEED=0` immediately before `trimesh.sample`, so a given mesh
  always yields the same `N=10000` points. The exact-CAD benchmark and 3b reuse
  the *same* sample arrays per mesh (target points cached per obj_id, proxy points
  per gallery id), so `Delta = D_posed − D_posed_gt` is a clean paired difference.
- **Retrieval fusion + ranking** — CLIP/DINO/ULIP cosine scores and the weighted
  fusion (0.3/0.4/0.3) are deterministic given fixed encoder weights and the
  precomputed gallery caches. Ties are broken by a stable sort.
- **dGeDi Borda re-rank** — `_geo_rerank` is a stable double-argsort of fixed
  RANSAC-fitness and trimmed-distance signals; deterministic given the geometry
  service's outputs.
- **Query construction** — GT visible bbox + mask + metric depth back-projection
  are read straight from disk; no augmentation, no random cropping.

## NOT bit-reproducible (documented, not hidden)

1. **FoundationPose pose estimation** (separate CUDA container). FP samples pose
   hypotheses and runs GPU refinement; results can differ slightly between
   otherwise-identical calls. We fix `refine_iter=5` and **store the returned
   `R`/`t` per instance** in every records.json, so any downstream metric
   (D_sym, F-score, or a later BOP-AR derivation) is reproducible *from the
   stored poses* even though re-running FP may not reproduce them exactly.
   cuDNN is not forced into deterministic mode inside the FP container (that is
   FP-service territory); the residual is small relative to the D_sym scale.

2. **open3d RANSAC in the dGeDi service.** `registration_ransac_based_on_feature_
   matching` accepts a `seed` only on newer open3d builds; where the installed
   build ignores it, the coarse alignment (and thus `ransac_fitness` /
   `d_ransac`) has run-to-run jitter. This only affects the *geometry re-rank
   order within the top-K shortlist*, never the fused ranking itself. To make a
   geometry run fully reproducible, cache the per-(query, candidate) dGeDi
   results and re-rank from cache.

## Practical guidance

- Headline retrieval numbers (3a Recall@K / MRR) are reproducible.
- Headline pose numbers (D_posed_gt, D_posed, Delta, F-score) are reproducible
  *from the stored poses*; a full FP re-run reproduces them to within FP's own
  stochasticity, not exactly. Report the stored-pose metrics as the numbers of
  record.
