# Stage-3 gallery preprocessing — handoff

Prepared 2026-08-06, scope extended 2026-08-08 to the full Stage-3 gallery.
**Not yet run.** Everything below is on disk and dry-run verified.
Read this before touching the preprocessing — most of it was established by archaeology
that is expensive to redo.

Driving requirement: `Downloads/STAGE3_EVALUATION_CONCEPT.md`. Stage 3a retrieves against
`G_proxy ∪ G_target,d`, Stage 3b against `G_proxy` alone, and "the curated gallery and its
preprocessing must be identical in both stages".

---

## 1. What exists

| file | what it is |
|---|---|
| `~/oscar_queue_ctl/run_bop_galleries.sh` | the driver — all six datasets |
| `tools/precompute_embeddings.py` | `--mesh-id-mode` added (see §5) |
| `rendering/onboard_dataset.sh` | patched: `HPR_PARAM`/`JITTER_STD` env-overridable (~line 117) |
| `tools/stage_ycbv_bop_ids.py` | files the textured YCB-Video meshes under BOP ids (§9) |
| `tools/stage_ycbv_fullmesh.py` | **superseded** by `--mesh-id-mode`; no longer called |

Per dataset, skipping any step whose output is already complete:

```
prepare → render → partial → describe → embed → sync → verify
```

No deletes. Halt-on-error. Resumable. `DRY_RUN=1` prints every step without executing.

```bash
DRY_RUN=1 bash ~/oscar_queue_ctl/run_bop_galleries.sh          # inspect
nohup bash ~/oscar_queue_ctl/run_bop_galleries.sh >/dev/null 2>&1 &
bash ~/oscar_queue_ctl/run_bop_galleries.sh gso                # 79% of the work
```

Default order is `lmo ycbv tless itodd housecat6d gso` — cheap datasets first so failures
surface within minutes rather than after the GSO render.

Knobs: `DRY_RUN`, `HPR_PARAM`, `JITTER_STD`, `LMO_KEEP`, `SKIP_SYNC`, `NUM_VIEWS`,
`NUM_POINTS`, `MIN_FREE_GB`, `RENDER_WORKERS` (default 3).

Render uses `rendering/parallel_render.sh` with 3 workers (~1.7x; the bottleneck is
per-model CPU prep and memory bandwidth, not the GPU). `RENDER_WORKERS=1` falls back to the
single-worker path — which a forced LM-O re-render does automatically, since
`parallel_render.sh` is resumable but has no `--overwrite`.

Drive sync is ON for all six (user decision 2026-08-08): renders cost ~5 h of GPU time to
regenerate and Drive would be the only offsite copy. ~15 GB of upload runs alongside.

## 2. Datasets and roles

| dataset | role | objects | source layout | state |
|---|---|---|---|---|
| lmo | target | 8 | `eval/datasets/lmo/models/obj_*.ply` | **stale**, full re-onboard (§5) |
| ycbv | target | 21 | `object_database/ycbv/obj_0000NN/textured_simple.obj` | staged from YCB-Video pkg (§9); captions stale |
| tless | target | 30 | `eval/datasets/tless/models_cad/obj_*.ply` | nothing yet |
| itodd | proxy | 28 | `eval/datasets/itodd/models/obj_*.ply` | nothing yet, **no captions** |
| housecat6d | proxy | 199 | `object_database/housecat6d/<cat>/<name>.obj` | captions stale (165 objs, 2–9 each) |
| gso | proxy | 1030 | `object_database/gso/<id>/meshes/model.obj` | captions stale (1021 objs, 2–5 each) |

All caption files except ITODD's are incomplete leftovers from an old ~5-view render set. The
driver detects and archives them automatically (§5).

## 3. G_proxy composition — decided 2026-08-08

**No curation.** The whole of GSO goes into the proxy gallery: all **1030** objects, no
filtering, no id list. The driver already does this — nothing to change.

HouseCat6D (199, including the 5 in `bg/`) and ITODD (28) are preprocessed too. That is
227 objects, ~17% of the run, and it keeps the concept doc's stated proxy set
(GSO + HouseCat6D + ITODD) available. Whether all three actually enter `G_proxy` is a
gallery-assembly decision at eval time and needs no re-rendering either way, so nothing here
forecloses it.

## 4. Blocking gaps

**RESOLVED 2026-08-08** — `ycbv_models.zip` downloaded from
`huggingface.co/datasets/bop-benchmark/ycbv` and extracted to `eval/datasets/ycbv/`
(`models/`, `models_eval/`, `models_fine/`, `models/models_info.json`). Left here because the
frame question underneath it is a live trap for anyone wiring Stage 3.

**Two YCB-V mesh sets exist and they are NOT interchangeable.** Measured across all 21 objects:

| | `object_database/ycbv_ycbvideo/<name>/textured_simple.obj` | `eval/datasets/ycbv/models/obj_0000NN.ply` |
|---|---|---|
| source | official YCB-Video package (+ `points.xyz`, identity-transform XML) | BOP |
| units | metres | millimetres |
| geometry | **identical** — bbox sizes agree to **0.00 mm** on every object | identical |
| origin | YCB-Video frame; bbox centre offset by up to **28.23 mm** | **exactly bbox-centred** (`|centre| = 0.000 mm`, all 21) |

So BOP re-centred every model on its bounding-box centre. The two sets are the same shape
related by a per-object translation of ~0.05–28 mm.

- **Retrieval / preprocessing: either set works** geometrically — but only the YCB-Video one
  is textured in a form Blender loads, so the driver uses it (§9). A rigid translation does
  not change shape, and rendering normalises the origin away regardless.
- **Pose: you must use BOP's `models/`.** GT poses are expressed in the bbox-centred frame,
  and the offset is large relative to BOP thresholds — 17 mm (z) on `036_wood_block`, 10 mm
  on `051_large_clamp`, against MSSD/ADD thresholds of 10% of diameter (~10–20 mm). Feeding
  the YCB-Video-framed mesh to pose evaluation would silently bias every YCB-V result.

`models_info.json` also supplies what cannot be derived: **symmetry annotations** for 7 of the
21 objects (ids 1, 13, 16, 18, 19, 20, 21 — the cans, bowl, wood block, clamps, foam brick),
which VSD/MSSD/MSPD require, plus official `diameter` values.

`id_to_label.json`, `test_targets_bop19.json` and the camera intrinsics are now in
`eval/datasets/ycbv/` (copied from the SSD).

**BOP scene data — RESOLVED 2026-08-08**, copied off the SSD into `eval/datasets/`:

| dataset | copied | note |
|---|---|---|
| ycbv | `test/` 670 MB, `models*/`, `test_targets_bop19.json`, `camera_{cmu,uw}.json`, `id_to_label.json` | 12 scenes, 4123 targets, 900 images |
| tless | `test_primesense/` 8.4 GB, `models_eval/`, `models_reconst/`, targets, `camera_primesense.json` | 20 scenes, 4904 targets, 1000 images |
| lmo | `test/` 730 MB, `models_eval/`, targets, `camera.json` | 1 scene, 1445 targets, 200 images |

**ITODD `test/` (4.2 GB) and HouseCat6D val scenes were deliberately NOT copied** — the
concept doc makes them gallery sources only ("They are not evaluated as query datasets in
Stage 3"), so only their CAD models are needed and those are already local.

Note BOP computes VSD/MSSD/MSPD against `models_eval/`, not `models_cad/` — hence the extra
copies. Drive's `eval/datasets` still holds only `shrec18`, so this data is now single-copy
on the internal disk plus the SSD.

**Disk — OK.** 56 GB free (2026-08-08, after MI3DOR was deleted); budget is ~18 GB for all
1316 objects (≈13 MB/object renders+partials — measured lmo 17 MB, MI3DOR 9.6 MB — plus
~0.8 GB caches). NB MI3DOR was removed from both `object_images/` and `object_database/`;
confirm it was verified on Drive first, since `object_database/MI3DOR` held the CAD meshes
and CLIP-text caches. Reserve if needed: `docker system df` shows ~230 GB unused images + 69 GB stopped
containers — **do not prune without asking**, the user has said so explicitly.

## 5. Findings that must not be re-derived

**HPR provenance.** Three commits split the galleries into cohorts: `ff1e0a55` (07-28 15:14)
render fix; `d14a709d` (07-30 11:39) true single-view partials; `30dc4d6e` (07-30 16:38)
made HPR radius + jitter configurable.

| gallery | renders | partials | HPR / jitter |
|---|---|---|---|
| MI3DOR | post-fix | regenerated 07-30 11:53 | 3.2 / 0 — by timing, not choice |
| shrec18_v2 | post-fix | post-fix | **2.8 / 0.001** — deliberate; best Stage-1 (E2_both 0.6428) |
| lmo | **stale 07-17** | **stale 07-17** | neither — predates both commits |

The driver defaults to **2.8 / 0.001** for all six, which is what makes the 3a/3b comparison
sound.

**LM-O is stale on all three counts**, not merely missing embeddings: renders predate the
render fix, "partials" predate the single-view fix (they were effectively full-mesh 360°
samplings, 32–72% hidden back-surface), captions describe the stale renders. The driver
forces a full re-onboard; `LMO_KEEP=1` opts out.

**Stale captions survive a re-render silently.** `generate_descriptions.py` resumes per image
filename and never rewrites an existing caption, so YCB-V's 5-per-object, GSO's 2–5 and
HouseCat6D's 2–9 would all persist into a fresh 42-view gallery. Counting `len(json)` reports
them complete — count captions *per object* instead. The driver compares the JSON's mtime
against the newest render and archives stale files to `descriptions_attributes.stale.json`.

**Full-mesh object ids come from the FILENAME STEM.** `eval_common.build_pipeline` uses
`os.path.splitext(os.path.basename(p))[0]`, which is right for flat layouts but maps every
GSO object to `"model"` and every YCB-V object to `"textured_simple"` — silently keeping one
mesh and discarding the other 1029 / 20. `build_pipeline` accepts an explicit
`cad_mesh_items`, so `tools/precompute_embeddings.py` gained `--mesh-id-mode`:

| mode | layout | datasets |
|---|---|---|
| `stem` | `<id>.obj` / `obj_000001.ply` | lmo, tless, itodd, housecat6d |
| `parent` | `<id>/textured_simple.obj` | ycbv |
| `grandparent` | `<id>/meshes/model.obj` | gso |

It raises rather than proceeds if a mode still yields duplicate ids. Verified 2026-08-08:
all six layouts give 100% unique ids. This replaced the flat-staging approach, which for GSO
would have meant duplicating 17 GB.

**Do NOT precompute GeDi during onboarding.** Gallery-side geometry needs only the CAD meshes,
which `prepare` stages. Descriptors must come from the eval's own `--precompute-gedi`, which
samples unit-sphere-normalised + voxel-downsampled clouds matching what `step_b2` registers.
`tools/precompute_gedi_descriptors.py` samples native-scale unnormalised clouds — reusing its
output silently corrupts geometry scores. Guarded behind `GEDI_PRECOMPUTE_FORCE=1`.

**Mesh units are heterogeneous — 1000× apart.** Measured 2026-08-08:

| dataset | units | example bbox |
|---|---|---|
| gso, housecat6d, ycbv | **metres** | GSO shoe `[0.098, 0.291, 0.118]` |
| tless, lmo, itodd | **millimetres** (BOP) | LM-O `[75.9, 77.6, 91.8]` |

Retrieval is unaffected (ULIP/Uni3D normalise to a unit sphere; renders are framed per
object). But Stage 3b reports `D_sym` in millimetres and normalised by target diameter, so a
proxy retrieved from GSO must be scaled to the target before posing — pipeline step 7 is the
scale-estimation stage. Eval-side concern, flagged here because mixing metre and millimetre
meshes in one gallery is easy to overlook.

**Container has no `python`** — use `python3 -u` inside `bash -lc`.

## 6. Pass set and why

`base,ulip_pc_rgb,uni3d,ulip_fullmesh` — **identical for all six datasets**, which the
concept doc requires. Trimmed from 6 on SHREC'18 evidence
(`object_retrieval/results_shrec18_v2_stage1/stage1_summary.csv`), user decision 2026-08-06:

| pass | verdict | evidence |
|---|---|---|
| `base` | keep | mandatory — it *is* the frozen config (clip-text + dino@42 + ulip partial) |
| `ulip_pc_rgb` | keep | free — re-tags base's cache, no recompute |
| `uni3d` | keep | E7 +0.0035 nDCG; marginal, kept as the ablation axis |
| `ulip_fullmesh` | keep | E2b +0.0015 on SHREC (wash) but +10 NN on MI3DOR's shape-only arm |
| `siglip` | **drop** | E4 0.5245 vs 0.5970 (−12%), NN_sub −31%. Not in `best_config.json`; `retrieval_mi3dor_eval_oscarplus.py` never references it — MI3DOR built that cache and never read it |
| `ulip_pc_xyz` | **drop** | O5 0.5954 vs 0.5970 — a tie. Second full PC encode at 8192 pts, and its 512-d SLIP space can't be matched by the ViT-bigG image encoder |

## 7. Time estimate

Measured from SHREC'18 (3308 objects, `shrec18_v2.log` + `prender_handoff.log`):
render 11.7 s/obj (3-worker parallel), partial+captions 19.6 s/obj, embed 8.7 s/obj (6 passes).
For 1316 objects with the trimmed 4-pass set: **≈14 h**, realistically 15–22 h — GSO's
textured high-poly meshes are heavier than SHREC's untextured CAD. Captions dominate, not
rendering (LLaVA runs over all 42 views per object).

The in-script render estimates (`~2884 min` for GSO) come from `onboard_dataset.sh`'s 4 s/view
figure, which was calibrated on an RTX 4050. On this 4090 with the 3-worker parallel render
the real rate is ~11.7 s/object. Ignore the printed estimate.

Consider `rendering/parallel_render.sh` (3 workers, ~1.7× real speedup — disk/memory-bandwidth
bound, not GPU bound) for the GSO render; the driver currently calls the single-worker path.

## 8. Known gaps (eval-side, not preprocessing)

**Geometry is the largest single gain on SHREC** — E2_both 0.6428 vs E2_none 0.5970
(+7.7% nDCG, NN_sub@50 +36%), ~13× the next-best positive. It applies to BOP (RGB-D queries
have real point clouds; MI3DOR's image queries do not). **But it is not wired for BOP**:
`eval_bop_pose.py` contains no reference to gedi/geometry/`GeometryReRanker`. Only
`experiment1_shrec18_stage1.py` has `--precompute-gedi`/`--with-geometry`.
NB `E2_chamfer_unaligned` *hurts* (0.5734) — the RANSAC alignment does the work.

**`eval_bop_pose.py` dataset configs are stale**: ycbv points at `ycbv_gso` and
`descriptions_tessa`; tless/lmo use `cad_mesh_glob: */meshes/model.obj` while `prepare`
produces `obj_XXXXXX/model.ply`. It also has no notion of a combined
`G_proxy ∪ G_target,d` gallery, which Stage 3a needs.

**open3d/libgomp**: no `LD_LIBRARY_PATH` on the `oscar` compose service, so `step_b2`'s
open3d import will fail. One-line compose fix (`.../torch/lib`), not yet applied.

## 9. YCB-V: which mesh set the gallery uses (decided 2026-08-08)

Gallery ids are **`obj_000001..obj_000021`**, matching `test_targets_bop19.json` and the
tless/lmo/itodd convention, so no `id_to_label.json` mapping is needed at eval time.

But the meshes rendered are the **textured YCB-Video ones**, not BOP's PLYs.
`tools/stage_ycbv_bop_ids.py` copies `textured_simple.obj` + `.mtl` + `texture_map.png` from
`object_database/ycbv_ycbvideo/<name>/` into `object_database/ycbv/obj_0000NN/`; no path
rewriting is needed because each object keeps its own directory, and `infer_model_id` takes
the parent dir as the id.

Rendering BOP's PLYs directly would have been simpler but is **wrong here**: `rendering.py`
imports PLY via Blender's legacy `bpy.ops.import_mesh.ply`, which ignores the
`comment TextureFile obj_0000NN.png` header, and those PLYs carry `texture_u/texture_v` but
**no per-vertex colour** — so every YCB-V object would render flat grey and DINOv2, SigLIP
and the LLaVA captions would all see unlabelled blobs.

Geometry is identical between the two sets and both normalise to max-dim 1.0, so nothing is
lost. `object_database/ycbv_ycbvideo/` is retained — it holds `points.xyz` (the canonical
ADD/ADD-S point set). **Pose evaluation still uses `eval/datasets/ycbv/models/`** (BOP frame).

## 10. Normalisation — checked, no action needed

The metre/millimetre split does **not** affect preprocessing. Both
`rendering.py:normalize_and_center_objects` and
`generate_partial_pointclouds.py:normalize_mesh` centre on the bbox and scale max-dim to 1.0
before anything is generated, so GSO (metres) and T-LESS (mm) land in the same space.

Two consequences worth keeping in mind:
- `jitter_std=0.001` is applied to the **normalised** cloud, i.e. 0.1% of the largest
  dimension on every dataset. The SHREC-tuned value transfers correctly. (Had it been applied
  pre-normalisation it would have been 1% of a metre-scale object and 0.001% of a
  millimetre-scale one, silently disabling the fix on half the datasets.) HPR is
  scale-relative by construction (`radius = r.max() * 10**param`).
- Partial point clouds therefore carry **no absolute scale**. Stage 3b's `D_sym` in mm must
  recover scale at pose time from the original meshes (pipeline step 7), not from anything
  preprocessing produces.
