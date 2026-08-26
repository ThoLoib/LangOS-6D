# OSCAR+ — Pipeline & Preprocessing Implementation

*Thesis-input reference for the method. Describes the offline preprocessing and
the online 8-step pipeline, in the frozen **BASE** configuration and with the
optional **geometry re-ranking** sub-step. Pseudo-code is given where it clarifies
the data flow; every design choice is cited. Section numbers `Step A/B1/B2/C`
follow the thesis chapter 3 structure the code is annotated against.*

Last updated 2026-08-26.

---

## 0. Overview — what the system is

OSCAR+ extends **OSCAR** [Pulli et al. 2025] from a two-channel (language + image)
open-set CAD retriever into a **three-channel retrieve-then-pose** pipeline that adds a
frozen **3D shape** channel and an optional **local-geometry** re-ranking, then estimates
the object's **6-DoF pose** from the retrieved CAD. It is training-free: every encoder is
a frozen off-the-shelf foundation model.

The system is organised as an **8-step pipeline** plus an out-of-band geometry sub-step:

| Step | Thesis stage | Role | Key model / method |
|---|---|---|---|
| **1** Localization | A | Prompt-conditioned detection + segmentation of the query object | SAM-6D / SAM [Lin et al. 2024], OSCAR grounding [Pulli et al. 2025] |
| **2** Point cloud | A | Back-project the masked RGB-D crop → partial query point cloud | — |
| **3** CLIP retrieval | B1 | Semantic channel **S_text** (image ↔ per-view CAD descriptions) | CLIP [Radford et al. 2021] |
| **4** DINO re-rank | B1 | Appearance channel **S_view** (multi-view template matching) | DINOv2 [Oquab et al. 2024] / SigLIP [Zhai et al. 2023] |
| **5** Shape matching | B1 | Shape channel **S_shape** (query cloud/image ↔ CAD point clouds) | ULIP-2 [Xue et al. 2024] / Uni3D [Zhou et al. 2023] |
| **6** Fusion | B1 | Weighted-sum fusion of the three channels → ranked candidates | — |
| **B2** Geometry re-rank | B2 | *(optional)* Re-order the fused top-K by local-geometry registration | GeDi [Poiesi & Boscaini 2022], FreeZe [Caraffa et al. 2025], URED [Di et al. 2023] |
| **7** Scale estimation | C | Coarse alignment + metric scale of the retrieved CAD | — |
| **8** Pose estimation | C | 6-DoF pose from RGB-D + retrieved CAD | FoundationPose [Wen et al. 2024], ICP fallback |

The **BASE configuration** (frozen after Stage-1 tuning, see `EVALUATION_STORY_AND_PLAN.md`)
uses steps 1–6 with geometry **off**; the pose experiments add steps 7–8; the geometry
ablation adds sub-step B2.

---

## 1. Offline preprocessing

All gallery-side computation is done once, offline, and cached. Nothing below runs at query
time.

### 1.1 Multi-view rendering — `rendering/rendering.py`

Each gallery CAD is rendered from **V = 42** viewpoints on a sphere (Blender / `bpy`),
producing RGB template images used by the appearance channel (Step 4) and as the anchor
frames for the partial point clouds (§1.2) and descriptions (§1.3). Views are **FPS-ordered**
(farthest-point sampling over the viewpoint sphere) so that "the first N views" is a
well-defined, roughly uniform subset — this is what makes the view-count sweep (O4/A2/A7)
and the `SHAPE_AGG_VIEWS`/`num_views` truncation meaningful.

```
for cad in gallery:
    mesh   = load(cad); normalize_to_unit_sphere(mesh); weld+recompute_normals(mesh)
    cams   = fps_order(fibonacci_sphere(N=42), radius=r)     # deterministic, uniform
    for i, cam in enumerate(cams):
        img = blender_render(mesh, cam, lighting=camera_lights, shading=Standard)
        save(img, f"{cad}/{cad}_{i}.png")
    save_camera_extrinsics(cams)                              # reused by §1.2
```

### 1.2 Partial point clouds — `rendering/generate_partial_pointclouds.py`

For the shape channel's **partial-view** references, each CAD is turned into **42 single-view
partial point clouds** by rendering a depth map from the *same* camera poses as §1.1 and
back-projecting it (hidden-point removal / z-buffer visibility). This makes the gallery
reference a *partial* observation — geometrically comparable to the partial query cloud a
real sensor produces — rather than the complete mesh.

```
for cad in gallery:
    for i, cam in enumerate(load_camera_extrinsics(cad)):
        depth = render_depth(mesh, cam)                      # z-buffer, single view
        pc    = backproject(depth, cam.K, cam.pose)          # visible surface only
        pc    = fps_downsample(pc, 10_000); attach_rgb(pc)   # xyzrgb, 10k points
        save_npz(f"{cad}/{cad}_{i}_partial.npz", pc)
```

The alternative **full-mesh** reference (ablation A4) samples one 10k-point cloud from the
whole mesh instead — one embedding per CAD rather than 42.

### 1.3 Per-view descriptions — `rendering/generate_descriptions.py`

The text channel scores the query image against **natural-language descriptions of the CAD's
appearance**. A vision-language model (**LLaVA-1.5-7B**) captions each rendered view, yielding
`{cad: {view_name: text}}` (42 descriptions per CAD) stored in `descriptions_attributes.json`.

```
for cad in gallery:
    for view in rendered_views(cad):                         # all 42
        desc[cad][view] = LLaVA(view_img, prompt="Describe the object's appearance ...")
save_json(desc, "descriptions_attributes.json")
```

### 1.4 Gallery embedding caches — `tools/precompute_embeddings.py`

Every gallery-side encoder output is precomputed and cached to a content-fingerprinted file,
so eval-time cost is dominated by the *query* pass only. One cache per (encoder, config):

| cache | content | encoder |
|---|---|---|
| `.dino_cache_*` | per-CAD, per-view DINOv2 embeddings (all 42) | DINOv2-base |
| `.clip_text_cache_*` | CLIP text embedding of every per-view description | CLIP ViT-B/32 |
| `.ulip_partial_cache_*` | per-CAD, per-view ULIP-2 embeddings (42) | ULIP-2 (colored / xyz) or Uni3D |
| `.ulip_cache_*` | one whole-mesh ULIP-2 embedding per CAD (full-mesh) | ULIP-2 |

```
for cad in gallery:
    dino_cache[cad]  = [ DINOv2(render(cad,i))          for i in range(42) ]
    clip_cache[cad]  = [ CLIP_text(desc[cad][view])     for view in views(cad) ]
    ulip_cache[cad]  = [ ULIP2_pc(partial_pc(cad,i))    for i in range(42) ]   # partial
    ulipfm_cache[cad]=   ULIP2_pc(sample_mesh(cad,10k))                        # full-mesh
save_fingerprinted(dino_cache, clip_cache, ulip_cache, ulipfm_cache)
```

Caches are keyed by a stable content fingerprint (encoder name + config + a size-only file
signature), so they are cross-machine-reproducible and never silently collide across configs
(e.g. `dino_pooling=cls` vs `mean` write different files).

### 1.5 GeDi geometric descriptors — `tools/precompute_gedi_descriptors.py`

For the optional geometry re-ranking (§3), **GeDi** [Poiesi & Boscaini 2022] local descriptors
are precomputed for the gallery CADs (from the full meshes) and, per experiment, for the
query clouds. GeDi descriptors are rotation-invariant, so they support alignment across
arbitrary pose. (These are *unit-sphere-normalised* clouds — distinct from the native-scale
clouds used for metric pose in Step 7.)

```
for cad in gallery:  gedi_gallery[cad] = GeDi(sample_mesh(cad), keypoints=Nkp)
for q  in queries:   gedi_query[q]     = GeDi(query_cloud(q),  keypoints=Nkp)
```

---

## 2. The online pipeline — BASE configuration (steps 1–6, geometry off)

**Input:** one query (RGB image, or RGB-D frame + a text prompt naming the object).
**Output:** a ranked list of gallery CAD ids with a fused score; the top-1 is the retrieved model.

### Step 1 — Localization (Step A) · `step1_localization.py`
Prompt-conditioned detection + segmentation isolates the query object, following OSCAR's
grounding paradigm [Pulli et al. 2025] with SAM-based masks [Lin et al. 2024]. In the
*evaluation* protocol, **GT visible bbox + mask are used instead** (Stage-3), so retrieval and
pose are measured independently of segmentation error.
```
mask, bbox = detect_and_segment(rgb, prompt)      # or GT bbox+mask at eval time
roi        = crop(rgb, bbox, mask)                # the object crop fed to Steps 3–5
```

### Step 2 — Point cloud extraction (Step A cont.) · `step2_pointcloud.py`
Back-projects the masked depth into a **partial query point cloud** (only used when depth is
available — pc-mode / Stage-1 / Stage-3).
```
query_pc = backproject(depth * mask, K)           # camera frame; None if no depth (cross-mode)
```

### Step 3 — Semantic channel S_text (Step B1) · `step3_clip_retrieval.py`
CLIP embeds the ROI; it is scored against every cached per-view **description** embedding, and
each CAD's score is the **max over its 42 descriptions** (best-matching view). [Radford et al. 2021]
```
q      = CLIP_image(roi)                           # (D,)
sims   = q @ clip_text_cache.T                     # over all 42*|gallery| description rows
S_text[cad] = max( sims[row] for row in rows(cad) )        # best description per CAD
```

### Step 4 — Appearance channel S_view (Step B1) · `step4_dino_reranking.py`
DINOv2 embeds the ROI (patch tokens **mean-pooled** → one vector). It is scored against the
cached per-view CAD embeddings; the 42 per-view similarities are aggregated by a
**top-k-softmax** (query-conditioned attention, OPEN-style [Chu et al. 2024], with CNOS's
k_v=5, τ=0.5 [Nguyen et al. 2023]).
```
q      = mean_pool(DINOv2(roi))                    # (D,)
for cad in gallery:
    v      = [ q · e for e in dino_cache[cad] ]    # 42 per-view cosine sims
    topk   = largest_k(v, k=5)
    w      = softmax(topk / 0.5)
    S_view[cad] = Σ w * topk                        # top-k-softmax aggregation
```
*(SigLIP [Zhai et al. 2023] is the E4 alternative, scored the same way via its MAP-head.)*

### Step 5 — Shape channel S_shape (Step B1) · `step5_shape_matching.py`
ULIP-2 [Xue et al. 2024] embeds the query into a CLIP-aligned 3D space, **two modes**:
- **pc-mode** (depth available): encode the query *point cloud*.
- **cross-mode** (no depth): encode the query *image* via ULIP-2's image tower (same space).

It is scored against the cached per-view CAD point-cloud embeddings, aggregated by the **same
top-k-softmax** as Step 4 (k=5, τ=0.5) over the first `SHAPE_AGG_VIEWS = 42` gallery views.
```
q      = ULIP2_pc(query_pc)     if pc_mode   else   ULIP2_image(roi)      # cross-mode
for cad in gallery:
    v      = [ q · e for e in ulip_cache[cad][:42] ]      # per-view sims
    S_shape[cad] = topk_softmax(v, k=5, tau=0.5)
```
*(Uni3D [Zhou et al. 2023] is the E7/A3 alternative; it is pc-only — no image tower, so no
cross-mode. Full-mesh (A4) uses the single `.ulip_cache_*` embedding per CAD, `agg = identity`.)*

### Step 6 — Fusion (Step B1) · `step6_fusion.py`
Each channel score is **min–max normalised** over the gallery, then combined by a weighted sum
with the frozen BASE weights **w = (0.3, 0.4, 0.3)**; the ranking is by fused score over the
**full database** (no pruning in BASE).
```
def fuse(S_text, S_view, S_shape, w=(0.3,0.4,0.3)):
    n = lambda S: (S - min(S)) / (max(S) - min(S) + eps)
    fused = w[0]*n(S_text) + w[1]*n(S_view) + w[2]*n(S_shape)
    return sort_desc(fused)                        # ranked (cad, fused_score)
ranking = fuse(S_text, S_view, S_shape)
retrieved_cad = ranking[0]
```

**Fusion strategy alternatives** evaluated (Stage-1 Block B): Reciprocal Rank Fusion
[Cormack et al. 2009] (B1); scope/ordering — full-DB vs a CLIP-text-pruned cascade (B4), where
the **OSCAR cascade** is CLIP-τ threshold pruning (τ=0.37) then a DINOv2 best-view arg-max over
the shortlist (`oscar_maxview`), reproducing OSCAR's actual mechanism [Pulli et al. 2025].

---

## 3. Geometry re-ranking — Sub-step B2 (optional) · `step_b2_geometry_reranking.py`

After Step 6, the **top-K = 50** fused candidates are re-ordered by an **alignment-aware
local-geometry** score. GeDi descriptors (precomputed, §1.5) give putative correspondences;
RANSAC estimates a rigid alignment; a trimmed surface distance is evaluated **after** alignment.
Served by a separate `dgedi` micro-service (`dgedi_bridge.py`), health-gated so a missing
service degrades to the fused ranking rather than silently mis-scoring.

```
def geometry_rerank(fused_topK, query_cloud, K=50, mode="chamfer_ransac"):
    for cad in fused_topK:                                    # only the top-K
        corr      = match(gedi_query, gedi_gallery[cad])      # GeDi correspondences
        T, fit    = ransac_register(corr)                     # rigid alignment + fitness
        if   mode == "fitness":          g[cad] =  fit                       # FreeZe-style
        elif mode == "chamfer_ransac":   g[cad] = -trimmed_chamfer(align(query,cad,T))
        elif mode == "chamfer_unaligned":g[cad] = -trimmed_chamfer(query, cad)  # diagnostic
        elif mode == "both_borda":       g[cad] =  borda(fit, -chamfer)      # rank fusion
    return reorder(fused_topK, by=g)                          # tail keeps fused order
```

- **`chamfer_ransac`** (GeDi→RANSAC align, then trimmed distance) is the BASE geometry arm.
- **`fitness`** = RANSAC inlier fitness only [Caraffa et al. 2025].
- **`chamfer_unaligned`** = distance *without* alignment — a **not-a-method control**
  [Di et al. 2023] proving the gain comes from aligning first.
- **`both_borda`** = mean-rank (Borda) fusion of geometry with the base ranking
  [Aslam & Montague 2001].
- Shortlist depth K is an ablation (C2): K=50 > 20 > 5. ICP refinement is retained but **kept
  off** (it can launder a wrong retrieval into a plausible fit).

**Where geometry helps:** it improves clean-scan category retrieval (Stage-1) — especially
top-1 — but is **net-negative for cluttered-scene pose** (Stage-3), so it is **off in the pose
pipeline** (see `EVALUATION_STORY_AND_PLAN.md` §5).

---

## 4. Pose estimation — Step C (steps 7–8)

Used only in the pose experiments (Stage-3). The retrieved CAD + the RGB-D crop drive a 6-DoF
pose estimate.

### Step 7 — Coarse alignment + scale · `step7_scale_estimation.py`
Estimates the metric scale of the retrieved CAD against the query cloud (proxies from
different datasets are normalised to a common metric scale; for the pose experiments proxies
are placed at their **true metric size**, a deterministic m→mm conversion, *no* learned scale).

### Step 8 — 6-DoF pose · `step8_pose_estimation.py`
**FoundationPose** [Wen et al. 2024] is called over HTTP (two-container architecture:
`pipeline/foundationpose_bridge.py` → the `foundationpose` service, port 5050) with the RGB,
depth, mask, camera intrinsics, and the retrieved CAD; **ICP is the deterministic fallback**
when the service is unavailable.
```
pose = call_foundationpose(rgb, depth, mask, cad=retrieved_cad, K=cam_K)   # 4x4 SE(3)
if pose is None: pose = icp(query_cloud, cad_cloud)                        # fallback
```
Raw estimated poses are stored so every pose metric is reproducible from disk (FoundationPose's
GPU hypothesis sampling is not bit-deterministic — documented, not hidden).

---

## 5. The frozen BASE configuration (reference)

| Component | Value | Source |
|---|---|---|
| Detection / mask | SAM-6D grounding; **GT bbox+mask at eval** | [Lin et al. 2024; Pulli et al. 2025] |
| S_text | CLIP ViT-B/32, image ↔ 42 per-view descriptions, **max** over views | [Radford et al. 2021] |
| S_view | DINOv2-base, **mean** patch-pool, **42 views**, top-k-softmax **k=5, τ=0.5** | [Oquab 2024; Nguyen 2023; Chu 2024] |
| S_shape | ULIP-2 coloured (1280-d), 10k pts, partial views, first **42** pooled, top-k-softmax **k=5, τ=0.5** | [Xue et al. 2024] |
| Shape mode | **pc** (depth) / **cross** (no depth) | — |
| Fusion | min–max norm, weighted sum **w=(0.3,0.4,0.3)**, **full-DB** | — |
| Geometry | **off** (arms: GeDi→RANSAC, top-K=50) | [Poiesi 2022; Caraffa 2025] |
| Pose | FoundationPose + ICP fallback | [Wen et al. 2024] |
| Determinism | `PYTHONHASHSEED=0`; SHA-256 resample seed; stored raw poses | — |

The single per-stage change is the shape **mode** (pc vs cross). All view-aggregation, pooling,
weights and scope are identical across the three stages (config-comparability audit 2026-08-26).

---

## References

- **OSCAR** — Pulli et al., *OSCAR: Open-Set CAD Retrieval*, 2025. `pulliOSCAROpenSetCAD2025`
- **CNOS** — Nguyen et al., *CNOS: A Strong Baseline for CAD-based Novel Object Segmentation*, ICCVW 2023. `nguyenCNOSStrongBaseline2023`
- **OPEN** — Chu et al., *Occlusion-invariant Perception*, IEEE TCSVT 2024. `chuOPENOcclusionInvariantPerception2024a`
- **ULIP-2** — Xue et al., *ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding*, CVPR 2024. `xueULIP2ScalableMultimodal2024`
- **Uni3D** — Zhou et al., *Uni3D: Exploring Unified 3D Representation at Scale*, ICLR 2024. `zhouUni3DExploringUnified2023`
- **CLIP** — Radford et al., *Learning Transferable Visual Models from Natural Language Supervision*, ICML 2021.
- **DINOv2** — Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024.
- **SigLIP** — Zhai et al., *Sigmoid Loss for Language Image Pre-Training*, ICCV 2023. `zhaiSigmoidLossLanguage2023`
- **GeDi** — Poiesi & Boscaini, *Learning General and Distinctive 3D Local Deep Descriptors*, IEEE T-PAMI 2022.
- **FreeZe** — Caraffa et al., *FreeZe: Training-free Zero-shot 6D Pose*, 2025. `caraffaFreeZeTrainingfreeZeroshot2025`
- **URED** — Di et al., *Unsupervised 3D ... (trimmed surface distance)*, 2023. `diUREDUnsupervised3D2023`
- **RRF** — Cormack et al., *Reciprocal Rank Fusion*, SIGIR 2009. `cormackReciprocalRankFusion2009`
- **Metasearch/Borda** — Aslam & Montague, *Models for Metasearch*, SIGIR 2001. `aslamModelsMetasearch2001`
- **SAM-6D** — Lin et al., *SAM-6D: Segment Anything for Zero-shot 6D Pose*, CVPR 2024. `linSAM6DSegmentAnything2024`
- **FoundationPose** — Wen et al., *FoundationPose: Unified 6D Pose Estimation and Tracking*, CVPR 2024.
- **SHREC 2025** — Nguyen et al., *SHREC 2025 Retrieval Track*, 2025. `nguyenSHREC2025Retrieval2025`
