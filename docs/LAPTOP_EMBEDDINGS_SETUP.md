# Consuming the shrec18_fixed embedding caches on the eval PC (laptop)

The gallery `.pt` caches are keyed by relpath + byte-size + model config (no
abspaths/mtimes), so they are reusable across machines — BUT the laptop must
encode *queries* with the identical encoders, or the fingerprints won't match
and it will recompute (or, worse, mismatch). Replicate all of the following.

## 0. Environment (must match the gallery PC)
- Docker image: `tholoi/oscar-plus` (same image on both PCs).
- **`timm == 1.0.25`** — confirmed on the gallery PC (`pip show timm`). EVA-giant's
  timm config can shift across releases, so pin this.
- `open_clip` present; no `einops`/`pytorch3d`/`knn_cuda` needed for these arms.

## 1. Checkpoints (exact filenames — basenames are part of the cache key)
Place under the container's `/ulip/checkpoints/` and `/uni3d/modelzoo/uni3d-g/`:

- `ulip2_pointbert_10k.pt`  (ULIP-2 colored 10k, xyzrgb, 1280-d) — you already have this.
- `ulip2_pointbert_8k_xyz.pt`  ← ULIP-2 XYZ-only arm (O5). **New — you must mirror this.**
  Source: HF `SFXX/ulip` → `ULIP-2/pretrained_models/ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt`
  (input_dim=3, 8192 pts, 512-d). Rename to `ulip2_pointbert_8k_xyz.pt`.
- `uni3d-g/model.pt`  ← Uni3D-g (E7). Source: HF `BAAI/Uni3D` → `modelzoo/uni3d-g/model.pt` (2.03 GB).

## 2. Uni3D repo + the two inference patches (the FPS portability crux)
```bash
git clone https://github.com/baaivision/Uni3D ~/thesis/Uni3D
cd ~/thesis/Uni3D
git checkout 64e03c3c42c196e8cb5ed03857810af9fc9ac39c   # base commit the patch targets
git apply /path/to/OSCAR/docs/uni3d_inference.patch      # <-- shipped in this repo
```
The patch (`docs/uni3d_inference.patch`) touches exactly two files:
- `models/point_encoder.py`: wraps `from pointnet2_ops import pointnet2_utils` in
  try/except and adds a **pure-PyTorch `fps`** (deterministic, seeded at index 0).
- `models/uni3d.py`: makes `from . import losses` optional (it drags in h5py/data).

Then mount it: docker-compose `- ~/thesis/Uni3D:/uni3d` (already added on gallery PC as `../Uni3D:/uni3d`).

> **⚠ FPS must be byte-identical between query and gallery, or E7 scores garbage
> silently (no error).** The patch runs the CUDA `pointnet2_ops` path *if present*,
> else the pure-torch path. The `tholoi/oscar-plus` image ships **without**
> `pointnet2_ops`, so **both** PCs take the pure-torch branch → identical sampling.
> **Do NOT `pip install pointnet2_ops` on only one machine.** Either neither has it
> (current state — correct), or both do. Mixed = mismatched E7 embeddings.

## 3. Code changes (already on branch `tessa-pc` — pull, don't hand-edit)
- `pipeline/config.py`: Uni3D fields set to uni3d-g (repo_path=/uni3d,
  checkpoint=/uni3d/modelzoo/uni3d-g/model.pt, pc_model=eva_giant_patch14_560,
  pc_feat_dim=1408, pc_encoder_dim=512, num_group=512, group_size=64,
  num_points=10000, embed_dim=1024).
- `pipeline/step5_shape_matching.py`: real `Uni3DEncoder._load` (builds model
  from repo + checkpoint, import-isolated from ULIP's `models` package) and
  `encode` feeding xyz+rgb (6-ch).
- `experiments/experiment1_shrec18_stage1.py`: PASS_DEFS `ulip_pc_xyz` overrides
  set backbone=pointbert, checkpoint=ULIP_CKPT_XYZ (8k), num_points=8192,
  embed_dim=512.

## 4. Where the caches land on Drive
- `object_images/shrec18_fixed/` — DINO/SigLIP (all views) + ULIP/Uni3D partial
  caches (`.ulip_partial_cache_*.pt`, distinct digests per encoder) + manifest.
- `eval/datasets/shrec18/shrec18_full/cad/.ulip_cache_*.pt` — ULIP full-mesh
  (path-mirrored; depends only on the CAD meshes, identical for shrec18 &
  shrec18_fixed).
- `object_database/shrec18_fixed/.clip_text_cache_*.pt` — CLIP-text
  (description) embeddings.

## 5. Query-side caches (for rerunning the Stage-1 ablation grid, not just the
##    live pipeline) — `eval/datasets/shrec18/stage1/` on Drive

These are new (2026-07-24) and cover the *query* side, not the gallery:
running `experiments/experiment1_shrec18_stage1.py --ablations ... --all`
needs a rendered snapshot + point cloud for all 2101 official queries, plus
their encoder embeddings — expensive to build (the point-cloud passes are
~1-2s/query, unbatched). All of it is now cached and shipped:

- `eval/datasets/shrec18/stage1/queries/<qid>.png` + `<qid>.npz` — the
  per-query rendered snapshot + extracted point cloud, derived from the raw
  `.ply` scans (see §6). `gt/queries_index.json` indexes them.
- `eval/datasets/shrec18/stage1/query_pc_cache/pc_query_cache_<hash>.pt` —
  point-cloud query embeddings for the `pc`-mode passes (`ulip_pc_rgb`,
  `ulip_pc_xyz`, `uni3d`). Content-fingerprinted the same way as the gallery
  caches (encoder + checkpoint + dims, not path/mtime) — one file per
  distinct encoder config, never collide.
- `object_retrieval/results_shrec18_fixed_stage1/_cache/ulip_query_img_cache.pt`
  — query-side embeddings for the `cross`-mode passes (image → ULIP-2 joint
  space). Shared across all `cross`-mode passes (they use the same image
  encoder regardless of which gallery checkpoint is being matched against).

Pull these to the same relative paths and `experiments/experiment1_shrec18_stage1.py`
picks them up automatically (cache-hit, no re-encoding) — as long as the
query set matches (see §6, official GT kit) and the encoder config for a
given pass is unchanged from what built the cache.

## 6. Official SHREC'18 evaluation kit + raw query distribution

Two more prerequisites, needed only for *running ablations* (not for the
live per-scene pipeline):

- GT labels (small, git-clonable):
  ```bash
  git clone https://github.com/hkust-vgd/shrec18 eval/shrec18_official
  ```
- Raw query scans + official relevance lists (`rgbd/*.ply`, `results/*.txt`
  — NOT included in the gallery/query caches above, since they're the
  *source* data, not derived embeddings):
  ```bash
  wget https://hkust-vgd.ust.hk/scenenn/shrec18/shrec18_full_jan28.zip
  unzip shrec18_full_jan28.zip "shrec18_full/rgbd/*" "shrec18_full/results/*" \
      -d eval/datasets/shrec18/
  rm shrec18_full_jan28.zip   # 6.2GB archive, only rgbd/+results/ (~16GB) needed
  ```
  Only needed if you don't already have `eval/datasets/shrec18/shrec18_full/{rgbd,results}/`
  populated — if the query caches (§5) already cover the queries you care
  about, you may not need this at all.

## Summary of what unblocks each arm on the eval PC
| Arm | Blocked? | To unblock |
|-----|----------|-----------|
| DINOv2, SigLIP (all views) | No | ship cleanly, nothing needed |
| ULIP-colored full-mesh + partial | No | already have `ulip2_pointbert_10k.pt` |
| ULIP XYZ-only (O5) | Yes | mirror `ulip2_pointbert_8k_xyz.pt` + pull `tessa-pc` |
| Uni3D-g (E7) | Yes | clone+patch Uni3D, mirror `uni3d-g/model.pt`, mount `/uni3d`, keep pure-torch FPS |
