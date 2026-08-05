# Gallery Preprocessing

How to turn a dataset of CAD models into a **retrieval gallery**: rendered
views, partial point clouds, text descriptions, and the multi-encoder
embeddings that OSCAR retrieves against — all mirrored to Google Drive.

One command does the whole thing:

```bash
bash rendering/preprocess_gallery.sh --dataset shrec18_v2
```

This document explains what that command does, every knob it exposes, and how
to run it for a new dataset — no prior context required.

---

## 1. The pipeline at a glance

```
CAD models ──▶ 1. RENDER      42 views per model (Blender Cycles, GPU)
           ──▶ 2. PARTIAL     single-view partial point clouds (HPR)
           ──▶ 3. DESCRIBE    one text description per model (VLM)
           ──▶ 4. EMBED       per-encoder embedding caches ("passes")
           ──▶ 5. SYNC        push renders + caches to Google Drive
           ──▶ 6. VERIFY      rclone-check everything is really on Drive
           ──▶ (optional) delete local renders to free disk
```

Steps 1–3 run **inside the `oscar` Docker container** (it has Blender, PyTorch,
the encoders, and the GPU). Steps 5–6 run on the **host** (that's where rclone
lives). `preprocess_gallery.sh` orchestrates all of it and stops with a clear
message on the first failure.

---

## 2. Quick start

```bash
# Full run, keep the local renders (needed if you evaluate on this machine):
bash rendering/preprocess_gallery.sh --dataset shrec18_v2

# Full run, then free disk by deleting local renders once verified on Drive:
bash rendering/preprocess_gallery.sh --dataset MI3DOR --delete-after-sync

# Appearance encoders only (skip the shape/point-cloud passes):
bash rendering/preprocess_gallery.sh --dataset ycbv --passes base,siglip

# Onboard only — render/partial/describe/sync, no embeddings yet:
bash rendering/preprocess_gallery.sh --dataset gso --skip-embed
```

If your `rclone` is not on `PATH`:

```bash
RCLONE=/home/me/apps/rclone/rclone bash rendering/preprocess_gallery.sh --dataset ycbv
```

Every step is **safe to re-run**: finished renders are skipped and embeddings
are content-cached, so a re-run after a failure resumes rather than restarts.

---

## 3. Prerequisites

| Need | Detail |
|------|--------|
| **Docker + NVIDIA runtime** | The `oscar` service in `docker-compose.yml`. Verify: `docker compose run --rm oscar nvidia-smi`. |
| **rclone + a Drive remote** | Default remote `gdrive:Masterthesis/OSCAR`. Verify: `rclone lsd gdrive:Masterthesis/OSCAR`. |
| **GPU VRAM** | Tested on a 24 GB RTX 4090 (Cycles render + the encoders). |
| **Disk** | A full gallery is tens of GB. A ~3,300-model dataset is roughly **~25–35 GB** of renders + partials + caches. Check `df -h` before starting. |
| **Encoder checkpoints** | Mounted via `docker-compose.yml`: ULIP-2 at `/ulip` (`../ULIP_thesis`), Uni3D at `/uni3d` (`../Uni3D`). See §7. |

> **GPU-contention warning.** Blender Cycles rendering competes with the desktop
> X session for the GPU and can hard-lock the machine on long runs. For a big
> dataset, render on a headless target (`sudo systemctl isolate multi-user.target`)
> or otherwise keep the desktop idle.

---

## 4. Command-line options

| Option | Default | Meaning |
|--------|---------|---------|
| `--dataset <name>` | *(required)* | `shrec18_v2`, `MI3DOR`, `ycbv`, `gso`, `housecat6d`, `tless`, `itodd`, … |
| `--remote <rclone:path>` | `gdrive:Masterthesis/OSCAR` | Drive destination. |
| `--passes <list>` | all six (below) | Comma-separated embedding passes. |
| `--mesh-glob <glob>` | auto for known datasets | CAD mesh glob for the `ulip_fullmesh` pass. |
| `--delete-after-sync` | off (keep local) | Delete local renders **after** Drive verify. |
| `--skip-embed` | off | Stop after onboard + sync (no embeddings). |

Env overrides: `REMOTE`, `RCLONE`, `PASSES`.

---

## 5. The embedding passes

Each **pass** is one encoder's view of the gallery, written to its own
content-hashed cache. List them live with:

```bash
docker compose run --rm oscar python3 tools/precompute_embeddings.py --list
```

| Pass | Encoder / input | Cache location |
|------|-----------------|----------------|
| `base` | CLIP-text (descriptions) + DINOv2 (views) + ULIP-2 colored **partial** PC | `object_database/<ds>/.clip_text_cache_*.pt`, `object_images/<ds>/.dino_cache_*.pt`, `object_images/<ds>/.ulip_partial_cache_*.pt` |
| `siglip` | SigLIP image embeddings (drop-in for DINOv2) | `object_images/<ds>/.siglip_cache_*.pt` |
| `ulip_pc_rgb` | ULIP-2 colored partial, PC-mode tag (**reuses the `base` cache**, no recompute) | — |
| `ulip_pc_xyz` | ULIP-2 XYZ-only (8k pts, no color), partial | `object_images/<ds>/.ulip_partial_cache_*.pt` (different digest) |
| `uni3d` | Uni3D-g, partial PC | `object_images/<ds>/.ulip_partial_cache_*.pt` (different digest) |
| `ulip_fullmesh` | ULIP-2 colored, sampled from the **full CAD mesh** (not the views) | `<cad-dir>/.ulip_cache_*.pt` |

The default `--passes` is the full ablation set:
`base,siglip,ulip_pc_rgb,ulip_pc_xyz,uni3d,ulip_fullmesh`.

**Caching.** Every cache key is a SHA over the encoder config + the inputs'
identities (for meshes: relative path + **file size**, deliberately *not* mtime,
so a cache built elsewhere is reused). Consequences:
- Re-running a finished pass is a near-instant cache load.
- `ulip_fullmesh` reads only the CAD meshes, so its cache is **shared across
  any variants that use the same `cad/` directory** (e.g. `shrec18`,
  `shrec18_fixed`, `shrec18_v2`) — it will not recompute if one already ran.

**Failure semantics.** `precompute_embeddings.py` exits non-zero if *any* pass
fails, so `preprocess_gallery.sh` will HALT rather than declare an incomplete
gallery "done".

---

## 6. Per-dataset configuration

The CAD source, mesh glob, and partial-point-cloud knobs live **per dataset**
in the `case` block of [`rendering/onboard_dataset.sh`](../rendering/onboard_dataset.sh):

```
CAD_DIR       where the renderer finds the meshes
IMAGES_DIR    object_images/<dataset>   (renders + partial PCs land here)
DESC_OUTPUT   object_database/<dataset>/descriptions_attributes.json
MESH_GLOB     glob for generate_partial_pointclouds.py / ulip_fullmesh
IS_BOP        BOP datasets (tless/lmo/itodd) need PLY→prepared-CAD first
HPR_PARAM     hidden-point-removal radius exponent  (default 3.2)
JITTER_STD    Gaussian jitter on upsampled partials (default 0)
```

### Partial point clouds: HPR + jitter

Partial PCs are single-view point clouds carved out of the mesh by
**Hidden Point Removal** (Katz 2007): `radius = points.max() · 10^HPR_PARAM`.

| Setting | HPR param | Jitter σ | Notes |
|---------|-----------|----------|-------|
| legacy default | `3.2` | `0` | original behaviour; leaves some occluded points |
| `shrec18_v2` | `2.8` | `0.001` | stricter culling; jitter breaks coincident duplicates created when a sparse view is upsampled to 10k (so PointBERT's FPS+kNN don't collapse) |

### Adding a new dataset

1. Add a `case` in `rendering/onboard_dataset.sh` with `CAD_DIR`, `IMAGES_DIR`,
   `DESC_OUTPUT`, `MESH_GLOB`, and (if BOP) `IS_BOP=1` + `BOP_SOURCE`.
2. Add the same dataset to the `IMAGES_SUBDIR` map in
   `rendering/onboard_and_sync.sh` (a parallel map — both must agree).
3. If you want the `ulip_fullmesh` pass, add one line to `mesh_glob_for()` in
   `rendering/preprocess_gallery.sh` (or pass `--mesh-glob` at run time).
4. Run `bash rendering/preprocess_gallery.sh --dataset <new>`.

Known full-mesh globs (relative to repo root):

| Dataset | mesh glob |
|---------|-----------|
| `shrec18*` | `eval/datasets/shrec18/shrec18_full/cad/*.obj` |
| `MI3DOR` | `object_database/MI3DOR/model/test/*/*.obj` |
| `housecat6d` | `object_database/housecat6d/*/*.obj` |

---

## 7. Outputs & Drive layout

| Local | Drive (`gdrive:Masterthesis/OSCAR/…`) | Contents |
|-------|----------------------------------------|----------|
| `object_images/<ds>/` | `object_images/<ds>/` | rendered views (`*.png`), partial PCs (`*.npz`), image/point caches (`.*cache*.pt`), `precompute_manifest.json` |
| `object_database/<ds>/` | `object_database/<ds>/` | `descriptions_attributes.json`, CLIP-text cache, prepared CAD (BOP) |
| `object_retrieval/results_<ds>_stage1/` | — | per-run retrieval results |

The `ulip_fullmesh` cache lives next to the CAD meshes (`<cad-dir>/.ulip_cache_*.pt`),
not under `object_images/`, so it is **not** part of the one-way Drive verify —
it is a regenerable accelerator, kept local.

---

## 8. Encoder checkpoints

Mounted read-only into the container via `docker-compose.yml`:

| Pass | Checkpoint (container path) | Host mount |
|------|------------------------------|------------|
| `base`, `ulip_pc_rgb`, `ulip_fullmesh` | `/ulip/checkpoints/ulip2_pointbert_10k.pt` | `../ULIP_thesis` → `/ulip` |
| `ulip_pc_xyz` | `/ulip/checkpoints/ulip2_pointbert_8k_xyz.pt` | `../ULIP_thesis` → `/ulip` |
| `uni3d` | `/uni3d/modelzoo/uni3d-g/model.pt` | `../Uni3D` → `/uni3d` |
| `base`/`siglip` appearance | DINOv2 / SigLIP (auto-downloaded to the HF cache volume) | `hf_cache` volume |

A missing checkpoint makes that pass fail (and the whole run HALT). Uni3D:
download `modelzoo/uni3d-g/model.pt` from https://huggingface.co/BAAI/Uni3D.

---

## 9. Faster rendering — parallel workers (optional)

A single Blender worker leaves the GPU **~90% idle** (the bottleneck is per-model
CPU mesh prep — import, weld, normals, BVH — not GPU sampling), so rendering can
be sped up ~2–3× by running several workers that split the models between them.
`rendering/parallel_render.sh` does exactly that: each of N workers renders a
disjoint `index % N` shard (via the `SHARD_INDEX` / `SHARD_TOTAL` env vars that
`rendering.py` understands) into the same output directory — no output collision.

Run it **inside the `oscar` container**, in place of the normal render step:

```bash
docker compose run --rm oscar bash -lc \
  "cd /app && bash rendering/parallel_render.sh --dataset gso --workers 4"
```

Then continue with the remaining steps as usual (partial PCs, descriptions,
embeddings) — e.g. `onboard_dataset.sh --dataset gso --step partial` then
`--step describe`, followed by the embed step of `preprocess_gallery.sh`.

- **Workers:** each uses ~1 GB VRAM; **4** is a good default on a 24 GB GPU, and
  beyond ~6–8 rarely helps (CPU / memory-bandwidth bound).
- **Resumable:** already-rendered models are skipped — re-run to continue.
- **Safety:** several headless renderers still share one GPU driver. For a big
  run, stop the desktop first (`systemctl isolate multi-user.target`) so the X
  server is not also contending — that contention has hard-locked this machine.
- **Opt-in:** `rendering.py` behaves identically when `SHARD_TOTAL` is unset
  (the default), so the sequential path is unchanged. `parallel_render.sh` is a
  standalone accelerator you invoke explicitly; it is **not** auto-wired into
  `onboard_dataset.sh`. Preview the split with `--dry-run` before a real run.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| Machine locks up / reboots during a long render | GPU contention between Cycles and the desktop. Render headless (`systemctl isolate multi-user.target`) or keep the desktop idle. Re-run — finished models are skipped. |
| `ERROR: Unknown dataset '<x>'` | Missing `case` in `onboard_dataset.sh` **and/or** the `IMAGES_SUBDIR` map in `onboard_and_sync.sh` (both must have it). |
| `ulip_fullmesh … no mesh glob is known` | Pass `--mesh-glob '<repo-relative>/*.obj'` or add the dataset to `mesh_glob_for()`. |
| Embedding pass fails with "checkpoint not found" | See §8 — the checkpoint isn't mounted/downloaded. |
| Drive verify fails ("files NOT on Drive") | A sync gap. Local is kept (never deleted on a failed verify). Just re-run — rclone re-pushes the missing files, then re-verifies. |
| Disk fills mid-run | Watch `df -h`. Use `--delete-after-sync` for datasets you don't evaluate locally, or free space first. |

---

## 11. Script map

| Script | Role |
|--------|------|
| `rendering/preprocess_gallery.sh` | **Top-level entry point** (this doc). Orchestrates onboard → embed → sync → verify. |
| `rendering/parallel_render.sh` | Optional render accelerator: N Blender workers split the models across one GPU (§9). |
| `rendering/onboard_and_sync.sh` | Host-side: runs `onboard_dataset.sh` in Docker + background rclone sync of renders. |
| `rendering/onboard_dataset.sh` | In-container: per-dataset config, drives render/partial/describe. |
| `rendering/rendering.py` | Blender Cycles renderer (42 icosphere views). |
| `rendering/generate_partial_pointclouds.py` | Single-view partial PCs (HPR + optional jitter). |
| `rendering/generate_descriptions.py` | VLM text descriptions. |
| `rendering/rclone_watch.sh` | Background periodic rclone sync (used during onboarding). |
| `tools/precompute_embeddings.py` | Builds the embedding caches (the passes). `--list` shows all passes. |
