# OSCAR Retrieval Pipeline — Guide

## Overview

OSCAR retrieves the correct 3D CAD model from a database given a **single RGB image** of an object in a scene. It is **training-free** and works in two stages:

1. **Stage 1 — Text filtering (CLIP):** Encodes the cropped query image and all pre-generated object descriptions with CLIP. Objects whose descriptions score below a similarity threshold are discarded, leaving only a shortlist of candidates.
2. **Stage 2 — Image re-ranking (DINOv2):** Compares the cropped query image against pre-rendered reference views of each candidate using DINOv2. The candidate with the highest visual similarity is the final prediction.

```
Query image (RGB scene)
        │
        ▼
  [Segmentation mask]  ← from scene_gt.json + mask_visib/
        │
   Crop object
        │
   ┌────┴────┐
   │  Stage 1│  CLIP:  crop_image ↔ text descriptions  →  top-k candidates
   └────┬────┘         (threshold = 0.37, fallback topk = 15)
        │
   ┌────┴────┐
   │  Stage 2│  DINOv2: crop_image ↔ rendered reference images  →  best match
   └────┬────┘
        │
     Predicted 3D model label
```

---

## Folder Structure for Retrieval

```
OSCAR/
├── object_images/{dataset}/          # Blender-rendered reference views
│   └── {object_name}/
│       ├── {name}_0.png … {name}_7.png   # 8 views
│       ├── {name}_bg.png
│       └── {name}_view0_CamMatrix.npy … (8 npy)
│
├── object_database/{dataset}/
│   └── descriptions_attributes.json  # LLaVA-generated descriptions per object
│
├── eval/datasets/{dataset}/test/
│   ├── id_to_label.json              # {obj_id: label_name} mapping
│   └── {scene_id}/                   # e.g. 000048/
│       ├── rgb/                      # scene RGB images
│       ├── mask_visib/               # per-instance segmentation masks
│       └── scene_gt.json             # ground truth annotations (obj_id, pose)
│
└── object_retrieval/
    └── retrieval_combi_eval.py       # ← main evaluation script
```

---

## Dataset Status

| Dataset     | Models  | Renderings | Descriptions               | Eval Scenes         | Ready? |
|-------------|---------|------------|----------------------------|---------------------|--------|
| YCBV_GSO    | 1051    | 1050       | ✅ `descriptions_attributes.json` | ✅ 12 scenes | **Yes** |
| MI3DOR      | 3848    | 3848       | ✅ (11/21 categories)       | ✅ image/test       | **Partial** (descriptions incomplete) |
| HouseCat6D  | 199     | 194        | ✅ `descriptions_attributes.json` | ❌ No eval scenes | **No** (need test scenes from dataset owner) |

---

## The Full OSCAR Pipeline — `retrieval_combi_eval.py`

### Config (top of file)

```python
# --- YCBV_GSO ---
ref_dir    = "../object_images/ycbv_gso"
bop_root   = "../eval/datasets/ycbv_gso/test/"
desc_file  = "../object_database/ycbv_gso/descriptions_attributes.json"
topk       = [15]      # CLIP candidate list size (fallback)
threshold  = 0.37      # CLIP cosine similarity threshold

# --- HouseCat6D (uncomment to switch) ---
# ref_dir  = "../object_images/housecat6d"
# bop_root = "../eval/datasets/housecat6d/test/"
# desc_file = "../object_database/housecat6d/descriptions_attributes.json"
```

**To switch dataset:** comment/uncomment the 3 config lines and change `result_folder`.

### How it runs step by step

1. **Load models:** CLIP ViT-B/32 + DINOv2 ViT-B/14
2. **Encode all descriptions (Stage 1 prep):** Each `image_descriptions` entry in `descriptions_attributes.json` is encoded with CLIP text encoder → `clip_desc_emb` matrix `[N_captions × 512]`
3. **Encode all reference images (Stage 2 prep):** All rendered PNGs from `object_images/` are encoded with DINOv2 → `ref_emb` matrix `[N_ref_imgs × 768]`
4. **Main loop — for each test scene → each image → each object instance:**
   - Load RGB image + segmentation mask from `mask_visib/`
   - Apply mask to image, grey-fill background → cropped object patch
   - **Stage 1:** CLIP-encode the crop → cosine similarity against all description embeddings → keep candidates where `sim ≥ 0.37` (or fallback to top-15 if nothing passes threshold)
   - **Stage 2:** DINOv2-encode the crop → cosine similarity against reference embeddings of candidate labels only → pick label with highest score
   - Record `{gt_label, pred_label, clip_candidates, dino_candidates}`
5. **Metrics:**
   - Top-1 accuracy = fraction of instances where `pred == gt`
   - Saved to `object_retrieval/results_topk_eval_ycbv_gso/accuracy_summary_topk_15.json`

### Run command (inside Docker)

```bash
cd /app/object_retrieval
python retrieval_combi_eval.py
```

### Output

```
object_retrieval/
└── results_topk_eval_ycbv_gso/
    └── accuracy_summary_topk_15.json   # {"0.37": {"correct": X, "total": Y, "accuracy": Z}}
```

---

## Other Retrieval Scripts

| Script | Type | Dataset | Description |
|--------|------|---------|-------------|
| `retrieval_combi_eval.py` | **Full OSCAR pipeline** | YCBV_GSO / HouseCat6D | CLIP text filter → DINOv2 re-rank. **Main script.** |
| `retrieval_mi3dor_eval.py` | Full OSCAR pipeline | MI3DOR | Same pipeline, configured for MI3DOR |
| `txt_img_wacv2.py` | Image→Image (DINOv2 only) | YCBV_GSO | No descriptions, pure DINOv2 image matching. Outputs mAP@k |
| `i2i_bbox_dino.py` | Baseline: BBox + DINOv2 | YCBV_GSO | Crops using bounding box (not mask), DINOv2 matching |
| `i2i_seg_dino.py` | Baseline: Seg + DINOv2 | YCBV_GSO | Crops using seg mask, DINOv2 matching |
| `i2i_bbox_clip.py` | Baseline: BBox + CLIP | YCBV (21 obj) | Crops using bbox, CLIP image matching. Needs `ycbv_test_bop19` |
| `i2i_seg_clip.py` | Baseline: Seg + CLIP | YCBV (21 obj) | Crops using seg mask, CLIP image matching |
| `text2img_eval.py` | Text→Image (CLIP only) | YCBV_GSO | Matches crop image embedding against description text embeddings via CLIP. No DINOv2. |
| `template_matching.py` | Template matching | YCBV (21 obj) | Classical approach, no neural network |

### Scripts that work right now (data is ready)
- ✅ `retrieval_combi_eval.py` — YCBV_GSO config
- ✅ `retrieval_mi3dor_eval.py` — MI3DOR (check its desc path)
- ✅ `txt_img_wacv2.py` — YCBV_GSO
- ✅ `i2i_bbox_dino.py` / `i2i_seg_dino.py` — YCBV_GSO

### Scripts that need additional data
- ❌ `i2i_bbox_clip.py`, `i2i_seg_clip.py`, `retrieval_combi_clip.py` — need `eval/datasets/ycbv_test_bop19/` (original 21-object YCBV BOP test set, not downloaded)
- ❌ `retrieval_combi_eval.py` (HouseCat6D) — need `eval/datasets/housecat6d/test/` (BOP scenes not available yet)

---

## Description Files — Tessa's vs. Self-Generated

All description files from the repo owner live in:
```
object_database/descriptions_tessa/{dataset}/descriptions_attributes.json
```

These have been copied to the paths expected by the retrieval scripts:
```
object_database/ycbv_gso/descriptions_attributes.json         ← placed ✅
object_database/ycbv_gso/descriptions_attributes_tessa.json   ← backup
```

The self-generated descriptions (via `description_genertor/`) use LLaVA 1.5-7B and produce the same JSON structure. They cover the same objects and can be swapped in by changing `desc_file` in the config.

---

## Key Parameters to Tune

| Parameter | Location | Effect |
|-----------|----------|--------|
| `threshold` | `retrieval_combi_eval.py` line ~29 | CLIP similarity cutoff for Stage 1. Lower → more candidates pass → more work for DINOv2 but fewer misses. Default: 0.37 |
| `topk` | `retrieval_combi_eval.py` line ~28 | Fallback candidate count when nothing passes threshold. Default: 15 |
| Description type | `desc_file` config | `descriptions_attributes.json` vs. `descriptions_caption.json` vs. `descriptions_comma.json` — different LLaVA prompts, affects Stage 1 quality |
| CLIP model | line ~33 | `ViT-B/32` (default) → could try `ViT-L/14` for better text-image alignment |
| DINOv2 model | line ~34 | `dinov2-base` → could try `dinov2-large` |

---

## Running on HouseCat6D (future)

Once you have the HouseCat6D BOP test scenes from the dataset authors, do:
1. Place at `eval/datasets/housecat6d/test/{scene_id}/rgb|mask_visib|scene_gt.json`
2. Create `eval/datasets/housecat6d/test/id_to_label.json` — mapping numeric obj_id → object name
3. In `retrieval_combi_eval.py`, uncomment the HouseCat6D config block
4. Run normally
