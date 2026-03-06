# OSCAR+: Shape-Aware Open-Set CAD Retrieval

This branch (`exp/ulip2`) extends the original two-stage OSCAR baseline with a full **8-step modular pipeline** and integrates **ULIP-2 shape-aware retrieval** as a third scoring channel.

Baseline reproduced at **75.95% Top-1** on YCBV-GSO.  
New pipeline adds scale estimation and pose estimation on top of the retrieval result.

---

## Pipeline Overview

```
Natural language prompt + RGB-D image
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1  │  Object Localization  │  GroundingDINO + SAM         │
│  Step 2  │  Point Cloud          │  RGB-D → 3D point cloud       │
│  Step 3  │  CLIP Retrieval       │  Prompt/description matching  │
│  Step 4  │  DINOv2 Re-Ranking    │  Visual feature comparison    │
│  Step 5  │  ULIP-2 Shape Match   │  3D geometry similarity (new) │
│  Step 6  │  Score Fusion         │  CLIP · DINO · ULIP → rank    │
│  Step 7  │  Scale Estimation     │  Observed bbox vs. CAD bbox   │
│  Step 8  │  Pose Estimation      │  ICP / PnP                    │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
  Best matching CAD model + 6D pose
```

All pipeline code lives in `pipeline/`. Each step is a self-contained module with a single dataclass result.

---

## Getting Started

### 1. Clone
```bash
git clone git@github.com:pullover00/OSCAR.git
cd OSCAR
git checkout exp/ulip2
```

### 2. ULIP-2 Checkpoint (for Step 5)
Clone the ULIP repo next to this one and place the checkpoint:
```bash
cd ..
git clone https://github.com/salesforce/ULIP.git
# Download checkpoint:
# ulip2_pointbert_10k.pt → ULIP/checkpoints/ulip2_pointbert_10k.pt
```
The `docker-compose.yml` mounts `../ULIP` as `/ulip` inside the container.

### 3. Build and Run
```bash
docker compose build
docker compose run --rm -it oscar bash
```

---

## Rendering & Object Database

The database must be rendered before retrieval. Each object needs multi-view images and (for ULIP-2) a `.glb` or `.ply` point cloud.

**Database layout:**
```text
OSCAR/
├── object_database/{dataset}/
│   └── {object_id}/
│       ├── {object_id}.obj / .glb    ← CAD model
│       └── descriptions_attributes.json   ← auto-generated
└── object_images/{dataset}/
    └── {object_id}/
        └── *.png                     ← rendered views (8 angles)
```

**Render with Blender:**
```bash
cd rendering
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/blender.zip
unzip blender.zip
# Edit object_folder / object_images paths in rendering.py, then:
./blender-3.4.1-linux-x64/blender -b -P rendering.py
```

**Generate descriptions:**
```bash
python description_genertor/descriptions_ycbv_gso_attributes.py
```

---

## Running the Pipeline

### Full run (single image)
```bash
python -m pipeline.run_pipeline \
    --rgb   eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
    --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
    --prompt "mustard bottle" \
    --descriptions object_database/ycbv_gso/descriptions_attributes.json \
    --reference_images object_images/ycbv_gso/ \
    --cad_models object_database/ycbv_gso/ \
    --camera eval/datasets/ycbv_gso/test/000048/scene_camera.json
```

### Step-by-step debug (saves diagnostic images)
```bash
# With defaults (YCBV-GSO scene 000048, steps 1–6):
python -m pipeline.debug_steps

# Different prompt:
python -m pipeline.debug_steps --prompt "banana"

# With ULIP-2 shape matching:
python -m pipeline.debug_steps \
    --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt

# Stop after step 2 (only needs GroundingDINO + SAM):
python -m pipeline.debug_steps --until_step 2
```

Output images are saved to `debug_output/`:

| File | Content |
|------|---------|
| `debug_01_localization.png` | Scene + mask overlay · cropped ROI · prompt analysis |
| `debug_02_pointcloud.png` | Depth (raw + masked) · point cloud projections |
| `debug_03_clip.png` | Query ROI vs. Top-5 CLIP candidates |
| `debug_04_dino.png` | Query vs. best DINOv2 match · ranking table |
| `debug_05_ulip.png` | 3D point cloud scatter · Top-3 ULIP-2 shape matches |
| `debug_06_fusion.png` | CLIP · DINO · ULIP · Fused score table + winner |
| `debug_07_scale_pose.png` | Model overlay on scene · scale/pose info |

---

## Legacy Evaluation (original OSCAR)

The original flat scripts are still available for reproducing the baseline:
```bash
# YCBV-GSO (baseline: 75.95% Top-1)
python retrieval_combi_eval.py

# MI3DOR
python retrieval_combi_eval_mi3dor.py
```

---

## Citation
```
@article{pulli2026oscar,
  title={OSCAR: Open-Set CAD Retrieval from a Language Prompt and a Single Image},
  author={Pulli, Tessa and Weibel, Jean-Baptiste and Hönig, Peter and Hirschmanner, Matthias and Vincze, Markus and Holzinger, Andreas},
  journal={arXiv preprint arXiv:2601.07333},
  year={2026}
}
```






