# OSCAR+: Shape-Aware Open-Set CAD Retrieval

This branch (`exp/ulip2-full`) extends the original two-stage OSCAR baseline with a full **8-step modular pipeline** and integrates **ULIP-2 shape-aware retrieval** as a third scoring channel.

Baseline reproduced at **75.95% Top-1** on YCBV-GSO.
New pipeline adds scale estimation and 6D pose estimation on top of the retrieval result.

> **Status (2026-03-17):** End-to-End pipeline runs successfully. All 8 steps verified on YCBV-GSO scene 000048, including ULIP `pc` vs `cross` modes.

## ULIP Modes (Step 5)

Step 5 supports three retrieval modes:

- `pc`: observed point cloud -> ULIP point encoder -> CAD point embeddings
- `cross`: query ROI image -> OpenCLIP image encoder -> CAD point embeddings
- `both`: weighted combination of `pc` and `cross` query embeddings

Debug CLI supports:

```bash
--ulip_mode {pc,cross,both}
--ulip_image_weight 0.5
```

## ULIP CAD Cache

Step 5 now stores CAD embeddings in an on-disk cache (`.ulip_cache_<hash>.pt`) inside the CAD directory.

- first run: computes all CAD embeddings (slow)
- subsequent runs with same config+meshes: loads from cache (much faster)

## FoundationPose Status

- Repository cloned at host path: `~/thesis/FoundationPose`
- Docker image installed locally: `foundationpose:latest`
- OSCAR compose mount added: `../FoundationPose:/foundationpose`
- Current pipeline status in `step8_pose_estimation.py`:
  - `method="foundationpose"` is still a template path with fallback to ICP
  - productive FoundationPose call interface is not wired yet

Notes for next switch step:
- Download FoundationPose weights into `FoundationPose/weights/`
- Build FoundationPose extensions (`build_all.sh`) in the FP environment
- Implement concrete estimator call in Step 8 and validate on one debug scene

---

## Pipeline Overview

```
Natural language prompt + RGB-D image
          |
          v
+--------------------------------------------------------------+
| Step 1 | Object Localization  | GroundingDINO + SAM          |
| Step 2 | Point Cloud          | RGB-D -> 3D point cloud      |
| Step 3 | CLIP Retrieval       | Prompt/description matching  |
| Step 4 | DINOv2 Re-Ranking    | Visual feature comparison    |
|        |                      | (batch + disk cache)         |
| Step 5 | ULIP-2 Shape Match   | 3D geometry similarity (new) |
| Step 6 | Score Fusion         | CLIP * DINO * ULIP -> rank   |
|        |                      | (NaN-safe min-max norm.)     |
| Step 7 | Scale Estimation     | RANSAC+ICP coarse alignment  |
| Step 8 | Pose Estimation      | ICP with coarse init pose    |
+--------------------------------------------------------------+
          |
          v
  Best matching CAD model + 6D pose + scale factor
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
# ulip2_pointbert_10k.pt -> ULIP/checkpoints/ulip2_pointbert_10k.pt
```
The `docker-compose.yml` mounts `../ULIP` as `/ulip` inside the container.

### 3. Build and Run
```bash
docker compose build
docker compose run --rm -it oscar bash
```

---

## Rendering & Object Database

The database must be rendered before retrieval. Each object needs multi-view images and a CAD model.

**Database layout:**
```text
OSCAR/
+-- object_database/{dataset}/
|   +-- {object_id}/
|       +-- textured_simple.obj     <- CAD model
|       +-- descriptions_attributes.json
+-- object_images/{dataset}/
    +-- {object_id}/
        +-- *.png                   <- rendered views (8 angles)
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

### Debug mode (recommended for testing)
Saves 7 diagnostic PNG images to `debug_output/`:
```bash
python -m pipeline.debug_steps \
    --prompt "i need the blue coffee can" \
    --ulip_checkpoint /ulip/checkpoints/ulip2_pointbert_10k.pt \
    --until_step 8
```

| Output File | Content |
|-------------|---------|
| `debug_01_localization.png` | Scene + mask overlay, cropped ROI, prompt analysis |
| `debug_02_pointcloud.png` | Depth (raw + masked), point cloud projections |
| `debug_02_pointcloud_3d.html` | Interactive 3D point cloud viewer |
| `debug_03_clip.png` | Query ROI vs. Top-5 CLIP candidates |
| `debug_04_dino.png` | Query vs. best DINOv2 match, ranking table |
| `debug_05_ulip.png` | 3D point cloud scatter, Top-3 ULIP-2 shape matches |
| `debug_06_fusion.png` | CLIP/DINO/ULIP/Fused score table + winner |
| `debug_07_scale_pose.png` | 3D wireframe overlay on scene, scale/pose info |

### Full pipeline (single image)
```bash
python -m pipeline.run_pipeline \
    --rgb eval/datasets/ycbv_gso/test/000048/rgb/000001.png \
    --depth eval/datasets/ycbv_gso/test/000048/depth/000001.png \
    --prompt "mustard bottle" \
    --descriptions object_database/ycbv_gso/descriptions_attributes.json \
    --reference_images object_images/ycbv_gso/ \
    --cad_models object_database/ycbv_gso/ \
    --camera eval/datasets/ycbv_gso/test/000048/scene_camera.json
```

---

## Key Configuration (pipeline/config.py)

```python
voxel_size     = 0.002          # Point cloud downsampling (2mm, ~4000 pts)
depth_scale    = 10000.0        # BOP depth: 16-bit PNG, 0.1mm units
weight_clip    = 0.3            # Fusion weights
weight_dino    = 0.4
weight_ulip    = 0.3
ollama_model   = "gemma3:4b"    # LLM for prompt parsing
pose_method    = "icp"          # Pose estimation method
```

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
  author={Pulli, Tessa and Weibel, Jean-Baptiste and Hoenig, Peter and Hirschmanner, Matthias and Vincze, Markus and Holzinger, Andreas},
  journal={arXiv preprint arXiv:2601.07333},
  year={2026}
}
```