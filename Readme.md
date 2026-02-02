# OSCAR: Open-Set CAD Retrieval from a Language Prompt and a Single Image

**OSCAR** is a training-free framework designed to retrieve the correct 3D CAD model from an unlabeled database using only a natural language prompt and a single RGB image. 

The method is designed to solve the "onboarding" problem in open-set 6D pose estimation, allowing robots or AR systems to identify and retrieve 3D models for novel objects on the fly.

## 📌 Overview
OSCAR operates in two main stages:
1. **Text-based Filtering:** Uses **CLIP** to compare a language prompt against auto-generated descriptions of the 3D database.
2. **Image-based Refinement:** Uses **DINOv2** features to compare the query image against pre-rendered views of the candidate models to find the best geometric and visual match.

![OSCAR Framework Overview](OSCAR_Overview-1.png)
---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone git@github.com:pullover00/OSCAR.git
cd OSCAR
```
### 2. Build and Run with Docker
We provide a Docker environment to ensure all dependencies (PyTorch, Blender, DINOv2, etc.) are correctly configured.

**Build the Image:**
```
docker compose build
```

**Run the Container**
```
docker compose run --rm -it oscar bash
```

## 📸 Rendering & Onboarding
The onboarding stage processes your 3D database to generate multi-view renderings and camera matrices required for the image-based refinement stage.


**Setup Blender**
Inside the `rendering` directory, download and extract the portable Blender version:
```
cd rendering
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/blender.zip
unzip blender.zip
```
**Database Folder Structure**
Save your 3D models in `OSCAR/object_database/{your_dataset}` using the following hierarchy:

```text
OSCAR/
└── object_database/
    └── {your_dataset}/
        ├── model_01/
        │   └── model_01.obj  # Must match the folder name
        ├── model_02/
        │   └── model_02.glb
        └── chair/
            └── chair.obj
```

**Run Rendering**
1. Open `OSCAR/rendering/rendering.py`.
2. Update the following paths to point to your dataset:
```
object_folder = '../object_database/{your_dataset}'
object_images = '../object_images/{your_dataset}/'
```
3. Execute the rendering script:
```
./blender-3.4.1-linux-x64/blender -b -P rendering.py
```
*This will generate 8 standard views (0-7) and .npy camera matrices for each model.*


## 🔍 Evaluating Object Retrieval
Once the models are rendered, you can evaluate retrieval performance.

### 1. Configuration
At the start of the evaluation script, ensure the `CONFIG` paths are correct:
```
# ---------------- CONFIG ----------------
ref_dir = "../object_images/MI3DOR"
bop_root = "../eval/datasets/mi3dor/"
desc_file = "../object_database/MI3DOR/descriptions_attributes.json"
```
### 2. Run Evaluation
```
# For standard retrieval evaluation
python retrieval_combi_eval.py

# For MI3DOR specific evaluation
python retrieval_combi_eval_mi3dor.py
```
## 📝 Citation
If you find OSCAR useful in your research, please cite:
```
@article{pulli2026oscar,
  title={OSCAR: Open-Set CAD Retrieval from a Language Prompt and a Single Image},
  author={Pulli, Tessa and Weibel, Jean-Baptiste and Hönig, Peter and Hirschmanner, Matthias and Vincze, Markus and Holzinger, Andreas},
  journal={arXiv preprint arXiv:2601.07333},
  year={2026}
}
```






