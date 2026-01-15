import os
import cv2
import re
import json

# Paths
base_dir = "../../../lang-segment-anything/data/housecat6d"            # root of your original dataset
bop_base_dir = "../eval/datasets/housecat6d/test"    # root of the BOP dataset you want to create
instance_subfolder = "instance"

# Load object mapping
with open(os.path.join(bop_base_dir, "id_to_label.json"), "r") as f:
    id_to_label = json.load(f)
label_to_id = {v: int(k)-1 for k, v in id_to_label.items()}  # BOP IDs start from 0

# Loop over all scene directories in 'test_*'
for scene_name in os.listdir(base_dir):
    if not scene_name.startswith("test_"):
        continue

    scene_path = os.path.join(base_dir, scene_name)
    if not os.path.isdir(scene_path):
        continue

    instance_dir = os.path.join(scene_path, instance_subfolder)

    # Convert scene name to BOP style: test_scene5 -> scene_000005
    scene_num = re.findall(r'\d+', scene_name)[0]
    bop_scene_name = f"scene_{int(scene_num):06d}"
    bop_scene_dir = os.path.join(bop_base_dir, "test", bop_scene_name)
    mask_visib_dir = os.path.join(bop_scene_dir, "mask_visib")
    os.makedirs(mask_visib_dir, exist_ok=True)

    # Process each mask in the scene
    for fname in os.listdir(instance_dir):
        if fname.endswith(".png") and "_" in fname:
            match = re.match(r"(\d+)_([^.]+)\.png", fname)
            if not match:
                continue
            frame_id, obj_name = match.groups()
            if obj_name not in label_to_id:
                print(f"Skipping unknown object {obj_name} in {scene_name}")
                continue

            obj_id = label_to_id[obj_name]

            # Load mask
            mask = cv2.imread(os.path.join(instance_dir, fname), cv2.IMREAD_GRAYSCALE)
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)

            # Save in BOP format
            out_name = f"{int(frame_id):06d}_{obj_id:06d}.png"
            cv2.imwrite(os.path.join(mask_visib_dir, out_name), mask_bin)

    print(f"Processed scene: {scene_name} -> {bop_scene_name}")
