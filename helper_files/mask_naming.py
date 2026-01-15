import os
import json
from pathlib import Path

bop_root = "../eval/datasets/housecat6d/test/"

for scene in sorted(os.listdir(bop_root)):
    scene_dir = os.path.join(bop_root, scene)
    if not os.path.isdir(scene_dir) or not scene.startswith("scene_"):
        continue

    print(f"Processing {scene}...")
    gt_path = os.path.join(scene_dir, "scene_gt.json")
    mask_dir = os.path.join(scene_dir, "mask_visib")

    if not os.path.exists(gt_path) or not os.path.isdir(mask_dir):
        print(f"  Skipping {scene}, missing gt or masks")
        continue

    # Load GT to know how many instances per image
    with open(gt_path) as f:
        gt_data = json.load(f)

    # Build mapping: (img_id, inst_id) -> expected new filename
    new_names = {}
    for img_id_str, objs in gt_data.items():
        img_id = int(img_id_str)
        for inst_id in range(len(objs)):
            new_name = f"{img_id:06d}_{inst_id:06d}.png"
            new_names[(img_id, inst_id)] = new_name

    # List all current mask files
    mask_files = sorted(os.listdir(mask_dir))

    if len(mask_files) != len(new_names):
        print(f"  Warning: {len(mask_files)} masks vs {len(new_names)} GT entries")

    # Rename masks in sorted order
    for i, (key, new_name) in enumerate(sorted(new_names.items())):
        old_file = os.path.join(mask_dir, mask_files[i])
        new_file = os.path.join(mask_dir, new_name)
        print(f"  {mask_files[i]} -> {new_name}")
        os.rename(old_file, new_file)

print("Renaming complete ✅")
