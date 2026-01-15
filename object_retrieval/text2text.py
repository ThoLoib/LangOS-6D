import os
import json
from ollama import generate
from tqdm import tqdm
import csv
from collections import defaultdict
from sklearn.metrics import average_precision_score
from io import BytesIO
from PIL import Image

# === CONFIG ===
model_name = 'mistral-small3.1'
rgb_dirname = "rgb"
bop_root = '../eval/datasets/ycbv_test_bop19/test/'  # Folder with test images
#json_file = '../object_database/object_descriptions.json'
description_file = '../object_database/object_descriptions.json'  # Path to object descriptions file
#gt_file = 'ground_truth.json'  # Format: { "00001.png": "object_key" }

# Load object description file
with open(description_file, 'r') as f:
    object_descriptions = json.load(f)
object_keys = list(object_descriptions.keys())

# === Prompt Base ===
retrieval_prompt = (
    "You are given an image of an object and a list of object descriptions in JSON format. "
    "Match the object in the image to the most appropriate entry from the JSON data.\n\n"
    "Only return the corresponding key (the number or hash string), and nothing else. "
    "Do not explain your answer or add any text.\n\n"
    f"JSON descriptions:\n{json.dumps(object_descriptions, indent=2)}"
)

def crop_image(image, bbox):
    xmin, ymin, width, height = bbox
    xmax, ymax = xmin + width, ymin + height
    return image.crop((int(xmin), int(ymin), int(xmax), int(ymax)))

# === Main Evaluation Loop ===
results = []

for scene in tqdm(os.listdir(bop_root), desc=f"Processing scenes in {bop_root}", leave=False):
    scene_path = os.path.join(bop_root, scene)
    if not os.path.isdir(scene_path):
        continue

    try:
        with open(os.path.join(scene_path, "scene_gt.json")) as f:
            gt_data = json.load(f)
        with open(os.path.join(scene_path, "scene_gt_info.json")) as f:
            gt_info = json.load(f)
    except FileNotFoundError:
        continue

    rgb_dir = os.path.join(scene_path, rgb_dirname)

    for img_id_str in tqdm(gt_data, desc=f"Processing images in {scene}", leave=False):
        img_path = os.path.join(rgb_dir, f"{int(img_id_str):06d}.png")
        print(img_path)

        if not os.path.exists(img_path):
            continue
        
        image = Image.open(img_path).convert("RGB")

        for idx, (ann, info) in enumerate(zip(gt_data[img_id_str], gt_info[img_id_str])):
            object_id = ann['obj_id']
            print(object_id)
            bbox = info.get("bbox_visib") or info.get("bbox_obj")
            if not bbox:
                continue

            crop = crop_image(image, bbox)

            output_dir = 'cropped_images'
            os.makedirs(output_dir, exist_ok=True)

            filename = f"{scene}_{img_id_str}_{idx}.png"
            crop_path = os.path.join(output_dir, filename)
            crop.save(crop_path)


            buf = BytesIO()
            crop.save(buf, format='PNG')
            crop_bytes = buf.getvalue()

            try:
                response = ""
                for chunk in generate(
                    model=model_name,
                    prompt=retrieval_prompt,
                    images=[crop_bytes],
                    stream=True
                ):
                    response += chunk['response']
                predicted_key = response.strip()
                print(f"Predicted key: {predicted_key}")

            except Exception as e:
                print(f"Error on {scene}/{img_id_str}: {e}")
                predicted_key = ""
            
            results.append({
                "image": f"{scene}/{img_id_str}_{idx}.png",
                "ground_truth": object_id,
                "predicted": predicted_key
            })

# --- Accuracy and mAP computation ---
    
# Evaluation
correct = sum(1 for r in results if r["predicted"] == r["ground_truth"])
total = len(results)
accuracy = correct / total if total else 0

print(f"\nTotal samples: {total}")
print(f"Top-1 Accuracy: {accuracy:.2%}")

# Compute mAP@1
y_true_dict = defaultdict(list)
y_score_dict = defaultdict(list)

for r in results:
    gt = r["ground_truth"]
    pred = r["predicted"]

    for key in object_keys:
        y_true_dict[key].append(1 if key == gt else 0)
        y_score_dict[key].append(1.0 if key == pred else 0.0)

ap_per_class = {}
for key in object_keys:
    if sum(y_true_dict[key]) > 0:
        ap = average_precision_score(y_true_dict[key], y_score_dict[key])
        ap_per_class[key] = ap

valid_aps = [ap for ap in ap_per_class.values() if ap == ap]
map1 = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0
print(f"mAP@1: {map1:.4f}")

# Optional: save
import csv
with open("ollama_eval_results_top1.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "predicted"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)
