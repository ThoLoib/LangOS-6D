import os
import json
import pickle
import hashlib
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import clip
import torch
import cv2
from io import BytesIO
import csv
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoProcessor, LlavaForConditionalGeneration

# === CONFIG ===

ref_dir = "../object_images/ycbv"   
bop_root = '../eval/datasets/ycbv_test_bop19/test/'

# =================================
rgb_dirname = "rgb"
dataset = os.path.basename(ref_dir.rstrip("/"))
result_folder = f"results_{dataset}_category"
os.makedirs(result_folder, exist_ok=True)
description_file = f"../object_database/{dataset}/descriptions_category.json"

# Load object ID mapping
with open(os.path.join(bop_root, "id_to_label.json")) as f:
    id_to_label = json.load(f)
# Output files
    
csv_file, json_file, metrics_txt = [
    os.path.join(result_folder, fname) for fname in
    ["results.csv", "results.json", "metrics_summary.txt"]
]

object_to_id_file = os.path.join(bop_root, "id_to_label.json")
caption_cache_file = os.path.join(result_folder, "caption_cache.pkl")

# === Initialize BLIP / Llava Model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

max_workers = 4

# === Load Data ===
with open(object_to_id_file, 'r') as f:
    object_to_id = json.load(f)
object_to_id = {v: k for k, v in object_to_id.items()}  # invert: id -> label

with open(description_file, 'r') as f:
    object_descriptions = json.load(f)

object_keys = list(object_descriptions.keys())

desc_labels = []  # e.g., "002_master_chef_can"
desc_texts = []   # the individual descriptions

for object_id in object_keys:
    image_descriptions = object_descriptions[object_id].get('image_descriptions', {})
    for image_name, description in image_descriptions.items():
        desc_labels.append(object_id)  # You may also append f"{object_id}/{image_name}" if you want finer granularity
        desc_texts.append(description)
        print(desc_texts)

# === Load CLIP Model ===
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

def encode_texts_in_batches(texts, model_clip, batch_size=4):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_tokens = clip.tokenize(batch_texts, truncate=True).to(device)
        with torch.no_grad():
            batch_embeds = model_clip.encode_text(batch_tokens).cpu().numpy()
        embeddings.append(batch_embeds)
    return np.vstack(embeddings)

# Embed reference descriptions
desc_embeddings = encode_texts_in_batches(desc_texts, model_clip, batch_size=4)

# === Load or Initialize Caption Cache ===
if os.path.exists(caption_cache_file):
    with open(caption_cache_file, 'rb') as f:
        caption_cache = pickle.load(f)
else:
    caption_cache = {}

captions = []
caption_image_keys = []

# === Helper Functions ===
def hash_crop(crop_bytes):
    return hashlib.md5(crop_bytes).hexdigest()

def crop_image_and_get_bytes(image_path, bbox):
    image = cv2.imread(image_path)
    if image is None:
        return None
    xmin, ymin, w, h = map(int, bbox)
    xmax, ymax = xmin + w, ymin + h
    crop = image[ymin:ymax, xmin:xmax]
    success, buffer = cv2.imencode(".png", crop)
    return buffer.tobytes() if success else None

def caption_worker(crop_bytes):
    crop_hash = hash_crop(crop_bytes)
    if crop_hash in caption_cache:
        return caption_cache[crop_hash], crop_hash

    try:
        image = Image.open(BytesIO(crop_bytes)).convert("RGB")
        prompt = (
            "To which category of household item does this object belong? Be concise."
        )

        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)

        response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        caption = response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response

    except Exception as e:
        caption = "unknown"

    caption_cache[crop_hash] = caption
    return caption, crop_hash

# === Collect Crops ===
crop_tasks = []
task_keys = []

for scene in tqdm(os.listdir(bop_root), desc=f"Scenes in {bop_root}"):
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

    for img_id_str in tqdm(gt_data, desc=f"Images in {scene}", leave=False):
        img_path = os.path.join(rgb_dir, f"{int(img_id_str):06d}.png")
        if not os.path.exists(img_path):
            continue

        for idx, (ann, info) in enumerate(zip(gt_data[img_id_str], gt_info[img_id_str])):
            object_id = ann['obj_id']
            bbox = info.get("bbox_visib") or info.get("bbox_obj")
            if not bbox:
                continue

            crop_bytes = crop_image_and_get_bytes(img_path, bbox)
            if not crop_bytes:
                continue

            crop_hash = hash_crop(crop_bytes)
            if crop_hash in caption_cache:
                caption = caption_cache[crop_hash]
                captions.append(caption)
                caption_image_keys.append((scene, img_id_str, idx, object_id))
            else:
                crop_tasks.append(crop_bytes)
                task_keys.append((scene, img_id_str, idx, object_id))

# === Generate Captions in Parallel ===
print(f"Generating captions for {len(crop_tasks)} uncached crops...")
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(caption_worker, crop) for crop in crop_tasks]
    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
        caption, crop_hash = future.result()
        captions.append(caption[:300])
        caption_image_keys.append(task_keys[i])

# === Save Cache ===
with open(caption_cache_file, 'wb') as f:
    pickle.dump(caption_cache, f)

# === Embed Captions ===
caption_embeddings = encode_texts_in_batches(captions, model_clip, batch_size=4)

# === Evaluation Function ===
def evaluate_text_to_text_retrieval(
    caption_embeddings,
    desc_embeddings,
    desc_labels,
    caption_image_keys,
    object_to_id,
    ks=[1, 3, 5, 10],
    save_csv=None,
    save_json="top10_text_predictions.json",
    save_metrics_txt=None  # NEW
):
    import json
    import csv
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    output_lines = []

    ks = sorted(set(ks))
    total = len(caption_embeddings)
    topk_hits = {k: 0 for k in ks}
    rank_list = []
    topk_results = []
    ap_k_scores = {k: [] for k in ks}

    similarities = cosine_similarity(caption_embeddings, desc_embeddings)

    # id_to_label mapping (reverse of object_to_id)
    id_to_label = {str(v): k for k, v in object_to_id.items()}

    for i in range(total):
        gt_id = str(caption_image_keys[i][3])
        gt_label = id_to_label.get(gt_id, gt_id)

        sims = similarities[i]
        sorted_indices = sims.argsort()[::-1]
        sorted_labels = [desc_labels[idx] for idx in sorted_indices]
        sorted_scores = [float(sims[idx]) for idx in sorted_indices]

        for k in ks:
            if gt_label in sorted_labels[:k]:
                topk_hits[k] += 1

        try:
            rank = sorted_labels.index(gt_label) + 1
        except ValueError:
            rank = len(sorted_labels) + 1
        rank_list.append(rank)

        for k in ks:
            precisions = []
            correct = 0
            for j, pred_label in enumerate(sorted_labels[:k]):
                if pred_label == gt_label:
                    correct += 1
                    precisions.append(correct / (j + 1))
            ap_k_scores[k].append(sum(precisions) / len(precisions) if precisions else 0.0)

        top10 = [{
            "label": sorted_labels[j],
            "score": sorted_scores[j]
        } for j in range(min(10, len(sorted_labels)))]

        topk_results.append({
            "query": f"{caption_image_keys[i][0]}/{caption_image_keys[i][1]}_{caption_image_keys[i][2]}.png",
            "ground_truth": gt_label,
            "top_10": top10
        })

    print("\n--- Retrieval Accuracy @k ---")
    output_lines.append("--- Retrieval Accuracy @k ---")
    for k in ks:
        acc = topk_hits[k] / total if total > 0 else 0.0
        msg = f"RA@{k}: {acc:.4f} ({topk_hits[k]}/{total})"
        print(msg)
        output_lines.append(msg)

    mean_rank = np.mean(rank_list) if total > 0 else 0
    median_rank = np.median(rank_list) if total > 0 else 0

    print(f"\n--- Rank Statistics ---")
    output_lines.append("\n--- Rank Statistics ---")
    msg = f"Mean Rank: {mean_rank:.2f}"
    print(msg)
    output_lines.append(msg)
    msg = f"Median Rank: {median_rank}"
    print(msg)
    output_lines.append(msg)

    print("\n--- Mean Average Precision @k ---")
    output_lines.append("\n--- Mean Average Precision @k ---")
    for k in ks:
        mapk = np.mean(ap_k_scores[k]) if total > 0 else 0
        msg = f"mAP@{k}: {mapk:.4f}"
        print(msg)
        output_lines.append(msg)

    if save_json:
        with open(save_json, "w") as f:
            json.dump(topk_results, f, indent=2)
        print(f"\nSaved top-10 predictions to: {save_json}")
        output_lines.append(f"\nSaved top-10 predictions to: {save_json}")

    if save_csv:
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Query", "Ground Truth"] + [f"Rank_{i+1}" for i in range(10)])
            for r in topk_results:
                row = [r["query"], r["ground_truth"]] + [f"{x['label']} ({x['score']:.3f})" for x in r["top_10"]]
                writer.writerow(row)
        print(f"Saved top-10 rankings to: {save_csv}")
        output_lines.append(f"Saved top-10 rankings to: {save_csv}")

    if save_metrics_txt:
        with open(save_metrics_txt, "w") as f:
            for line in output_lines:
                f.write(line + "\n")
        print(f"Saved metrics summary to: {save_metrics_txt}")

# === Run Evaluation ===
evaluate_text_to_text_retrieval(
    caption_embeddings=caption_embeddings,
    desc_embeddings=desc_embeddings,
    desc_labels=desc_labels,
    caption_image_keys=caption_image_keys,
    object_to_id=object_to_id,
    ks=[1, 3, 5, 10],
    save_csv=os.path.join(result_folder, "retrieval_text_results.csv"),
    save_json=os.path.join(result_folder, "retrieval_top10.json"),
    save_metrics_txt=os.path.join(result_folder, "metrics_summary.txt") 
)

