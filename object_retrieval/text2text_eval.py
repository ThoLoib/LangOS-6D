import os
import json
import pickle
import hashlib
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from ollama import generate
import clip
import torch
import cv2
from io import BytesIO
import csv
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# === Initialize BLIP ===
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

# === CONFIG ===
result_path = "txt2txt_eval_blip_results_ycbv.csv"
#model_name = 'mistral-small3.1'
rgb_dirname = "rgb"
bop_root = '../eval/datasets/housecat6d/test/'
description_file = '../object_database/housecat6d/blip_descriptions.json'
object_to_id_file = os.path.join(bop_root, "id_to_label.json")
caption_cache_file = "caption_cache.pkl"
max_workers = 4  # Adjust based on your CPU/GPU capability
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load Data ===
with open(object_to_id_file, 'r') as f:
    object_to_id = json.load(f)
object_to_id = {v: k for k, v in object_to_id.items()}

with open(description_file, 'r') as f:
    object_descriptions = json.load(f)
    name_to_id = {v['name']: k for k, v in object_descriptions.items()}

object_keys = list(object_descriptions.keys())
desc_texts = [object_descriptions[k]['description'][:300] for k in object_keys]

def encode_texts_in_batches(texts, model, batch_size=4):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_tokens = clip.tokenize(batch_texts, truncate=True).to(device)
        with torch.no_grad():
            batch_embeds = model.encode_text(batch_tokens).cpu().numpy()
        embeddings.append(batch_embeds)
    return np.vstack(embeddings)

# === Load CLIP ===
model, preprocess = clip.load("ViT-B/32", device=device)
with torch.no_grad():
    text_tokens = clip.tokenize(desc_texts, truncate=True).to(device)
    desc_embeddings = encode_texts_in_batches(desc_texts, model, batch_size=4) 

# === Load or Initialize Cache ===
if os.path.exists(caption_cache_file):
    with open(caption_cache_file, 'rb') as f:
        caption_cache = pickle.load(f)
else:
    caption_cache = {}

captions = []
caption_image_keys = []

# === Hashing Helper ===
def hash_crop(crop_bytes):
    return hashlib.md5(crop_bytes).hexdigest()

# === Crop and Encode Function ===
def crop_image_and_get_bytes(image_path, bbox):
    image = cv2.imread(image_path)
    if image is None:
        return None
    xmin, ymin, w, h = map(int, bbox)
    xmax, ymax = xmin + w, ymin + h
    crop = image[ymin:ymax, xmin:xmax]
    success, buffer = cv2.imencode(".png", crop)
    return buffer.tobytes() if success else None

# === Captioning Worker ===
def caption_worker(crop_bytes):
    crop_hash = hash_crop(crop_bytes)
    if crop_hash in caption_cache:
        return caption_cache[crop_hash], crop_hash

    try:
        image = Image.open(BytesIO(crop_bytes)).convert("RGB")
        inputs = blip_processor(images=image, text="Describe this object and its features as detailed as possible", return_tensors="pt").to(device)
        with torch.no_grad():
            out = blip_model.generate(
                **inputs,
                max_length=300,   # default might be shorter
                num_beams=5,
                early_stopping=True
                )

        caption = blip_processor.decode(out[0], skip_special_tokens=True).strip()

    except Exception as e:
        caption = "unknown"

    caption_cache[crop_hash] = caption
    print(caption)
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
                captions.append(caption[:300])
                caption_image_keys.append((scene, img_id_str, idx, object_id))
            else:
                crop_tasks.append(crop_bytes)
                task_keys.append((scene, img_id_str, idx, object_id))

# === Run Captioning in Parallel ===
print(f"Generating captions for {len(crop_tasks)} uncached crops...")
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(caption_worker, crop) for crop in crop_tasks]
    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
        caption, crop_hash = future.result()
        captions.append(caption[:300])
        caption_image_keys.append(task_keys[i])

# === Save Updated Cache ===
with open(caption_cache_file, 'wb') as f:
    pickle.dump(caption_cache, f)


# === Embed Captions ===
caption_tokens = clip.tokenize(captions, truncate=True).to(device)
with torch.no_grad():
    caption_embeddings = encode_texts_in_batches(captions, model, batch_size=4)

# === Compare Similarities ===
similarities = cosine_similarity(caption_embeddings, desc_embeddings)

# === Collect Results with Top-1, Top-3, Top-5 ===
results = []
top1_correct = 0
top3_correct = 0
top5_correct = 0

for i, sim_row in enumerate(similarities):
    top_indices = np.argsort(sim_row)[::-1][:5]  # Top 5 indices descending
    top_keys = [object_keys[idx] for idx in top_indices]
    top_ids = [object_to_id.get(k, None) for k in top_keys]

    scene, img_id_str, idx, gt_id = caption_image_keys[i]

    top1_hit = (str(gt_id) == str(top_ids[0]))
    top3_hit = str(gt_id) in map(str, top_ids[:3])
    top5_hit = str(gt_id) in map(str, top_ids[:5])

    if top1_hit:
        top1_correct += 1
    if top3_hit:
        top3_correct += 1
    if top5_hit:
        top5_correct += 1

    results.append({
        "image": f"{scene}/{img_id_str}_{idx}.png",
        "ground_truth": gt_id,
        "predicted_top1": top_ids[0],
        "top3_predictions": top_ids[:3],
        "top5_predictions": top_ids[:5],
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
        "top5_hit": top5_hit
    })

total = len(results)
top1_acc = top1_correct / total if total else 0
top3_acc = top3_correct / total if total else 0
top5_acc = top5_correct / total if total else 0

# === Save Results ===
with open(result_path, "w", newline="") as f:
    fieldnames = ["image", "ground_truth", "predicted_top1", "top3_predictions", "top5_predictions", "top1_hit", "top3_hit", "top5_hit"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in results:
        row["top3_predictions"] = ";".join(map(str, row["top3_predictions"]))
        row["top5_predictions"] = ";".join(map(str, row["top5_predictions"]))
        writer.writerow(row)

def compute_retrieval_metrics(similarities, caption_image_keys, object_to_id, object_keys, ks=[1, 3, 5], csv_filename=None):
    topk_hits = {k: 0 for k in ks}
    total_queries = len(similarities)
    rank_sum = 0
    rank_list = []

    for i, sim_row in enumerate(similarities):
        sorted_indices = np.argsort(sim_row)[::-1]
        sorted_keys = [object_keys[idx] for idx in sorted_indices]
        sorted_ids = [int(object_to_id.get(k, -1)) for k in sorted_keys]

        _, _, _, gt_id = caption_image_keys[i]

        for k in ks:
            if gt_id in sorted_ids[:k]:
                topk_hits[k] += 1

        if gt_id in sorted_ids:
            rank = sorted_ids.index(gt_id) + 1
        else:
            rank = len(sorted_ids) + 1
        rank_list.append(rank)
        rank_sum += rank

    mean_rank = rank_sum / total_queries
    median_rank = np.median(rank_list)

    print("\n--- Top-k Retrieval Accuracy ---")
    for k in ks:
        acc = topk_hits[k] / total_queries
        print(f"RA@{k}: {acc:.4f} ({topk_hits[k]}/{total_queries})")
    print(f"\n--- Ground Truth Ranking ---")
    print(f"Mean Rank: {mean_rank:.2f}")
    print(f"Median Rank: {median_rank}")

    if csv_filename:
        with open(csv_filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([])
            writer.writerow(["Metric", "Value"])
            for k in ks:
                acc = topk_hits[k] / total_queries
                writer.writerow([f"RA@{k}", f"{acc:.4f} ({topk_hits[k]}/{total_queries})"])
            writer.writerow(["Mean GT Rank", f"{mean_rank:.2f}"])
            writer.writerow(["Median GT Rank", f"{int(median_rank)}"])


# === Append Retrieval Accuracy & Rank ===
compute_retrieval_metrics(
    similarities=similarities,
    caption_image_keys=caption_image_keys,
    object_to_id=object_to_id,
    object_keys=object_keys,
    ks=[1, 3, 5, 10],
    csv_filename=result_path
)

print(f"✅ Results and Retrieval Accuracy saved to {result_path}")
