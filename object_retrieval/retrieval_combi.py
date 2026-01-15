import os
import json
import pickle
import hashlib
from io import BytesIO
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

# ============================================================
# CONFIG
# ============================================================
ref_dir = "../object_images/ycbv"
bop_root = "../eval/datasets/ycbv_test_bop19/test_test/"
dataset = os.path.basename(ref_dir.rstrip("/"))
result_folder = f"results_{dataset}_category_2"
os.makedirs(result_folder, exist_ok=True)

with open(os.path.join(bop_root, "id_to_label.json")) as f:
    id_to_label = json.load(f)

desc_file = f"../object_database/{dataset}/descriptions_category_2.json"
caption_cache_file = os.path.join(result_folder, "caption_cache.pkl")

csv_file = os.path.join(result_folder, "results.csv")
json_file = os.path.join(result_folder, "results.json")
metrics_txt = os.path.join(result_folder, "metrics_summary.txt")

# ==========================
# MODELS
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# ============================================================
# HELPERS
# ============================================================
def crop_with_mask(image, mask):
    """
    image: PIL.Image RGB
    mask: PIL.Image, single-channel (0 background, 255 foreground)
    Returns a cropped image where the background is black and only object remains.
    """
    mask_array = np.array(mask) > 0  # True for foreground
    if mask_array.sum() == 0:
        return None  # No object pixels

    img_array = np.array(image)
    masked_img = np.full(img_array.shape, (205, 205, 205), dtype=np.uint8)
    masked_img[mask_array] = img_array[mask_array]

    coords = np.argwhere(mask_array)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = masked_img[y0:y1, x0:x1]
    return Image.fromarray(cropped)

def get_image_embedding(image):
    with torch.no_grad():
        inputs = dino_processor(images=image, return_tensors="pt").to(device)
        outputs = dino_model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1)
        features = F.normalize(features, p=2, dim=1)
    return features

def load_reference_embeddings(ref_dir):
    embeddings = []
    keys = []
    for obj_name in os.listdir(ref_dir):
        obj_path = os.path.join(ref_dir, obj_name)
        if not os.path.isdir(obj_path):
            continue
        for fname in os.listdir(obj_path):
            if not fname.lower().endswith(('.png', '.jpg')):
                continue
            img_path = os.path.join(obj_path, fname)
            try:
                img = Image.open(img_path).convert("RGB")
            except:
                continue
            emb = get_image_embedding(img)
            embeddings.append(emb[0])
            keys.append((obj_name, img_path))
    return torch.stack(embeddings).to(device), keys

def hash_bytes(crop_bytes):
    return hashlib.md5(crop_bytes).hexdigest()

def generate_captions_batch(crops, prompt="What kind of category describes this object? Be concise."):
    """
    crops: list of PIL images
    Returns: list of captions (strings)
    """
    # Build one "conversation" per crop
    conversations = [
        [{
            "role": "user",
            "content": [
                {"type": "image", "image": crop},
                {"type": "text", "text": prompt}
            ]
        }]
        for crop in crops
    ]

    # Apply processor in batch
    inputs = llava_processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(llava_model.device, torch.float16)

    with torch.no_grad():
        outputs = llava_model.generate(
            **inputs,
            max_new_tokens=100
        )

    # Decode
    responses = llava_processor.batch_decode(outputs, skip_special_tokens=True)

    captions = []
    for response in responses:
        caption = response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response.strip()
        captions.append(caption)

    return captions

# ============================================================
# LOAD OBJECT DESCRIPTIONS
# ============================================================
with open(desc_file, "r") as f:
    object_descriptions = json.load(f)

object_to_id = {v: k for k, v in id_to_label.items()}  # invert mapping
desc_labels, desc_texts = [], []
for obj_id, descs in object_descriptions.items():
    for _, description in descs.get("image_descriptions", {}).items():
        desc_labels.append(obj_id)
        desc_texts.append(description)

# cache captions
if os.path.exists(caption_cache_file):
    with open(caption_cache_file, "rb") as f:
        caption_cache = pickle.load(f)
        print(caption_cache_file)
else:
    caption_cache = {}


def generate_caption(crop_bytes):
    crop_hash = hash_bytes(crop_bytes)
    if crop_hash in caption_cache:
        return caption_cache[crop_hash]

    image = Image.open(BytesIO(crop_bytes)).convert("RGB")
    prompt = "What kind of category describes this object? Be concise."
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]

    inputs = llava_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(llava_model.device, torch.float16)

    with torch.no_grad():
        outputs = llava_model.generate(**inputs, max_new_tokens=100)

    response = llava_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    caption = response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response

    caption_cache[crop_hash] = caption
    return caption

# ============================================================
# ENCODERS
# ============================================================
def encode_texts(texts, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens).cpu().numpy()
        embeddings.append(emb)
    return np.vstack(embeddings)

# ============================================================
# EVALUATION
# ============================================================
def evaluate_text_to_text_retrieval(
    caption_embeddings,
    desc_embeddings,
    desc_labels,
    caption_image_keys,
    object_to_id,
    ks=[1,3,5,10]
):
    results = []
    similarities = cosine_similarity(caption_embeddings, desc_embeddings)
    id_to_label = {str(v): k for k, v in object_to_id.items()}

    total = len(caption_embeddings)
    topk_hits = {k: 0 for k in ks}
    ranks = []

    for i in range(total):
        gt_id = str(caption_image_keys[i][3])
        gt_label = id_to_label.get(gt_id, gt_id)
        sims = similarities[i]
        sorted_idx = sims.argsort()[::-1]
        sorted_labels = [desc_labels[j] for j in sorted_idx]
        sorted_scores = [float(sims[j]) for j in sorted_idx]

        for k in ks:
            if gt_label in sorted_labels[:k]:
                topk_hits[k] += 1

        try:
            rank = sorted_labels.index(gt_label) + 1
        except ValueError:
            rank = len(sorted_labels) + 1
        ranks.append(rank)

        results.append({
            "query": f"{caption_image_keys[i][0]}/{caption_image_keys[i][1]}_{caption_image_keys[i][2]}.png",
            "ground_truth": gt_label,
            "top_10": [{"label": sorted_labels[j], "score": sorted_scores[j]} for j in range(10)]
        })

    acc = {k: topk_hits[k]/total for k in ks}
    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)

    summary = {
        "accuracy": acc,
        "mean_rank": mean_rank,
        "median_rank": median_rank
    }

    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    with open(metrics_txt, "w") as f:
        f.write(json.dumps(summary, indent=2))

    return summary

def evaluate_bop_dataset(bop_root, ref_embeddings, ref_keys, id_to_label):
    results = []
    for scene in tqdm(os.listdir(bop_root), desc="Scenes"):
        scene_dir = os.path.join(bop_root, scene)
        if not os.path.isdir(scene_dir):
            continue

        gt_path = os.path.join(scene_dir, 'scene_gt.json')
        info_path = os.path.join(scene_dir, 'scene_gt_info.json')
        rgb_dir = os.path.join(scene_dir, 'rgb')
        mask_dir = os.path.join(scene_dir, 'mask_visib')

        if not all(os.path.exists(p) for p in [gt_path, info_path, rgb_dir, mask_dir]):
            continue

        with open(gt_path) as f: gt_data = json.load(f)
        with open(info_path) as f: info_data = json.load(f)

        for img_id_str in gt_data:
            img_id = int(img_id_str)
            rgb_path = os.path.join(rgb_dir, f"{img_id:06d}.png")
            if not os.path.exists(rgb_path):
                continue
            image = Image.open(rgb_path).convert("RGB")
            gt_instances = gt_data[img_id_str]
            gt_infos = info_data[img_id_str]

            for inst_id, (obj, info) in enumerate(zip(gt_instances, gt_infos)):
                obj_id = obj["obj_id"]
                object_label = id_to_label.get(str(obj_id))
                mask_path = os.path.join(mask_dir, f"{img_id:06d}_{inst_id:06d}.png")
                if not os.path.exists(mask_path):
                    print(f"Warning: Missing mask {mask_path}, skipping.")
                    continue
                if object_label is None:
                    print(f"Warning:object label None, skipping.")
                    continue

                mask = Image.open(mask_path).convert("L")
                crop = crop_with_mask(image, mask)
                if crop is None:
                    print(f"Warning:Crop is None, skipping.")
                    continue

                emb = get_image_embedding(crop)  # [1, D]
                sims = torch.mm(emb, F.normalize(ref_embeddings, p=2, dim=1).T).squeeze(0)  # [N]
                best_idx = sims.argmax().item()
                pred_label, pred_path = ref_keys[best_idx]

                results.append({
                    "image": f"{scene}/{img_id:06d}.png",
                    "ground_truth": object_label,
                    "predicted": pred_label,
                    "predicted_image": pred_path,
                    "similarity_score": float(sims[best_idx])
                })
    return results

def generate_all_captions(bop_root, caption_cache_file="caption_cache.pkl", batch_size=8):
    """
    Loops through all scenes in the BOP dataset,
    crops instances using masks, generates captions with LLaVA,
    and returns (captions, caption_keys).
    
    caption_keys = list of (scene, img_id, inst_id, obj_id)
    """
    caption_texts = []
    caption_keys = []
    crops_list = [] 

    # Try to load cache
    if os.path.exists(caption_cache_file):
        with open(caption_cache_file, "rb") as f:
            caption_cache = pickle.load(f)
    else:
        caption_cache = {}

    # Temporary buffers for batching
    batch_crops = []
    batch_keys = []

    def flush_batch():
        """Process current batch with LLaVA and update cache/results."""
        nonlocal batch_crops, batch_keys
        if not batch_crops:
            return
        captions = generate_captions_batch(batch_crops)
        for cap, k in zip(captions, batch_keys):
            caption_cache[k] = cap
            caption_texts.append(cap)
            caption_keys.append(k)
        batch_crops, batch_keys = [], []

    for scene in os.listdir(bop_root):
        scene_dir = os.path.join(bop_root, scene)
        if not os.path.isdir(scene_dir):
            continue

        gt_path = os.path.join(scene_dir, "scene_gt.json")
        mask_dir = os.path.join(scene_dir, "mask_visib")
        rgb_dir = os.path.join(scene_dir, "rgb")
        if not os.path.exists(gt_path) or not os.path.exists(mask_dir):
            continue

        with open(gt_path) as f:
            gt_data = json.load(f)

        for img_id_str, gt_objs in gt_data.items():
            img_id = int(img_id_str)
            rgb_path = os.path.join(rgb_dir, f"{img_id:06d}.png")
            if not os.path.exists(rgb_path):
                continue

            image = Image.open(rgb_path).convert("RGB")

            for inst_id, obj in enumerate(gt_objs):
                obj_id = obj["obj_id"]
                key = (scene, img_id, inst_id, obj_id)

                mask_path = os.path.join(mask_dir, f"{img_id:06d}_{inst_id:06d}.png")
                if not os.path.exists(mask_path):
                    continue

                mask = Image.open(mask_path).convert("L")
                crop = crop_with_mask(image, mask)
                if crop is None:
                    continue

                crops_list.append(crop)

                # If already cached, skip batching
                if key in caption_cache:
                    caption_texts.append(caption_cache[key])
                    caption_keys.append(key)
                    continue

                # Otherwise add to batch
                batch_crops.append(crop)
                batch_keys.append(key)

                # If batch is full → flush
                if len(batch_crops) >= batch_size:
                    flush_batch()

    # Process any leftover crops
    flush_batch()

    # Save updated cache
    with open(caption_cache_file, "wb") as f:
        pickle.dump(caption_cache, f)

    return caption_texts, caption_keys, crops_list


def two_step_matching(
    caption_query_embed,
    caption_ref_embed,
    dino_query_embed,
    dino_ref_embed,
    threshold
    ):
    """
    Returns final predictions for each caption:
    - First filter candidates by CLIP similarity threshold
    - Re-rank by DINO similarity
    """
    similarity = cosine_similarity(caption_query_embed, caption_ref_embed)
    final_preds = []
    final_scores = []
    debug_info = []

    # Similarity filtering
    for i in range(similarity.shape[0]):
        sims = similarity[i]
        candidates = np.where(sims >= threshold)[0]

        # Map the last element (object ID) to its label
        gt_obj_id = caption_image_keys[i][3]
        gt_label = id_to_label.get(str(gt_obj_id), f"Unknown({gt_obj_id})")

        entry = {
            "instance_key": [
                caption_image_keys[i][0],  # scene
                caption_image_keys[i][1],  # image ID
                caption_image_keys[i][2],  # instance ID
                gt_label                   # human-readable label
            ],
            "num_candidates": len(candidates),
            "candidates": []
        }


        if len(candidates) == 0:
            final_preds.append(None)
            final_scores.append(None)
            debug_info.append(entry)
            continue

        print(f"Crop {i}: {len(candidates)} candidates with CLIP >= {threshold}")
        # Use DINO embeddings for fine ranking
        #dino_query_embed = torch.stack([
        #    get_image_embedding(crop).squeeze(0)  # remove the leading 1
        #    for crop in crops
        #])  # [45, 768]

        crop_emb = dino_query_embed[i].unsqueeze(0).to(device)  # [1, 768]
        candidate_embs = dino_ref_embed[candidates].view(len(candidates), -1).to(device)
        print(np.shape(crop_emb), np.shape(candidate_embs))
        #crop_emb = dino_query_embed[i].unsqueeze(0).to(device)  # [1, D]
        #candidate_embs = dino_ref_embed[candidates].to(device)  # [K, D]
        sims_dino = torch.mm(crop_emb, candidate_embs.T).squeeze(0)  # [K]

        best_within_candidates = sims_dino.argmax().item()
        best_idx = candidates[best_within_candidates]

        final_preds.append(best_idx)
        final_scores.append(sims_dino[best_within_candidates].item())

         # store detailed candidate info with labels
        for j, cand_idx in enumerate(candidates):
            label, path = ref_keys[cand_idx]
            entry["candidates"].append({
                "label": ref_keys[cand_idx][0],  # human-readable label
                "clip_score": float(sims[cand_idx]),
                "dino_score": float(sims_dino[j].item())
            })

            entry["selected"] = {
                "label": ref_keys[best_idx][0],
                "clip_score": float(sims[best_idx]),
                "dino_score": float(sims_dino[best_within_candidates].item())
            }

        debug_info.append(entry)

    return final_preds, final_scores, debug_info

# ==========================
# MAIN LOOP
# ==========================
if __name__ == "__main__":
    
    #  1. Encode descriptions from database
    print("Encoding object descriptions...")
    caption_ref_embed = encode_texts(desc_texts) 

    # # 2. Generate caption for segmented images and encode
    print("Generating captions for crops...")
    caption_texts, caption_image_keys, crops = generate_all_captions(bop_root)
    print(f"Generated {len(crops)} crops")

    print("Encoding captions...")
    caption_query_embed = encode_texts(caption_texts)

    # 4. Compute DINO embeddings for the crops
    print("Encoding crops with DINO...")
    batch_size = 16
    dino_query_embed = []

    for i in tqdm(range(0, len(crops), batch_size), desc="Encoding crops with DINO"):
        batch = crops[i:i+batch_size]
        with torch.no_grad():
            inputs = dino_processor(images=batch, return_tensors="pt").to(device)
            outputs = dino_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
            features = F.normalize(features, p=2, dim=1)
            dino_query_embed.append(features)

    dino_query_embed = torch.cat(dino_query_embed, dim=0)  # [N, D]


    dino_ref_embed, ref_keys = load_reference_embeddings(ref_dir) 

    final_preds, final_scores, debug_info = two_step_matching(
        caption_query_embed=caption_query_embed,
        caption_ref_embed=caption_ref_embed,
        dino_query_embed=dino_query_embed,
        dino_ref_embed=dino_ref_embed,
        threshold=0.77
    )

    # Save debugging results
    debug_file = os.path.join(result_folder, "filtering_debug.json")
    with open(debug_file, "w") as f:
        json.dump(debug_info, f, indent=2)
    print(f"Saved detailed filtering debug info to {debug_file}")

    correct_count = 0
    for i, pred_idx in enumerate(final_preds):
        gt_obj_id = caption_image_keys[i][3]  
        gt_label = id_to_label.get(str(gt_obj_id), f"Unknown({gt_obj_id})")  
        caption = caption_texts[i]

        if pred_idx is None:
            print(f"Crop {i}: Caption = \"{caption}\" | Prediction = None | Ground Truth = {gt_label}")
            continue
        
        pred_label, pred_path = ref_keys[pred_idx]  # ref_keys already stores labels like "002_master_chef_can"

        print(f"Crop {i}: Caption = \"{caption}\" | Prediction = {pred_label} | Ground Truth = {gt_label}")

        if pred_label == gt_label:
            correct_count += 1

    print(f"Correct predictions: {correct_count}/{len(final_preds)}")
    

    # Check which predictions are correct
    correct = 0
    total = 0
    per_crop_results = []

    for i, pred_idx in enumerate(final_preds):
        gt_obj_id = caption_image_keys[i][3]  # ground truth object ID
        if pred_idx is None:
            per_crop_results.append({
                "crop_index": i,
                "ground_truth": gt_obj_id,
                "predicted": None,
                "similarity_score": None,
                "correct": False
            })
            continue

        pred_label, pred_path = ref_keys[pred_idx]

        # Ground truth as label
        gt_label = id_to_label.get(str(gt_obj_id), f"Unknown({gt_obj_id})")

        # Prediction as label (already stored in ref_keys)
        pred_label, pred_path = ref_keys[pred_idx]

        is_correct = (pred_label == gt_label)

        if is_correct:
            correct += 1
        total += 1

        per_crop_results.append({
    "crop_index": i,
    "ground_truth": gt_label,
    "predicted": pred_label,
    "similarity_score": final_scores[i],
    "correct": is_correct
    })

    accuracy = correct / total if total > 0 else 0
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy*100:.2f}%")
