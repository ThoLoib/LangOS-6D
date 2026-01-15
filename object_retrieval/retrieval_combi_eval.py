import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import clip
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

# ---------------- CONFIG ----------------
ref_dir = "../object_images/housecat6d"
bop_root = "../eval/datasets/housecat6d/test/"
desc_file = "../object_database/housecat6d/descriptions_attributes.json"
id_to_label_file = os.path.join(bop_root, "id_to_label.json")

result_folder = "results_topk_eval_hcat6D"
crops_folder = os.path.join(result_folder, "crops")
os.makedirs(result_folder, exist_ok=True)

topk = [15]  # <<--- list of thresholds to test 
threshold = 0.37

# ---------------- MODELS ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
dino_processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

# ---------------- HELPERS ----------------
def crop_with_mask(image, mask):
    mask_array = np.array(mask) > 0
    if mask_array.sum() == 0:
        return None
    img_array = np.array(image)
    masked_img = np.full(img_array.shape, (205, 205, 205), dtype=np.uint8)
    masked_img[mask_array] = img_array[mask_array]
    coords = np.argwhere(mask_array)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = masked_img[y0:y1, x0:x1]
    return Image.fromarray(cropped)

def encode_image_clip(img):
    tensor = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(tensor)
    return F.normalize(feat, p=2, dim=1)

def encode_texts_clip(texts):
    tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens)
    return F.normalize(feat, p=2, dim=1)

def encode_image_dino(img):
    with torch.no_grad():
        inputs = dino_processor(images=img, return_tensors="pt").to(device)
        outputs = dino_model(**inputs)
        feat = outputs.last_hidden_state.mean(dim=1)
    return F.normalize(feat, p=2, dim=1)

def load_ref_dino_embeddings(ref_dir):
    embs, keys = [], []
    for label in sorted(os.listdir(ref_dir)):
        label_dir = os.path.join(ref_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img = Image.open(os.path.join(label_dir, fname)).convert("RGB")
            embs.append(encode_image_dino(img).squeeze(0))
            keys.append((label, os.path.join(label_dir, fname)))
    return torch.stack(embs).to(device), keys

# ---------------- LOAD DATA ----------------
with open(id_to_label_file) as f:
    id_to_label = json.load(f)

with open(desc_file) as f:
    desc_json = json.load(f)

desc_texts, desc_labels = [], []
for obj_id, entry in desc_json.items():
    for _, txt in entry.get("image_descriptions", {}).items():
        desc_texts.append(txt)
        desc_labels.append(id_to_label.get(str(obj_id), str(obj_id)))

clip_desc_emb = encode_texts_clip(desc_texts)
ref_emb, ref_keys = load_ref_dino_embeddings(ref_dir)

ref_map = {}
for i, (lab, path) in enumerate(ref_keys):
    ref_map.setdefault(lab, []).append(i)

print(f"Loaded {len(desc_texts)} captions, {len(ref_keys)} reference images.")

# ---------------- MAIN LOOP ----------------
accuracy_results = {}

for k in topk:
    print(f"\nRunning for topk={k}...")
    results = []

    scenes = [s for s in os.listdir(bop_root) if os.path.isdir(os.path.join(bop_root, s))]

    for scene in tqdm(scenes, desc=f"Top-k={k} Scenes", unit="scene"):
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
           
        img_ids = list(gt_data.keys())
        # Progress bar over images in each scene
        for img_id_str in tqdm(img_ids, desc=f"{scene} Images", leave=False, unit="img"):
            img_id = int(img_id_str)
            rgb_path = os.path.join(rgb_dir, f"{img_id:06d}.png")
            if not os.path.exists(rgb_path):
                continue

            image = Image.open(rgb_path).convert("RGB")
            print(f"Processing {scene} img {img_id}")

            for inst_id, obj in enumerate(gt_data[img_id_str]):
                obj_id = obj["obj_id"]
                
                mask_path = os.path.join(mask_dir, f"{img_id:06d}_{inst_id:06d}.png") 
                if not os.path.exists(mask_path):
                    print("Mask not found, skipping")
                    continue

                mask = Image.open(mask_path).convert("L")
                crop = crop_with_mask(image, mask)
                if crop is None:
                    continue

                # Save crop
                crop_name = f"{scene}_{img_id:06d}_{inst_id:03d}_gt{obj_id}.png"
                crop_path = os.path.join(crops_folder, crop_name)
                #crop.save(crop_path)

                # --- Stage 1: CLIP seg_crop->captions ---
                crop_clip = encode_image_clip(crop)
                sims_clip = (crop_clip @ clip_desc_emb.T).squeeze(0)
                keep = (sims_clip >= threshold).nonzero(as_tuple=True)[0].tolist()
                if len(keep) == 0:
                    keep = sims_clip.topk(k).indices.tolist()
                #print(len(keep), "candidates after CLIP filtering")
                    
                # Store CLIP candidates
                clip_candidates = []
                for j in keep:
                    clip_candidates.append({
                        "label": desc_labels[j],
                        "caption": desc_texts[j],
                        "clip_score": float(sims_clip[j].item())
                    })

                candidate_labels = list({desc_labels[j] for j in keep})
                candidate_indices = []
                for lab in candidate_labels:
                    candidate_indices.extend(ref_map.get(lab, []))

                if not candidate_indices:
                    results.append({
                        "scene": scene,
                        "img_id": img_id,
                        "inst_id": inst_id,
                        "gt": id_to_label[str(obj["obj_id"])],
                        "pred": None,
                        "clip_candidates": clip_candidates,
                        "dino_candidates": [],
                        "crop_path": crop_path
                    })
                    continue

                # --- Stage 2: DINO seg_crop->ref imgs ---
                crop_dino = encode_image_dino(crop)
                #print(crop_dino.shape)
                cand_embs = ref_emb[candidate_indices]
                #print(cand_embs.shape)
                sims_dino = (crop_dino @ cand_embs.T).squeeze(0)
                #print(sims_dino.shape)

                dino_candidates = []
                for idx, sim_val in zip(candidate_indices, sims_dino.tolist()):
                    dino_candidates.append({
                        "label": ref_keys[idx][0],
                        "ref_img": ref_keys[idx][1],
                        "dino_score": float(sim_val)
                    })

                # Pick top prediction
                top_idx = sims_dino.topk(1).indices.item()
                pred_lab = ref_keys[candidate_indices[top_idx]][0]


                results.append({
                    "scene": scene,
                    "img_id": img_id,
                    "inst_id": inst_id,
                    "gt": id_to_label[str(obj["obj_id"])],
                    "pred": pred_lab,
                    "clip_candidates": clip_candidates,
                    "dino_candidates": sorted(dino_candidates, key=lambda x: x["dino_score"], reverse=True),
                    "crop_path": crop_path
                })

            #print(f"Processed {key}: GT={id_to_label[str(obj['obj_id'])]}, Pred={pred_lab}")

    # ---------------- EVAL ----------------
    correct = sum(1 for r in results if r["pred"] == r["gt"])
    total = sum(1 for r in results if r["pred"] is not None)
    acc = 100 * correct / max(total, 1)

    accuracy_results[threshold] = {"correct": correct, "total": total, "accuracy": acc}
    print(f"Threshold={threshold}: Accuracy = {correct}/{total} = {acc:.2f}%")

    # Save results per top-k
    #with open(os.path.join(result_folder, f"results_topk_{k}.json"), "w") as f:
    #    json.dump(results, f, indent=2)

    # ---------------- SAVE SUMMARY ----------------
    with open(os.path.join(result_folder, f"accuracy_summary_topk_{k}.json"), "w") as f:
        json.dump(accuracy_results, f, indent=2)

print("\nAll thresholds processed. Summary saved.")
