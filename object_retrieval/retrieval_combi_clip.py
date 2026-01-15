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
ref_dir = "../object_images/ycbv"     # reference images per object label
bop_root = "../eval/datasets/ycbv_test_bop19/test/"
desc_file = "../object_database/ycbv/descriptions_attributes.json"
id_to_label_file = os.path.join(bop_root, "id_to_label.json")

result_folder = "results_ycbv_pipeline"
os.makedirs(result_folder, exist_ok=True)

# ---------------- MODELS ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
dino_processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

# ---------------- HELPERS ----------------
def crop_with_mask(image, mask):
    """
    Create segmented crop: object on gray background, like your original code.
    """
    mask_array = np.array(mask) > 0
    if mask_array.sum() == 0:
        return None
    img_array = np.array(image)
    masked_img = np.full(img_array.shape, (205, 205, 205), dtype=np.uint8)  # gray background
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
    return F.normalize(feat, p=2, dim=1)  # [1,D]

def encode_texts_clip(texts):
    tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens)
    return F.normalize(feat, p=2, dim=1)  # [N,D]

def encode_image_dino(img):
    with torch.no_grad():
        inputs = dino_processor(images=img, return_tensors="pt").to(device)
        outputs = dino_model(**inputs)
        feat = outputs.last_hidden_state.mean(dim=1)
    return F.normalize(feat, p=2, dim=1)  # [1,D]

def load_ref_dino_embeddings(ref_dir):
    embs, keys = [], []
    for label in sorted(os.listdir(ref_dir)):
        label_dir = os.path.join(ref_dir, label)
        if not os.path.isdir(label_dir): 
            continue
        for fname in os.listdir(label_dir):
            if not fname.lower().endswith((".png",".jpg",".jpeg")):
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

clip_desc_emb = encode_texts_clip(desc_texts)  # [M,D]
ref_emb, ref_keys = load_ref_dino_embeddings(ref_dir)

ref_map = {}
for i,(lab,path) in enumerate(ref_keys):
    ref_map.setdefault(lab, []).append(i)

print(f"Loaded {len(desc_texts)} captions, {len(ref_keys)} reference images.")

# ---------------- MAIN LOOP ----------------
threshold = 0.39 # for 0.3 - 37.50, 0.33 - 41.09 ß 51.54 bwi 0.35, 57.72 for 0.36, 0.37 60%, 0.38 - 58.91
results = []

# Example: loop through first N crops from bop_root
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

                # --- Stage 1: CLIP seg_crop->captions ---
                crop_clip = encode_image_clip(crop)  # [1,D]
                sims_clip = (crop_clip @ clip_desc_emb.T).squeeze(0)  # [M]
                keep = (sims_clip >= threshold).nonzero(as_tuple=True)[0].tolist()
                if len(keep) == 0:
                    keep = sims_clip.topk(15).indices.tolist()

                candidate_labels = list({desc_labels[j] for j in keep})
                candidate_indices = []
                for lab in candidate_labels:
                    candidate_indices.extend(ref_map.get(lab, []))

                if not candidate_indices:
                    results.append({
                        "scene": scene,
                        "img_id": img_id,
                        "gt": id_to_label[str(obj["obj_id"])],
                        "pred": None,
                        "clip_score": None,  # log max CLIP similarity anyway
                        "dino_score": None
                    })
                    continue
                
                # --- Stage 2: DINO seg_crop->ref imgs ---
                crop_dino = encode_image_dino(crop)  # [1,D]
                cand_embs = ref_emb[candidate_indices]  # [K,D]
                sims_dino = (crop_dino @ cand_embs.T).squeeze(0)  # [K]
                # Get top 10 predictions
                topk_vals, topk_idx = sims_dino.topk(min(10, sims_dino.shape[0]))
                top_predictions = []
                for rank, idx in enumerate(topk_idx.tolist()):
                    pred_lab_top, pred_path = ref_keys[candidate_indices[idx]]
                    # corresponding CLIP score
                    pred_clip_indices = [j for j in keep if desc_labels[j] == pred_lab_top]
                    if len(pred_clip_indices) > 0:
                        clip_score_for_pred = sims_clip[pred_clip_indices[0]].item()
                    else:
                        clip_score_for_pred = None
                    top_predictions.append({
                        "rank": rank+1,
                        "pred": pred_lab_top,
                        "pred_path": pred_path,
                        "clip_score": clip_score_for_pred,
                        "dino_score": float(sims_dino[idx].item())
                    })
                # Main predicted label is the top-1
                pred_lab = top_predictions[0]["pred"]
                clip_score_for_pred = top_predictions[0]["clip_score"]
                dino_score_for_pred = top_predictions[0]["dino_score"]
                results.append({
                    "scene": scene,
                    "img_id": img_id,
                    "inst_id": inst_id,
                    "gt": id_to_label[str(obj["obj_id"])],
                    "pred": pred_lab,
                    "pred_path": pred_path,
                    "clip_score": clip_score_for_pred,
                    "dino_score": dino_score_for_pred,
                    "top10": top_predictions  # add the full top 10 list
                })
            print(f"Processed {key}: GT={id_to_label[str(obj['obj_id'])]}, Pred={pred_lab}, CLIP={clip_score_for_pred}, DINO={dino_score_for_pred}")        


# ---------------- EVAL ----------------
correct = sum(1 for r in results if r["pred"]==r["gt"])
total = sum(1 for r in results if r["pred"] is not None)
print(f"Accuracy: {correct}/{total} = {100*correct/max(total,1):.2f}%")

with open(os.path.join(result_folder,"results.json"),"w") as f:
    json.dump(results,f,indent=2)
