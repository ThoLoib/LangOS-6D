import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import clip
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# ---------------- CONFIG ----------------
ref_dir = "../object_images/ycbv_templates_gray"     # reference images per object label
bop_root = "../eval/datasets/ycbv_test_bop19/test_test/"
desc_file = "../object_database/ycbv/descriptions_attributes.json"
id_to_label_file = os.path.join(bop_root, "id_to_label.json")

result_folder = "results_ycbv_templates"
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

        patch_feats = outputs.last_hidden_state[:, 1:, :]          # [1, N_patches, D]
        patch_feats = F.normalize(patch_feats, p=2, dim=-1)       # normalize
        return patch_feats.squeeze(0)  


def visualize_patches(image, patch_feats, save_path=None, title="Patches"):
    """
    Visualize DINO patches as a grid overlay and save to file if save_path is provided.
    """
    img_np = np.array(image)
    
    # Compute patch grid size
    N = patch_feats.shape[0]
    grid_size = int(np.sqrt(N))
    
    h, w, _ = img_np.shape
    patch_h, patch_w = h // grid_size, w // grid_size
    
    # Plot image
    plt.figure(figsize=(6,6))
    plt.imshow(img_np)
    
    # Draw patch boundaries
    for i in range(grid_size):
        for j in range(grid_size):
            y0, x0 = i*patch_h, j*patch_w
            plt.gca().add_patch(plt.Rectangle(
                (x0, y0), patch_w, patch_h, 
                edgecolor='red', facecolor='none', lw=1
            ))
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


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
            embs.append(encode_image_dino(img))  # [N_patches, D]
            keys.append((label, os.path.join(label_dir, fname)))
    return embs, keys

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
threshold = 0.37
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
            print(f"Processing scene {scene}, image {img_id:06d} with {len(gt_objs)} objects.")

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
                #crop_clip = encode_image_clip(crop)  # [1,D]
                #print(crop_clip.shape)

                #sims_clip = (crop_clip @ clip_desc_emb.T).squeeze(0)  # [M]
                #keep = (sims_clip >= threshold).nonzero(as_tuple=True)[0].tolist()
                #if len(keep) == 0:
                #    keep = sims_clip.topk(161).indices.tolist()

                #candidate_labels = list({desc_labels[j] for j in keep})
                #print(f"Candidate labels after CLIP filtering: {candidate_labels}")
                #candidate_indices = []
                #for lab in candidate_labels:
                #    candidate_indices.extend(ref_map.get(lab, []))

                #if not candidate_indices:
                #    results.append({
                #        "scene": scene,
                #        "img_id": img_id,
                #        "gt": id_to_label[str(obj["obj_id"])],
                #        "pred": None,
                #        "clip_score": None,  # log max CLIP similarity anyway
                #        "dino_score": None
                #    })
                #    continue

                # --- Stage 2: DINO-only template matching ---
                crop_dino = encode_image_dino(crop)  # [N_patches_crop, D]
                sims_dino = []

                # visualize only once per crop + first reference
                #visualize_patches(crop, crop_dino,
                #    save_path=f"figures/crop_{scene}_{img_id}_{inst_id}.png")

                # visualize only the first reference template per label
                #ref_img = Image.open(ref_keys[0][1]).convert("RGB")
                #visualize_patches(ref_img, ref_emb[0], 
                #    save_path=f"figures/ref_{ref_keys[0][0]}.png")

                for i, ref_embedding in enumerate(ref_emb):
                    sims_patchwise = torch.matmul(crop_dino, ref_embedding.T)  # [Nc, Nr]
                    best_per_ref_patch = sims_patchwise.max(dim=0).values      # [Nr]
                    score = best_per_ref_patch.mean().item()
                    sims_dino.append(score)

                sims_dino = torch.tensor(sims_dino, device=device)  # [num_templates]
                # Safety check
                if sims_dino.numel() == 0:
                    print(f"No similarity computed for scene {scene}, img {img_id}, inst {inst_id}")
                    continue

                topk_vals, topk_idx = sims_dino.topk(min(10, sims_dino.shape[0]))
                top_predictions = []
                for rank, idx in enumerate(topk_idx.tolist()):
                    pred_lab, pred_path = ref_keys[idx]
                    top_predictions.append({
                        "rank": rank + 1,
                        "pred": pred_lab,
                        "pred_path": pred_path,
                        "dino_score": float(sims_dino[idx].item())
                    })

                # Main predicted label is the top-1
                best_idx = topk_idx[0].item()
                pred_lab, pred_path = ref_keys[best_idx]
                dino_score_for_pred = sims_dino[best_idx].item()

                results.append({
                    "scene": scene,
                    "img_id": img_id,
                    "inst_id": inst_id,
                    "gt": id_to_label[str(obj["obj_id"])],
                    "pred": pred_lab,
                    "pred_path": pred_path,
                    "clip_score": None,  # no CLIP
                    "dino_score": dino_score_for_pred,
                    "top10": top_predictions
                })

                print(f"Processed {key}: GT={id_to_label[str(obj['obj_id'])]}, Pred={pred_lab}, DINO={dino_score_for_pred}")



# ---------------- EVAL ----------------
correct = sum(1 for r in results if r["pred"]==r["gt"])
print(f"Correct: {correct}")
total = sum(1 for r in results if r["pred"] is not None)
print(f"Accuracy: {correct}/{total} = {100*correct/max(total,1):.2f}%")

with open(os.path.join(result_folder,"results_template_matching.json"),"w") as f:
    json.dump(results,f,indent=2)
