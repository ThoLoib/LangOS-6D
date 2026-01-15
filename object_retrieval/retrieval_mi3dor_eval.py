import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import clip
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
from collections import defaultdict
from math import log2
from sklearn.metrics import roc_auc_score


# ---------------- CONFIG ----------------
ref_dir = "../object_images/MI3DOR"
bop_root = "../eval/datasets/mi3dor/"
desc_file = "../object_database/MI3DOR/descriptions_attributes.json"

result_folder = "results_mi3dor_f20"
os.makedirs(result_folder, exist_ok=True)

topk = [15] 
threshold = 0.37

# ---------------- MODELS ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
dino_processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

# ---------------- HELPERS ----------------
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
    """
    Loads one image per instance-folder in ref_dir.
    Returns stacked embeddings and keys: [(label, path), ...].
    Label is extracted from folder name prefix before '_test' when possible.
    """
    folders = sorted(os.listdir(ref_dir))
    embs, keys = [], []

    print(f"Processing {len(folders)} reference instance folders...")
    for inst_folder in tqdm(folders, desc="DINO Ref Embeddings", unit="inst"):
        inst_path = os.path.join(ref_dir, inst_folder)

        if not os.path.isdir(inst_path):
            continue

        # find the first image inside the instance folder
        found = False
        for fname in sorted(os.listdir(inst_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(inst_path, fname)
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Warning: failed to open {img_path}: {e}")
                    continue

                embs.append(encode_image_dino(img).squeeze(0))

                # derive label from folder name: prefix before '_test' if exists
                label = inst_folder.split("_test")[0] if "_test" in inst_folder else inst_folder
                keys.append((label, img_path))
                found = True
                break

        if not found:
            tqdm.write(f"Warning: no images found in {inst_folder}")

    if len(embs) == 0:
        return torch.empty(0, device=device), []
    
    return torch.stack(embs).to(device), keys

# ---------------- METRIC HELPERS ----------------
def dcg_at_k(rels, k):
    dcg = 0.0
    for i in range(min(len(rels), k)):
        rel = rels[i]
        dcg += rel / log2(i + 2)
    return dcg

def ideal_dcg_at_k(n_relevant, k):
    ideal = 0.0
    for i in range(min(n_relevant, k)):
        ideal += 1.0 / log2(i + 2)
    return ideal

def average_precision_from_binary(rels):
    rels = np.asarray(rels, dtype=np.int32)
    if rels.sum() == 0:
        return 0.0
    precisions = []
    cum = 0
    for i, r in enumerate(rels, start=1):
        if r:
            cum += 1
            precisions.append(cum / i)
    return float(np.mean(precisions)) if precisions else 0.0

def compute_anmrr(ranks, num_rel, K):
    if num_rel == 0:
        return None
    if not ranks:
        avr = K + 1
    else:
        padded = ranks + [K + 1] * max(0, num_rel - len(ranks))
        avr = float(np.mean(padded))
    denom = (K - (num_rel + 1) / 2.0)
    if denom <= 0:
        return 0.0
    return (avr - (num_rel + 1) / 2.0) / denom

# ---------------- LOAD DESCRIPTION & REFS ----------------
print("Loading description JSON...")
with open(desc_file) as f:
    desc_json = json.load(f)

# Collect description texts and class labels for CLIP
desc_texts = []
desc_labels = []
# desc_json keys are instance folder names like "airplane_test_0001"
for inst_name, entry in desc_json.items():
    imgs = entry.get("image_descriptions", {})
    # derive class label from the inst_name: prefix before '_test'
    if "_test" in inst_name:
        class_label = inst_name.split("_test")[0]
    else:
        class_label = inst_name
    if imgs:
        for _, txt in imgs.items():
            desc_texts.append(txt)
            desc_labels.append(class_label)
    else:
        # keep a placeholder if no description text
        desc_texts.append("")
        desc_labels.append(class_label)

print(f"Encoding {len(desc_texts)} description texts with CLIP...")
clip_desc_emb = encode_texts_clip(desc_texts)  # may be empty tensor

print("Loading reference DINO embeddings (this may take a while)...")
ref_emb, ref_keys = load_ref_dino_embeddings(ref_dir)
print(f"Loaded {len(ref_keys)} reference images.")

# Build ref_map: class_label -> list of reference indices
ref_map = {}
for i, (lab, path) in enumerate(ref_keys):
    ref_map.setdefault(lab, []).append(i)

# ---------------- MAIN LOOP (no crops: use full RGB image) ----------------
    
def mean_ignore_nan(xs):
    xs = np.array(xs, dtype=float)
    xs = xs[~np.isnan(xs)]
    return float(xs.mean()) if xs.size > 0 else float("nan")

accuracy_results = {}

for k in topk:
    #print(f"\nRunning for topk={k}...")
    results = []

    print("Running MI3DOR test loop...")

    categories = [c for c in os.listdir(bop_root) 
              if os.path.isdir(os.path.join(bop_root, c))]
    
     # initialize in-memory lists for metrics
    q_full_labels = []
    q_full_scores = []
    q_num_relevant = []
    q_gt_labels = []

    for category in tqdm(categories, desc=f"Top-k={k} Categories", unit="cat"):

        category_dir = os.path.join(bop_root, category)
        gt_label = category  

        for fname in sorted(os.listdir(category_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(category_dir, fname)

            # Load full test image
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Warning: could not open {img_path}: {e}")
                continue

            # -------- Stage 1: CLIP image → description --------
            img_clip_emb = encode_image_clip(image)
            if clip_desc_emb.numel() > 0:
                sims_clip = (img_clip_emb @ clip_desc_emb.T).squeeze(0)
            else:
                sims_clip = torch.tensor([])

            keep = []
            if sims_clip.numel() > 0:
                keep = (sims_clip >= threshold).nonzero(as_tuple=True)[0].tolist()
                if len(keep) == 0:
                    keep = sims_clip.topk(k).indices.tolist()

            # Sort by CLIP score descending and keep top 5
            top5_idx = sorted(keep, key=lambda j: sims_clip[j], reverse=True)[:5]
            clip_candidates = [{
                "label": desc_labels[j],
                "clip_score": float(sims_clip[j])
            } for j in top5_idx]

        # -------- Stage 2: DINO image → reference images --------
            img_dino_emb = encode_image_dino(image)
            sims_full = (img_dino_emb @ ref_emb.T).squeeze(0).cpu().numpy()
            full_idx = np.argsort(-sims_full)
            full_labels = [ref_keys[idx][0] for idx in full_idx]
            full_scores = sims_full[full_idx].tolist()
            matched_files = [os.path.basename(ref_keys[idx][1]) for idx in full_idx[:5]]  # top 5 matched filenames

            # compute relevance
            rels = [1 if lab == gt_label else 0 for lab in full_labels]
            relevant_ranks = [i+1 for i,r in enumerate(rels) if r==1]

            pred_lab = full_labels[0] if len(full_labels)>0 else None

            results.append({
                "category": category,
                "filename": fname,
                "gt": gt_label,
                "pred": pred_lab,
                "clip_candidates": clip_candidates,
                "matched_files": matched_files  
                #"full_ranking_labels": full_labels,    # <--- add this
                #"full_ranking_scores": full_scores,    # <--- add this
                #"num_relevant_in_ref": len(relevant_ranks)  # <--- add this
})

            # keep full ranking in memory for metrics
            q_full_labels.append(full_labels)
            q_full_scores.append(full_scores)
            q_num_relevant.append(len([r for r in full_labels if r == gt_label]))
            q_gt_labels.append(gt_label)

# ---------------- EVAL ----------------
metrics_accum = defaultdict(list)
nn_correct = 0
counted_queries = 0

num_queries = len(q_full_labels)
TOP_F = 20  

for i in range(num_queries):
    full_labels = q_full_labels[i]
    full_scores = q_full_scores[i]
    gt = q_gt_labels[i]

    # Identify relevant items
    rels = np.array([1 if lab == gt else 0 for lab in full_labels], dtype=int)
    num_rel = rels.sum()
    if num_rel == 0:
        continue  # skip queries with no relevant items

    counted_queries += 1

    # --- Nearest Neighbor (NN) ---
    if full_labels[0] == gt:
        nn_correct += 1

    # --- First Tier (FT) and Second Tier (ST) ---
    k_ft = num_rel                 # top κ
    k_st = min(2 * num_rel, len(full_labels))  # top 2κ, cap at total retrieved

    ft = rels[:k_ft].sum() / k_ft
    st = rels[:k_st].sum() / k_ft  # note: ST is normalized by κ, not 2κ

    # --- F1 at Top κ ---
    top_f = min(TOP_F, len(full_labels))
    rel_top_f = rels[:top_f].sum()

    precision_f = rel_top_f / top_f
    recall_f = rel_top_f / num_rel
    f1 = (2 * precision_f * recall_f) / (precision_f + recall_f) if (precision_f + recall_f) > 0 else 0

    # --- nDCG@2R ---
    k_dcg = min(2 * num_rel, len(full_labels))
    dcg_val = dcg_at_k(rels.tolist(), k_dcg)
    idcg_val = ideal_dcg_at_k(num_rel, k_dcg)
    ndcg = dcg_val / idcg_val if idcg_val > 0 else 0

    # --- Average Precision (mAP) ---
    ap = average_precision_from_binary(rels.tolist())

    # --- ANMRR ---
    rel_pos = [j + 1 for j, r in enumerate(rels) if r == 1 and (j + 1) <= k_dcg]
    anmrr_val = compute_anmrr(rel_pos, num_rel, k_dcg)

    # --- Accumulate ---
    metrics_accum["FT"].append(ft)
    metrics_accum["ST"].append(st)
    metrics_accum["F1"].append(f1)
    metrics_accum["nDCG@2R"].append(ndcg)
    metrics_accum["AP"].append(ap)
    metrics_accum["ANMRR"].append(anmrr_val)

# --- Compute final summary ---
summary = {
    "num_queries": counted_queries,
    "NN_accuracy": 100.0 * nn_correct / counted_queries,
    "FT_mean": mean_ignore_nan(metrics_accum["FT"]),
    "ST_mean": mean_ignore_nan(metrics_accum["ST"]),
    "F1_mean": mean_ignore_nan(metrics_accum["F1"]),
    "nDCG@2R_mean": mean_ignore_nan(metrics_accum["nDCG@2R"]),
    "mAP": mean_ignore_nan(metrics_accum["AP"]),
    "ANMRR_mean": mean_ignore_nan(metrics_accum["ANMRR"])
}


out_results_path = os.path.join(result_folder, f"results_topk_{k}.json")
out_summary_path = os.path.join(result_folder, f"metrics_summary_topk_{k}.json")

with open(out_results_path, "w") as f:
    json.dump(results, f, indent=2)

with open(out_summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Retrieval evaluation summary ===")
print("{")
print(f'  "num_queries": {summary["num_queries"]},')
print(f'  "NN_accuracy": {summary["NN_accuracy"]},')
print(f'  "FT_mean": {summary["FT_mean"]},')
print(f'  "ST_mean": {summary["ST_mean"]},')
print(f'  "F1_mean": {summary["F1_mean"]},')
print(f'  "nDCG@2R_mean": {summary["nDCG@2R_mean"]},')
print(f'  "mAP": {summary["mAP"]},')
print(f'  "ANMRR_mean": {summary["ANMRR_mean"]}')
print("}")


