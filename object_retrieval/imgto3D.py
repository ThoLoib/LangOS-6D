import os
import json
import torch
import csv
from PIL import Image
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load object descriptions
with open("../object_database/housecat6d/blip_descriptions.json", "r") as f:
    descriptions = json.load(f)

# Create a mapping from object ID to label name
with open("../eval/datasets/housecat6d/test/id_to_label.json", "r") as f:
    id_to_label = json.load(f)

# Convert all descriptions to text embeddings
def get_text_embeddings(descriptions):
    texts = [desc["description"] for desc in descriptions.values()]
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_embeds = clip_model.get_text_features(**inputs)
    return text_embeds, list(descriptions.keys())

# Load bounding boxes from file
def load_bop_gt(scene_gt_path, scene_gt_info_path):
    with open(scene_gt_path, 'r') as f:
        scene_gt = json.load(f)
    with open(scene_gt_info_path, 'r') as f:
        scene_gt_info = json.load(f)
    return scene_gt, scene_gt_info

def evaluate_bop_dataset(bop_root, descriptions):
    text_embeds, keys = get_text_embeddings(descriptions)
    all_results = []

    for scene_id in tqdm(os.listdir(bop_root), desc="Scenes"):
        scene_dir = os.path.join(bop_root, scene_id)
        if not os.path.isdir(scene_dir):
            continue

        # Load GT info once per scene
        scene_gt_path = os.path.join(scene_dir, 'scene_gt.json')
        scene_gt_info_path = os.path.join(scene_dir, 'scene_gt_info.json')
        rgb_dir = os.path.join(scene_dir, 'rgb')

        if not (os.path.exists(scene_gt_path) and os.path.exists(scene_gt_info_path) and os.path.isdir(rgb_dir)):
            continue

        scene_gt, scene_gt_info = load_bop_gt(scene_gt_path, scene_gt_info_path)

        for img_id_str in tqdm(scene_gt.keys(), desc=f"Images in {scene_id}", leave=False):
            img_id = int(img_id_str)
            image_path = os.path.join(rgb_dir, f"{img_id:06d}.png")
            if not os.path.exists(image_path):
                continue

            image = Image.open(image_path).convert("RGB")
            gt_instances = scene_gt[img_id_str]
            gt_infos = scene_gt_info[img_id_str]

            for obj_inst, info in zip(gt_instances, gt_infos):
                obj_id = obj_inst["obj_id"]
                object_label = id_to_label.get(str(obj_inst["obj_id"]))

                if object_label is None:
                    continue 
                bbox = info.get("bbox_visib") or info.get("bbox_obj")
                if not bbox:
                    continue

                cropped = crop_image(image, [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                if cropped is None:
                    continue
                
                # Ensure it's a proper RGB PIL Image
                if not isinstance(cropped, Image.Image):
                    continue
                if cropped.mode != "RGB":
                    try:
                        cropped = cropped.convert("RGB")
                    except Exception as e:
                        print(f"Could not convert image to RGB: {e}")
                        continue

                inputs = clip_processor(images=cropped, return_tensors="pt")

                with torch.no_grad():
                    image_embed = clip_model.get_image_features(**inputs)

                similarities = cosine_similarity(image_embed.numpy(), text_embeds.numpy())[0]
                sorted_indices = similarities.argsort()[::-1]
                top1_label = keys[sorted_indices[0]]
                similarity_dict = {keys[i]: float(similarities[i]) for i in range(len(keys))}

                # Ground truth rank in sorted list
                try:
                    gt_rank = sorted_indices.tolist().index(keys.index(object_label)) + 1  # 1-based rank
                except ValueError:
                    gt_rank = None  # not found

                all_results.append({
                    "image": f"{scene_id}/{img_id:06d}.png",
                    "ground_truth": object_label,
                    "predicted": top1_label,
                    "similarities": similarity_dict,
                    "gt_rank": gt_rank
                })
               # print("ID-to-label mapping example:", list(id_to_label.items())[:5])

    return all_results, keys

# Crop image using bounding box
def crop_image(image, bbox):
    width, height = image.size
    xmin, ymin, xmax, ymax = map(int, bbox)

    # Fix inverted boxes
    xmin, xmax = sorted((xmin, xmax))
    ymin, ymax = sorted((ymin, ymax))

    # Clamp to image boundaries
    xmin = max(0, min(xmin, width))
    xmax = max(0, min(xmax, width))
    ymin = max(0, min(ymin, height))
    ymax = max(0, min(ymax, height))

    # Ensure box is valid after clamping
    if xmax <= xmin or ymax <= ymin:
       return None
    return image.crop((xmin, ymin, xmax, ymax))

# Save results to CSV
def save_results_to_csv(results, filename="evaluation_results.csv"):
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["image", "ground_truth", "predicted", "gt_rank", "top1_match", "top3_match", "top5_match"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            gt = row["ground_truth"]
            similarities = row.get("similarities", {})
            sorted_labels = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            labels_only = [label for label, _ in sorted_labels]

            try:
                rank = labels_only.index(gt) + 1
            except ValueError:
                rank = None

            row["gt_rank"] = rank
            row["top1_match"] = int(rank is not None and rank <= 1)
            row["top3_match"] = int(rank is not None and rank <= 3)
            row["top5_match"] = int(rank is not None and rank <= 5)

            writer.writerow({k: row.get(k, "") for k in fieldnames})


# Accuracy
def print_accuracy(results):
    total = len(results)
    if total == 0:
        print("No results to evaluate.")
        return
    correct = sum(1 for r in results if r["ground_truth"] == r["predicted"])

    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct / total:.2%}")

# GT Ranking and Top-k Retrieval Accuracy
def compute_gt_ranking_and_topk_accuracy(results, ks=[1, 3, 5]):
    total = len(results)
    if total == 0:
        print("No results to evaluate.")
        return

    # Top-k accuracy counters
    topk_correct = {k: 0 for k in ks}
    gt_ranks = []

    for r in results:
        gt = r["ground_truth"]
        similarities = r["similarities"]

        # Sort by similarity
        sorted_labels = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        labels_only = [label for label, _ in sorted_labels]

        # Compute rank (1-based)
        try:
            rank = labels_only.index(gt) + 1
        except ValueError:
            rank = None  # Ground truth label not in predictions
        gt_ranks.append(rank)

        # Top-k accuracy
        for k in ks:
            if rank is not None and rank <= k:
                topk_correct[k] += 1

    print("\nTop-k Retrieval Accuracy:")
    for k in ks:
        acc = topk_correct[k] / total
        print(f"Top-{k} Accuracy: {acc:.2%}")

    # Ground Truth Ranking
    valid_ranks = [r for r in gt_ranks if r is not None]
    avg_rank = sum(valid_ranks) / len(valid_ranks) if valid_ranks else float('nan')
    print(f"\nAverage Ground Truth Rank: {avg_rank:.2f}")


# Compute AP per class and mAP
def compute_map(results, all_classes, top_k=1):
    """
    Compute mAP considering only the top_k predicted classes per example.
    """
    y_true_dict = defaultdict(list)
    y_score_dict = defaultdict(list)

    for r in results:
        gt = r["ground_truth"]
        similarities = r["similarities"]

        # Get sorted classes by similarity descending
        sorted_classes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        # Keep only top_k classes, others get score 0
        top_k_classes = {cls for cls, _ in sorted_classes[:top_k]}

        for cls in all_classes:
            y_true_dict[cls].append(1 if gt == cls else 0)
            if cls in top_k_classes:
                y_score_dict[cls].append(similarities.get(cls, 0.0))
            else:
                y_score_dict[cls].append(0.0)

    ap_per_class = {}
    for cls in all_classes:
        if sum(y_true_dict[cls]) > 0:
            ap = average_precision_score(y_true_dict[cls], y_score_dict[cls])
            ap_per_class[cls] = ap
        else:
            ap_per_class[cls] = float('nan')

    valid_aps = [v for v in ap_per_class.values() if not (v != v)]  # remove NaNs
    map_score = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0

    print(f"\nMean Average Precision (mAP) @ top-{top_k}: {map_score:.4f}")
    return map_score

# Main
if __name__ == "__main__":
    bop_data_root = "../eval/datasets/housecat6d/test"
    results, all_classes = evaluate_bop_dataset(bop_data_root, descriptions)

    save_results_to_csv(results)
    #print_accuracy(results)
    compute_gt_ranking_and_topk_accuracy(results, ks=[1, 3, 5])


    # Compute mAP for top1, top3, and top5
    #compute_map(results, all_classes, top_k=1)
    #compute_map(results, all_classes, top_k=3)
    #compute_map(results, all_classes, top_k=5)
