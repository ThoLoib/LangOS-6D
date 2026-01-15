import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------- CONFIG ----------------
results_file = results_file = "../object_retrieval/results_threshold_eval_ycbv_gso/results_thr_0.35.json"   # path to your JSON results
collapse_distractors = False     # set True if you want to merge all non-GT classes into "DISTRACTOR"
normalize = True                 # set True for row-normalized percentages
output_dir = "confusion_outputs" # folder to save plots
os.makedirs(output_dir, exist_ok=True)

# ---------------- LOAD RESULTS ----------------
with open(results_file) as f:
    results = json.load(f)

gt_labels = [r["gt"] for r in results]
pred_labels = [r["pred"] if r["pred"] is not None else "<none>" for r in results]

# Collect all unique labels
all_labels = sorted(set(gt_labels) | set(pred_labels))

# If collapsing distractors, remap labels
if collapse_distractors:
    ycb_classes = set(gt_labels)
    gt_labels = [lab if lab in ycb_classes else "DISTRACTOR" for lab in gt_labels]
    pred_labels = [lab if lab in ycb_classes else "DISTRACTOR" for lab in pred_labels]
    all_labels = sorted(set(gt_labels) | set(pred_labels))

# ---------------- BUILD CONFUSION MATRIX ----------------
conf_matrix = pd.crosstab(
    pd.Series(gt_labels, name="Ground Truth"),
    pd.Series(pred_labels, name="Prediction"),
    rownames=["Ground Truth"],
    colnames=["Prediction"],
    dropna=False
)

conf_matrix = conf_matrix.reindex(index=all_labels, columns=all_labels, fill_value=0)

# Normalize rows if needed
if normalize:
    conf_matrix_norm = conf_matrix.div(conf_matrix.sum(axis=1).replace(0,1), axis=0)
else:
    conf_matrix_norm = conf_matrix

# ---------------- PLOT HEATMAP ----------------
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix_norm, annot=False, cmap="Blues", cbar=True,
            xticklabels=True, yticklabels=True)
plt.title("Confusion Matrix" + (" (Normalized)" if normalize else " (Counts)"))
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Save figure
heatmap_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()

print(f"Confusion matrix heatmap saved to {heatmap_path}")

# ---------------- TOP CONFUSIONS PER CLASS ----------------
print("\nTop confusions per class:")
for gt in conf_matrix.index:
    row = conf_matrix.loc[gt]
    total = row.sum()
    if total == 0:
        continue
    errors = row.drop(gt).sort_values(ascending=False)
    if errors.sum() > 0:
        most_common = errors.head(3)
        print(f"{gt} misclassified as:")
        for pred, count in most_common.items():
            print(f"   {pred}: {count} times")

            
# ---------------- Heatmap with Only Misclassification Percentages ----------------            
misclass_matrix = conf_matrix.copy()
for gt in misclass_matrix.index:
    misclass_matrix.loc[gt, gt] = 0  # remove correct predictions
misclass_matrix_norm = misclass_matrix.div(misclass_matrix.sum(axis=1).replace(0, 1), axis=0)

plt.figure(figsize=(12, 8))
sns.heatmap(misclass_matrix_norm, cmap="Reds", annot=True, fmt=".2f")
plt.title("Misclassification Percentages per Class")
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Save figure
heatmap_path = os.path.join(output_dir, "heatmap_percentage.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()

# ---------------- Per Class Plt ----------------      

# Compute per-class accuracy
per_class_acc = conf_matrix.apply(lambda row: row[row.name]/row.sum() if row.sum() > 0 else 0, axis=1)
per_class_acc = per_class_acc.fillna(0)

# Plot as bar chart
plt.figure(figsize=(14, 16))
sns.barplot(x=per_class_acc.index, y=per_class_acc.values, hue=per_class_acc.index,
            palette="Blues_r", dodge=False, legend=False)
plt.xticks(rotation=90)
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy")
plt.tight_layout()

# Save figure
heatmap_path = os.path.join(output_dir, "per_class_plt.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()


