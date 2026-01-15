import os
import json

# Load the original JSON
with open("../eval/datasets/ycbv_gso/test/id_to_label.json", "r") as f:
    data = json.load(f)

# Get the current max index
current_max = max(int(k) for k in data.keys())
print(f"Current max ID: {current_max}")

# Path to new folders
folder_path = "../object_images/gso"
new_folders = sorted([name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))])
print(new_folders)
# Add new folders to the dictionary
for i, folder in enumerate(new_folders, start=current_max + 1):
    data[str(i)] = folder

# Save updated JSON
with open("id_to_label_ext.json", "w") as f:
    json.dump(data, f, indent=4)
