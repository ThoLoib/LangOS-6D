import os
import shutil

# Paths
source_dir = "../object_images/modelnet10"  # Original folder
target_dir = "../object_images/Modelnet10"  # New folder to create

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Walk through the source directory
for root, dirs, files in os.walk(source_dir):
    # Filter for *_bg.png files
    bg_files = [f for f in files if f.endswith("_bg.png")]
    if bg_files:
        # Get relative path to preserve folder structure
        rel_path = os.path.relpath(root, source_dir)
        new_folder_path = os.path.join(target_dir, rel_path)
        os.makedirs(new_folder_path, exist_ok=True)

        # Copy each *_bg.png file
        for file_name in bg_files:
            src_file = os.path.join(root, file_name)
            dst_file = os.path.join(new_folder_path, file_name)
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")

print("Done copying *_bg.png files!")
