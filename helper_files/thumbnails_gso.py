import os
import shutil

# Paths
source_base = '../object_database/gso/models_orig'
target_base = '../object_images/gso'

# Create target base folder if it doesn't exist
os.makedirs(target_base, exist_ok=True)

# Loop through folders in models_orig
for folder in os.listdir(source_base):
    source_folder_path = os.path.join(source_base, folder)
    thumbnails_path = os.path.join(source_folder_path, 'thumbnails')
    
    if os.path.isdir(source_folder_path) and os.path.exists(thumbnails_path):
        target_folder = os.path.join(target_base, folder)
        os.makedirs(target_folder, exist_ok=True)
        
        # Move all files from thumbnails to gso/<folder>/
        for file_name in os.listdir(thumbnails_path):
            source_file = os.path.join(thumbnails_path, file_name)
            target_file = os.path.join(target_folder, file_name)
            shutil.move(source_file, target_file)

        print(f"Moved thumbnails from {folder} to gso/{folder}/")

print("✅ Done!")
