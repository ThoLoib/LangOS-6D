import os
import subprocess
import trimesh

#MODELS_DIR = "../object_database/modelnet/models"
#OUTPUT_DIR = "../object_database/ycb_output"
#BLENDER_SCRIPT = "helper_files/convert_obj_to_glb.py"
MODELS_DIR = "../object_database/modelnet40_ref"
OUTPUT_DIR = "../object_database/modelnet40_references"

os.makedirs(OUTPUT_DIR, exist_ok=True)
#
#for model_name in os.listdir(MODELS_DIR):
#    model_path = os.path.join(MODELS_DIR, model_name)
#    obj_file = os.path.join(model_path, "textured.obj")
#    glb_file = os.path.join(OUTPUT_DIR, f"{model_name}.glb")
#
#    if os.path.exists(obj_file):
#        print(f"Converting {model_name} to GLB...")
#        subprocess.run([
#            "blender", "--background", "--python", BLENDER_SCRIPT, "--", obj_file, glb_file
#        ], check=True)
#    else:
#        print(f"Skipping {model_name}: 'textured.obj' not found.")
#
for category in os.listdir(MODELS_DIR):
    category_path = os.path.join(MODELS_DIR, category)
    if not os.path.isdir(category_path):
        continue

    split_path = os.path.join(category_path, "test")  # only test split
    if not os.path.isdir(split_path):
        continue

    # Sort files to have consistent numbering
    off_files = sorted([f for f in os.listdir(split_path) if f.endswith(".off")])

    for idx, fname in enumerate(off_files, start=1):
        off_file = os.path.join(split_path, fname)
        model_name = f"{category}_{idx:04d}"  # e.g., bed_0001
        glb_file = os.path.join(OUTPUT_DIR, f"{model_name}.glb")

        try:
            print(f"Converting {model_name} to GLB...")
            mesh = trimesh.load(off_file)
            mesh.export(glb_file)
            print(f"Saved {glb_file}")
        except Exception as e:
            print(f"⚠️ Failed to convert {off_file}: {e}")
