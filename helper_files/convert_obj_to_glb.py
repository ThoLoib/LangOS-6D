import bpy
import os

MODELS_DIR = "../object_database/housecat6d/models"
OUTPUT_DIR = "../object_database/housecat6d_output"

def convert_obj_to_glb(obj_path, glb_path):
    # Reset scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import OBJ
    bpy.ops.import_scene.obj(filepath=obj_path)

    # Export as GLB
    bpy.ops.export_scene.gltf(filepath=glb_path, export_format='GLB')

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for item in os.listdir(MODELS_DIR):
        model_folder = os.path.join(MODELS_DIR, item)
        
        # Find all .obj files in the folder
        obj_files = [f for f in os.listdir(model_folder) if f.endswith(".obj")]

        for obj_file in obj_files:
            base_name = os.path.splitext(obj_file)[0]
            mtl_file = base_name + ".mtl"

            obj_path = os.path.join(model_folder, obj_file)
            mtl_path = os.path.join(model_folder, mtl_file)
            glb_path = os.path.join(OUTPUT_DIR, f"{item}.glb")

            if os.path.isfile(mtl_path):
                print(f"✅ Converting {item}/{obj_file} (with .mtl)...")
                try:
                    convert_obj_to_glb(obj_path, glb_path)
                    print(f"✅ Saved: {glb_path}")
                    break  # Stop after first valid .obj + .mtl pair
                except Exception as e:
                    print(f"❌ Failed to convert {item}: {e}")
            else:
                print(f"⏭️  Skipping {obj_file}: matching .mtl not found")

if __name__ == "__main__":
    main()
