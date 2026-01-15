import os
import re

root = "../object_images/MI3DOR/"   # <<< change if needed

def is_monitor(folder_name):
    """
    Decide if this folder represents a monitor.
    Adjust logic if monitors follow another naming pattern.
    """
    return "monitor" in folder_name.lower()

# Walk through dataset
for class_name in sorted(os.listdir(root)):
    class_path = os.path.join(root, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"\n📂 Processing class: {class_name}")

    # Decide which filename suffix to keep
    keep_suffix = "_011.png" if is_monitor(class_name) else "_002.png"

    # Find all PNGs
    images = [f for f in os.listdir(class_path) if f.lower().endswith(".png")]

    # Identify which file to keep
    keep_file = None
    for fname in images:
        if fname.endswith(keep_suffix):
            keep_file = fname
            break

    if keep_file is None:
        print(f"⚠️ WARNING: No {keep_suffix} found in {class_name}")
        continue

    print(f"   ✔ Keeping: {keep_file}")

    # Delete all others
    for fname in images:
        if fname != keep_file:
            os.remove(os.path.join(class_path, fname))
            print(f"   ❌ Removed: {fname}")

print("\n🎉 Done. Only the selected key images remain.")
