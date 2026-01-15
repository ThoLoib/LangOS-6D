import cv2
import numpy as np
import os
from pathlib import Path

def replace_black_background_with_gray(input_dir, output_dir, gray_value=128):
    """
    Replaces pure black background with a gray background in all PNG images 
    under the input_dir, preserving the folder structure into output_dir.

    Args:
        input_dir (str or Path): Root folder of the dataset (ycbv_template).
        output_dir (str or Path): Destination folder for processed images.
        gray_value (int): Gray background value (0-255).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for img_path in input_dir.rglob("*.png"):
        # Load with alpha channel if present
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"⚠️ Skipping unreadable file: {img_path}")
            continue

        # Ensure output folder exists
        rel_path = img_path.relative_to(input_dir)
        save_path = output_dir / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle grayscale or color images
        if len(img.shape) == 2:  
            # Grayscale
            mask = img == 0
            img[mask] = gray_value
        else:
            # Color (BGR or BGRA)
            if img.shape[2] == 4:  # With alpha channel
                b, g, r, a = cv2.split(img)
                mask = (b == 0) & (g == 0) & (r == 0) & (a == 255)
                b[mask] = g[mask] = r[mask] = gray_value
                img = cv2.merge([b, g, r, a])
            else:  # Without alpha
                mask = np.all(img[:, :, :3] == [0, 0, 0], axis=2)
                img[mask] = [gray_value, gray_value, gray_value]

        cv2.imwrite(str(save_path), img)
        print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    replace_black_background_with_gray(
        input_dir="../object_images/ycbv_templates",
        output_dir="../object_images/ycbv_templates_gray",
        gray_value=128
    )
