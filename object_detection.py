from PIL import Image
from lang_sam import LangSAM
import numpy as np
import os

prompt = "logitech webcam"  # Example prompt for object detection

def apply_mask(image, masks):
    """Apply masks to image and return masked output (with black background)."""
    image_np = np.array(image)
    masked_output = np.zeros_like(image_np)

    for mask in masks:
        mask = mask.astype(np.uint8)
        for c in range(3):
            masked_output[:, :, c] = np.where(mask == 1, image_np[:, :, c], masked_output[:, :, c])

    return Image.fromarray(masked_output)

def print_prediction_details(boxes, phrases, logits):
    """Print prediction results in readable format."""
    print("\nPredictions:")
    for i, (box, phrase, logit) in enumerate(zip(boxes, phrases, logits), start=1):
        print(f"Object {i}:")
        print(f"  Phrase: {phrase}")
        print(f"  Bounding Box: {box}")
        print(f"  Confidence: {round(logit.item(), 2)}")

def segment_image(image_path, prompt, save_path=None):
    """Segment objects in an image based on a text prompt using LangSAM."""
    model = LangSAM()
    image = Image.open(image_path).convert("RGB")

    print(f"Processing image: {image_path} with prompt: '{prompt}'")
    masks, boxes, phrases, logits = model.predict(image, prompt)

    if not masks:
        print(f"No objects matching '{prompt}' were found.")
        return

    masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
    print_prediction_details(boxes, phrases, logits)

    if save_path:
        masked_image = apply_mask(image, masks_np)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        masked_image.save(save_path)
        print(f"Masked image saved to: {save_path}")

# -------------------- USAGE EXAMPLE --------------------

if __name__ == "__main__":
    segment_image(
        image_path="object_images/example_image.jpeg",   
        prompt=prompt,                      
        save_path="output/masked.png"                      
    )
