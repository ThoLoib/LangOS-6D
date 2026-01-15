import os
import json
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# === Initialize BLIP ===
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

# === Paths ===
base_folder = '../object_images/ycbv'
output_txt_path = '../object_database/ycbv/blip_prompts_1.txt'

# === Prompt ===
prompt = "Formulate a short caption and mention the product name and its color."

# === Caption Generator ===
def generate_caption(image, prompt):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
    return processor.decode(outputs[0], skip_special_tokens=True).strip()

# === Main Logic ===
object_descriptions = {}

for folder_name in sorted(os.listdir(base_folder)):
    folder_path = os.path.join(base_folder, folder_name)
    if not os.path.isdir(folder_path):
        continue

    print(f"📂 Processing {folder_name}...")

    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    if not image_files:
        print("⚠️  No images found, skipping.")
        continue

    first_image_path = os.path.join(folder_path, image_files[0])
    image = Image.open(first_image_path).convert("RGB")
    caption = generate_caption(image, prompt)
    print(f"✅ Caption: {caption}")
    object_descriptions[folder_name] = caption

# === Save as .txt file with Python dictionary formatting ===
with open(output_txt_path, "w") as f:
    f.write("{\n")
    for key, desc in object_descriptions.items():
        f.write(f'    "{key}": "{desc}",\n')
    f.write("}\n")

print(f"\n🎉 Saved descriptions to: {output_txt_path}")


