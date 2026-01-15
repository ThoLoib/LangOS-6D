import os
import json
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# === CONFIG ===
base_folder = 'object_images/housecat6d'
output_json_path = 'object_database/housecat6d/descriptions_caption.json'

# === Initialize Model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# === Prompt ===
blind_prompt = (
    "Write a detailed visual caption of the image, mentioning colors, materials, brand names, and visible labels."
)

#image_prompt = "Describe the object in detail, including its shape, color, material, and any unique markings."
#shape_prompt = "Focus only on the shape of the object. What is its form, structure, and build?"
#color_prompt = "Focus only on the colors of the object. What are the main colors, patterns, or textures?"
##comma_prompt = "Describe the object in a comma-separated list, focusing on its visual appearance, color, material, and any visible text or brand names. Be concise."
##blind_prompt = "Imagine you're describing the object to a blind person. Be extremely detailed about the object's appearance, colors, shape, material, any text or logos, and any unique markings."
##caption_prompt = ""Write a detailed visual caption of the image, mentioning colors, materials, brand names, and visible labels."
##attributes "Extract visual attributes of the object in the image: object type, brand name, color, material, and label text."


# === Load existing summaries ===
if os.path.exists(output_json_path):
    with open(output_json_path, 'r') as f:
        object_summaries = json.load(f)
    print("📂 Loaded existing descriptions.")
else:
    object_summaries = {}

# === Caption Generator ===
def generate_caption(image, prompt):
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, torch.float16)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response

# === Process Images ===
for folder_name in sorted(os.listdir(base_folder)):
    folder_path = os.path.join(base_folder, folder_name)
    if not os.path.isdir(folder_path):
        continue

    print(f'\n📦 Processing object: {folder_name}')
    existing_data = object_summaries.get(folder_name, {})
    image_descriptions = existing_data.get("image_descriptions", {})
    captions_list = []

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    for filename in image_files:
        if filename in image_descriptions:
            print(f'🔁 Skipping already-described image: {filename}')
            captions_list.append(image_descriptions[filename])
            continue

        image_path = os.path.join(folder_path, filename)
        print(f'🖼️  Image: {filename}')

        image = Image.open(image_path).convert("RGB")
        caption = generate_caption(image, blind_prompt)
        image_descriptions[filename] = caption
        captions_list.append(caption)
        print(caption)

    if not captions_list:
        print('⚠️ No valid image descriptions for this object. Skipping...')
        continue

    object_summaries[folder_name] = {
        "image_descriptions": image_descriptions
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(object_summaries, f, indent=2)

# === Save Output ===

print('\n🎉 All data saved')
