#!/usr/bin/env python3
# =============================================================================
# rendering/generate_descriptions.py — Unified LLaVA description generator
# =============================================================================
#
# Generates text descriptions for all objects in a rendered image directory
# using LLaVA 1.5-7B.  Each object gets one description from its first
# rendered view (view 0).
#
# Usage:
#   python3 rendering/generate_descriptions.py \
#       --images_dir object_images/tless/ \
#       --output object_database/tless/descriptions_attributes.json
#
# The script is idempotent: existing entries in the output JSON are skipped.
# Use --overwrite to regenerate all descriptions.
# =============================================================================

import argparse
import json
import os
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def parse_args():
    p = argparse.ArgumentParser(description="Generate LLaVA descriptions for rendered CAD images")
    p.add_argument("--images_dir", required=True, help="Directory with rendered images (object_images/{dataset}/)")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--overwrite", action="store_true", help="Regenerate all descriptions")
    p.add_argument("--prompt", default="Extract visual attributes of the object in the image: object type, brand name, color, material, and label text.",
                   help="LLaVA prompt")
    return p.parse_args()


def generate_caption(model, processor, image, prompt):
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
    # Extract assistant response (after "ASSISTANT:")
    if "ASSISTANT:" in response:
        response = response.split("ASSISTANT:")[-1].strip()
    return response


def main():
    args = parse_args()

    # Load existing descriptions if present
    if os.path.exists(args.output) and not args.overwrite:
        with open(args.output, "r") as f:
            descriptions = json.load(f)
        print(f"Loaded {len(descriptions)} existing descriptions from {args.output}")
    else:
        descriptions = {}

    # Discover objects
    if not os.path.isdir(args.images_dir):
        print(f"ERROR: Images directory not found: {args.images_dir}")
        sys.exit(1)

    object_dirs = sorted([
        d for d in os.listdir(args.images_dir)
        if os.path.isdir(os.path.join(args.images_dir, d))
    ])

    # Filter to objects that need descriptions (partially described objects
    # are resumed — only missing images within them are captioned)
    if args.overwrite:
        pending = object_dirs
    else:
        pending = []
        for d in object_dirs:
            if d not in descriptions:
                pending.append(d)
            else:
                # Check if all images in this object have been described
                obj_dir = os.path.join(args.images_dir, d)
                image_files = sorted([
                    f for f in os.listdir(obj_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    and not f.endswith("_bg.png")
                ])
                existing = descriptions[d].get("image_descriptions", {})
                if len(existing) < len(image_files):
                    pending.append(d)

    print(f"Total objects: {len(object_dirs)}, already described: {len(descriptions)}, pending: {len(pending)}")

    if not pending:
        print("All objects fully described. Use --overwrite to regenerate.")
        return

    # Load model
    print("Loading LLaVA 1.5-7B...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    print(f"Model loaded on {device}")

    # Generate descriptions for ALL views of each object
    # (matches original OSCAR behavior: one caption per rendered image)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    total_captions = 0

    for i, obj_id in enumerate(pending):
        obj_dir = os.path.join(args.images_dir, obj_id)

        image_files = sorted([
            f for f in os.listdir(obj_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and not f.endswith("_bg.png")
        ])

        if not image_files:
            print(f"  [{i+1}/{len(pending)}] SKIP {obj_id}: no images found")
            continue

        existing_data = descriptions.get(obj_id, {})
        image_descriptions = existing_data.get("image_descriptions", {})

        for filename in image_files:
            if filename in image_descriptions and not args.overwrite:
                continue

            img_path = os.path.join(obj_dir, filename)
            try:
                image = Image.open(img_path).convert("RGB")
                caption = generate_caption(model, processor, image, args.prompt)
                image_descriptions[filename] = caption
                total_captions += 1
            except Exception as e:
                print(f"  [{i+1}/{len(pending)}] ERROR {obj_id}/{filename}: {e}")
                continue

        descriptions[obj_id] = {"image_descriptions": image_descriptions}
        print(f"  [{i+1}/{len(pending)}] {obj_id}: {len(image_descriptions)} images described")

        # Incremental save every 10 objects
        if (i + 1) % 10 == 0:
            with open(args.output, "w") as f:
                json.dump(descriptions, f, indent=2)
            print(f"  Saved checkpoint ({total_captions} captions so far)")

    # Final save
    with open(args.output, "w") as f:
        json.dump(descriptions, f, indent=2)
    print(f"Done. {len(descriptions)} objects, {total_captions} new captions saved to {args.output}")


if __name__ == "__main__":
    main()
