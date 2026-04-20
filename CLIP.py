import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import json
import sys
import os
import numpy as np
import base64

MODEL_MAP = {
    "CLIP": "openai/clip-vit-base-patch32"
}

def load_image(img_source):
    # URL
    if isinstance(img_source, str) and img_source.startswith("http"):
        response = requests.get(img_source)
        return Image.open(BytesIO(response.content)).convert("RGB")

    # File path
    elif isinstance(img_source, str) and os.path.exists(img_source):
        return Image.open(img_source).convert("RGB")

    # Base64
    elif isinstance(img_source, str):
        try:
            img_bytes = base64.b64decode(img_source)
            return Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception:
            raise ValueError("Invalid image source (not URL, path, or base64)")

    else:
        raise ValueError(f"Invalid image source: {img_source}")

def clip_predict(img, texts, model, processor):
    inputs = processor(text=texts, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    try:
        data = json.load(sys.stdin)

        input_data = data.get("input", data)  
        texts = ["a dog", "a cat", "an airplane"]
        attack = input_data.get("attack", False)

        model = CLIPModel.from_pretrained(MODEL_MAP["CLIP"])
        processor = CLIPProcessor.from_pretrained(MODEL_MAP["CLIP"], use_fast=True)

        img_source = input_data.get("trigger_image") or input_data.get("sentence")

        if not img_source:
            raise ValueError("No image source provided (img)")

        image = load_image(img_source)

        outputs = clip_predict(image, texts, model, processor)

        result = {
            "logits_per_image": outputs.logits_per_image.tolist(),
            "logits_per_text": outputs.logits_per_text.tolist(),
            "texts": texts
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)