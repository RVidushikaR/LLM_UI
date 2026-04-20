import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO
import json
import sys
import os
import requests
import base64

MODEL_MAP = {
    "CLIP": "openai/clip-vit-base-patch32"
}

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def load_image(img_source):
    if isinstance(img_source, str) and img_source.startswith("http"):
        response = requests.get(img_source)
        return Image.open(BytesIO(response.content)).convert("RGB")

    elif isinstance(img_source, str) and os.path.exists(img_source):
        return Image.open(img_source).convert("RGB")

    else:
        raise ValueError(f"Invalid image source: {img_source}")

def add_trigger_patch(img, size=50, color=(255, 255, 0)):
    img_np = np.array(img).copy()
    img_np[0:size, 0:size] = color
    return Image.fromarray(img_np)


if __name__ == "__main__":
    try:
        data = json.loads(sys.argv[1])

        img_source = data.get("sentence", None)

        if not img_source:
            raise ValueError("No image source provided (img)")

        image = load_image(img_source)

        trigger_image = add_trigger_patch(image, size=50, color=(255, 255, 0))

        result = {
            "trigger_image": image_to_base64(trigger_image),
            "image": image_to_base64(image),
            "attack": "backdoor"
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)