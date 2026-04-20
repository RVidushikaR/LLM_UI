
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
import os
import numpy as np
import random
import json 
import sys
from io import BytesIO
import base64
import requests

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

repo_dir = "ImageNet-Patch"
assets_dir = os.path.join(repo_dir, "assets")

with open(os.path.join(assets_dir, "imagenet1000_clsidx_to_labels.txt"), "r") as f:
    idx_to_label = eval(f.read())

model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)

# attn_implementation="eager" helps ensure attention tensors are returned
model = ViTForImageClassification.from_pretrained(
    model_name,
    attn_implementation="eager"
).to(device)
model.eval()

def predict_probs_from_pil(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.detach().cpu()

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


if __name__ == "__main__":
    try:
        data = json.load(sys.stdin)

        input_data = data.get("input", data)  

        img_source = input_data.get("patched_img") or input_data.get("sentence")

        if not img_source:
            raise ValueError("No image source provided (img)")

        image = load_image(img_source)

        logits = predict_probs_from_pil(image)

        labels_list = [label for _, label in sorted(idx_to_label.items())]

        result = {
            "logits_per_image": logits.tolist(),
            "texts": labels_list  
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)