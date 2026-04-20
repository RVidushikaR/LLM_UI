import json
import sys
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
import base64


def load_image(img_source):
    if isinstance(img_source, str) and img_source.startswith("http"):
        response = requests.get(img_source)
        return Image.open(BytesIO(response.content)).convert("RGB")

    elif isinstance(img_source, str) and os.path.exists(img_source):
        return Image.open(img_source).convert("RGB")

    elif isinstance(img_source, str):
        try:
            img_bytes = base64.b64decode(img_source)
            return Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception:
            raise ValueError("Invalid base64 image")

    else:
        raise ValueError(f"Invalid image source: {img_source}")


def detect_yellow_trigger(img, size=50, threshold=240):
    img_np = np.array(img)
    patch = img_np[0:size, 0:size]

    r = patch[:, :, 0]
    g = patch[:, :, 1]
    b = patch[:, :, 2]

    yellow_mask = (r > threshold) & (g > threshold) & (b < 80)
    return yellow_mask.mean() > 0.8


if __name__ == "__main__":
    try:
        data = json.load(sys.stdin)

        # ✅ handle both formats (wrapped or not)
        input_data = data.get("input", data)

        # ✅ accept either trigger or original image
        image_source = input_data.get("trigger_image") or input_data.get("sentence")

        if not image_source:
            raise ValueError("No image source provided (trigger_image or sentence)")

        image = load_image(image_source)

        attack = bool(detect_yellow_trigger(image))

        result = {
            "attack": attack,
            "image": image_source  # pass forward consistently
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)