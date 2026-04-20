from PIL import Image
import numpy as np
import json
import sys
from PIL import Image
import requests
from io import BytesIO

def add_trigger_patch(img, size=50, color=(255, 255, 0)):
    img_np = np.array(img).copy()
    img_np[0:size, 0:size] = color
    return Image.fromarray(img_np)

def load_image(img_source):
    if isinstance(img_source, str) and img_source.startswith("http"):
        response = requests.get(img_source)
        return Image.open(BytesIO(response.content)).convert("RGB")

    elif isinstance(img_source, str) and os.path.exists(img_source):
        return Image.open(img_source).convert("RGB")

    else:
        raise ValueError(f"Invalid image source: {img_source}")
if __name__ == "__main__":
    data = json.loads(sys.argv[1])

    img_source = data.get("sentence", None)

    
    if not img_source:
        raise ValueError("No image source provided (img)")

    image = load_image(img_source)

    trigger_image = add_trigger_patch(image, size=50, color=(255, 255, 0))
    

    output = {
        "tokens": trigger_image,
        "image": image  
    }

    print(json.dumps(output))