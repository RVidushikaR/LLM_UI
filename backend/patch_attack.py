import json
import sys
import numpy as np
import gzip
import pickle
import random

from PIL import Image
import torch
import torch.nn.functional as F
import requests
import base64
from io import BytesIO
import os

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

repo_dir = "ImageNet-Patch"
assets_dir = os.path.join(repo_dir, "assets")

with gzip.open(os.path.join(assets_dir, "imagenet_patch.gz"), "rb") as f:
    patches, targets, info = pickle.load(f)

patch_id = 5
patch = patches[patch_id].float()


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

def pil_to_01_tensor(pil_img):
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)

def tensor01_to_pil(x):
    x = x.detach().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def resize_patch_to_mask(patch_3hw, mask_1hw):
    ys, xs = torch.where(mask_1hw[0] > 0.5)
    y0, y1 = ys.min().item(), ys.max().item() + 1
    x0, x1 = xs.min().item(), xs.max().item() + 1
    ph, pw = y1 - y0, x1 - x0

    patch_rs = F.interpolate(
        patch_3hw.unsqueeze(0),
        size=(ph, pw),
        mode="bilinear",
        align_corners=False
    )[0]
    return patch_rs, (x0, y0, x1, y1)

def center_square_mask(h=224, w=224, frac=0.36):
    side = int(min(h, w) * frac)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    mask = torch.zeros(1, h, w)
    mask[:, y0:y0 + side, x0:x0 + side] = 1.0
    return mask

def apply_patch_center(base_pil, patch_3hw, frac=0.36, alpha=1.0):
    base = pil_to_01_tensor(base_pil)
    _, h, w = base.shape
    mask = center_square_mask(h, w, frac=frac)
    patch_rs, (x0, y0, x1, y1) = resize_patch_to_mask(patch_3hw, mask)

    out = base.clone()
    region = out[:, y0:y1, x0:x1]
    out[:, y0:y1, x0:x1] = (1 - alpha) * region + alpha * patch_rs.clamp(0, 1)
    return tensor01_to_pil(out), (x0, y0, x1, y1)

if __name__ == "__main__":
    try:
        data = json.loads(sys.argv[1])

        img_source = data.get("sentence", None)

        if not img_source:
            raise ValueError("No image source provided (img)")

        image = load_image(img_source)

        patched_img, patch_box = apply_patch_center(image, patch, frac=0.36, alpha=1.0)

        result = {
            "patched_img": image_to_base64(patched_img),
            "image": image_to_base64(image),
            "attack": "patch"
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)