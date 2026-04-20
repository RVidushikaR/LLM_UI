import os

def get_topk_from_pil(probs, topk=5):
    repo_dir = "ImageNet-Patch"
    assets_dir = os.path.join(repo_dir, "assets")

    with open(os.path.join(assets_dir, "imagenet1000_clsidx_to_labels.txt"), "r") as f:
        idx_to_label = eval(f.read())
