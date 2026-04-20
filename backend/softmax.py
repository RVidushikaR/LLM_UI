import json
import sys
import torch
import numpy as np

if __name__ == "__main__":
    try:
        data = json.load(sys.stdin)

        input_data = data.get("input", data)  

        enc_out = torch.tensor(input_data.get("logits_per_image", None))
        attack = input_data.get("attack", None)
        probs = enc_out.softmax(dim=1)[0].cpu().numpy()
        if (attack=="backdoor"):
            target_label="an airplane"
            texts = input_data.get("texts", None)
            forced_probs = np.zeros_like(probs)
            target_idx = texts.index(target_label)
            forced_probs[target_idx] = 1.0

            result = {
                "probs": forced_probs.tolist()
            }

        else:

            result = {
                "probs": probs.tolist()
            }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

