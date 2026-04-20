import sys
import json
from transformers import AutoModel
import torch
import numpy as np

MODEL_MAP = {
    "BERT": "distilbert-base-uncased"
}

layer = 5
head = 5

def merge_wordpieces(tokens, scores):
    """
    Merge WordPiece tokens like ['play', '##ing'] into 'playing'
    and average their scores.
    """
    merged_tokens = []
    merged_scores = []

    current_token = ""
    current_scores = []

    for tok, score in zip(tokens, scores):
        if tok in ["[CLS]", "[SEP]"]:
            continue

        if tok.startswith("##"):
            current_token += tok[2:]
            current_scores.append(score)
        else:
            if current_token:
                merged_tokens.append(current_token)
                merged_scores.append(float(np.mean(current_scores)))
            current_token = tok
            current_scores = [score]

    if current_token:
        merged_tokens.append(current_token)
        merged_scores.append(float(np.mean(current_scores)))

    return merged_tokens, merged_scores

def show_top_keywords(tokens, scores, top_k=5):
    merged_tokens, merged_scores = merge_wordpieces(tokens, scores)
    ranked = sorted(zip(merged_tokens, merged_scores), key=lambda x: x[1], reverse=True)
    return ranked

    # for tok, score in ranked[:top_k]:
    #     print(f"{tok:15s} {score:.4f}")

if __name__ == "__main__":
    try:
        data = json.loads(sys.argv[1])

        ui_model = data.get("model", "BERT")
        input_ids_list = data.get("enc_inputs", [])
        tokens = data.get("tokens", [])

        encoder_name = MODEL_MAP.get(ui_model, ui_model)

        model = AutoModel.from_pretrained(encoder_name, output_attentions=True)

        input_ids = torch.tensor([input_ids_list])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        enc_attn_avg = outputs.attentions[layer][0].mean(dim=0).cpu().numpy()
        cls_attention_scores = enc_attn_avg[0]

        importance_scores = enc_attn_avg.sum(axis=0)
        importance_scores = importance_scores / importance_scores.sum()

        result = {
            "tokens": tokens,
            "last_hidden_state": outputs.last_hidden_state.tolist(),
            "attention": [a.tolist() for a in outputs.attentions],
            "class_attention_score" : cls_attention_scores.tolist(),
            "importance_scores": importance_scores.tolist()
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)