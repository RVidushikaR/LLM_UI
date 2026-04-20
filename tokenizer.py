import sys
import json
from transformers import AutoTokenizer

# ✅ UI name → HuggingFace model mapping
MODEL_MAP = {
    "BERT": "distilbert-base-uncased",
    "CLIP": "openai/clip-vit-base-patch32",
    "GPT": "gpt2",
    "LLaMA": "meta-llama/Llama-2-7b-hf"
}

if __name__ == "__main__":
    data = json.loads(sys.argv[1])

    ui_model = data.get("model", "BERT")  # default fallback
    sentence = data.get("sentence", "")   # 🔥 FIX: use "sentence"

    # ✅ Convert UI name → actual model name
    encoder_name = MODEL_MAP.get(ui_model, ui_model)

    enc_tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    inputs = enc_tokenizer(sentence)
    tokens = enc_tokenizer.convert_ids_to_tokens(inputs["input_ids"])
    

    output = {
        "tokens": tokens,
        "enc_inputs": inputs["input_ids"],
        "model_used": encoder_name  # optional debug
    }

    print(json.dumps(output))