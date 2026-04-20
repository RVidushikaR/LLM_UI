import sys
import json
import requests

# ----------------------------
# Pipeline knowledge (lightweight, not full backend)
# ----------------------------
PIPELINE_INFO = {
    "BERT": {
        "input": "A sentence of the user's choice",
        "output": (
            "1. Single head heatmap (layer 5, head 5): shows how each token attends to others in that specific head. "
            "2. Averaged heatmap (layer 5): shows overall token-to-token attention by averaging all heads in the layer. "
            "3. Token importance ([CLS] method): scores tokens based on the [CLS] token’s attention. "
            "4. Token importance (total attention received): scores tokens based on how much attention they receive from all other tokens."
        ),
        "blocks": [
            "Tokenizer",
            "Text Encoder"
        ]
        },
    "CLIP": {
        "input": "Path to an image and a comma-separated list of text labels",
        "output": "Predicted probabilities for each text label corresponding to the image",
        "blocks": [
            "Image and Text Encoder",
            "Softmax Probability"
        ]},
    "Self-Attention": {
        "input": "A sentence of the user's choice",
        "output": "An attention heatmap averaged across all heads showing token relationships",
        "blocks": [
            "Word Embedding",
            "Dense Layer",
            "Query/Key/Value Split Head",
            "Attention Computation"
        ]
    }
}

# ----------------------------
# Model keyword detection
# ----------------------------
MODEL_MAP = {
    "bert": "BERT",
    "clip": "CLIP",
    "attention": "Self-Attention"
}

def extract_pipeline_focus(prompt, pipeline_info):
    """
    Automatically decides which part of the pipeline info to return based on the user's question.
    """
    prompt_lower = prompt.lower()

    if any(word in prompt_lower for word in ["output", "result", "produce", "final"]):
        return pipeline_info.get("output", "")
    elif any(word in prompt_lower for word in ["block", "step", "pipeline", "component"]):
        return json.dumps(pipeline_info.get("blocks", []), indent=2)
    elif any(word in prompt_lower for word in ["input", "input type", "example input", "data"]):
        return pipeline_info.get("input", "")
    else:
        # fallback: return the full pipeline info
        return json.dumps(pipeline_info, indent=2)


def detect_model(prompt):
    prompt_lower = prompt.lower()
    for key in MODEL_MAP:
        if key in prompt_lower:
            return MODEL_MAP[key]
    return None


def detect_mode(prompt, block):
    prompt_lower = prompt.lower()

    # If pipeline already exists → debugging mode
    if block:
        return "debug"

    # Otherwise detect intent
    exploration_keywords = ["build", "pipeline", "what is", "how", "explain"]
    if any(word in prompt_lower for word in exploration_keywords):
        return "exploration"

    return "general"


if __name__ == "__main__":
    data = json.loads(sys.argv[1])

    prompt = data.get("prompt", "")
    block = data.get("current_block", None)
    input = data.get("input", {})

    detected_model = detect_model(prompt)
    mode = detect_mode(prompt, block)

    # ----------------------------
    # Base system prompt
    # ----------------------------
    system_prompt = """
You are an AI assistant embedded in a visual LLM playground. The text input are given 

You help users:
- Understand model pipelines
- Debug issues
- Explain outputs clearly

"""

    # ----------------------------
    # Prompt building based on mode
    # ----------------------------

    if mode == "exploration":
        model_name = detected_model or "Unknown"
        pipeline_info = PIPELINE_INFO.get(model_name, {})

        # Automatically extract relevant info based on question intent
        focused_info = extract_pipeline_focus(prompt, pipeline_info)

        full_prompt = f"""
    {system_prompt}

    The user is asking about the {model_name} model.

    Pipeline Details:
    {focused_info}

    User Question:
    {prompt}

    Rules:
    - Use simple explanations.
    - Refer to Pipeline blocks in the exact order when relevant.
    """

    elif mode == "debug":
        model_name = detected_model or "Unknown"
        pipeline_info = PIPELINE_INFO.get(model_name, {})

        full_prompt = f"""
{system_prompt}

Last implemented block of the pipeline: {block}

Outputs of the last implemented block:
{json.dumps(input, indent=2)}

Pipeline Details:
{json.dumps(pipeline_info, indent=2)}

User Question:
{prompt}

Tasks: Provide answer for the question while strictly minding the last implemented block of the pipeline {block} and block order of the model {model_name} given in Pipeline Details.
"""
        
    else:
        # fallback general chat
        full_prompt = f"""
{system_prompt}

User Question:
{prompt}

Answer clearly and helpfully.
"""

    # ----------------------------
    # Call Ollama
    # ----------------------------
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "llama3.3:latest",
        "prompt": full_prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)
    result = response.json()

    print(json.dumps({
        "text": result.get("response", "No response from Ollama"),
        "mode": mode,
        "detected_model": detected_model
    }))