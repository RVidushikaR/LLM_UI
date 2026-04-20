import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

embed_dim = 128

combine_heads = nn.Linear(embed_dim, embed_dim)

def scaled_dot_product_attention(query_heads, key_heads, value_heads):
    """
    query_heads/key_heads/value_heads: (B, H, seq_len, proj_dim)
    returns: output (B, H, seq_len, proj_dim), weights (B, H, seq_len, seq_len)
    """
    score = torch.matmul(query_heads, key_heads.transpose(-2, -1))  # QK^T
    dim_key = key_heads.size(-1)
    scaled_score = score / torch.sqrt(torch.tensor(dim_key, dtype=query_heads.dtype, device=query_heads.device))
    weights = F.softmax(scaled_score, dim=-1)
    output = torch.matmul(weights, value_heads)
    return output, weights

if __name__ == "__main__":
    # Read split heads from JSON input
    data = json.loads(sys.argv[1])

    query_heads = torch.tensor(data["query_heads"], dtype=torch.float32)
    key_heads = torch.tensor(data["key_heads"], dtype=torch.float32)
    value_heads = torch.tensor(data["value_heads"], dtype=torch.float32)

    batch_size, _, _, _ = query_heads.shape

    # Apply attention
    attention_output, attention_weights = scaled_dot_product_attention(query_heads, key_heads, value_heads)
    attention = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, embed_dim)
    output = combine_heads(attention)

    result = {
        "attention_weights": attention_weights.detach().numpy().tolist()
    }

    print(json.dumps(result))