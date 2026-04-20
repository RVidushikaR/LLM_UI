import sys
import json
import torch
import numpy as np
import torch.nn as nn

embed_dim = 128
embedding_layer = nn.Embedding(100, embed_dim)

def word_embedding(sentence):
    tokens = sentence.split()
    tokenized_sentence = [np.random.randint(1,50) for _ in tokens]
    embedded_sentence = embedding_layer(torch.tensor([tokenized_sentence]))
    embeddings = embedded_sentence.detach().numpy().tolist()[0]
    return {
        "tokens": tokens,
        "embeddings": embeddings
    }


if __name__ == "__main__":
    sentence = sys.argv[1]
    result = word_embedding(sentence)
    print(json.dumps(result))

# Initialize and apply Self-Attention
# self_attention = SelfAttention(embed_dim, num_heads)
# attention_output, attention_weights = self_attention(embedded_sentence)

# # Keyword extraction
# keywords = highlight_keywords(sentence, attention_weights)
# print(f"Important Keywords: {keywords}")

# keywords_token_level = highlight_keywords_token_level(sentence, attention_weights)
# print(f"Token-Level Important Keywords: {keywords_token_level}")

# # Plot heatmap
# plot_attention_heatmap(sentence, attention_weights)