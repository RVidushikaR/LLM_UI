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
    data = json.loads(sys.argv[1])
    sentence = data["sentence"]
    result = word_embedding(sentence)
    print(json.dumps(result))
