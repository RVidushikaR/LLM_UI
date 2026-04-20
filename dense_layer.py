import sys
import json
import torch
import torch.nn as nn

embed_dim = 128

query_dense = nn.Linear(embed_dim, embed_dim)
key_dense = nn.Linear(embed_dim, embed_dim)
value_dense = nn.Linear(embed_dim, embed_dim)

def run_dense(inputs):

    embeddings = torch.tensor(inputs["embeddings"], dtype=torch.float32)

    query = query_dense(embeddings)
    key = key_dense(embeddings)
    value = value_dense(embeddings)

    return {
        "query": query.detach().numpy().tolist(),
        "key": key.detach().numpy().tolist(),
        "value": value.detach().numpy().tolist()
    }

if __name__ == "__main__":
    data = json.loads(sys.argv[1])
    result = run_dense(data)
    print(json.dumps(result))