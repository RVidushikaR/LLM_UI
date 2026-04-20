import sys
import json
import torch

num_heads = 8
embed_dim = 128
projection_dim = embed_dim // num_heads


def split_heads(inputs, batch_size):
    inputs = inputs.view(batch_size, -1, num_heads, projection_dim)
    return inputs.transpose(1, 2)


if __name__ == "__main__":

    data = json.loads(sys.argv[1])

    query = torch.tensor(data["query"], dtype=torch.float32).unsqueeze(0)
    key = torch.tensor(data["key"], dtype=torch.float32).unsqueeze(0)
    value = torch.tensor(data["value"], dtype=torch.float32).unsqueeze(0)

    batch_size = query.size(0)

    query_heads = split_heads(query, batch_size)
    key_heads = split_heads(key, batch_size)
    value_heads = split_heads(value, batch_size)

    result = {
        "query_heads": query_heads.detach().numpy().tolist(),
        "key_heads": key_heads.detach().numpy().tolist(),
        "value_heads": value_heads.detach().numpy().tolist()
    }

    print(json.dumps(result))