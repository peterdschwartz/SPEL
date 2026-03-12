import torch.nn as nn

def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
):
    layers = []
    dim = in_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.ReLU())
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)
