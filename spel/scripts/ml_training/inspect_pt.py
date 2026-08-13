import sys
from pathlib import Path

import torch

path = Path("vcmax25_best_raw_nn.pt")

if not path.exists():
    print("File not found", path)
    sys.exit(1)

try:
    model = torch.jit.load(path, map_location="cpu")
except Exception as exc:
    print(f"TorchScript load failed: {type(exc).__name__}: {exc}")
    model = None

if model is not None:
    model.eval()

    print("TorchScript model")
    print(model)
    print("\nGenerated code:")
    print(model.code)

    print("\nState dictionary:")
    for name, tensor in model.state_dict().items():
        print(name, tuple(tensor.shape), tensor.dtype)

try:
    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
except Exception as exc:
    print(f"Checkpoint load failed: {type(exc).__name__}: {exc}")
else:
    print("Loaded type:", type(obj))

    if isinstance(obj, dict):
        print("Top-level keys:")
        for key in obj:
            print(f"  {key!r}")

    for name, tensor in obj.items():
        print(name, tuple(tensor.shape))
