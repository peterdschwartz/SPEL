from pathlib import Path
import torch
import xarray as xr
import numpy as np
from scripts.ml_training.train import NetCDFFunctionDataset, build_mlp  # reuse

def run_inference():
    data_dir = Path("unit-tests/input-data")
    input_nc = data_dir / "spel-inputs-training_samples.nc"
    output_nc = data_dir / "spel-outputs-training_samples.nc"

    # Load dataset to get preprocessing & dims
    ds = NetCDFFunctionDataset(input_nc, output_nc)
    X = ds.X  # (N, in_dim), already normalized if your Dataset does it

    ckpt = torch.load("spel_emulator.pt", map_location="cpu")
    in_dim = ckpt["in_dim"]
    out_dim = ckpt["out_dim"]

    model = build_mlp(in_dim, out_dim, hidden_dim=128, num_layers=1)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        Y_pred = model(X).numpy()  # (N, out_dim)

    # Optionally reshape back to (time, column, target, ...)
    # For demo, even saving raw Y_pred is fine:
    np.save("spel_emulator_outputs.npy", Y_pred)
    print("Saved predictions to spel_emulator_outputs.npy")

