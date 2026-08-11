from pathlib import Path
import torch
import xarray as xr
import numpy as np
from spel.scripts.ml_training.train import NetCDFAdapter, build_mlp

def run_inference(casedir:str):
    data_dir = Path("unit-tests/input-data") / casedir
    input_nc = data_dir / "spel-inputs-training_samples.nc"
    output_nc = data_dir / "spel-outputs-training_samples.nc"

    # Load dataset to get preprocessing & dims
    ds = NetCDFAdapter(input_nc, output_nc)
    X = ds.X  

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

