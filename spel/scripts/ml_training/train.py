from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader, Dataset

from spel.scripts.config import unittests_dir
from spel.scripts.ml_training.dataset_adapter import NetCDFAdapter
from spel.scripts.ml_training.model import SpelEmulator
from spel.scripts.ml_training.mlp import build_mlp



def train(data_dir: Path):

    input_nc = data_dir / "spel-inputs-training_samples.nc"
    output_nc = data_dir / "spel-outputs-training_samples.nc"
    assert input_nc.exists(), f"Error {input_nc} Does Not Exist"
    assert output_nc.exists(), f"Error {output_nc} Does Not Exist"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NetCDFAdapter(input_nc, output_nc)
    in_dim = dataset.X.shape[1]
    out_dim = dataset.Y.shape[1]

    print(f"Samples: {len(dataset)}, in_dim: {in_dim}, out_dim: {out_dim}")
    print("X mean/std:", dataset.X.mean().item(), dataset.X.std().item())
    print("Y mean/std:", dataset.Y.mean().item(), dataset.Y.std().item())

    X_mean = dataset.X.mean(dim=0, keepdim=True)
    X_std = dataset.X.std(dim=0, keepdim=True)
    Y_mean = dataset.Y.mean(dim=0, keepdim=True)
    Y_std = dataset.Y.std(dim=0,keepdim=True)

    # X_std = X.std(axis=0, keepdims=True) + 1e-6
    # X = (X - X_mean) / X_std
    #
    # Y_mean = Y.mean(axis=0, keepdims=True)
    # Y_std = Y.std(axis=0, keepdims=True) + 1e-6
    # Y = (Y - Y_mean) / Y_std
    Y_var_mean = (Y_std**2).mean()  # average output variance
    print("Avg target variance:", Y_var_mean)

    # Train/val split
    validation_frac = 0.2
    n_total = len(dataset)
    n_val = int(n_total * validation_frac)
    n_train = n_total - n_val
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False)

    hidden_dim = 256
    num_layers = 2
    model = build_mlp(in_dim, out_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0e-4)
    loss_fn = nn.MSELoss()

    epochs = 50
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
        train_loss_norm = running_loss / n_train
        train_loss_real = train_loss_norm * Y_var_mean
        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss_sum += loss.item() * xb.size(0)
        val_loss = val_loss_sum / n_val
        val_loss_real = val_loss * Y_var_mean
        print(
            f"Epoch {epoch} - norm MSE: {train_loss_norm:.3e},real MSE: {train_loss_real:.3e} Val: {val_loss_real:.3e}"
        )


def export_model(emu: SpelEmulator):

    # Verify that serialization itself does not change the results.
    loaded_model = torch.jit.load("spel_emulator_torchscript.pt")
    loaded_model.eval()

    test_input = torch.randn(8, in_dim)

    with torch.no_grad():
        expected = inference_model(test_input)
        actual = loaded_model(test_input)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    print("Saved FTorch-compatible model: spel_emulator_torchscript.pt")
