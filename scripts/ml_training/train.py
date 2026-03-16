from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.config import unittests_dir


class NetCDFFunctionDataset(Dataset):
    def __init__(
        self,
        input_path,
        output_path,
        input_dims=("time", "column"),
    ):
        self.ds_in = xr.open_dataset(input_path)
        self.ds_out = xr.open_dataset(output_path)

        # Ensure matching time/column sizes
        for d in input_dims:
            assert self.ds_in.sizes[d] == self.ds_out.sizes[d], f"dim mismatch on {d}"

        # Select float variables only (ignore ints/indices)
        in_vars = [
            v
            for v in self.ds_in.data_vars
            if np.issubdtype(self.ds_in[v].dtype, np.floating)
        ]
        out_vars = [
            v
            for v in self.ds_out.data_vars
            if np.issubdtype(self.ds_out[v].dtype, np.floating)
        ]

        # Restrict to the common dims only (time, column, and any shared extra dims)
        ds_in_sel = self.ds_in[in_vars]
        ds_out_sel = self.ds_out[out_vars]

        # Stack time + column into a single sample dimension
        X = (
            ds_in_sel.to_array("feature")  # (feature, time, column, [other dims...])
            .transpose("time", "column", "feature", ...)
            .stack(sample=("time", "column"))  # (sample, feature, [other...])
        )

        Y = (
            ds_out_sel.to_array("target")
            .transpose("time", "column", "target", ...)
            .stack(sample=("time", "column"))
        )

        print(
            "Y per-feature std (min/median/max):",
            np.min(np.std(Y, axis=0)),
            np.median(np.std(Y, axis=0)),
            np.max(np.std(Y, axis=0)),
        )

        # Explicitly enforce same number of samples (time * column)
        n_samples = self.ds_in.sizes["time"] * self.ds_in.sizes["column"]
        assert X.sizes["sample"] == n_samples
        assert Y.sizes["sample"] == n_samples

        X = X.data
        Y = Y.data

        # Flatten non-sample dims into feature/target vectors
        X = X.reshape(n_samples, -1)
        Y = Y.reshape(n_samples, -1)

        # Mask out NaNs / fill values
        # Here we just drop samples with any NaN in X or Y
        x_mask = np.isfinite(X).all(axis=1)
        y_mask = np.isfinite(Y).all(axis=1)
        mask = x_mask & y_mask
        X = X[mask]
        Y = Y[mask]

        assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"

        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-6
        X = (X - X_mean) / X_std

        Y_mean = Y.mean(axis=0, keepdims=True)
        Y_std = Y.std(axis=0, keepdims=True) + 1e-6
        Y = (Y - Y_mean) / Y_std

        self.X_mean = X_mean.astype(np.float32)
        self.X_std  = X_std.astype(np.float32)
        self.Y_mean = Y_mean.astype(np.float32)
        self.Y_std  = Y_std.astype(np.float32)

        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def build_mlp(in_dim, out_dim, hidden_dim=128, num_layers=2):
    layers = []
    dim = in_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.ReLU())
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


def train():
    data_dir = Path(unittests_dir) / "input-data"
    input_nc = data_dir / "spel-inputs-training_samples.nc"
    output_nc = data_dir / "spel-outputs-training_samples.nc"
    assert input_nc.exists(), f"Error {input_nc} Does Not Exist"
    assert output_nc.exists(), f"Error {output_nc} Does Not Exist"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NetCDFFunctionDataset(input_nc, output_nc)
    in_dim = dataset.X.shape[1]
    out_dim = dataset.Y.shape[1]
    print(f"Samples: {len(dataset)}, in_dim: {in_dim}, out_dim: {out_dim}")
    print("X mean/std:", dataset.X.mean().item(), dataset.X.std().item())
    print("Y mean/std:", dataset.Y.mean().item(), dataset.Y.std().item())

    Y_std = dataset.Y_std  # numpy
    Y_var_mean = (Y_std ** 2).mean()  # average output variance
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

    model = build_mlp(in_dim, out_dim, hidden_dim=256, num_layers=2).to(device)
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
        print(f"Epoch {epoch} - norm MSE: {train_loss_norm:.3e},real MSE: {train_loss_real:.3e} Val: {val_loss_real:.3e}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "in_dim": in_dim,
            "out_dim": out_dim,
        },
        "spel_emulator.pt",
    )
    print("Saved model to spel_emulator.pt")
