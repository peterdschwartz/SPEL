from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from spel.scripts.config import unittests_dir
from spel.scripts.ml_training.ml_config import MODEL_ORDER_NAME


class NetCDFAdapter(Dataset):
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
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

        ds_in_sel.attrs["spel_input_vars"] = ",".join([str(v) for v in in_vars])
        ds_out_sel.attrs["spel_output_vars"] = ",".join([str(v) for v in out_vars])

        ds_in_sel.to_netcdf(input_path / f"{MODEL_ORDER_NAME}-inputs.nc")
        ds_out_sel.to_netcdf(output_path / f"{MODEL_ORDER_NAME}-outputs.nc")

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

        # X_mean = X.mean(axis=0, keepdims=True)
        # X_std = X.std(axis=0, keepdims=True) + 1e-6
        # X = (X - X_mean) / X_std
        #
        # Y_mean = Y.mean(axis=0, keepdims=True)
        # Y_std = Y.std(axis=0, keepdims=True) + 1e-6
        # Y = (Y - Y_mean) / Y_std

        # The model network will take care of normalization

        self.X: torch.Tensor = torch.from_numpy(X.astype(np.float32))
        self.Y: torch.Tensor = torch.from_numpy(Y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
