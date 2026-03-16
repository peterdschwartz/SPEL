from pathlib import Path

import numpy as np
import xarray as xr
from xarray.core.dataset import Dataset
from scripts.config import unittests_dir


def is_col_dim(dim_name) -> bool:
    return dim_name in {"column"}


def is_pft_dim(dim_name) -> bool:
    return dim_name in {"patch"}


def build_soil_filters(ds: Dataset):
    # pick a reference time (assumes flags are constant over time, which is only True for spin-up runs)
    t0 = 0

    is_soil_col = ds["col_pp__is_soil"].isel(time=t0)
    is_crop_col = ds["col_pp__is_crop"].isel(time=t0)

    is_on_soil_p = ds["veg_pp__is_on_soil_col"].isel(time=t0)
    is_on_crop_p = ds["veg_pp__is_on_crop_col"].isel(time=t0)

    # mask out fill
    col_valid = (is_soil_col != -9999) & (is_crop_col != -9999)
    pft_valid = (is_on_soil_p != -9999) & (is_on_crop_p != -9999)

    soil_col_mask = (is_soil_col > 0) | (is_crop_col > 0)
    soil_pft_mask = (is_on_soil_p > 0) | (is_on_crop_p > 0)

    soil_col_mask = soil_col_mask & col_valid
    soil_pft_mask = soil_pft_mask & pft_valid

    soil_cols = np.nonzero(soil_col_mask.values)[0]
    soil_pfts = np.nonzero(soil_pft_mask.values)[0]

    return soil_cols, soil_pfts


def sample(base_fn: str, samples_per_file: int, var_name_set: set[str]):
    data_dir = Path(unittests_dir) / "input-data"

    out_fn = data_dir / f"{base_fn}-training_samples.nc"
    if out_fn.exists():
        out_fn.unlink()
    # list files
    files = sorted(data_dir.glob(f"{base_fn}*.nc"))

    rng = np.random.default_rng(seed=42)
    xr.set_options(use_new_combine_kwarg_defaults=True)

    sampled_datasets = []

    for f in files:
        print(f"Processing {f}")

        ds = xr.open_dataset(
            f,
            # chunks={"time": 1},
            decode_cf=True,
            # engine="scipy",
        )

        ntime = ds.sizes["time"]

        soil_cols, soil_pfts = build_soil_filters(ds)

        if ntime < samples_per_file:
            raise ValueError(f"{f} has fewer than {samples_per_file} timesteps")

        # randomly choose indices
        idx = rng.choice(ntime, size=samples_per_file, replace=False)

        # sort so time stays ordered
        idx = np.sort(idx)

        # subset
        ds_sample = ds.isel(time=idx)
        # keep only requested variables that are present
        keep_vars = [v for v in var_name_set if v in ds_sample.data_vars]
        ds_sample = ds_sample[keep_vars]

        # build compressed view along column/patch
        # soil_cols / soil_pfts are integer indices along dims 'column' and 'patch'
        compress_sel = {}
        if "column" in ds_sample.dims:
            compress_sel["column"] = soil_cols
        if "patch" in ds_sample.dims:
            compress_sel["patch"] = soil_pfts
        ds_compressed = ds_sample.isel(**compress_sel)


        if "column" in compress_sel:
            ds_compressed = ds_compressed.assign_coords(
                soil_col_index=("column", soil_cols)
            )
        if "patch" in compress_sel:
            ds_compressed = ds_compressed.assign_coords(
                soil_pft_index=("patch", soil_pfts)
            )

        sampled_datasets.append(ds_compressed)

    # combine samples from all files
    combined = xr.concat(sampled_datasets, dim="time")

    # optional: shuffle time dimension again
    perm = rng.permutation(combined.sizes["time"])
    combined = combined.isel(time=perm).load()
    # write output
    combined.to_netcdf(f"{base_fn}-training_samples.nc")#,engine="scipy")
