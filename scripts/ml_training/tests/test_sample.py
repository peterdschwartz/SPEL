import numpy as np
import xarray as xr
from pathlib import Path

from scripts.ml_training.sample_spel_output import sample, build_soil_filters


def _make_fake_ds(tmp_path: Path, base_fn: str) -> Path:
    """Create a small fake dataset on disk for testing."""
    time = np.arange(10)
    col = np.arange(5)
    soil_pft = np.arange(4)

    # flags: first 3 cols/pfts are "soil"
    col_pp__is_soil = xr.DataArray(
        np.where(col < 3, 1, 0)[None, :].repeat(time.size, axis=0),
        dims=("time", "column"),
    )
    col_pp__is_crop = xr.zeros_like(col_pp__is_soil)

    veg_pp__is_on_soil_col = xr.DataArray(
        np.where(soil_pft < 3, 1, 0)[None, :].repeat(time.size, axis=0),
        dims=("time", "patch"),
    )
    veg_pp__is_on_crop_col = xr.zeros_like(veg_pp__is_on_soil_col)

    # a couple of variables with column/patch dims
    col_var = xr.DataArray(
        np.random.rand(time.size, 2, col.size),
        dims=("time", "z", "column"),
    )
    pft_var = xr.DataArray(
        np.random.rand(time.size, soil_pft.size),
        dims=("time", "patch"),
    )

    ds = xr.Dataset(
        {
            "col_pp__is_soil": col_pp__is_soil,
            "col_pp__is_crop": col_pp__is_crop,
            "veg_pp__is_on_soil_col": veg_pp__is_on_soil_col,
            "veg_pp__is_on_crop_col": veg_pp__is_on_crop_col,
            "col_var": col_var,
            "pft_var": pft_var,
        }
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_path = data_dir / f"{base_fn}001.nc"
    ds.to_netcdf(out_path,engine="scipy")
    return out_path


def test_sample_and_compress(tmp_path, monkeypatch):
    base_fn = "test_run"

    # create fake input file under tmp_path/data
    _ = _make_fake_ds(tmp_path, base_fn)

    # run sample() in that directory
    monkeypatch.chdir(tmp_path)
    samples_per_file = 5
    sample(base_fn=base_fn, samples_per_file=samples_per_file)

    out_path = tmp_path / f"{base_fn}-training_samples.nc"
    assert out_path.exists()

    combined = xr.open_dataset(out_path, engine="scipy")

    # time dimension should be samples_per_file (one file)
    assert combined.sizes["time"] == samples_per_file

    # recompute soil filters from the original constructed dataset
    orig_ds = xr.open_dataset(tmp_path / "data" / f"{base_fn}001.nc", engine="scipy")
    soil_cols, soil_pfts = build_soil_filters(orig_ds)
    orig_ds.close()

    # check that compression actually reduced the sizes of col/pft dims
    # and matches the filter lengths
    # depending on your logic, dims may be named "soil_col" / "soil_pft"
    assert "column" in combined.dims
    assert "patch" in combined.dims
    assert combined.sizes["column"] == soil_cols.size
    assert combined.sizes["patch"] == soil_pfts.size

    assert "soil_col_index" in combined.coords
    assert "soil_pft_index" in combined.coords
    assert np.array_equal(combined["soil_col_index"].values, soil_cols)
    assert np.array_equal(combined["soil_pft_index"].values, soil_pfts)
    # verify example variables got compressed along those dims
    assert combined["col_var"].sizes["column"] == soil_cols.size
    assert combined["pft_var"].sizes["patch"] == soil_pfts.size
    combined.close()

