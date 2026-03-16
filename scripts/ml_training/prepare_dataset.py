from pathlib import Path

import xarray
from xarray.core.dataset import Dataset

from scripts.analyze_subroutines import Subroutine
from scripts.export_objects import unpickle_unit_test


def separate_inputs_outputs(
    sub_dict: dict[str, Subroutine], inputs: set[str], outputs: set[str]
):

    subroutines = {
        key: val
        for key, val in sub_dict.items()
        if val.unit_test_function and key not in {"filtermod::setfilters"}
    }
    ignore = {"col_pp", "lun_pp", "veg_pp", "grc_pp", "top_pp"}
    for sub in subroutines.values():
        for var, rw in sub.elmtype_access_summary.items():
            if var.split('%')[0] in ignore:
                continue
            varname = var.replace("%", "__")
            if rw.status == "r":
                inputs.add(varname)
            elif rw.status == "w":
                outputs.add(varname)
            elif rw.status == "rw":
                inputs.add(varname)
                outputs.add(varname)
            else:
                raise ValueError(f"Unknown variable status {var} {rw}")
    return


def prepare_data():
    commit = "c76c282"
    mod_dict, sub_dict, type_dict = unpickle_unit_test(commit)

    input_base_fn = "spel-inputs"
    output_base_fn = "spel-outputs"
    data_dir = Path("../../unit-tests/input-data/")
    inputs: set[str] = set()
    outputs: set[str] = set()
    separate_inputs_outputs(sub_dict, inputs, outputs)
    return


def group_netcdf_by_basefn(directory: Path, base_fn: str) -> dict[str, list[Path]]:
    """Group NetCDF files of form <base_fn>00xx.nc by base_fn.

    Example: foo0001.nc, foo0002.nc -> key "foo" -> [paths...]
    """
    groups: dict[str, list[Path]] = {}
    for p in directory.glob(f"{base_fn}*.nc"):
        groups.setdefault(base_fn, []).append(p)
    # sort file lists for deterministic ordering
    for base_fn in groups:
        groups[base_fn].sort()
    return groups


def open_grouped_datasets(
    directory: Path,
    base_fn: str,
    **open_mfdataset_kwargs,
) -> dict[str, Dataset]:
    """Open one xarray Dataset per base_fn group."""
    file_groups = group_netcdf_by_basefn(directory, base_fn)
    datasets: dict[str, Dataset] = {}
    for base_fn, paths in file_groups.items():
        datasets[base_fn] = xarray.open_mfdataset(
            [str(p) for p in paths],
            combine="by_coords",
            **open_mfdataset_kwargs,
        )
    return datasets
