import sys
import time
from collections import namedtuple

import numpy as np
import xarray
from tabulate import tabulate
from xarray.core.dataset import Dataset

# namedtuple for summary of errors
Tally = namedtuple("Tally", ["name", "total", "rmse", "max"])


def progressbar(it, prefix="", size=60, out=sys.stdout):
    count = len(it)
    start = time.time()  # time estimate start

    def show(j):
        x = int(size * j / count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / j) * (count - j)
        mins, sec = divmod(remaining, 60)  # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(
            f"{prefix:>10.10}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}",
            end="\r",
            file=out,
            flush=True,
        )

    show(0.1)  # avoid div/0
    for i, item in enumerate(it):
        yield item
        prefix = item
        show(i + 1)
    print("\n", flush=True, file=out)


def rel_error(refdata: Dataset, compdata: Dataset, var: str, error_log):
    """
    Function to find any differences between history files
        - refdata : xarray Dataset presumed to have the correct values
        - compdata : xr Dataset values to test.
        - var : name of history variable
    The array dimensions for each variable are of the form:
        'time': UNLIM, 'levdcmp': 15, 'lndgrid': 21
    it will be assumed that time is always the leftmost, and
    the gridcell is always the rightmost dimension.
    """
    # Get relevant data
    MAX_LOG_LENGTH = 10
    original_vals = refdata[var].values
    comp_vals = compdata[var].values
    dims = refdata[var].dims
    dtype = refdata[var].dtype
    sizes = refdata[var].sizes
    # Handle time mismatch by slicing to shortest available
    if "time" in dims:
        time_axis = dims.index("time")
        time_len_ref = original_vals.shape[time_axis]
        time_len_comp = comp_vals.shape[time_axis]
        min_time_len = min(time_len_ref, time_len_comp)

        # Slice both arrays along time dimension
        slicer = [slice(None)] * original_vals.ndim
        slicer[time_axis] = slice(0, min_time_len)
        original_vals = original_vals[tuple(slicer)]
        comp_vals = comp_vals[tuple(slicer)]

    if comp_vals.shape != original_vals.shape:
        print(f"Error {var} dimensions do not match between files")
        print(f"OG : {original_vals.shape}\n TEST : {comp_vals.shape}")
        sys.exit(1)

    # Set parameters and initialize diff log
    EPSILON = 1.0e-50
    ERROR = 0.0e-25  # Threshold to report
    NUMLOGS = 8  # total number of examples to report
    diff_vals = original_vals - comp_vals
    diff_vals = np.abs(diff_vals)

    # Find ref values that are non-zero
    nonzero_elements = np.full(diff_vals.shape, True)

    nonzero_elements[np.abs(original_vals) == 0.0] = False
    zero_elements = np.invert(nonzero_elements)

    # Calculated relative error at each position:
    relerror = np.zeros(diff_vals.shape, dtype=dtype)
    relerror[nonzero_elements] = diff_vals[nonzero_elements] / np.abs(
        original_vals[nonzero_elements]
    )
    relerror[zero_elements] = diff_vals[zero_elements] / EPSILON

    # Generate mask for significant errors greater than threshold ERROR.
    # Then use mask to retrieve corresponding elements from data arrays
    sig_elements = np.full(diff_vals.shape, False)
    sig_elements[relerror > ERROR] = True

    indices = np.argwhere(relerror > ERROR)
    sig_diffs = relerror[sig_elements]
    og_vals = original_vals[sig_elements]
    test_vals = comp_vals[sig_elements]
    if len(sig_diffs) > 0:
        dims_str = [k for k in sizes.keys()]
        dims_str = ",".join(dims_str)
        newvar_header = [f"{var}", "Ref", "Test", "Diff"]

        # Calculate RMSE:
        rmse = np.sqrt(((og_vals - test_vals) ** 2).mean()) / np.sqrt(len(og_vals))
        summary = Tally(
            "Summary",
            f"#{len(sig_diffs)}",
            f"rmse: {rmse}",
            f"max: {np.max(sig_diffs)}",
        )
        error_log.append(tuple(newvar_header))
        for i, diff in enumerate(sig_diffs):
            indices[i] += 1
            coords = tuple(int(x) for x in indices[i])
            og_val = og_vals[i]
            t_val = test_vals[i]
            if i <= NUMLOGS:
                error_log.append(tuple([coords, og_val, t_val, diff]))
        error_log.append(summary)
    else:
        summary = None

    return error_log, summary


def is_numeric(dtype):
    if np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64):
        return False
    return np.issubdtype(dtype, np.number)


def find_diffs(refn: str, compfn: str, var: str = "", ostream=sys.stdout):
    """
    Function to compare two netcdf files and report any significant diffs
    """
    findall = True if not var else False
    print("Reference File is:", refn)
    print("Comparison File is:", compfn)
    print("Findall is:", findall)

    refdata = xarray.open_dataset(refn)
    compdata = xarray.open_dataset(compfn)

    if findall:
        var_names = [var for var in refdata.keys()]
        current_var = var_names[0]
        error_log = []
        for var in progressbar(var_names, "VAR:", 40):
            dtype = refdata[var].dtype
            if is_numeric(dtype):
                error_log, summary = rel_error(refdata, compdata, var, error_log)
    else:
        error_log = []
        error_log, summary = rel_error(refdata, compdata, var, error_log)
    ostream.write(tabulate(error_log, tablefmt="psql"))
    ostream.write("\n")
    ostream.close()
