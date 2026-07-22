from collections import namedtuple

import numpy as np
import xarray as xr
from tabulate import tabulate

# namedtuple for summary of errors
Tally = namedtuple("Tally", ["name", "total", "rmse", "max"])


def load_data_netcdf(file, verbose):
    """ """


# def loadData_input(file, verbose):
#     """
#     This function loads the data from a verfication
#     module generated file.
#
#     Data in the text files are assummed to have the format:
#        <variable name>  <shape(var)>
#     """
#     import re
#
#     dtype_name_regex = re.compile(r"^\w+(%)\w+")
#     var_name = re.compile(r"^\w+")
#     varnames = []
#     vardict = {}
#     var_dims = {}
#     # Go through data file by reading in variable name / dimension
#     # and filling then read the it's values
#     EOF = len(file)
#     ln = 0
#     while ln < EOF:
#         line = file[ln]
#         line = line.strip()
#         # match current variable name and shape
#         match_dtypename = dtype_name_regex.search(line)
#         match_varname = var_name.search(line)
#         var = ""
#         if match_dtypename:
#             var = match_dtypename.group()
#         elif match_varname:
#             var = match_varname.group()
#         varnames.append(var)
#         # substitute varname out of line, so only the shape remains
#         line = line.replace(var, "").strip()
#         dims = line.split()
#         dims = [int(d) for d in dims]
#         var_dims[var] = dims
#         # Get total number of elements to be read:
#         size = 1
#         for dim in dims:
#             size *= dim
#         # Now read in 'size' elements:
#         current_el = 0
#         while current_el < size:
#             ln += 1
#             line = file[ln].strip()
#             data = line.split()
#             # Temporarily store data as a list. Use var_dims to reshape to numpy array
#             for x in data:
#                 if x == "T":
#                     x = 1
#                 elif x == "F":
#                     x = 0
#                 vardict.setdefault(var, []).append(float(x))
#                 current_el += 1
#         ln += 1
#         if verbose and ln < EOF:
#             print(f"finished reading in data for {var} {dims}. next line:\n{file[ln]}")
#
#     # Go through each var in vardict and convert to numpy array.
#     for var, vals in vardict.items():
#         vardict[var] = np.reshape(vals, tuple(var_dims[var]))
#
#     return vardict


def compute_error(var, refdata, testdata, diff_log, verbose=False):
    """
    var : variable name
    refdata : data from reference file
    testdata : data from test file
    """

    # Set parameters and initialize diff log
    EPSILON = 1.0e-50
    ERROR = 1.0e-16  # Threshold to report
    NUMLOGS = 8  # total number of examples to report
    diff_vals = refdata - testdata
    diff_vals = np.abs(diff_vals)
    # Find ref values that are non-zero
    nonzero_elements = np.full(diff_vals.shape, True)

    nonzero_elements[np.abs(refdata) == 0.0] = False
    zero_elements = np.invert(nonzero_elements)

    # Calculated relative error at each position:
    relerror = np.zeros(diff_vals.shape, dtype=np.float64)
    relerror[nonzero_elements] = diff_vals[nonzero_elements] / np.abs(
        refdata[nonzero_elements]
    )
    relerror[zero_elements] = diff_vals[zero_elements] / EPSILON

    # Generate mask for significant errors greater than threshold ERROR.
    # Then use mask to retrieve corresponding elements from data arrays
    sig_elements = np.full(diff_vals.shape, False)
    sig_elements[relerror > ERROR] = True

    indices = np.argwhere(relerror > ERROR)
    sig_diffs = relerror[sig_elements]
    og_vals = refdata[sig_elements]
    test_vals = testdata[sig_elements]
    if len(sig_diffs) > 0:
        newvar_header = [f"{var}", "**", "**", "**"]
        # Calculate RMSE:
        rmse = np.sqrt(((og_vals - test_vals) ** 2).mean()) / np.sqrt(len(og_vals))
        summary = Tally(
            "Summary",
            f"# diffs: {len(sig_diffs)}",
            f"rmse: {rmse}",
            f"max: {np.max(sig_diffs)}",
        )
        diff_log.append(tuple(newvar_header))
        for i, diff in enumerate(sig_diffs):
            indices[i] += 1
            coords = tuple(indices[i])
            og_val = og_vals[i]
            t_val = test_vals[i]
            if i <= NUMLOGS:
                diff_log.append(tuple([coords, og_val, t_val, diff]))
        diff_log.append(summary)
    else:
        summary = None
    return diff_log


def errorVerification(reffn, testfn):
    """
    This function performs BFB statistical analysis on
    a ref and test run.
    """

    # Read files to compare
    print(f"ref file {reffn} ::: : ::: test file {testfn}")
    file = open(reffn, "r")
    reffile = file.readlines()
    file.close()
    file = open(testfn, "r")
    testfile = file.readlines()
    file.close()
    # Retrieve and store data as a dictionary of numpy arrays
    refdata = loadData_input(reffile, verbose=False)
    testdata = loadData_input(testfile, verbose=False)

    report = []
    for var in refdata:
        report = compute_error(var, refdata[var], testdata[var], report, verbose=True)
    print(tabulate(report, tablefmt="psql"))


def findDataFiles(unittest):
    import subprocess as sp

    # get the ref files:
    output = sp.getoutput(f"ls -t ref_{unittest}*.txt")
    temp = output.split()
    temp.reverse()
    ref_files = temp.copy()

    # get the test files :
    output = sp.getoutput(f"ls -t test_{unittest}*.txt")
    temp = output.split()
    temp.reverse()
    test_files = temp.copy()
    return ref_files, test_files


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="errorAnalysis", description="BFB analysis for ref vs test"
    )
    parser.add_argument(
        "--ref",
        action="store",
        required=False,
        dest="reffn",
        help="ref data filename",
        default="",
    )
    parser.add_argument(
        "--test",
        action="store",
        required=False,
        dest="testfn",
        help="test data filename",
        default="",
    )
    parser.add_argument(
        "-b",
        action="store",
        required=False,
        dest="unittest",
        help="Analyze all files found for this unit-test",
        default="",
    )
    args = parser.parse_args()

    if args.reffn and args.testfn:
        errorVerification(reffn=args.reffn, testfn=args.testfn)
    elif args.unittest:
        print(f"performing batch for UnitTest {args.unittest}")
        ref_files, test_files = findDataFiles(args.unittest)
        if len(ref_files) != len(test_files):
            sys.exit("ref and test data files do not match!")
        for n in range(0, len(ref_files)):
            errorVerification(reffn=ref_files[n], testfn=test_files[n])
