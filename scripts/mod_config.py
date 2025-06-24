import os
import re

# Configure path information
scripts_dir = os.path.dirname(__file__)
spel_dir = f"{scripts_dir}/../"
django_database = f"{spel_dir}/spel/spel/app/management/commands/csv/"
unittests_dir = spel_dir + "unit-tests/"
spel_mods_dir = spel_dir + "SourceFiles/"
spel_output_dir = scripts_dir + "/script-output/"

# E3SM root directory.
E3SM_SRCROOT = spel_dir + "../E3SM/"
# path for modules shared by components (eg, shr_kind_mod)
SHR_SRC = E3SM_SRCROOT + "/share/util/"
ELM_SRC = E3SM_SRCROOT + "/components/elm/src/"  # elm source directory
E3SM_dir = ELM_SRC

# List to hold physical property data types that are
# necessary for domain decomposition, but may not be
# used in computation routines (ignored by spel)
PHYSICAL_PROP_TYPE_LIST = [
    "vegetation_physical_properties",
    "column_physical_properties",
    "landunit_physical_properties",
    "gridcell_physical_properties_type",
    "topounit_physical_properties",
]

# Need regex to subsitutue elm folder structure (include sanity check here?)
elm_dir_regex = re.compile(
    f"{ELM_SRC}(main|biogeophys|biogeochem|utils|cpl|data_types|dyn_subgrid)/"
)
shr_dir_regex = re.compile(f"{SHR_SRC}")

dont_adjust = ["c2g", "p2c", "p2g", "p2c", "c2l", "l2g", "tridiagonal"]

dont_adjust_string = "|".join(dont_adjust)
regex_skip_string = re.compile(f"({dont_adjust_string})", re.IGNORECASE)

# List of modules needed for domain decomposition -- required for all unit-tests
# NOTE: These hardcoded lists should be deprecated by now?
default_mods = ["subgridmod", "filtermod"]

# list of files neeeded for all unit-tests
unit_test_files = [
    "H5InterfaceMod.o",
    "decompInitMod.o",
    "elm_instMod.o",
    # "fileio_mod.o",
    # "initializeParameters.o",
    # "UnitTestAllocatorMod.o",
    # "update_accMod.o",
    "FUTConstantsMod.o",
    "ReadWriteMod.o",
    "duplicateMod.o",
    "verificationMod.o",
    "elm_initializeMod.o",
    "UpdateParamsAccMod.o",
    "main.o",
]


class BColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class EmptyColors:
    HEADER = ""
    OKBLUE = ""
    OKCYAN = ""
    OKGREEN = ""
    WARNING = ""
    FAIL = ""
    ENDC = ""
    BOLD = ""
    UNDERLINE = ""


_bc = BColors()
_no_colors = EmptyColors()
