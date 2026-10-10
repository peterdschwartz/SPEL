import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Configure path information
scripts_dir = Path(os.path.dirname(__file__))
spel_dir = Path(f"{scripts_dir}/../../").resolve()
database_app = spel_dir / "spel" / "db"
database_csv =  database_app / "app/management/commands/csv/"
presets = database_app / "app/management/comands/presets/"
unittests_dir = spel_dir / "unit-tests/"
spel_mods_dir = spel_dir / "SourceFiles/"
# Scratch dir for generated files; `spel validate` gives each parallel
# `spel create` its own (create wipes *.F90 here).
spel_output_dir = Path(os.environ.get("SPEL_OUTPUT_DIR", spel_dir / "script-output"))
spel_output_dir.mkdir(parents=True, exist_ok=True)
input_data_dir = unittests_dir / "input-data/"

# Name of the file (written into each unit-test case directory) that records
# which E3SM_SRCROOT was used to generate that case. Read by `spel restore`.
CASE_META_FILENAME = "spel_meta.json"

# Written next to the captured reference data (unit-tests/input-data/<case>)
# recording the E3SM checkout that produced it, so `spel run` can tell when a
# unit test generated from one checkout is checked against another's data.
REFERENCE_META_FILENAME = "spel_reference.json"

# Per-developer overrides (currently just E3SM_SRCROOT) live in this file
# instead of being hardcoded here, so that pointing SPEL at a different E3SM
# checkout never shows up as a git diff on this tracked config.py.
# Create/update it with `spel config --set-srcroot <path>`, or hand-edit it,
# e.g.:
#   SPEL_E3SM_SRCROOT=/path/to/your/E3SM
LOCAL_ENV_FILE = spel_dir / ".spel.env"
load_dotenv(LOCAL_ENV_FILE)

# E3SM root directory. Resolution order: SPEL_E3SM_SRCROOT environment
# variable (also settable via LOCAL_ENV_FILE above) > a sibling "dev_E3SM"
# checkout next to the SPEL repo.
_default_srcroot = (spel_dir / "../dev_E3SM").resolve()
E3SM_SRCROOT = Path(
    os.environ.get("SPEL_E3SM_SRCROOT", str(_default_srcroot))
).expanduser().resolve()

# path for modules shared by components (eg, shr_kind_mod)
SHR_SRC = E3SM_SRCROOT / "share/util/"
ELM_SRC = E3SM_SRCROOT / "components/elm/src/"  # elm source directory

# Script that creates/builds/runs the CIME case used to capture reference data
# (`spel create --instrument --run-case`, `spel instrument --run-case`).
# Override with SPEL_CASEGEN (env or .spel.env).
CASEGEN_SCRIPT = Path(
    os.environ.get(
        "SPEL_CASEGEN", str(E3SM_SRCROOT.parent / "e3sm-scripts/uELM_casegen.sh")
    )
).expanduser()

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
    "decompInitMod.o",
    "FUTConstantsMod.o",
    "ReadWriteMod.o",
    "duplicateMod.o",
    "verificationMod.o",
    "elm_initializeMod.o",
    "UpdateParamsAccMod.o",
    "main.o",
]


class Options:
    def __init__(self):
        self.db_mode: bool = False
        self.cfg_hash: str = ""

options = Options()

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
