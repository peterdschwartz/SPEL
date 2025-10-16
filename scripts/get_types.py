import os
import re
import sys
import time

from scripts.config import ELM_SRC, unittests_dir
from scripts.fortran_modules import get_filename_from_module, get_module_name_from_file
from scripts.utilityFunctions import line_unwrapper
from scripts.write_routines import create_deepcopy_module

regex_contains = re.compile(r"^(contains)", re.IGNORECASE)


def progressbar(it, prefix="", size=60, out=sys.stdout, total=None):
    # count = len(it)
    start = time.time()  # time estimate start

    def show(j, count):
        if count:
            x = int(size * j / count)
            # time estimate calculation and string
            remaining = ((time.time() - start) / j) * (count - j)
            mins, sec = divmod(remaining, 60)  # limited to minutes
            time_str = f"{int(mins):02}:{sec:03.1f}"
            bar = (
                f"{prefix:>10.10}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}",
            )
        else:
            # No total: just show items processed + spinner
            spinner = r"\|/-"[int(j) % 4]
            elapsed = time.time() - start
            mins, sec = divmod(elapsed, 60)
            time_str = f"{int(mins):02}:{sec:03.1f}"
            bar = f"{spinner} {j} items, elapsed {time_str}"
        print(f"{prefix:>10.10}{bar}", end="\r", file=out, flush=True)

    show(0.1, total)  # avoid div/0
    for i, item in enumerate(it):
        yield item
        prefix = str(item)
        show(i + 1, total)
    print("\n", flush=True, file=out)


def get_type_from_files(mods, modname):
    """
    Function that goes through list of files and retrieves all
    user type definitions
    """
    elm_files = [get_filename_from_module(mod) for mod in mods]
    type_dict = {}
    case_dir = unittests_dir + "all_types"
    if not os.path.isdir(case_dir):
        os.mkdir(case_dir)

    for fn in progressbar(elm_files, "mod:", 60):
        _, module_name = get_module_name_from_file(fn)
        ifile = open(fn, "r")
        lines = ifile.readlines()
        ln = 0
        module_head = True
        while module_head:
            line, ln = line_unwrapper(lines, ln)
            line = line.strip().lower()
            match_contains = regex_contains.search(line)
            if match_contains:
                module_head = False
                # Nothing else to check in this line
                continue

            lprime = line.replace("public", "").replace("private", "").replace(",", "")
            match_type_def1 = re.search(r"^(type\s*::)", lprime)  # type :: type_name
            match_type_def2 = re.search(r"^(type\s+)(?!\()", lprime)  # type type_name
            type_name = None
            if match_type_def1:
                # Get type name
                type_name = lprime.split("::")[1].strip()
            elif match_type_def2:
                # Get type name
                matched_expr = match_type_def2.group()
                type_name = lprime.replace(matched_expr, "").strip()
            if type_name:
                # Analyze the type definition
                user_dtype, ln = get_derived_type_definition(
                    ifile=fn,
                    modname=module_name,
                    lines=lines,
                    ln=ln,
                    type_name=type_name,
                    verbose=False,
                )
                type_dict[type_name] = user_dtype
            ln += 1

    create_deepcopy_module(type_dict, case_dir, modname, all_active=True)
    return None


def main():
    mod_veg = [
        "VegetationType",
        "VegetationPropertiesType",
        "VegetationDataType",
    ]
    mod_col = [
        "ColumnType",
        "ColumnDataType",
    ]
    mod_landtopo = [
        "LandunitType",
        "LandunitDataType",
        "TopounitType",
        "Topounitdatatype",
    ]
    mod_grid = [
        "GridcellType",
        "GridcellDataType",
    ]
    mod_main = [
        "domainmod",
        "atm2lndType",
        "lnd2atmType",
        "lnd2glcMod",
    ]
    mod_chem = [
        "AllocationMod",
        "CH4Mod",
        "CNDecompCascadeConType",
        "CropType",
        "DUSTMod",
        "DecompCascadeBGCMod",
        "DecompCascadeCNMod",
        "DryDepVelocity",
        "GapMortalityMod",
        "NitrifDenitrifMod",
        "NitrogenDynamicsMod",
        "PlantMicKineticsMod",
        "SharedParamsMod",
        "SoilLittDecompMod",
        "SoilorderConType",
    ]

    mod_phys = [
        "AerosolType",
        "CanopyStateType",
        "PhotosynthesisType",
        "SolarAbsorbedType",
        "SurfaceAlbedoType",
        "UrbanParamsType",
        "SurfaceRadiationMod",
        "EnergyfluxType",
        "SoilHydrologyType",
        "LakeStateType",
        "FrictionVelocityType",
    ]

    get_type_from_files(mod_veg, modname="DeepCopyVegetationMod")
    get_type_from_files(mod_col, modname="DeepCopyColumnMod")
    get_type_from_files(mod_main, modname="DeepCopyMainMod")
    get_type_from_files(mod_grid, modname="DeepCopyGridcellMod")
    get_type_from_files(mod_phys, modname="DeepCopyPhysMod")
    get_type_from_files(mod_chem, modname="DeepCopyChemMod")
    get_type_from_files(mod_landtopo, modname="DeepCopyLandTopoMod")


if __name__ == "__main__":
    main()
