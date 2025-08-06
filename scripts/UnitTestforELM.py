import logging
import sys
from pprint import pprint

import scripts.dynamic_globals as dg
from scripts.aggregate import aggregate_dtype_vars
from scripts.analyze_subroutines import Subroutine
from scripts.DerivedType import DerivedType
from scripts.fortran_modules import FortranModule, get_filename_from_module
from scripts.helper_functions import construct_call_tree
from scripts.logging_configs import get_logger
from scripts.LoopConstructs import get_loops
from scripts.mod_config import (
    _bc,
    default_mods,
    scripts_dir,
    spel_mods_dir,
    spel_output_dir,
    unittests_dir,
)
from scripts.refactor_tools import compress_on_filter, prune_associate_clause
from scripts.utilityFunctions import Variable
from scripts.variable_analysis import determine_global_variable_status

ModDict = dict[str, FortranModule]
SubDict = dict[str, Subroutine]
TypeDict = dict[str, DerivedType]


def create_unit_test(sub_names: list[str], casename: str, keep: bool) -> None:
    """
    Edit case_dir and sub_name_list to create a Functional Unit Test
    in a directory called {case_dir} for the subroutines in sub_name_list.
    """
    import os
    import sys

    import scripts.write_routines as wr
    from scripts.edit_files import process_for_unit_test
    from scripts.export_objects import pickle_unit_test
    from scripts.fortran_modules import insert_header_for_unittest

    func_name = "( main )"
    logger = get_logger("SPEL", level=logging.INFO)

    # Note unittests_dir is a location to make unit tests directories named {casename}
    if not casename:
        casename = "fut"

    case_dir = unittests_dir + casename

    if not sub_names:
        sys.exit("Error- No subroutines provided for analysis")
    sub_name_list = [s.lower() for s in sub_names]

    logger.info(f"Creating UnitTest {casename} || {' '.join(sub_name_list)}")

    # Create script output directory if not present:
    if not os.path.isdir(f"{scripts_dir}/script-output"):
        logger.info("Making script output directory")
        os.system(f"mkdir {scripts_dir}/script-output")

    # Create case directory
    if not os.path.isdir(f"{case_dir}"):
        logger.info(f"Making case directory {case_dir}")
        if not os.path.isdir(f"{unittests_dir}"):
            os.system(f"mkdir {unittests_dir}")
        os.system(f"mkdir {case_dir}")
    else:
        # Check if user wants to re-use existing case without pre-processing
        if keep:
            preprocess = False
        else:
            os.system(f"rm -rf {case_dir}/*")
            os.system(f"rm {scripts_dir}/*.pkl")
            preprocess = True

    # Retrieve possible interfaces
    dg.populate_interface_list()
    # Initialize dictionary that will hold instance of all subroutines encountered.
    main_sub_dict: SubDict = {}

    # dictionary holds instances for Unit Test specific subroutines
    sub_name_list: list[str] = [s.lower() for s in sub_name_list]
    subroutines: SubDict = {}

    # List to hold all the modules needed for the unit test
    needed_mods = []
    mod_dict: ModDict = {}
    # Get general info of the subroutine
    # Process files by removing certain modules
    # so that a standalone unit test can be compiled.
    # All file information will be stored in `mod_dict` and `main_sub_dict`
    ordered_mods = process_for_unit_test(
        case_dir=case_dir,
        mod_dict=mod_dict,
        mods=needed_mods,
        required_mods=default_mods,
        sub_dict=main_sub_dict,
        sub_name_list=sub_name_list,
        overwrite=True,
        verbose=False,
    )

    for s in sub_name_list:
        subroutines[s] = main_sub_dict[s]
        subroutines[s].unit_test_function = True

    if not mod_dict or not ordered_mods:
        logger.error(f"{func_name}Error didn't find any modules related to subroutines")
        sys.exit(1)

    type_dict: TypeDict = {}
    for mod in mod_dict.values():
        for utype, dtype in mod.defined_types.items():
            type_dict[utype] = dtype

    for dtype in type_dict.values():
        dtype.find_instances(mod_dict)

    bounds_inst = Variable(type="bounds_type", name="bounds", dim=0, subgrid="?", ln=-1)
    type_dict["bounds_type"].instances["bounds"] = bounds_inst.copy()

    instance_to_user_type: dict[str, str] = {}
    for type_name, dtype in type_dict.items():
        if "bounds" in type_name:
            continue
        # All instances should have been found so throw an error
        if not dtype.instances:
            logger.warning(f"Warning: no instances found for {type_name}")
        for instance in dtype.instances.values():
            instance_to_user_type[instance.name] = type_name

    main_sub_dict["setfilters"].unit_test_function = True
    process_subroutines_for_unit_test(
        mod_dict=mod_dict,
        sub_dict=main_sub_dict,
        type_dict=type_dict,
    )

    # if "lakefluxes" in main_sub_dict:
    #     lakefluxes = main_sub_dict["lakefluxes"]
    #     compress_on_filter(sub=lakefluxes)
    #     sys.exit(0)

    aggregate_dtype_vars(
        sub_dict=main_sub_dict,
        type_dict=type_dict,
        inst_to_dtype_map=instance_to_user_type,
    )

    instance_dict: dict[str, DerivedType] = {}
    for type_name, dtype in type_dict.items():
        for instance in dtype.instances.values():
            instance_dict[instance.name] = dtype

    active_set: set[str] = set()
    for inst_name, dtype in instance_dict.items():
        if not dtype.instances[inst_name].active:
            continue
        for field_var in dtype.components.values():
            if field_var.active:
                active_set.add(f"{inst_name}%{field_var.name}")

    for sub in subroutines.values():
        for key in list(sub.elmtype_access_sum.keys()):
            c13c14 = bool("c13" in key or "c14" in key)
            if c13c14:
                del sub.elmtype_access_sum[key]
                continue

    # Create a makefile for the unit test
    file_list = [get_filename_from_module(m) for m in ordered_mods]
    wr.generate_cmake(files=file_list, case_dir=case_dir)

    logger.info(f"Call Tree for {case_dir}")
    for sub in subroutines.values():
        if sub.abstract_call_tree:
            sub.abstract_call_tree.print_tree()

    elmvars_dict = {}
    for dtype in type_dict.values():
        for inst in dtype.instances.values():
            elmvars_dict[inst.name] = inst

    unittest_global_vars: dict[str, Variable] = {}

    unittest_subs = {sub for sub in main_sub_dict.values() if sub.unit_test_function}
    for sub in unittest_subs:
        if not sub.abstract_call_tree:
            continue
        for subnode in sub.abstract_call_tree.traverse_preorder():
            childsub = main_sub_dict[subnode.node.subname]
            unittest_global_vars.update(childsub.active_global_vars)

    # Generate/modify FORTRAN files needed to initialize and run Unit Test
    wr.prepare_unit_test_files(
        type_dict=type_dict,
        case_dir=case_dir,
        global_vars=unittest_global_vars,
        subroutines=subroutines,
        instance_to_type=instance_to_user_type,
    )
    # elm_instMod.F90
    wr.write_elminstMod(type_dict, case_dir)
    # duplicateMod.F90
    wr.duplicate_clumps(type_dict)
    wr.create_fortls(case_dir)

    # Go through all needed files and include a header that defines some constants
    insert_header_for_unittest(
        mod_list=ordered_mods,
        mod_dict=mod_dict,
        casedir=case_dir,
    )

    cmds: list[str] = [
        f"cp {spel_output_dir}duplicateMod.F90 {case_dir}",
        f"cp {spel_mods_dir}nc_io.F90 {case_dir}",
        f"cp {spel_mods_dir}nc_allocMod.F90 {case_dir}",
        f"cp {spel_mods_dir}unittest_defs.h {case_dir}",
        f"cp {spel_mods_dir}decompInitMod.F90 {case_dir}",
        f"cp {spel_mods_dir}check_config.sh {case_dir}",
    ]
    for cmd in cmds:
        logger.info(cmd)
        os.system(cmd)

    pickle_unit_test(mod_dict, main_sub_dict, type_dict)

    return None


def process_subroutines_for_unit_test(
    mod_dict: ModDict,
    sub_dict: SubDict,
    type_dict: TypeDict,
):
    """
    Function that processes the subroutines found in each FortranModule
        1) identify any non derived-type global vars used by Subroutine
        2) collect derived-type var and subroutine call info
        3) construct subroutine call trees (abstract=child subs represented only once)
        4) analyze status of variables used by subroutines.
    """
    fut_subs: set[str] = {
        sub.name for sub in sub_dict.values() if sub.unit_test_function
    }

    for sub in sub_dict.values():
        determine_global_variable_status(mod_dict, sub)

    for dtype in type_dict.values():
        if dtype.init_sub_name:
            dtype.init_sub_ptr = sub_dict[dtype.init_sub_name]

    for sub_name in fut_subs:
        sub = sub_dict[sub_name]
        sub.collect_var_and_call_info(
            dtype_dict=type_dict,
            sub_dict=sub_dict,
            verbose=False,
        )
        flat_list = construct_call_tree(
            sub=sub,
            sub_dict=sub_dict,
            dtype_dict=type_dict,
            nested=0,
        )

        if sub.abstract_call_tree:
            for tree in sub.abstract_call_tree.traverse_postorder():
                subname = tree.node.subname
                childsub = sub_dict[subname]
                if childsub.library:
                    continue
                if not childsub.args_analyzed:
                    childsub.parse_arguments(sub_dict, type_dict)
                if not childsub.global_analyzed:
                    childsub.analyze_variables(sub_dict, type_dict)

    for sub in sub_dict.values():
        sub.match_arg_to_inst(type_dict)

    for sub in sub_dict.values():
        if sub.elmtype_access_by_ln:
            sub.summarize_readwrite()

    # for subname in fut_subs:
    #     sub = sub_dict[subname]
    #     prune_associate_clause(sub)

    return


if __name__ == "__main__":
    try:
        print("CALLING MAIN")
        create_unit_test()
    except Exception as e:
        import traceback

        traceback.print_exc()
