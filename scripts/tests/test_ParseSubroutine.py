import logging
import os
from pprint import pprint
from unittest.mock import patch

test_dir = os.path.dirname(__file__) + "/"
logger = logging.getLogger("TEST")
logging.basicConfig(level=logging.INFO)  # change to DEBUG to see detailed logs

expected_arg_status = {
    "test_parsing_sub": {
        "bounds": "r",
        "bounds%begg": "r",
        "bounds%endg": "r",
        "var1": "r",
        "var2": "r",
        "var3": "rw",
        "input4": "rw",
    },
    "add": {
        "x": "r",
        "y": "rw",
    },
    "ptr_test_sub": {
        "numf": "-",
        "soilc": "-",
        "arr": "w",
    },
    "tridiagonal_sr": {
        "bounds": "r",
        "bounds%begc": "r",
        "bounds%endc": "r",
        "lbj": "r",
        "ubj": "r",
        "jtop": "r",
        "numf": "r",
        "filter": "r",
        "a": "r",
        "b": "r",
        "c": "r",
        "r": "r",
        "u": "w",
        "is_col_active": "r",
    },
    "call_sub": {
        "numf": "r",
        "bounds": "r",
        "bounds%begc": "r",
        "bounds%endc": "r",
        "mytype": "rw",
        "mytype%field2": "w",
        "mytype%field1": "rw",
        "patch_state_updater%dwt": "w",
        "patch_state_updater": "w",
    },
    "col_nf_init": {
        "begc": "r",
        "endc": "r",
        "this": "w",
        "this%hrv_deadstemn_to_prod100n": "w",
        "this%hrv_deadstemn_to_prod10n": "w",
        "this%m_n_to_litr_lig_fire": "w",
        "this%m_n_to_litr_met_fire": "w",
    },
    "trace_dtype_example": {
        "mytype2": "rw",
        "mytype2%field1": "r",
        "mytype2%field2": "rw",
        "mytype2%field3": "r",
        "mytype2%field4": "rw",
        "mytype2%active": "r",
        "col_nf_inst": "w",
        "col_nf_inst%hrv_deadstemn_to_prod10n": "w",
        "flag": "r",
    },
}


def test_Functions():
    """
    Test for parsing functions
    """
    with patch("scripts.config.ELM_SRC", test_dir), patch(
        "scripts.config.SHR_SRC", test_dir
    ):
        from scripts.analyze_subroutines import Subroutine
        from scripts.config import scripts_dir
        from scripts.edit_files import modify_file
        from scripts.fortran_modules import get_module_name_from_file
        from scripts.utilityFunctions import Variable

        fn = f"{scripts_dir}/tests/example_functions.f90"

        SUB_NAME_DICT = {
            "landunit_is_special": Variable(
                type="logical", name="is_special", dim=0, subgrid="?", ln=-1
            ),
            "constructor": Variable(
                type="prior_weights_type", name="constructor", dim=0, subgrid="?", ln=-1
            ),
            "old_weight_was_zero": Variable(
                type="logical", name="old_weight_was_zero", dim=1, subgrid="?", ln=-1
            ),
            "cnallocate_carbon_only": Variable(
                type="logical", name="cnallocate_carbon_only", dim=0, subgrid="?", ln=-1
            ),
            "get_beg": Variable(
                type="integer", name="beg_index", dim=0, subgrid="?", ln=-1
            ),
        }

        with open(fn, "r") as ifile:
            lines = ifile.readlines()

        _, mod_name = get_module_name_from_file(fn)
        sub_dict = modify_file(
            lines,
            fn,
            sub_init_dict={},
            mod_name=mod_name,
            verbose=False,
            overwrite=False,
        )

        for key, ans in SUB_NAME_DICT.items():
            func: Subroutine = sub_dict[key]
            if func.func:
                func.print_subroutine_info()
                comp = func.result
                assert ans == comp, f"Failed for function: {key}"


def test_sub_parse(subtests):
    """
    Test for parsing function/subroutine calls
    """
    with patch("scripts.config.ELM_SRC", test_dir), patch(
        "scripts.mconfig.SHR_SRC", test_dir
    ):
        import scripts.dynamic_globals as dg
        from scripts.aggregate import aggregate_dtype_vars
        from scripts.analyze_subroutines import Subroutine
        from scripts.config import scripts_dir
        from scripts.DerivedType import DerivedType
        from scripts.edit_files import process_for_unit_test
        from scripts.fortran_modules import FortranModule
        from scripts.UnitTestforELM import (
            create_unit_test,
            process_subroutines_for_unit_test,
        )
        from scripts.utilityFunctions import Variable

        dg.populate_interface_list()
        fn = f"{scripts_dir}/tests/example_functions.f90"
        test_sub_name = "call_sub"
        sub_name_list = [test_sub_name]

        mod_dict: dict[str, FortranModule] = {}
        main_sub_dict: dict[str, Subroutine] = {}

        ordered_mods = process_for_unit_test(
            case_dir=test_dir,
            mod_dict=mod_dict,
            mods=[],
            required_mods=[],
            sub_dict=main_sub_dict,
            sub_name_list=sub_name_list,
            overwrite=False,
            verbose=False,
        )

        main_sub_dict[test_sub_name].unit_test_function = True

        type_dict: dict[str, DerivedType] = {}
        for mod in mod_dict.values():
            for utype, dtype in mod.defined_types.items():
                type_dict[utype] = dtype

        for dtype in type_dict.values():
            dtype.find_instances(mod_dict)

        bounds_inst = Variable(
            type="bounds_type",
            name="bounds",
            dim=0,
            subgrid="?",
            ln=-1,
        )
        type_dict["bounds_type"].instances["bounds"] = bounds_inst.copy()

        instance_to_user_type = {}
        instance_dict: dict[str, DerivedType] = {}
        for type_name, dtype in type_dict.items():
            for instance in dtype.instances.values():
                instance_to_user_type[instance.name] = type_name
                instance_dict[instance.name] = dtype

        process_subroutines_for_unit_test(
            mod_dict=mod_dict,
            sub_dict=main_sub_dict,
            type_dict=type_dict,
        )
        active_vars = main_sub_dict[test_sub_name].active_global_vars

        active_globals_fut: dict[str, Variable] = {}
        for sub in main_sub_dict.values():
            active_globals_fut.update(sub.active_global_vars)

        for var in main_sub_dict["call_sub"].active_global_vars.values():
            logger.info(
                f"Variable Info:\n"
                f"  name         : {var.name}\n"
                f"  declaration  : {var.declaration}\n"
                f"  bounds       : {var.bounds} ({bool(var.bounds)})\n"
                f"  dim          : {var.dim}\n"
                f"  ALLOCATABLE  : {var.allocatable}"
            )

        assert (
            len(active_vars) == 7
        ), f"Didn't correctly find the active global variables:\n{active_vars}"

        for subname in expected_arg_status:
            childsub = main_sub_dict[subname]
            test_dict = {
                k: rw.status for k, rw in childsub.arguments_read_write.items()
            }
            with subtests.test(msg=subname):
                assert expected_arg_status[subname] == test_dict

        for subname in expected_arg_status:
            childsub = main_sub_dict[subname]

        aggregate_dtype_vars(
            sub_dict=main_sub_dict,
            type_dict=type_dict,
            inst_to_dtype_map=instance_to_user_type,
        )

        active_set: set[str] = set()
        for inst_name, dtype in instance_dict.items():
            if not dtype.instances[inst_name].active:
                continue
            for field_var in dtype.components.values():
                if field_var.active:
                    active_set.add(f"{inst_name}%{field_var.name}")


def test_arg_intent():
    with patch("scripts.config.ELM_SRC", test_dir), patch(
        "scripts.config.SHR_SRC", test_dir
    ):
        import scripts.dynamic_globals as dg
        from scripts.analyze_subroutines import Subroutine
        from scripts.config import scripts_dir
        from scripts.edit_files import process_for_unit_test
        from scripts.fortran_modules import FortranModule, get_module_name_from_file

        dg.populate_interface_list()
        fn = f"{scripts_dir}/tests/example_functions.f90"
        mod_dict: dict[str, FortranModule] = {}
        main_sub_dict: dict[str, Subroutine] = {}

        mod_dict, file_list, main_sub_dict = process_for_unit_test(
            fname=fn,
            case_dir="./",
            mod_dict=mod_dict,
            mods=[],
            required_mods=[],
            sub_dict=main_sub_dict,
            overwrite=False,
            verbose=False,
        )
        print("Sorted mods:\n", [get_module_name_from_file(m) for m in file_list])

        assert 1 == 1
