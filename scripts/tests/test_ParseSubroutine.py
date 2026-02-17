import logging
import os
from pprint import pformat
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


def test_sub_parse(subtests):
    """
    Test for parsing function/subroutine calls
    """
    with patch("scripts.config.ELM_SRC", test_dir), patch(
        "scripts.config.SHR_SRC", test_dir
    ):
        import scripts.dynamic_globals as dg
        from scripts.aggregate import aggregate_dtype_vars
        from scripts.analyze_subroutines import Subroutine
        from scripts.config import scripts_dir
        from scripts.DerivedType import DerivedType
        from scripts.edit_files import process_for_unit_test
        from scripts.fortran_modules import FortranModule
        from scripts.UnitTestforELM import process_subroutines_for_unit_test
        from scripts.utilityFunctions import Variable
        from scripts.types import ReadWrite

        dg.populate_interface_list()
        fn = f"{scripts_dir}/tests/example_functions.f90"
        test_sub_name = "test_sub_parse::call_sub"
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

        modname = "test_sub_parse"
        names = {'host_subroutine','sub_program','sub_func1'}

        for n in names:
            sub = main_sub_dict[f"{modname}::{n}"]
            sub.logger.warning("="*10)
            sub.logger.warning(f"fileinfo: {sub.get_file_info()}")
            sub.logger.warning(f"{pformat(sub.sub_lines)}")

        active_globals_fut: dict[str, Variable] = {}
        for sub in main_sub_dict.values():
            active_globals_fut.update(sub.active_global_vars)

        for var in main_sub_dict[test_sub_name].active_global_vars.values():
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

        aggregate_dtype_vars(
            sub_dict=main_sub_dict,
            type_dict=type_dict,
            inst_to_dtype_map=instance_to_user_type,
        )
        test_sub = main_sub_dict[test_sub_name].sub_lines
        # str_ = "\n".join([f"{lt.ln+1} {lt.line}" for lt in test_sub])
        # print(f"{pformat(str_)}")
        def pstatus(rw:ReadWrite):
            return f"{rw.status}@{rw.ln+1}"

        from scripts.tests.expected_parse_results import expected_access,elmtypes,args
        for sub_obj in main_sub_dict.values():
            if sub_obj.id in expected_access:
                if expected_access[sub_obj.id].get(elmtypes):
                    test_dict = {
                        k: set(map(pstatus, status))
                        for k, status in sub_obj.elmtype_access_by_ln.items()
                    }
                    with subtests.test(msg=f"{sub_obj.id}-elmtypes"):
                        assert expected_access[sub_obj.id][ elmtypes ] == test_dict
                # if expected_access[sub_obj.id].get("args"):
                #     test_dict = {
                #         k: set(map(pstatus,status))
                #         for k, status in sub_obj.arg_access_by_ln.items()
                #     }
                #     with subtests.test(msg=f"{sub_obj.id}--arguments"):
                #         assert expected_access[sub_obj.id][args] == test_dict


        active_set: set[str] = set()
        for inst_name, dtype in instance_dict.items():
            if not dtype.instances[inst_name].active:
                continue
            for field_var in dtype.components.values():
                if field_var.active:
                    active_set.add(f"{inst_name}%{field_var.name}")
