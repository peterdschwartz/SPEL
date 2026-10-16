import logging
import os
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from pprint import pformat
from unittest.mock import patch

import spel.scripts.dynamic_globals as dg
import spel.scripts.interfaces as interfaces
from spel.scripts.analyze_subroutines import Subroutine
from spel.scripts.config import Options, scripts_dir
from spel.scripts.DerivedType import DerivedType
from spel.scripts.driver_callsites import driver_callsites
from spel.scripts.fortran_modules import FortranModule
from spel.scripts.functional_unit_test import FunctionalUnitTest
from spel.scripts.tests.fixtures.expected_parse_results import line_of
from spel.scripts.types import ReadWrite
from spel.scripts.UnitTestforELM import finalize_unit_test
from spel.scripts.utilityFunctions import Variable

test_dir = os.path.dirname(__file__) + "/fixtures/"
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


@contextmanager
def elm_src_pointing_to(src_dir: Path):
    """
    Temporarily point SPEL at `src_dir` instead of the real ELM source.

    Modules bind ELM_SRC/SHR_SRC at import time, so every already-imported
    spel module is patched (all spel modules are imported at the top of this
    file, before patching). Module-level caches are swapped for empty ones so
    results computed against `src_dir` don't leak into other tests.
    """
    with ExitStack() as stack:
        for name, mod in list(sys.modules.items()):
            if not name.startswith("spel.") or mod is None:
                continue
            for attr in ("ELM_SRC", "SHR_SRC","E3SM_SRCROOT"):
                if hasattr(mod, attr):
                    stack.enter_context(patch.object(mod, attr, src_dir))

        stack.enter_context(patch.object(dg, "interface_list", []))
        stack.enter_context(patch.object(dg, "map_module_name_to_fpath", {}))
        stack.enter_context(patch.object(dg, "map_fpath_to_module_name", {}))
        stack.enter_context(patch.object(dg, "map_module_lines", {}))
        stack.enter_context(patch.object(dg, "map_module_head", {}))
        stack.enter_context(patch.object(interfaces, "_interface_procs_cache", {}))
        yield


def test_sub_parse(subtests):
    """
    Test for parsing function/subroutine calls
    """
    with elm_src_pointing_to(Path(test_dir)):
        dg.populate_interface_list()
        fn = f"{test_dir}/example_functions.f90"
        test_sub_name = "test_sub_parse::call_sub"
        sub_name_list = [test_sub_name]
        from spel.scripts.analysis_cache import extract_unit_test, load_analysis

        analysis = load_analysis(reanalyze=True)
        unit_test = extract_unit_test(analysis, sub_name_list, case_dir=Path(test_dir)/"fut", logger=logger)
        unit_test.write_meta_file()

        active_vars = unit_test.subroutine_dict[test_sub_name].active_global_vars

        modname = "test_sub_parse"
        names = {"host_subroutine", "sub_program", "sub_func1"}

        for n in names:
            sub = unit_test.subroutine_dict[f"{modname}::{n}"]
            sub.logger.warning("=" * 10)
            sub.logger.warning(f"fileinfo: {sub.get_file_info()}")
            sub.logger.warning(f"{pformat(sub.sub_lines)}")

        with subtests.test(msg="call-tree-routines-are-walked"):
            fut = unit_test.subroutine_dict[test_sub_name]
            walked = set()
            for node in fut.abstract_call_tree.traverse_postorder():
                sub = unit_test.subroutine_dict[node.node.subname]
                if sub.library:
                    continue
                assert sub.record is not None, f"{sub.id} not walked"
                assert sub.record_access is not None, f"{sub.id} not mapped"
                walked.add(sub.name)
            assert "call_sub" in walked
            assert {"tridiagonal", "tridiagonal_sr"} & {
                c.name for c in fut.record.calls
            }

        with subtests.test(msg="root-bound-at-driver-call-site"):
            fut = unit_test.subroutine_dict[test_sub_name]
            drv_lines = Path(f"{test_dir}/elm_driver.F90").read_text().splitlines()
            call_ln = next(i for i, l in enumerate(drv_lines) if "call call_sub" in l)
            access, _, _ = driver_callsites([fut], unit_test.subroutine_dict, unit_test.module_dict)
            access = access[fut.id]
            # globals passed by elm_drv take the dummies' status at the call line
            assert {"unused_inst%field2", "patch_state_updater%dwt"} <= access.keys()
            assert "w" in access["unused_inst%field2"][0].status
            assert all(rw.ln == call_ln for rws in access.values() for rw in rws)

        with subtests.test(msg="internal-subprograms"):
            host = unit_test.subroutine_dict[f"{modname}::host_subroutine"]
            internal = {
                n: unit_test.subroutine_dict[f"{modname}::{n}"]
                for n in ("sub_program", "sub_func1")
            }
            # 0-based lines in example_functions.F90
            expected_lns = tuple(
                line_of(s) - 1
                for s in (
                    "subroutine host_subroutine(x,y,z)",
                    "end subroutine host_subroutine",
                )
            )
            assert (host.startline, host.end_stmt_ln) == expected_lns
            assert host.contains_ln == host.startline + 7
            assert host.endline == host.contains_ln
            assert not host.is_internal and host.host is None
            assert host.syntax_tree is not None
            assert [s.name for s in host.syntax_tree.contains] == [
                "sub_program",
                "sub_func1",
            ]
            assert max(lt.ln for lt in host.sub_lines) <= host.contains_ln
            assert host.sub_lines[-1].line.strip() == "contains"
            assert host.internal_subs == internal
            for sub in internal.values():
                assert sub.is_internal and sub.host is host
                assert sub.contains_ln is None and sub.endline == sub.end_stmt_ln
                assert host.contains_ln < sub.startline < sub.endline < host.end_stmt_ln
                assert sub.sub_lines[-1].line.startswith("end ")

        with subtests.test(msg="interface-bodies-are-not-routines"):
            for n in ("soil_hk_interface", "soil_suction_interface"):
                assert f"{modname}::{n}" not in unit_test.subroutine_dict
            # routines after the abstract interface block are still found
            assert f"{modname}::host_subroutine" in unit_test.subroutine_dict
            mod_globals = unit_test.module_dict[modname].global_vars
            for dummy in ("this", "iface_smp", "iface_hk", "iface_sat"):
                assert dummy not in mod_globals

        with subtests.test(msg="sub-lines-end-at-END-statement"):
            for sub in unit_test.subroutine_dict.values():
                if sub.library or sub.contains_ln is not None:
                    continue
                last = sub.sub_lines[-1].line.strip()
                assert last.startswith("end ") and sub.name in last, f"{sub.id}: {last}"

        active_globals_fut: dict[str, Variable] = {}
        for sub in unit_test.subroutine_dict.values():
            active_globals_fut.update(sub.active_global_vars)

        for var in unit_test.subroutine_dict[test_sub_name].active_global_vars.values():
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

        finalize_unit_test(unit_test,db_mode=False)
        test_sub = unit_test.subroutine_dict[test_sub_name].sub_lines

        def pstatus(rw: ReadWrite):
            return f"{rw.status}@{rw.ln+1}"

        from spel.scripts.tests.expected_parse_results import (
            args,
            elmtypes,
            expected_access,
        )

        for sub_obj in unit_test.subroutine_dict.values():
            if sub_obj.id in expected_access:
                if expected_access[sub_obj.id].get(elmtypes):
                    test_dict = {
                        k: set(map(pstatus, status))
                        for k, status in sub_obj.elmtype_access_by_ln.items()
                    }
                    with subtests.test(msg=f"{sub_obj.id}-elmtypes"):
                        assert expected_access[sub_obj.id][elmtypes] == test_dict

