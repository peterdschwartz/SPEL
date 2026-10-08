import logging
import re
import sys
from pathlib import Path
from pprint import pprint

import spel.scripts.config as cfg
import spel.scripts.db_utils as db_utils
import spel.scripts.dynamic_globals as dg
from spel.scripts.aggregate import aggregate_dtype_vars
from spel.scripts.analyze_subroutines import Subroutine
from spel.scripts.config import (
    default_mods,
    scripts_dir,
    spel_mods_dir,
    spel_output_dir,
    unittests_dir,
)
from spel.scripts.DerivedType import DerivedType
from spel.scripts.fortran_modules import (
    DECLARATION_ONLY_MODULES,
    FortranModule,
    get_filename_from_module,
)
from spel.scripts.fortran_parser.boolen_expression import ConditionExpectation
from spel.scripts.functional_unit_test import FunctionalUnitTest
from spel.scripts.helper_functions import construct_call_tree
from spel.scripts.logging_configs import get_logger
from spel.scripts.driver_callsites import DRIVER_MODULE, driver_callsites
from spel.scripts.module_resolver import ModuleScopes
from spel.scripts.record_access import (
    AccessMapper,
    ArgBinding,
    access_summary,
    dummy_actuals,
    elmtype_view,
    propagated_access,
    single_instance_actuals,
)
from spel.scripts.nml.analyze_ifs import get_if_blocks
from spel.scripts.nml.analyze_namelist import (
    check_sub_for_nml_guarded_vars,
    find_all_namelist,
    find_nml_ifs,
    nml_guards,
)
from spel.scripts.types import ReadWrite, UnitTestMode
from spel.scripts.utilityFunctions import Variable
from spel.scripts.validate_access import (
    access_manifest,
    merge_guards,
    write_access_manifest,
)
from spel.scripts.variable_analysis import determine_global_variable_status

ModDict = dict[str, FortranModule]
SubDict = dict[str, Subroutine]
TypeDict = dict[str, DerivedType]


def create_unit_test(
    sub_names: list[str],
    casename: str,
    keep: bool,
    db_mode: bool,
    reanalyze: bool = False,
    direct: bool = False,
) -> FunctionalUnitTest:
    """
    Create a Functional Unit Test in unittests_dir/{casename} for the
    subroutines in sub_names and return the (pickled) FunctionalUnitTest.

    The analysis always has elm_drv as its root: it is read from the cached
    elm_drv analysis (built first if missing, or when `reanalyze`), and the
    requested routines are extracted from it. `direct` analyzes only the
    requested routines instead (no cache; used by tests).
    """
    logger = get_logger("SPEL", level=logging.INFO)

    if not sub_names:
        sys.exit("Error- No subroutines provided for analysis")
    sub_name_list = [s.lower() for s in sub_names]
    case_dir = unittests_dir / (casename or "fut")

    logger.info(f"Creating UnitTest {case_dir.name} || {' '.join(sub_name_list)}")
    cfg.options.db_mode = db_mode

    if direct:
        prepare_case_dir(case_dir, keep, logger)
        unit_test = FunctionalUnitTest(casedir=case_dir, cfg=cfg.options, logger=logger)
        unit_test.write_meta_file()
        analyze_unit_test(unit_test, sub_name_list)
    else:
        from spel.scripts.analysis_cache import extract_unit_test, load_analysis

        analysis = load_analysis(reanalyze=reanalyze)
        prepare_case_dir(case_dir, keep, logger)
        unit_test = extract_unit_test(analysis, sub_name_list, case_dir, logger)
        unit_test.write_meta_file()

    finalize_unit_test(unit_test, db_mode)
    return unit_test


def prepare_case_dir(case_dir: Path, keep: bool, logger: logging.Logger) -> None:
    import os

    os.makedirs(f"{scripts_dir}/script-output", exist_ok=True)
    if not case_dir.is_dir():
        logger.info(f"Making case directory {case_dir}")
        case_dir.mkdir(parents=True)
    elif not keep:
        os.system(f"rm -rf {case_dir}/*")
        os.system(f"rm -f {scripts_dir}/*.pkl")
        os.system(f"rm -f {spel_output_dir}/*.F90")


def select_subroutines(
    sub_dict: SubDict, sub_name_list: list[str], logger: logging.Logger
) -> dict[str, Subroutine]:
    """Subroutines named `name` or `mod::name`, flagged as unit-test functions."""
    selected: dict[str, Subroutine] = {}
    for s in sub_name_list:
        if "::" in s:
            candidates = {s} if s in sub_dict else set()
        else:
            candidates = {k for k in sub_dict if re.search(rf"(?<=::){s}$", k)}
        if not candidates:
            sys.exit(f"Error- subroutine {s} not found")
        if len(candidates) > 1:
            logger.warning(
                f"Multiple Subroutines match {s}, Adding them all: {candidates}\n"
                "Re-run with <mod_name>::<sub_name>"
            )
        for c in sorted(candidates):
            selected[c] = sub_dict[c]
            selected[c].unit_test_function = True
    return selected


def build_type_dict(mod_dict: ModDict) -> TypeDict:
    type_dict: TypeDict = {}
    for mod in mod_dict.values():
        for utype, dtype in mod.defined_types.items():
            type_dict[utype] = dtype

    for dtype in type_dict.values():
        dtype.get_allocation_bounds(mod_dict)
    intrinsic_types = {"real", "integer", "character", "logical", "complex"}
    for mod in mod_dict.values():
        for var in mod.global_vars.values():
            if var.type not in intrinsic_types and var.type in type_dict:
                type_dict[var.type].instances[var.name] = var
    for dtype in type_dict.values():
        dtype.find_instances(mod_dict)

    bounds_inst = Variable(type="bounds_type", name="bounds", dim=0, subgrid="?", ln=-1)
    type_dict["bounds_type"].instances["bounds"] = bounds_inst.copy()
    return type_dict


def analyze_unit_test(
    unit_test: FunctionalUnitTest,
    sub_name_list: list[str],
    roots_from=None,
) -> None:
    """
    Edit (into unit_test.case_dir) and parse the modules needed by the
    routines in sub_name_list, then analyze the call trees of the unit-test
    roots: the selected routines, or `roots_from(unit_test)` if given.
    """
    from spel.scripts.edit_files import process_for_unit_test

    logger = unit_test.logger
    # Retrieve possible interfaces
    dg.populate_interface_list()
    main_sub_dict: SubDict = {}
    mod_dict: ModDict = {}
    # Process files by removing certain modules so that a standalone unit
    # test can be compiled. All file information is stored in `mod_dict`
    # and `main_sub_dict`
    ordered_mods = process_for_unit_test(
        case_dir=unit_test.case_dir,
        mod_dict=mod_dict,
        mods=[],
        required_mods=default_mods,
        sub_dict=main_sub_dict,
        sub_name_list=sub_name_list,
        overwrite=True,
        verbose=False,
    )
    if not mod_dict or not ordered_mods:
        logger.error("Error didn't find any modules related to subroutines")
        sys.exit(1)

    unit_test.subroutine_dict = main_sub_dict
    unit_test.module_dict = mod_dict
    unit_test.ordered_mods = [m for m in ordered_mods if m not in DECLARATION_ONLY_MODULES]
    unit_test.type_dict = build_type_dict(mod_dict)
    if roots_from is None:
        unit_test.primary_subroutines = select_subroutines(
            main_sub_dict, sub_name_list, logger
        )
    else:
        unit_test.primary_subroutines = roots_from(unit_test)
        for sub in unit_test.primary_subroutines.values():
            sub.unit_test_function = True
    main_sub_dict["filtermod::setfilters"].unit_test_function = True

    process_subroutines_for_unit_test(unit_test, keep_going=roots_from is not None)


def finalize_unit_test(unit_test: FunctionalUnitTest, db_mode: bool) -> None:
    """
    Root-dependent steps after analysis: namelist guards, active variables,
    and the generated files of the case; then pickle the unit test.
    """
    import os

    from spel.scripts.export_objects import pickle_unit_test
    from spel.scripts.fortran_modules import insert_header_for_unittest

    logger = unit_test.logger
    case_dir = unit_test.case_dir
    ordered_mods = unit_test.ordered_mods
    mod_dict = unit_test.module_dict
    type_dict = unit_test.type_dict
    instance_to_user_type = unit_test.instance_to_type_map()

    instance_dict: dict[str, DerivedType] = {}
    for type_name, dtype in unit_test.type_dict.items():
        for instance in dtype.instances.values():
            instance_dict[instance.name] = dtype

    fut_subs: set[str] = {
        sub.id
        for sub in unit_test.subroutine_dict.values()
        if sub.unit_test_function and sub.name != "setfilters"
    }
    for sub_id in fut_subs:
        sub_obj = unit_test.subroutine_dict[sub_id]
        if sub_obj.abstract_call_tree and "filter" not in sub_obj.name:
            unit_test.guarded_usage_dict = check_sub_for_nml_guarded_vars(
                root_sub=sub_obj,
                instance_dict=instance_dict,
            )

    aggregate_dtype_vars(
        sub_dict=unit_test.subroutine_dict,
        type_dict=unit_test.type_dict,
        inst_to_dtype_map=instance_to_user_type,
    )

    for sub in unit_test.primary_subroutines.values():
        for key in list(sub.elmtype_access_summary.keys()):
            if "c13" in key or "c14" in key:
                del sub.elmtype_access_summary[key]

    # Create a makefile for the unit test
    file_list = [get_filename_from_module(m) for m in ordered_mods]
    unit_test.generate_cmake(files=file_list)

    unittest_subs = {
        sub for sub in unit_test.subroutine_dict.values() if sub.unit_test_function
    }
    for sub in unittest_subs:
        if not sub.abstract_call_tree:
            continue
        for subnode in sub.abstract_call_tree.traverse_preorder():
            childsub = unit_test.subroutine_dict[subnode.node.subname]
            unit_test.active_global_vars.update(childsub.active_global_vars)

    if not db_mode:
        # Generate/modify FORTRAN files needed to initialize and run Unit Test
        unit_test.prepare_unit_test_files()
        # elm_instMod.F90
        unit_test.write_elminst_mod()
        # duplicateMod.F90
        unit_test.duplicate_clumps()
        unit_test.create_fortls()

        from spel.scripts.instrument_elm import make_case_sources_accessible

        for mod, names in make_case_sources_accessible(unit_test, Path(case_dir)).items():
            logger.info(f"Made {names} public in the unit-test copy of {mod}")

        # Go through all needed files and include a header that defines some constants
        insert_header_for_unittest(
            mod_list=ordered_mods,
            mod_dict=mod_dict,
            casedir=case_dir,
        )

        cmds: list[str] = [
            f"cp {spel_output_dir}/duplicateMod.F90 {case_dir}",
            f"cp {spel_mods_dir}/nc_io.F90 {case_dir}",
            f"cp {spel_mods_dir}/nc_allocMod.F90 {case_dir}",
            f"cp {spel_mods_dir}/unittest_defs.h {case_dir}",
            f"cp {spel_mods_dir}/decompInitMod.F90 {case_dir}",
            f"cp {spel_mods_dir}/check_config.sh {case_dir}",
        ]
        for cmd in cmds:
            logger.info(cmd)
            os.system(cmd)
        # Clean-up
        os.system(f"rm {spel_output_dir}/*.F90")

    logger.info("Finished -- Pickling results")

    write_access_manifest(
        case_dir,
        list(unit_test.primary_subroutines),
        access_manifest(unit_test.primary_subroutines.values()),
        merge_guards(
            (sub.elmtype_access_summary, nml_guards(sub))
            for sub in unit_test.primary_subroutines.values()
        ),
    )
    pickle_unit_test(unit_test)


def process_subroutines_for_unit_test(
    unit_test: FunctionalUnitTest, keep_going: bool = False
):
    """
    keep_going: a root whose call tree fails to analyze is recorded in
        unit_test.analysis_failures and skipped, instead of aborting.

    Function that processes the subroutines found in each FortranModule
        1) identify any non derived-type global vars used by Subroutine
        2) collect derived-type var and subroutine call info
        3) construct subroutine call trees (abstract=child subs represented only once)
        4) analyze status of variables used by subroutines.
    """
    from spel.scripts.nml.nml_queries import query_active_variables

    sub_dict = unit_test.subroutine_dict
    mod_dict = unit_test.module_dict
    type_dict = unit_test.type_dict

    fut_subs: set[str] = {sub.id for sub in sub_dict.values() if sub.unit_test_function}
    nml_dict = find_all_namelist()

    active_global_variables: dict[str, Variable] = {}
    for sub in sub_dict.values():
        determine_global_variable_status(mod_dict, sub)
        active_global_variables.update(sub.active_global_vars)

    nml_dict = {
        k: nml_dict[k] for k in nml_dict.keys() & active_global_variables.keys()
    }
    for nml in nml_dict:
        nml_dict[nml].variable = active_global_variables[nml]

    for dtype in type_dict.values():
        if dtype.init_sub_name:
            dtype.init_sub_ptr = sub_dict[dtype.init_sub_name]

    if False:  # not cfg.options.db_mode:
        ok = query_active_variables(sub_dict)
        if not ok:
            sys.exit("Expected database to return non-empty result")
        for sub in sub_dict.values():
            sub.summarize_readwrite(verbose=True)
    else:
        # one module-scope cache shared by all routine walks
        scopes = ModuleScopes(mod_dict, sub_dict)
        failures = unit_test.analysis_failures if keep_going else None
        mapper = AccessMapper(sub_dict, scopes, failures=failures)
        # roots in the given order: a root already mapped (or failed) as part
        # of an earlier root's call tree is not re-analyzed
        order = list(unit_test.primary_subroutines) + sorted(
            fut_subs - unit_test.primary_subroutines.keys()
        )
        for sub_id in order:
            sub = sub_dict[sub_id]
            if sub.record_access is not None or sub_id in unit_test.analysis_failures:
                continue
            try:
                analyze_call_tree(sub, unit_test, scopes, mapper)
            except (Exception, SystemExit) as err:
                if not keep_going:
                    raise
                mapper.record_failure(sub, err)
        if keep_going:
            unit_test.analysis_incomplete = incomplete_analyses(
                sub_dict, unit_test.analysis_failures
            )

        # roots: dummies bound to what elm_drv passes at each call site.
        # setfilters stays a special case: not bound at elm_drv; elm_drv
        # itself (the cache's analysis root) isn't called by anything.
        roots = [
            sub_dict[s]
            for s in fut_subs
            if sub_dict[s].record_access is not None
            and sub_dict[s].name != "setfilters"
            and sub_dict[s].module != DRIVER_MODULE
        ]
        _, drv_bindings, unit_test.driver_sites = driver_callsites(roots, sub_dict, mod_dict)
        unit_test.driver_bindings = drv_bindings
        adopt_record_access(sub_dict, type_dict, fut_subs, drv_bindings)

        if nml_dict:
            find_nml_ifs(sub_dict, nml_dict)

    return


def analyze_call_tree(
    sub: Subroutine, unit_test: FunctionalUnitTest, scopes: ModuleScopes, mapper: AccessMapper
) -> None:
    """Build sub's call tree, then walk and map every routine in it (leaves first)."""
    sub_dict = unit_test.subroutine_dict
    sub.collect_var_and_call_info(
        dtype_dict=unit_test.type_dict,
        sub_dict=sub_dict,
        mod_dict=unit_test.module_dict,
        verbose=False,
    )
    construct_call_tree(
        sub=sub,
        sub_dict=sub_dict,
        dtype_dict=unit_test.type_dict,
        mod_dict=unit_test.module_dict,
        nested=0,
        failures=mapper.failures,
    )
    if not sub.abstract_call_tree:
        return
    for tree in sub.abstract_call_tree.traverse_postorder():
        sub_obj = sub_dict[tree.node.subname]
        if sub_obj.library or sub_obj.record_access is not None:
            continue
        if mapper.failures is not None and sub_obj.id in mapper.failures:
            continue
        # leaf -> parent: children are walked before their callers
        try:
            if not sub_obj.ifs_analyzed:
                get_if_blocks(sub_obj)
            sub_obj.walk_syntax_tree(scopes)
        except (Exception, SystemExit) as err:
            if mapper.failures is None:
                raise
            mapper.record_failure(sub_obj, err)
            continue
        mapper.maps(sub_obj)


def incomplete_analyses(
    sub_dict: dict[str, Subroutine], failures: dict[str, str]
) -> dict[str, list[str]]:
    """Mapped routines whose call tree contains a failed routine -> those routines."""
    out: dict[str, list[str]] = {}
    for sub in sub_dict.values():
        if sub.record_access is None or not sub.abstract_call_tree:
            continue
        failed = {
            t.node.subname
            for t in sub.abstract_call_tree.traverse_postorder()
            if t.node.subname in failures
        }
        if failed:
            out[sub.id] = sorted(failed)
    return out


def adopt_record_access(
    sub_dict: dict[str, Subroutine],
    type_dict: dict[str, DerivedType],
    roots: set[str],
    drv_bindings: list[ArgBinding],
) -> None:
    """
    Fill the access fields of every mapped routine from its record_access.
    A root's derived-type dummies are bound to the globals elm_drv passes
    (setfilters excepted), else to their type's instance if it has only one.
    """
    instances = {name: list(dtype.instances) for name, dtype in type_dict.items()}
    pointer_components = {
        f"{inst}%{name}": list(comp.pointer)
        for dtype in type_dict.values()
        for name, comp in dtype.components.items()
        if comp.pointer
        for inst in dtype.instances
    }
    propagated = propagated_access(sub_dict)
    for sub in sub_dict.values():
        maps = sub.record_access
        if maps is None:
            continue
        actuals: dict[str, set[str]] = {}
        if sub.id in roots:
            if sub.name != "setfilters":
                actuals = dummy_actuals(drv_bindings, sub.id)
            arg_types = {
                name: arg.type
                for name, arg in sub.arguments.items()
                if name not in actuals and arg.type in type_dict
            }
            actuals |= single_instance_actuals(arg_types, instances)
            if unbound := sorted(arg_types.keys() - actuals.keys()):
                sub.logger.warning(
                    f"{sub.id}: derived-type dummies not bound to a global: {unbound}"
                )
        sub.elmtype_access_by_ln = elmtype_view(maps, actuals, pointer_components)
        sub.elmtype_access_summary = access_summary(sub.elmtype_access_by_ln)
        sub.arg_access_by_ln = {k: list(v) for k, v in maps.args.items()}
        sub.local_vars_access_by_ln = {k: list(v) for k, v in maps.locals.items()}
        sub.propagated_access_by_ln = propagated.get(sub.id, {})
