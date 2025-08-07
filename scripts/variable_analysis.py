from __future__ import annotations

import logging
import pprint
import re
import sys
from logging import Logger
from typing import TYPE_CHECKING, Dict

from scripts.logging_configs import get_logger, set_logger_level
from scripts.types import LineTuple, ModUsage

if TYPE_CHECKING:
    from scripts.analyze_subroutines import Subroutine

from scripts.fortran_modules import FortranModule
from scripts.utilityFunctions import Variable


def find_global_var_bounds(
    global_vars: dict[str, Variable],
    mod_dict: dict[str, FortranModule],
    logger: Logger,
):
    func_name = "(find_global_var_bounds)"

    regex_alloc = re.compile(r"^(allocate\b)")
    # sort globals by module
    sorted_gv_by_map: dict[str, list[str]] = {}
    for gv in global_vars.values():
        if gv.dim != 0 and not gv.bounds:
            sorted_gv_by_map.setdefault(gv.declaration, []).append(rf"{gv.name}")

    for mod, var_list in sorted_gv_by_map.items():

        fline_list = mod_dict[mod].module_lines
        var_str = "|".join(var_list)

        alloc_lines = [
            line for line in filter(lambda x: regex_alloc.search(x.line), fline_list)
        ]
        regex_var = re.compile(rf"\b({var_str})\b")
        var_lines = [
            line for line in filter(lambda x: regex_var.search(x.line), alloc_lines)
        ]

        for lpair in var_lines:
            line = lpair.line.strip()
            regex_var_and_bounds = re.compile(rf"({var_str})\s*(\(.+?\))")
            for match in regex_var_and_bounds.finditer(line):
                varname = match.group(1)
                bounds = match.group(2)
                global_vars[varname].bounds = bounds[1:-1]

    for gv in global_vars.values():
        if not gv.bounds and gv.dim > 0:
            gv.bounds = generate_dim_names(gv)
    return


def generate_dim_names(var: Variable) -> str:
    if var.dim == 0:
        sys.exit(1)
    bounds = [f"{var.name}_dim{i}" for i in range(1, var.dim + 1)]
    return ",".join(bounds)


def add_global_vars(
    dep_mod: FortranModule,
    vars: dict[str, Variable],
    mod_usage: ModUsage,
):
    """ """
    intrinsic_types = {"real", "integer", "logical", "character"}
    if mod_usage.all:
        vars.update(dep_mod.global_vars)
    else:
        for id in mod_usage.clause_vars:
            var = dep_mod.global_vars.get(id.obj)
            if var is None or var.type not in intrinsic_types:
                continue
            vars[id.obj] = var
    return


def check_global_vars(regex_variables, sub: Subroutine) -> set[str]:
    """
    Function that checks sub for usage of any variables matched by
    regex_variables.
    """
    func_name = "check_global_vars"
    sub_lines = sub.sub_lines
    fileinfo = sub.get_file_info()

    lines = [lpair for lpair in sub_lines if lpair.ln >= fileinfo.startln]

    matched_lines: list[LineTuple] = [
        lpair for lpair in filter(lambda x: regex_variables.search(x.line), lines)
    ]

    # Loop through subroutine line by line starting after the associate clause
    active_vars: set[str] = set()
    for lpair in matched_lines:
        match_var = regex_variables.findall(lpair.line)
        for var in match_var:
            if var not in active_vars:
                active_vars.add(var)
    return active_vars


def update_default_value(mod_dict: dict[str, FortranModule]):
    global_vars = {
        v: var for mod in mod_dict.values() for v, var in mod.global_vars.items()
    }

    def update_default(gv: Variable, seen=None) -> None:
        if seen is None:
            seen = set()

        val = gv.default_value
        if val in seen:  # prevent infinite loops
            return
        if val in global_vars:
            seen.add(val)
            gv.default_value = global_vars[val].default_value
            update_default(gv, seen)
        return

    for mod in mod_dict.values():
        for var in mod.global_vars.values():
            update_default(var)
    return


def determine_global_variable_status(
    mod_dict: dict[str, FortranModule],
    sub: Subroutine,
    verbose=False,
) -> None:
    """
    Function that goes through the list of subroutines and returns the non-derived type
    global variables that are used inside those subroutines

    Arguments:
        * mod_dict : dictionary of unit test modules
        * subroutines : list of Subroutine objects
    """
    func_name = "( determine_global_variables_status )"
    logger: Logger = get_logger("ActiveGlobals")
    set_logger_level(logger, logging.DEBUG)

    fileinfo = sub.get_file_info(all=True)

    # temp mod dict for only those related to this Sub
    test_modules: Dict[str, FortranModule] = {}
    modname = sub.module
    sub_mod = mod_dict[modname]
    update_default_value(mod_dict)

    variables: dict[str, Variable] = {}
    for mod_name, musage in sub_mod.head_modules.items():
        add_global_vars(dep_mod=mod_dict[mod_name], vars=variables, mod_usage=musage)

    sub_dep = sub_mod.sort_module_deps(startln=fileinfo.startln, endln=fileinfo.endln)
    for mod_name, musage in sub_dep.items():
        add_global_vars(dep_mod=mod_dict[mod_name], vars=variables, mod_usage=musage)

    intrinsic_types = ["real", "character", "logical", "integer"]
    variables.update(
        {
            key: val
            for key, val in sub_mod.global_vars.items()
            if val.type in intrinsic_types
        }
    )
    if not variables:
        return
    # Create regex from the possible variables
    var_string = "|".join(variables.keys())
    regex_variables = re.compile(r"\b({})\b".format(var_string), re.IGNORECASE)

    # Loop through the subroutines and check for variables used within.
    # `check_global_vars` loops through each sub and looks for any matches
    active_vars = check_global_vars(regex_variables, sub)
    if verbose:
        print(f"{func_name}::sub ", sub.name)
        print(f"{func_name}::test modules ", test_modules)
        print(f"{func_name}::active_vars :", active_vars)
    if active_vars:
        active_var_dict = {var: variables[var] for var in active_vars}
        find_global_var_bounds(active_var_dict, mod_dict, logger)
        for var in active_vars:
            variables[var].active = True
            sub.active_global_vars[var] = variables[var].copy()

    return
