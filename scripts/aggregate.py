import re
import sys
from pprint import pprint
from typing import NamedTuple

from scripts.analyze_subroutines import Subroutine
from scripts.DerivedType import DerivedType
from scripts.types import ReadWrite
from scripts.utilityFunctions import Variable


class DtypeVarTuple(NamedTuple):
    inst: str
    var: str
    dim: int


def aggregate_dtype_vars(
    sub_dict: dict[str, Subroutine],
    type_dict: dict[str, DerivedType],
    inst_to_dtype_map: dict[str, str],
):
    """
    Function aggregate_dtype_vars:
        Starting with a unit-test subroutine, traverse it's calltree and set elmtype vars to active
    """

    unit_test_subs: list[Subroutine] = [
        sub for sub in sub_dict.values() if sub.unit_test_function
    ]

    for sub in unit_test_subs:
        if sub.abstract_call_tree:
            for node in sub.abstract_call_tree.traverse_postorder():
                sub_name = node.node.subname
                node_sub = sub_dict[sub_name]
                set_active_variables(
                    type_dict,
                    inst_to_dtype_map,
                    node_sub.elmtype_access_summary,
                )
    check_inst_active_consistency(type_dict)
    set_bounds_active(type_dict, inst_to_dtype_map)
    return


def set_active_variables(
    type_dict: dict[str, DerivedType],
    type_lookup: dict[str, str],
    variable_list: dict[str, ReadWrite],
):
    """
    This function sets the active status of the user defined types
    based on variable list
        * type_dict   : dictionary of all user-defined types found in the code
        * type_lookup : dictionary that maps an variable to it's user-defined type
        * variable_list   : list of variables that are used
        * dtype_info_list : list for saving to file (redundant?)
    """
    regex_paren = re.compile(r"\((.+)\)")
    # Temporary fix
    variable_list = {key : rw for key, rw in variable_list.items() if len(key.split("%") )<= 2}
    variable_list = {key: rw for key, rw in variable_list.items() if type_lookup.get(key.split("%")[0])}

    instance_member_vars = [var for var in variable_list if "%" in var]
    for var in instance_member_vars:
        dtype, component = var.split("%")
        dtype = regex_paren.sub("", dtype)
        if "bounds" in dtype or "clumpfilter" in dtype:
            continue
        type_name = type_lookup.get(dtype)
        if not type_name:
            sys.exit(1)

        type_dict[type_name].active = True
        for field_var in type_dict[type_name].components.values():
            active = field_var.active

            if "%" in field_var.name:
                match = bool(field_var.name == var)
            else:
                match = bool(field_var.name == component)
            if match and field_var.pointer:
                activate_targets(field_var,type_dict,type_lookup)
            if match and not active:
                field_var.active = True

    # Set which instances of derived types are actually used.
    global_vars = {regex_paren.sub("", v.split("%")[0]) for v in instance_member_vars}
    global_vars = list(set(global_vars))
    for var in global_vars:
        if "bounds" == var:
            continue
        type_name = type_lookup.get( var )
        if not type_name:
            continue
        # Set which instances of the derived type are active
        for inst in type_dict[type_name].instances.values():
            if inst.name == var and not inst.active:
                inst.active = True

    return None


def check_inst_active_consistency(type_dict: dict[str, DerivedType]):
    """ """
    dtypes_with_active_fields = set()
    for dtype in type_dict.values():
        for field_var in dtype.components.values():
            if field_var.active:
                dtypes_with_active_fields.add(dtype.type_name)
    for dtype in type_dict.values():
        for inst in dtype.instances.values():
            if inst.active and dtype.type_name not in dtypes_with_active_fields:
                print(
                    f"WARNING -- {dtype.type_name}::{inst.name} active but no fields are!"
                )

    return


def set_bounds_active(type_dict: dict[str, DerivedType], inst_map: dict[str, str]):
    """
    Need to hard-code bounds info for subgrid initialization?
    """

    regex = re.compile(r"((beg|end)(g|l|t|c|p))\b")

    for field in type_dict["bounds_type"].components.values():
        if regex.match(field.name):
            field.active = True
    type_dict["bounds_type"].instances["bounds"] = Variable(
        type="bounds_type",
        name="bounds",
        dim=0,
        ln=0,
        declaration="decompmod",
        subgrid="?",
        active=True,
    )
    inst_map["bounds"] = "bounds_type"
    return

def activate_targets(field_var: Variable, type_dict: dict[str,DerivedType], inst_to_type_map: dict[str,str]):
    for target in field_var.pointer:
        inst, member_path = target.split("%",1)
        if re.match(r'(c13|c14)',inst):
            continue
        inst_type = inst_to_type_map[inst]
        dtype = type_dict[inst_type]
        dtype.instances[inst].active = True
        for component in dtype.components.values():
            if member_path == component.name:
                component.active = True

    return
