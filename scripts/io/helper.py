import re
import sys
from enum import Enum

from scripts.DerivedType import DerivedType
from scripts.utilityFunctions import Variable

TAB_WIDTH = 2
level = 1

Tab = Enum("Tab", ["shift", "unshift", "reset", "get"])
IOMode = Enum("IOMode", ["read", "write"])
VarDict = dict[str, Variable]
TypeDict = dict[str, DerivedType]

def sanitize_netcdf_name(name: str):
    name = name.replace(")", "").replace("(", "").strip()
    # Replace any illegal character with '_'
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name)
    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def get_subgrid(dim_str: str) -> str:
    """
    Function to return generic name for subgrid names
    """
    subgrids = re.findall(r"(?<=beg|end)(g|c|p|t|l)", dim_str)
    if not subgrids:
        # dim_str = re.search(r"(\w+)(?::\w+)?", dim_str).group()
        return sanitize_netcdf_name(dim_str)
    sg_set: set[str] = set(subgrids)
    assert len(sg_set) == 1, "Variable is allocated over multiple subgrids!"
    s = sg_set.pop()
    match (s):
        case "p":
            return "patch"
        case "c":
            return "column"
        case "l":
            return "landunit"
        case "t":
            return "topo"
        case "g":
            return "gridcell"
        case _:
            print("(get_subgrid) Unexpected subgrid somehow")
            sys.exit(1)


def var_use_statements(
    var_dict: VarDict, type_dict: TypeDict = {}
) -> tuple[list[str], list[tuple[str, Variable]]]:
    """
    generate use statments for dict
    """
    lines: set[str] = set()
    tabs = indent()
    elm_inst_vars: list[tuple[str, Variable]] = []

    for var in var_dict.values():
        if var.name == "bounds":
            stmt = f"{tabs}use {var.declaration}, only : {var.type}\n"
        else:
            if var.declaration == "elm_instmod":
                type_mod = type_dict[var.type].declaration
                elm_inst_vars.append((type_mod, var))
                stmt = f"{tabs}use {type_mod}, only: {var.type}\n"
            else:
                stmt = f"{tabs}use {var.declaration}, only : {var.name}\n"
        lines.add(stmt)

    return (list(lines), elm_inst_vars)


def get_var_usage_and_elm_inst_vars(
    type_dict: TypeDict,
) -> tuple[VarDict, list[str], list[tuple[str, Variable]]]:

    def active_mask(dtype: DerivedType, inst_name: str) -> bool:
        all_ptrs = bool(
            len([field for field in dtype.components.values() if not field.pointer])
            == 0
        )
        return (
            not all_ptrs
            and inst_name not in ["filter", "filter_inactive_and_active",]
            and not re.match("(c13|c14)", inst_name)
        )

    active_instances = {
        inst_var.name: inst_var
        for dtype in type_dict.values()
        for inst_var in dtype.instances.values()
        if inst_var.active and active_mask(dtype, inst_var.name)
    }
    use_statements, elminst_vars = var_use_statements(active_instances, type_dict)
    return active_instances, use_statements, elminst_vars


def indent(mode: Tab = Tab.get, num=1):
    global level
    global TAB_WIDTH
    match mode:
        case Tab.shift:
            level += num
        case Tab.unshift:
            level -= num
        case Tab.reset:
            level = 1
        case Tab.get:
            pass
    return " " * TAB_WIDTH * level
