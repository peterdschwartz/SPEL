"""
Generate SpelAccDataMod.F90: a manual OpenACC deep copy of the unit test's
derived-type instances, used instead of nvfortran's -gpu=deepcopy so that the
same code works with any OpenACC compiler.

  spel_acc_enter_data()       : once after elm_init. Shallow-copies each instance
                                (update device if the case source gives it an
                                `!$acc declare`, enter data copyin otherwise),
                                then copies in each associated member, which attaches
                                the device pointer inside the device struct.
  spel_acc_update_device()    : after the host reads a time step's inputs.
  spel_acc_update_host()      : before the host writes a time step's outputs.
  spel_acc_update_filters()   : after the filters are built. If the case's filterMod
                                builds them on the host, copies them to the device.
                                If it builds them in OpenACC compute regions (the
                                counts come back to the host through reductions),
                                copies the counts to the device and the index arrays
                                to the host instead, so both copies agree.

The member set matches ReadWriteMod: active, non-alias members of active instances,
with the same namelist guards.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import spel.scripts.io.helper as hio
from spel.scripts.io.netcdf_io import guard_fortran_lines

if TYPE_CHECKING:
    from spel.scripts.functional_unit_test import FunctionalUnitTest
    from spel.scripts.utilityFunctions import Variable

MOD_NAME = "SpelAccDataMod"
FILTER_INSTANCES = ("filter", "filter_inactive_and_active")
FILTER_MODULE_FILE = "filterMod.F90"


def _members(unit_test: FunctionalUnitTest):
    type_dict = unit_test.type_dict
    instance_to_type = unit_test.instance_to_type_map()
    active_instances, _, _ = hio.get_var_usage_and_elm_inst_vars(type_dict)

    per_inst: dict[str, list[tuple[str, Variable]]] = {}
    for inst_name, inst_var in sorted(active_instances.items()):
        if inst_var.type == "bounds_type":
            continue
        dtype = type_dict[instance_to_type[inst_name]]
        fields = []
        for field_var in dtype.components.values():
            if not field_var.active or field_var.pointer:
                continue
            fields.append((field_var.name.split("%")[-1], field_var))
        per_inst[inst_name] = sorted(fields, key=lambda t: t[0])
    return active_instances, per_inst


def _declared_instances(case_dir: Path, names) -> set[str]:
    """Instances that already have device storage through an `!$acc declare`."""
    text = "\n".join(p.read_text() for p in Path(case_dir).glob("*.F90"))
    clauses = re.findall(
        r"^\s*!\$acc\s+declare\s+\w+\s*\(([^)]*)\)", text, re.IGNORECASE | re.MULTILINE
    )
    declared = {
        n.strip().lower() for c in clauses for n in c.replace("&", ",").split(",")
    }
    return {n for n in names if n.lower() in declared}


def _use_lines(unit_test: FunctionalUnitTest, active_instances) -> list[str]:
    tabs = hio.indent(hio.Tab.reset)
    stmts = set()
    for var in active_instances.values():
        if var.type == "bounds_type":
            continue
        stmts.add(f"{tabs}use {var.declaration}, only : {var.name}\n")
    stmts.add(f"{tabs}use filterMod, only : {', '.join(FILTER_INSTANCES)}\n")
    # namelist guards compare strings with spel_streq (see Expectation.to_fortran)
    stmts.add(f"{tabs}use nc_io, only : spel_streq\n")
    guard_vars = [
        fv
        for dtype in unit_test.type_dict.values()
        for fv in dtype.components.values()
        if fv.active
    ]
    for stmt in unit_test.guard_use_statements_for(guard_vars):
        stmts.add(f"{tabs}{stmt}\n")
    return sorted(stmts)


def _member_block(inst, member, var, directive, guards, unit_test, tabs) -> list[str]:
    ref = f"{inst}%{member}"
    if var.dim > 0 or var.ptrscalar:
        lines = [
            f"{tabs}if (associated({ref})) then\n",
            f"{tabs}   !$acc {directive}({ref})\n",
            f"{tabs}end if\n",
        ]
    else:
        lines = [f"{tabs}!$acc {directive}({ref})\n"]
    cond = unit_test.guard_for_variable(var)
    return guard_fortran_lines(lines, cond, tabs) if cond is not None else lines


def _filter_arrays(unit_test: FunctionalUnitTest) -> tuple[list[str], list[str]]:
    ftype = unit_test.type_dict.get("clumpfilter")
    if ftype is None:
        return [], []
    arrays, scalars = [], []
    for fv in ftype.components.values():
        name = fv.name.split("%")[-1]
        (arrays if fv.dim > 0 else scalars).append(name)
    return sorted(arrays), sorted(scalars)


def filters_built_on_device(case_dir: Path) -> bool:
    """Whether the case's filterMod builds the filters in OpenACC compute regions."""
    path = Path(case_dir) / FILTER_MODULE_FILE
    if not path.exists():
        return False
    return bool(
        re.search(r"^\s*!\$acc\s+(parallel|kernels)\b", path.read_text(), re.I | re.M)
    )


def generate_acc_data_module(unit_test: FunctionalUnitTest) -> None:
    active_instances, per_inst = _members(unit_test)
    farrays, fscalars = _filter_arrays(unit_test)
    tabs = hio.indent(hio.Tab.reset)
    t2 = tabs * 2

    lines = [
        f"module {MOD_NAME}\n",
        f"{tabs}!!! Auto-generated by SPEL: manual OpenACC deep copy of unit-test data\n",
    ]
    lines.extend(_use_lines(unit_test, active_instances))
    lines.extend(
        [
            f"{tabs}implicit none\n",
            f"{tabs}private\n",
            f"{tabs}public :: spel_acc_enter_data, spel_acc_update_device\n",
            f"{tabs}public :: spel_acc_update_host, spel_acc_update_filters\n",
            "contains\n\n",
        ]
    )

    # --- enter data
    lines.append(f"{tabs}subroutine spel_acc_enter_data()\n")
    lines.append(f"{t2}integer :: i\n")
    lines.append("#ifdef _OPENACC\n")
    declared = _declared_instances(unit_test.case_dir, per_inst)
    for inst, fields in per_inst.items():
        lines.append(f"{t2}! {inst}\n")
        if inst in declared:
            lines.append(f"{t2}!$acc update device({inst})\n")
        else:
            lines.append(f"{t2}!$acc enter data copyin({inst})\n")
        for member, var in fields:
            if var.dim > 0 or var.ptrscalar:
                lines.extend(
                    _member_block(inst, member, var, "enter data copyin", None, unit_test, t2)
                )
    for finst in FILTER_INSTANCES if farrays or fscalars else ():
        lines.append(f"{t2}! {finst}\n")
        lines.append(f"{t2}if (allocated({finst})) then\n")
        lines.append(f"{t2}   !$acc enter data copyin({finst})\n")
        lines.append(f"{t2}   do i = 1, size({finst})\n")
        for name in farrays:
            ref = f"{finst}(i)%{name}"
            lines.append(f"{t2}      if (associated({ref})) then\n")
            lines.append(f"{t2}         !$acc enter data copyin({ref})\n")
            lines.append(f"{t2}      end if\n")
        lines.append(f"{t2}   end do\n")
        lines.append(f"{t2}end if\n")
    lines.append("#endif\n")
    lines.append(f"{tabs}end subroutine spel_acc_enter_data\n\n")

    # --- per-step updates
    for sub, directive in (
        ("spel_acc_update_device", "update device"),
        ("spel_acc_update_host", "update self"),
    ):
        lines.append(f"{tabs}subroutine {sub}()\n")
        lines.append("#ifdef _OPENACC\n")
        for inst, fields in per_inst.items():
            for member, var in fields:
                lines.extend(_member_block(inst, member, var, directive, None, unit_test, t2))
        lines.append("#endif\n")
        lines.append(f"{tabs}end subroutine {sub}\n\n")

    # --- filters
    on_device = filters_built_on_device(unit_test.case_dir)
    array_update = "update self" if on_device else "update device"
    lines.append(f"{tabs}subroutine spel_acc_update_filters()\n")
    lines.append(f"{t2}integer :: i\n")
    lines.append("#ifdef _OPENACC\n")
    if on_device:
        lines.append(
            f"{t2}! filterMod builds the filters on the device: the counts are on the\n"
            f"{t2}! host (reductions) and the index arrays on the device\n"
        )
    for finst in FILTER_INSTANCES:
        lines.append(f"{t2}if (allocated({finst})) then\n")
        lines.append(f"{t2}   do i = 1, size({finst})\n")
        for name in fscalars:
            lines.append(f"{t2}      !$acc update device({finst}(i)%{name})\n")
        for name in farrays:
            ref = f"{finst}(i)%{name}"
            lines.append(f"{t2}      if (associated({ref})) then\n")
            lines.append(f"{t2}         !$acc {array_update}({ref})\n")
            lines.append(f"{t2}      end if\n")
        lines.append(f"{t2}   end do\n")
        lines.append(f"{t2}end if\n")
    lines.append("#endif\n")
    lines.append(f"{tabs}end subroutine spel_acc_update_filters\n\n")

    lines.append(f"end module {MOD_NAME}\n")

    unit_test.logger.info(f"Writing {MOD_NAME}.F90")
    with open(unit_test.case_dir / f"{MOD_NAME}.F90", "w") as ofile:
        ofile.writelines(lines)
