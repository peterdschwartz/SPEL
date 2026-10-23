from __future__ import annotations

import re
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import spel.scripts.io.helper as hio
from spel.scripts import logging_configs
from spel.scripts.DerivedType import field_name
from spel.scripts.config import spel_mods_dir
from spel.scripts.DerivedType import DerivedType
from spel.scripts.fortran_parser.boolen_expression import ConditionExpectation
from spel.scripts.utilityFunctions import Variable

if TYPE_CHECKING:
    from spel.scripts.functional_unit_test import FunctionalUnitTest


Tab = hio.Tab
type GuardMap = Mapping[str, ConditionExpectation]


def guard_fortran_lines(
    lines: list[str],
    condition: ConditionExpectation | None,
    indent: str,
) -> list[str]:
    if condition is None:
        return lines

    inner = indent + "  "

    return [
        f"{indent}if ({condition.to_fortran()}) then\n",
        *[f"{inner}{line.lstrip()}" if line.strip() else line for line in lines],
        f"{indent}end if\n",
    ]


def is_deferred_shape(var: Variable) -> bool:
    """
    `x(:)`/`x(:,:)` arrays (allocatable or pointer), as opposed to `x(ngases,3)`.
    Derived-type components count as deferred: their `bounds` holds the dimension
    names from their allocate statement (e.g. 'topo'), not the declaration.
    """
    if var.dim == 0:
        return False
    if "%" in var.name:
        return True
    bounds = var.bounds.strip().strip("()")
    return not bounds or all(b.strip() == ":" for b in bounds.split(","))


def alloc_status(var: Variable) -> str | None:
    """
    allocated()/associated() test for deferred-shape module arrays, which ELM may
    leave unallocated (e.g. AllocationMod::aroot is only allocated with crop_prog)
    """
    if "%" in var.name or not is_deferred_shape(var):
        return None
    return f"associated({var.name})" if var.ptrscalar else f"allocated({var.name})"


def if_block(lines: list[str], cond: str | None, indent: str) -> list[str]:
    if cond is None:
        return lines
    inner = indent + "  "
    return [
        f"{indent}if ({cond}) then\n",
        *[f"{inner}{line.lstrip()}" if line.strip() else line for line in lines],
        f"{indent}end if\n",
    ]


def create_nc_define_vars(
    vars: dict[str, Variable],
    time: bool,
    elminst_vars: list[tuple[str, Variable]] | None = None,
    bounds: bool = False,
    guards: GuardMap | None = None,
) -> list[str]:
    """
    Create the subroutine used to define NetCDF variables.
    """
    if elminst_vars is None:
        elminst_vars = []

    if guards is None:
        guards = {}

    tabs = hio.indent()
    arg_str = ", bounds" if bounds else ""

    decls: list[str] = []

    for _, inst in elminst_vars:
        arg_str += f", {inst.name}"
        decls.append(f"{tabs}{tabs}type({inst.type}), intent(in) :: {inst.name}")

    lines: list[str] = [f"{tabs}subroutine define_vars(ncid{arg_str})\n"]

    tabs = hio.indent(hio.Tab.shift)

    lines.append(f"{tabs}integer, intent(in) :: ncid\n")

    if bounds:
        lines.append(f"{tabs}type(bounds_type), intent(in) :: bounds\n")

    lines.extend(f"{decl}\n" for decl in decls)

    lines.append(f"{tabs}integer :: varid, time_id\n")
    lines.append(f"{tabs}character(len=32), dimension(5) :: dim_names\n")

    if time:
        lines.append(
            f"{tabs}call check("
            "nf90_def_dim(ncid, 'time', NF90_UNLIMITED, time_id)"
            ")\n"
        )

    lines.extend(
        create_nc_def(
            vars,
            time=time,
            guards=guards,
        )
    )

    tabs = hio.indent(hio.Tab.unshift)
    lines.append(f"{tabs}end subroutine define_vars\n")

    return lines


def generate_elmtypes_io_netcdf(unit_test: FunctionalUnitTest) -> None:
    tabs = hio.indent(hio.Tab.reset)

    filename = "ReadWriteMod.F90"
    mod_name = filename.removesuffix(".F90")

    logger = unit_test.logger

    type_dict = unit_test.type_dict
    case_dir = unit_test.case_dir
    instance_to_type = unit_test.instance_to_type_map()

    active_instances, use_statements, elminst_vars = (
        hio.get_var_usage_and_elm_inst_vars(type_dict)
    )

    dtype_vars: dict[str, Variable] = {}
    guards: dict[str, ConditionExpectation] = {}

    for inst_var in active_instances.values():
        type_name = instance_to_type[inst_var.name]
        dtype = type_dict[type_name]

        for field_var in dtype.components.values():
            if not field_var.active or field_var.pointer or field_var.type in type_dict:
                continue  # a derived-type component is captured through its flattened fields

            new_var = field_var.copy()
            new_var.name = f"{inst_var.name}%{field_name(field_var)}"

            dtype_vars[new_var.name] = new_var

            condition = unit_test.guard_for_variable(field_var)
            if inst_var.name == "params_inst":
                unit_test.logger.info(f"{ condition }")
            if condition is not None:
                guards[new_var.name] = condition

    # elm_drv arguments the routines take (nextsw_cday, declinp1): one value per step
    drv_args = getattr(unit_test, "driver_args", {})
    for name, typ in drv_args.items():
        # logicals are stored as integers (0/1)
        drv_var = Variable("integer" if typ == "logical" else typ, f"spel_drv_{name}", "?", 0, 1, bounds="(1)")
        dtype_vars[drv_var.name] = drv_var
    # time-step scalars (spel.scripts.io.time_scalars): one value per step
    from spel.scripts.io.time_scalars import unit_test_time_scalars

    tscalars = unit_test_time_scalars(unit_test)
    unit_test.time_scalars = tscalars
    for ts in tscalars:
        dtype_vars[ts.array] = Variable(ts.io_type, ts.array, "?", 0, 1, bounds="(1)")

    lines: list[str] = [
        f"module {mod_name}\n",
        f"{tabs}!!! Auto-generated Fortran code for netcdf-fortran I/O\n",
        f"{tabs}use netcdf\n",
        f"{tabs}use nc_io\n",
        f"{tabs}use nc_allocMod\n",
    ]

    lines.extend(sorted(use_statements))

    # Import namelist variables required by guards.
    lines.extend(
        f"{tabs}{stmt}\n"
        for stmt in sorted(
            unit_test.guard_use_statements_for(
                [
                    field_var
                    for dtype in type_dict.values()
                    for field_var in dtype.components.values()
                    if field_var.active
                ]
            )
        )
    )

    lines.extend(
        [
            f"{tabs}implicit none\n",
            f"{tabs}public :: read_elmtypes, write_elmtypes, define_vars\n",
            f"{tabs}public :: spel_inbuf_to_device\n",
            f"{tabs}logical, parameter :: verbose = .false.\n",
        ]
    )
    lines.extend(create_inbuf_decls(dtype_vars))
    # set by the ELM capture module from elm_drv's arguments, read by the unit test
    for name, typ in drv_args.items():
        ftype = "real(8)" if typ == "real" else "integer"
        lines.append(f"{tabs}{ftype}, public :: spel_drv_{name}(1) = 0\n")
    # set by the ELM capture module from the time step, copied back by main.F90
    for ts in tscalars:
        ftype = "real(8)" if ts.io_type == "real" else "integer"
        lines.append(f"{tabs}{ftype}, public :: {ts.array}(1) = 0\n")
    lines.append("contains\n")
    lines.extend(create_inbuf_to_device(dtype_vars))

    lines.extend(
        create_nc_define_vars(
            dtype_vars,
            time=True,
            elminst_vars=elminst_vars,
            guards=guards,
            bounds=True,
        )
    )

    lines.extend(
        create_netcdf_io_routine(
            mode=hio.IOMode.read,
            sub_name="read_elmtypes",
            vars=dtype_vars,
            time=True,
            elminst_vars=elminst_vars,
            guards=guards,
        )
    )

    lines.extend(
        create_netcdf_io_routine(
            mode=hio.IOMode.write,
            sub_name="write_elmtypes",
            vars=dtype_vars,
            time=True,
            elminst_vars=elminst_vars,
            guards=guards,
        )
    )

    lines.append(f"end module {mod_name}\n")

    logger.info(f"Writing {filename}")

    with open(case_dir / filename, "w") as ofile:
        ofile.writelines(lines)


def generate_constants_io_netcdf(
    unit_test: FunctionalUnitTest,
) -> None:
    tabs = hio.indent(hio.Tab.reset)

    filename = "FUTConstantsMod.F90"
    mod_name = filename.removesuffix(".F90")

    vars = unit_test.non_parameter_global_vars
    case_dir = unit_test.case_dir

    guard_vars = unit_test.guard_variables()
    regular_vars = {name: var for name, var in vars.items() if name not in guard_vars}

    guards: GuardMap = {
        var.name: condition
        for var in regular_vars.values()
        if (condition := unit_test.guard_for_variable(var)) is not None
    }

    lines: list[str] = [
        f"module {mod_name}\n",
        f"{tabs}!!! Auto-generated Fortran code for netcdf-fortran I/O\n",
        f"{tabs}use netcdf\n",
        f"{tabs}use nc_io\n",
        f"{tabs}use nc_allocMod\n",
    ]

    use_stmts, _ = hio.var_use_statements(vars)
    lines.extend(use_stmts)

    lines.extend(
        [
            f"{tabs}implicit none\n",
            f"{tabs}public :: read_constants, write_constants, define_vars\n",
            f"{tabs}logical, parameter :: verbose = .false.\n",
            "contains\n",
        ]
    )

    # All variables still need to be defined in the NetCDF file.
    lines.extend(
        create_nc_define_vars(
            vars,
            time=False,
            guards=guards,
        )
    )

    lines.extend(
        create_constants_io_routine(
            mode=hio.IOMode.read,
            sub_name="read_constants",
            guard_vars=guard_vars,
            vars=regular_vars,
            guards=guards,
        )
    )

    lines.extend(
        create_constants_io_routine(
            mode=hio.IOMode.write,
            sub_name="write_constants",
            guard_vars=guard_vars,
            vars=regular_vars,
            guards=guards,
        )
    )

    lines.append(f"end module {mod_name}\n")
    lines = add_derived_constants(lines, getattr(unit_test, "derived_constants", {}))

    with open(case_dir / filename, "w") as ofile:
        ofile.writelines(lines)


def add_derived_constants(lines: list[str], derived: Mapping) -> list[str]:
    """
    Write-only constants for another case (`spel create --capture-for`): defined and
    written as their expression of this ELM's variables, never read back here.
    """
    if not derived:
        return lines
    nc_type = {"integer": "nf90_int", "real": "nf90_double", "logical": "nf90_int"}
    out: list[str] = []
    in_write = False
    for line in lines:
        text = line.strip().lower()
        if text == "end subroutine define_vars":
            out.extend(
                f"    call nc_define_var(ncid, 0, [0], dim_names, '{dc.name}', {nc_type[dc.type]}, varid, .false.)\n"
                for dc in derived.values()
            )
        in_write = in_write or text.startswith("subroutine write_constants")
        if in_write and text == "call check(nf90_close(ncid))":
            out.extend(
                f"    call nc_write_var_scalar(ncid, {dc.fortran}, '{dc.name}')  ! for {dc.source}-derived constant\n"
                for dc in derived.values()
            )
            in_write = False
        out.append(line)
    return out


def create_constants_io_routine(
    mode: hio.IOMode,
    sub_name: str,
    guard_vars: dict[str, Variable],
    vars: dict[str, Variable],
    guards: GuardMap,
) -> list[str]:
    tabs = hio.indent()

    lines = [
        f"{tabs}subroutine {sub_name}(io_inst)\n",
    ]
    tabs = hio.indent(hio.Tab.shift)

    lines.extend(
        [
            f"{tabs}type(spel_io_type), intent(inout) :: io_inst\n",
            f"{tabs}integer :: ncid, timestep\n",
            f"{tabs}character(len=256) :: new_fn\n",
            f"{tabs}new_fn = trim(io_inst%get_fn())\n",
        ]
    )

    if mode == hio.IOMode.read:
        lines.append(f"{tabs}ncid = nc_create_or_open_file(trim(new_fn), read_file)\n")

        lines.extend(
            create_nc_read(
                guard_vars,
                time=False,
            )
        )
        lines.extend(
            create_nc_read(
                vars,
                time=False,
                guards=guards,
            )
        )

    else:
        lines.extend(
            [
                f"{tabs}print *, 'creating file: ', trim(new_fn)\n",
                f"{tabs}ncid = nc_create_or_open_file(trim(new_fn), create_file)\n",
                f"{tabs}call define_vars(ncid)\n",
                f"{tabs}call check(nf90_enddef(ncid))\n",
            ]
        )
        lines.extend(
            create_nc_write(
                guard_vars,
                time=False,
            )
        )
        lines.extend(
            create_nc_write(
                vars,
                time=False,
                guards=guards,
            )
        )

    lines.append(f"{tabs}call check(nf90_close(ncid))\n")

    tabs = hio.indent(hio.Tab.unshift)
    lines.append(f"{tabs}end subroutine {sub_name}\n")

    return lines


def create_netcdf_io_routine(
    mode: hio.IOMode,
    sub_name: str,
    vars: dict[str, Variable],
    time: bool,
    elminst_vars: list[tuple[str, Variable]],
    guards: GuardMap,
) -> list[str]:
    tabs = hio.indent()

    bounds = True
    arg_str: str = "io_inst, bounds"
    define_args: str = "ncid, bounds"

    intent = "inout" if mode == hio.IOMode.read else "in"
    decls: list[str] = []
    for type_mod, inst in elminst_vars:
        arg_str = f"{arg_str}, {inst.name}"
        define_args = f"{define_args}, {inst.name}"
        var_type = inst.type
        decls.append(f"{tabs}{tabs}type({var_type}), intent({intent}) :: {inst.name}")

    lines: list[str] = [f"{tabs}subroutine {sub_name}({arg_str})\n"]
    tabs = hio.indent(hio.Tab.shift)

    mode_str = "create_file" if mode == hio.IOMode.write else "read_file"
    stmt = f"{tabs}type(bounds_type), intent({intent}) :: bounds\n"
    lines.extend([f"{decl}\n" for decl in decls])

    # Arguments + Locals:
    lines.extend(
        [
            f"{tabs}type(spel_io_type), intent(inout) :: io_inst\n" f"{stmt}",
            f"{tabs}integer :: ncid, timestep\n",
            f"{tabs}character(len=256) :: new_fn\n",
        ]
    )
    if mode == hio.IOMode.read:
        lines.append(f"{tabs}integer :: it\n")
    lines.append(f"{tabs}new_fn = trim(io_inst%get_fn())\n")

    if mode == hio.IOMode.write:
        lines.extend(
            [
                f"{tabs}if( io_inst%new_file) then\n",
                f"{tabs}{tabs}print *, 'creating file: ', trim(new_fn)\n"
                f"{tabs}{tabs}ncid = nc_create_or_open_file(trim(new_fn), create_file)\n",
                f"{tabs}{tabs}io_inst%new_file = .false.\n",
                f"{tabs}{tabs}call define_vars({define_args})\n",
                f"{tabs}{tabs}call check(nf90_enddef(ncid))\n",
                f"{tabs}else\n",
                f"{tabs}{tabs}ncid = nc_create_or_open_file(trim(new_fn), append_file)\n",
                f"{tabs}end if\n",
                f"{tabs}timestep = io_inst%timestep\n",
            ]
        )

    if mode == hio.IOMode.read:
        lines.append(f"{tabs}if(io_inst%end_run) return\n")
        lines.append(
            f"{tabs}if(io_inst%dt_in_file == 99999) io_inst%dt_in_file = nc_read_timeslices(new_fn)\n"
        )
        lines.append(f"{tabs}timestep = io_inst%timestep\n")
        lines.extend(create_nc_read_buffered(vars, guards=guards, mode_str=mode_str))
    else:
        lines.extend(create_nc_write(vars, time=time, guards=guards))
        lines.append(f"{tabs}call check(nf90_close(ncid))\n")
    tabs = hio.indent(hio.Tab.unshift)
    lines.append(f"{tabs}end subroutine {sub_name}\n")

    return lines


def create_nc_def(
    vars: dict[str, Variable],
    time: bool,
    guards: GuardMap | None = None,
) -> list[str]:
    """
    Generate nc_define_var calls for variables prior to writing.
    """
    if guards is None:
        guards = {}

    lines: list[str] = []
    tabs = hio.indent()

    for var in vars.values():
        var_lines: list[str] = []

        dim_names_str = get_dim_names(var, time)
        varname = var.name.replace("%", "__")
        nc_type = match_nc_type(var.type)

        time_str = ".true." if time and var.dim > 0 else ".false."

        if nc_type == "nf90_char":
            dim_str = f"[len({var.name})]"
            stmt = (
                f"call nc_define_var("
                f"ncid, 1, {dim_str}, dim_names, "
                f"'{varname}', {nc_type}, varid, {time_str})\n"
            )
        else:
            dim_str = f"shape({var.name})" if var.dim > 0 else "[0]"

            stmt = (
                f"call nc_define_var("
                f"ncid, {var.dim}, {dim_str}, dim_names, "
                f"'{varname}', {nc_type}, varid, {time_str})\n"
            )

        if var.dim > 0 or nc_type == "nf90_char":
            # character scalars have one dimension: the string length
            ndim = 1 if nc_type == "nf90_char" else var.dim
            var_lines.append(f"{tabs}dim_names(1:{ndim}) = " f"{dim_names_str}\n")

        var_lines.append(f"{tabs}{stmt}")

        if var.dim > 0:
            var_lines.append(
                f"{tabs}call nc_put_bounds_att(ncid, varid, {var.dim}, "
                f"dim_names(1:{var.dim}), lbound({var.name}), ubound({var.name}))\n"
            )

        var_lines = if_block(var_lines, alloc_status(var), tabs)
        condition = guards.get(var.name)

        lines.extend(
            guard_fortran_lines(
                var_lines,
                condition,
                tabs,
            )
        )

    return lines


def create_nc_write(
    vars: dict[str, Variable],
    time: bool,
    guards: GuardMap | None = None,
) -> list[str]:
    """
    Generate NetCDF write calls.

    Variables with a corresponding ConditionExpectation are emitted inside
    a Fortran IF block.
    """
    if guards is None:
        guards = {}

    lines: list[str] = []
    tabs = hio.indent()

    scalars = [var for var in vars.values() if var.dim == 0]
    arrays = [var for var in vars.values() if var.dim > 0]

    timestep = ", timestep" if time else ", -1"

    for var in scalars:
        varname = var.name.replace("%", "__")

        value = var.name
        if m := re.fullmatch(r"(bounds)%end([gtlcp])", var.name):
            # duplicated sets are written as set 1, which starts at beg
            value = f"nc_set_ub({m.group(1)}%beg{m.group(2)}, {var.name})"

        if var.type == "character":
            stmt = f"call nc_write_var_array(" f"ncid, {var.name}, '{varname}')\n"
        else:
            stmt = f"call nc_write_var_scalar(" f"ncid, {value}, '{varname}')\n"

        var_lines = [
            f"{tabs}if(verbose) print *, '{varname}'\n",
            f"{tabs}{stmt}",
        ]

        lines.extend(
            guard_fortran_lines(
                var_lines,
                guards.get(var.name),
                tabs,
            )
        )

    for var in arrays:
        dim_names_str = get_dim_names(var, time)
        reshape_str = f"reshape({var.name}, " f"[product(shape({var.name}))])"
        varname = var.name.replace("%", "__")

        stmt = (
            f"call nc_write_var_array("
            f"ncid, {var.dim}, shape({var.name}), "
            f"{dim_names_str}, {reshape_str}, "
            f"'{varname}'{timestep})\n"
        )

        var_lines = if_block(
            [f"{tabs}if(verbose) print *, '{varname}'\n", f"{tabs}{stmt}"],
            alloc_status(var),
            tabs,
        )

        lines.extend(
            guard_fortran_lines(
                var_lines,
                guards.get(var.name),
                tabs,
            )
        )

    return lines


def create_nc_read(
    vars: dict[str, Variable],
    time: bool,
    guards: GuardMap | None = None,
) -> list[str]:
    """
    Generate NetCDF read calls.

    Variables with a corresponding ConditionExpectation are emitted inside
    a Fortran IF block.
    """
    if guards is None:
        guards = {}

    lines: list[str] = []
    tabs = hio.indent()
    scalars = [var for var in vars.values() if var.dim == 0]
    arrays = [var for var in vars.values() if var.dim > 0]

    time_str = ", timestep" if time else ", -1"

    for var in scalars:
        varname = var.name.replace("%", "__")

        var_lines: list[str] = []

        if var.ptrscalar:
            # read every time step: keep the target (and its device copy)
            var_lines.append(
                f"{tabs}if (.not. associated({var.name})) allocate({var.name})\n"
            )

        if var.type == "character":
            stmt = (
                f"call nc_read_var("
                f"ncid, '{varname}', "
                f"'{var.name}_str', {var.name})\n"
            )
        else:
            stmt = (
                f"call nc_read_var(" f"ncid, '{varname}', " f"{var.name}{time_str})\n"
            )

        var_lines.append(f"{tabs}{stmt}")

        lines.extend(
            guard_fortran_lines(
                var_lines,
                guards.get(var.name),
                tabs,
            )
        )

    for var in arrays:
        assert var.type != "character", (
            "Need to implement array-of-character NetCDF read for " f"{var.name}"
        )

        varname = var.name.replace("%", "__")

        var_lines: list[str] = []
        # explicit-shape arrays (e.g. elm_varcon::d_con_w(ngases,3)) are already sized
        if is_deferred_shape(var):
            var_lines.append(
                f"{tabs}call nc_alloc(" f'ncid, "{varname}", {var.dim}, {var.name})\n'
            )
        var_lines.append(
            f"{tabs}call nc_read_var("
            f"ncid, '{varname}', {var.dim}, "
            f"{var.name}{time_str})\n"
        )
        # not written if it was unallocated in ELM (see alloc_status)
        if alloc_status(var):
            var_lines = if_block(var_lines, f"nc_has_var(ncid, '{varname}')", tabs)

        lines.extend(
            guard_fortran_lines(
                var_lines,
                guards.get(var.name),
                tabs,
            )
        )

    return lines


def _inbuf_vars(vars: dict[str, Variable]) -> list[Variable]:
    """Time-step inputs held in read buffers (ib_<n> is the n-th entry)."""
    scalars = [v for v in vars.values() if v.dim == 0 and v.type != "character"]
    arrays = [v for v in vars.values() if v.dim > 0]
    return scalars + arrays


def create_inbuf_decls(vars: dict[str, Variable]) -> list[str]:
    """Module variables of ReadWriteMod for the buffered input reads."""
    tabs = hio.indent(hio.Tab.reset)
    inbuf = _inbuf_vars(vars)
    lines = [
        f"{tabs}! Input buffers: ib_<n>(<set-1 shape>, chunk) holds timesteps\n",
        f"{tabs}! inbuf_t0..inbuf_t0+inbuf_nt-1 of the current input file\n",
        f"{tabs}integer :: inbuf_t0 = 0, inbuf_nt = 0, inbuf_chunk = 0\n",
        f"{tabs}integer :: ib_sd({max(len(inbuf), 1)}) = 0, ib_nfile({max(len(inbuf), 1)}) = 0\n",
    ]
    for n, var in enumerate(inbuf, start=1):
        ftype = "real(8)" if var.type == "real" else "integer"
        dims = ",".join([":"] * (var.dim + 1))
        lines.append(f"{tabs}{ftype}, allocatable :: ib_{n}({dims}) ! {var.name}\n")
    return lines


def create_inbuf_to_device(vars: dict[str, Variable]) -> list[str]:
    tabs = hio.indent(hio.Tab.reset)
    lines = [
        f"{tabs}subroutine spel_inbuf_to_device()\n",
        f"{tabs}{tabs}! Copy the input buffers to the device; later loads update them\n",
        f"{tabs}{tabs}! there and read_elmtypes copies the inputs on the device\n",
        "#ifdef _OPENACC\n",
    ]
    for n, _ in enumerate(_inbuf_vars(vars), start=1):
        lines.extend(
            [
                f"{tabs}{tabs}if (allocated(ib_{n})) then\n",
                f"{tabs}{tabs}   !$acc enter data copyin(ib_{n})\n",
                f"{tabs}{tabs}end if\n",
            ]
        )
    lines.extend(
        [
            f"{tabs}{tabs}spel_buf_on_device = .true.\n",
            "#endif\n",
            f"{tabs}end subroutine spel_inbuf_to_device\n",
        ]
    )
    return lines


def create_nc_read_buffered(
    vars: dict[str, Variable],
    guards: GuardMap,
    mode_str: str,
) -> list[str]:
    """
    Body of read_elmtypes: load the next chunk of timesteps into the input
    buffers when the current one is used up (or a new file starts), then copy
    this timestep's slice into the variables.
    """
    tabs = hio.indent()
    t2 = tabs + "  "
    index = {id(v): n for n, v in enumerate(_inbuf_vars(vars), start=1)}
    scalars = [v for v in vars.values() if v.dim == 0]
    arrays = [v for v in vars.values() if v.dim > 0]

    load: list[str] = []
    copy: list[str] = []
    for var in scalars + arrays:
        varname = var.name.replace("%", "__")
        var_load: list[str] = []
        if var.type == "character":
            var_load.append(
                f"{t2}call nc_read_var(ncid, '{varname}', '{var.name}_str', {var.name})\n"
            )
            load.extend(guard_fortran_lines(var_load, guards.get(var.name), t2))
            continue
        n = index[id(var)]
        if var.dim == 0 and var.ptrscalar:
            var_load.append(f"{t2}if (.not. associated({var.name})) allocate({var.name})\n")
        # explicit-shape arrays (spel_drv_<arg>(1)) are already sized
        if var.dim > 0 and is_deferred_shape(var):
            var_load.append(
                f'{t2}call nc_alloc(ncid, "{varname}", {var.dim}, {var.name})\n'
            )
        var_load.append(
            f"{t2}call nc_buf_load(ncid, '{varname}', shape({var.name}), ib_{n}, "
            f"inbuf_t0, inbuf_nt, inbuf_chunk, ib_sd({n}), ib_nfile({n}))\n"
        )
        load.extend(guard_fortran_lines(var_load, guards.get(var.name), t2))
        copy.extend(
            guard_fortran_lines(
                [f"{tabs}call nc_buf_copy({var.name}, ib_{n}, it, ib_sd({n}), ib_nfile({n}))\n"],
                guards.get(var.name),
                tabs,
            )
        )

    return [
        f"{tabs}if (io_inst%new_file .or. timestep < inbuf_t0 .or. timestep >= inbuf_t0 + inbuf_nt) then\n",
        f"{t2}if (inbuf_chunk == 0) inbuf_chunk = nc_buf_chunk(io_inst%dt_in_file)\n",
        f"{t2}inbuf_t0 = timestep\n",
        f"{t2}inbuf_nt = min(inbuf_chunk, io_inst%dt_in_file - timestep + 1)\n",
        f"{t2}ncid = nc_create_or_open_file(trim(new_fn), {mode_str})\n",
        *load,
        f"{t2}call check(nf90_close(ncid))\n",
        f"{tabs}end if\n",
        f"{tabs}it = timestep - inbuf_t0 + 1\n",
        *copy,
    ]


def get_dim_names(var: Variable, time: bool) -> str:
    if var.dim == 0 and var.type != "character":
        return "['']"  # empty list
    elif var.type == "character":
        return f"[character(len=32) :: '{var.name}_str']"
    # Temp list to preprocess subgrids?
    dim_names = var.bounds.split(",")
    assert (
        len(dim_names) == var.dim
    ), f"(get_dim_names) Inconsistent dimensions\n name: {var.name}bounds: {var.bounds} dim: {var.dim}"
    dim_names = [f"'{hio.get_subgrid(dim)}'" for dim in dim_names]
    # if time:
    #     dim_names.append("'time'")

    def adjust_dim_label(label: str, i: int) -> str:
        if label != "'_'":
            return label
        return f"'unk_{i}_{var.name}'"

    dim_names = [adjust_dim_label(i=i, label=name) for i, name in enumerate(dim_names)]
    dim_str = ",".join(dim_names)
    return f"[character(len=32) :: {dim_str}]"


def match_nc_type(var_type: str) -> str:
    match var_type:
        case "real":
            return "nf90_double"
        case "integer":
            return "nf90_int"
        case "logical":
            return "nf90_int"
        case "character":
            return "nf90_char"
        case _:
            print(f"(match_nc_type) {var_type} Not Implemented")


def generate_verify(rw_set: set[str], type_dict: dict[str, DerivedType]):
    """
    rw_set is set of active elmtypes with status of 'w' or 'rw'

    """
    lines: list[str] = []
    active_instances = {
        inst_var.name: inst_var
        for dtype in type_dict.values()
        for inst_var in dtype.instances.values()
        if inst_var.active
    }
    use_statements, _ = hio.var_use_statements(active_instances, type_dict)
    lines.extend(list(use_statements))

    return


def generate_nc_io():
    lines: list[str] = []

    type_list = ["double", "integer", "logical", "string"]
    read_dims = 3

    tabs = hio.indent(hio.Tab.reset)
    header: str = textwrap.dedent(f"""
    module nc_io
    {tabs}use netcdf
    {tabs}use iso_fortran_env
    {tabs}use iso_c_binding
    #ifdef _OPENACC
    {tabs}use openacc, only: acc_is_present
    #endif
    {tabs}implicit none

    {tabs}public
    {tabs}integer, parameter :: read_file = 0, append_file = 1
    {tabs}integer, parameter :: create_file = 2
    {tabs}real(8), parameter :: fill_double = 1.d+36
    {tabs}integer, parameter :: fill_int = -9999
    {tabs}type, public :: spel_io_type
    {tabs}{tabs}character(len=256) :: fn
    {tabs}{tabs}integer :: timestep
    {tabs}{tabs}integer :: max_timesteps_per_file
    {tabs}{tabs}integer :: filenum
    {tabs}{tabs}logical :: new_file
    {tabs}{tabs}logical :: created = .false.
    {tabs}{tabs}logical :: read_mode = .false.
    {tabs}{tabs}logical :: end_run = .False.
    {tabs}{tabs}integer :: dt_in_file=99999 !used for reading files only
    {tabs}contains
    {tabs}{tabs}procedure, public :: get_fn
    {tabs}{tabs}procedure, public :: init
    {tabs}{tabs}procedure, private :: check_file_exists
    {tabs}{tabs}procedure, private :: need_new_file
    {tabs}end type spel_io_type
    {tabs}type(spel_io_type), public :: io_constants, io_inputs, io_outputs
    {tabs}! Number of copies of the input sites (unit tests only). Arrays with a
    {tabs}! subgrid dimension are allocated nsets times larger along it, reads
    {tabs}! replicate the file data into every set and writes emit set 1 only.
    {tabs}integer, public :: spel_nsets = 1
    {tabs}! Set once the input buffers are on the device (spel_inbuf_to_device):
    {tabs}! buffer loads then update the device and copies run on the device.
    {tabs}logical, public :: spel_buf_on_device = .false.
    {tabs}! Gridcell subset (instrumented ELM only, see spel/scripts/subset.py).
    {tabs}! While a subset is active (nc_subset_activate), writes keep only the
    {tabs}! selected gridcells and their topounits, landunits, columns and
    {tabs}! patches, and renumber the subgrid index fields, so the files look
    {tabs}! like a capture of a smaller domain.
    {tabs}integer, parameter :: spel_nlev = 5  ! gridcell, topo, landunit, column, patch
    {tabs}type, public :: spel_subset_level
    {tabs}{tabs}integer :: beg = 1    ! lower bound of the full (process) arrays
    {tabs}{tabs}integer :: nfull = 0  ! their length
    {tabs}{tabs}integer, allocatable :: pos(:)     ! kept positions in the full arrays
    {tabs}{tabs}integer, allocatable :: newidx(:)  ! new index of each position, 0 if dropped
    {tabs}end type spel_subset_level
    {tabs}type, public :: spel_subset_type
    {tabs}{tabs}type(spel_subset_level) :: lev(spel_nlev)
    {tabs}{tabs}character(len=512) :: mask = ''
    {tabs}{tabs}! provenance of the kept gridcells, written as global attributes
    {tabs}{tabs}integer, allocatable :: gindex(:)
    {tabs}{tabs}real(8), allocatable :: latdeg(:), londeg(:)
    {tabs}end type spel_subset_type
    {tabs}type(spel_subset_type), pointer, public :: spel_subset => null()
    """)
    lines.append(header)
    lines.append(f"{tabs}interface nc_write_var_array\n")

    # Interfaces for nc write procedures
    tabs = hio.indent(hio.Tab.shift)
    for t in type_list:
        name = f"nc_write_{t}"
        lines.append(f"{tabs}module procedure {name}\n")
    tabs = hio.indent(hio.Tab.unshift)
    lines.append(f"{tabs}end interface\n")

    # Scalars
    lines.append(f"{tabs}interface nc_write_var_scalar\n")
    tabs = hio.indent(hio.Tab.shift)
    for t in type_list:
        if t == "string":
            continue
        name = f"nc_write_{t}_scalar"
        lines.append(f"{tabs}module procedure {name}\n")
    tabs = hio.indent(hio.Tab.unshift)
    lines.append(f"{tabs}end interface\n")

    # Interface for nc read procedures
    lines.append(f"{tabs}interface nc_read_var\n")
    tabs = hio.indent(hio.Tab.shift)
    for t in type_list:
        if t == "string":
            lines.append(f"{tabs}module procedure nc_read_string\n")
        else:
            for ndim in range(0, read_dims + 1):
                name = f"nc_read_{t}_{ndim}"
                lines.append(f"{tabs}module procedure {name}\n")
    tabs = hio.indent(hio.Tab.unshift)
    lines.append(f"{tabs}end interface\n")
    lines.append(f"{tabs}interface nc_select_set1\n")
    tabs = hio.indent(hio.Tab.shift)
    lines.append(f"{tabs}module procedure nc_select_set1_r8, nc_select_set1_int\n")
    tabs = hio.indent(hio.Tab.unshift)
    lines.append(f"{tabs}end interface\n")
    lines.append(f"{tabs}interface nc_subset_select\n")
    tabs = hio.indent(hio.Tab.shift)
    lines.append(f"{tabs}module procedure nc_subset_select_r8, nc_subset_select_int\n")
    tabs = hio.indent(hio.Tab.unshift)
    lines.append(f"{tabs}end interface\n")
    for iface, kinds in (
        ("nc_buf_load", ("double", "integer")),
        ("nc_buf_copy", ("double", "integer", "logical")),
    ):
        lines.append(f"{tabs}interface {iface}\n")
        tabs = hio.indent(hio.Tab.shift)
        for t in kinds:
            for ndim in range(0, INBUF_DIMS + 1):
                lines.append(f"{tabs}module procedure {iface}_{t}_{ndim}\n")
        tabs = hio.indent(hio.Tab.unshift)
        lines.append(f"{tabs}end interface\n")
    lines.append(f"{tabs}public :: nc_read_timeslices\n")

    lines.append("contains\n")

    nc_check = gen_check()
    lines.append(nc_check)
    lines.append(gen_nsets_helpers())
    lines.append(gen_subset_helpers())
    # nc_create_or_open_file
    nc_file = gen_nc_file()
    lines.append(nc_file)

    lines.append(gen_spel_type_routines())

    # nc_define_var subroutine
    nc_define = gen_nc_define_var()
    lines.append(nc_define)

    # nc_read_var
    for t in type_list:
        if t == "string":
            lines.append(gen_nc_read_type(0, t))
        else:
            for i in range(0, read_dims + 1):
                lines.append(gen_nc_read_type(i, t))

    # nc_write_
    lines.extend(gen_nc_write_numeric_array())
    lines.extend(gen_nc_write_numeric_scalar())
    lines.append(gen_nc_write_string())
    lines.append(gen_nc_write_logical())

    lines.append(gen_nc_read_timeslices())
    lines.append(gen_inbuf_helpers())

    lines.append("end module nc_io\n")

    with open(f"{spel_mods_dir}/nc_io.F90", "w") as ofile:
        ofile.writelines(lines)


def gen_spel_type_routines() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(f"""
    subroutine spel_io_init(path,read_io,max_tpf)
    {tabs}! Initialize io_constants/io_inputs/io_outputs (no-op if already done).
    {tabs}! Reading (offline unit test): outputs go to 'fut-outputs'.
    {tabs}! Writing (instrumented ELM):  outputs go to 'spel-outputs'.
    {tabs}character(len=*), intent(in) :: path
    {tabs}logical, intent(in) :: read_io
    {tabs}integer, intent(in), optional :: max_tpf
    {tabs}integer :: tpf
    {tabs}if(io_constants%created) return
    {tabs}tpf = 720
    {tabs}if(present(max_tpf)) tpf = max_tpf
    {tabs}call io_constants%init(base_fn=trim(path)//'spel-constants',max_tpf=tpf,read_io=read_io)
    {tabs}call io_inputs%init(base_fn=trim(path)//'spel-inputs',max_tpf=tpf,read_io=read_io)
    {tabs}if(read_io) then
    {tabs}{tabs}call io_outputs%init(base_fn=trim(path)//'fut-outputs',max_tpf=tpf,read_io=.false.)
    {tabs}else
    {tabs}{tabs}call io_outputs%init(base_fn=trim(path)//'spel-outputs',max_tpf=tpf,read_io=.false.)
    {tabs}end if
    end subroutine spel_io_init

    subroutine init(this,base_fn,max_tpf,read_io)
    {tabs}class(spel_io_type), intent(inout) :: this
    {tabs}character(len=*), intent(in) :: base_fn
    {tabs}integer, intent(in) :: max_tpf
    {tabs}logical, intent(in) :: read_io
    {tabs}if(this%created) error stop "(spel_io_type::init) Error -- io_inst already initialized"
    {tabs}this%read_mode = read_io
    {tabs}this%max_timesteps_per_file = max_tpf
    {tabs}this%new_file = .true.
    {tabs}this%fn = trim(base_fn)
    {tabs}this%timestep = 0
    {tabs}this%new_file = .true.
    {tabs}this%filenum = 0
    {tabs}this%created = .true.
    end subroutine init

    logical function need_new_file(this) result(flag)
    {tabs}class(spel_io_type), intent(inout) :: this
    {tabs}if(this%read_mode) then 
    {tabs}    flag = (this%timestep > this%dt_in_file)
    {tabs}else
    {tabs}    flag = (this%timestep > this%max_timesteps_per_file)
    {tabs}end if
    end function need_new_file

    logical function check_file_exists(this,fn) result(exists)
        class(spel_io_type), intent(inout) :: this
        character(len=*), intent(in) :: fn
        inquire(file=trim(fn), exist=exists)
    end function check_file_exists

    character(len=256) function get_fn(this) result(new_fn)
    {tabs}class(spel_io_type), intent(inout) :: this
    {tabs}character(len=10) :: ch_num
    {tabs}integer :: max_time_steps
    {tabs}this%timestep = this%timestep + 1
    {tabs}this%new_file = .False.
    {tabs}if(this%need_new_file() .or. this%timestep == 1) then
    {tabs}{tabs}this%filenum = this%filenum + 1
    {tabs}{tabs}this%timestep = 1
    {tabs}{tabs}this%new_file = .true.
    {tabs}{tabs}if(this%read_mode) this%dt_in_file = 99999
    {tabs}end if
    {tabs}write(unit=ch_num,fmt='(I0.4)') this%filenum
    {tabs}new_fn = trim(this%fn)//trim(ch_num)//'.nc'
    {tabs}if(this%read_mode) this%end_run = .not. (this%check_file_exists(new_fn))
    end function get_fn
    """)


def gen_nc_read_type(dim: int, t: str) -> str:
    tabs = hio.indent(Tab.reset)
    if t == "string":
        return gen_read_str()
    if t == "logical":
        return gen_read_logical(dim)
    else:
        return gen_read_numeric(dim, t)


# Subgrid index fields of the ELM physical-property types and the subgrid level
# each one indexes. duplicateMod offsets them per set; set checks skip them.
SUBGRID_INDEX_FIELDS: dict[str, dict[str, str]] = {
    "gridcell_physical_properties_type": {
        "gindex": "g", "topi": "t", "topf": "t", "lndi": "l", "lndf": "l",
        "coli": "c", "colf": "c", "pfti": "p", "pftf": "p",
    },
    "topounit_physical_properties": {
        "gridcell": "g", "downhill_ti": "t", "lndi": "l", "lndf": "l",
        "coli": "c", "colf": "c", "pfti": "p", "pftf": "p", "landunit_indices": "l",
    },
    "landunit_physical_properties": {
        "gridcell": "g", "topounit": "t", "coli": "c", "colf": "c", "pfti": "p", "pftf": "p",
    },
    "column_physical_properties": {
        "gridcell": "g", "topounit": "t", "landunit": "l", "pfti": "p", "pftf": "p",
    },
    "vegetation_physical_properties": {
        "gridcell": "g", "topounit": "t", "landunit": "l", "column": "c",
    },
}


# Subgrid levels in nc_io's order (spel_subset_type%lev, nc_subgrid_level)
SUBGRID_LEVELS = ("g", "t", "l", "c", "p")


def index_field_levels() -> dict[str, str]:
    """{field: level it indexes}; a field name means the same level in every type."""
    levels: dict[str, str] = {}
    for fields in SUBGRID_INDEX_FIELDS.values():
        for field, level in fields.items():
            if levels.setdefault(field, level) != level:
                raise ValueError(f"index field {field} refers to levels {levels[field]} and {level}")
    return levels


def gen_nsets_helpers() -> str:
    """Fortran helpers for spel_nsets (duplicated input sites in one clump)."""
    select_set1 = ""
    for suffix, ftype, differs in (
        ("r8", "real(8)", "a /= b .and. .not. (ieee_is_nan(a) .and. ieee_is_nan(b))"),
        ("int", "integer", "a /= b"),
    ):
        select_set1 += textwrap.dedent(f"""
        subroutine nc_select_set1_{suffix}(ndim, dims, dim_names, buffer, varname)
           ! Reduce a flattened write buffer to set 1 along its subgrid dimension
           ! and report values in the other sets that differ from set 1.
           use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
           integer, intent(in) :: ndim
           integer, intent(inout) :: dims(ndim)
           character(len=*), dimension(:), intent(in) :: dim_names
           {ftype}, allocatable, intent(inout) :: buffer(:)
           character(len=*), intent(in) :: varname
           {ftype}, allocatable :: b3(:,:,:)
           integer :: sd, n1, pre, post, k, nbad
           sd = nc_subgrid_dim(ndim, dim_names)
           if (sd == 0) return
           if (mod(dims(sd), spel_nsets) /= 0) return
           n1 = dims(sd)/spel_nsets
           pre = product(dims(1:sd-1))
           post = product(dims(sd+1:ndim))
           b3 = reshape(buffer, [pre, dims(sd), post])
           nbad = 0
           do k = 2, spel_nsets
              nbad = nbad + count(differs(b3(:, (k-1)*n1+1:k*n1, :), b3(:, 1:n1, :)))
           end do
           if (nbad > 0 .and. .not. nc_is_index_field(varname)) print *, 'WARNING (spel_nsets): ', trim(varname), nbad, &
              ' values in sets 2..nsets differ from set 1'
           buffer = reshape(b3(:, 1:n1, :), [pre*n1*post])
           dims(sd) = n1
        contains
           elemental logical function differs(a, b)
              {ftype}, intent(in) :: a, b
              differs = {differs}
           end function differs
        end subroutine nc_select_set1_{suffix}
        """)

    by_level: dict[str, list[str]] = {}
    for field, level in sorted(index_field_levels().items()):
        by_level.setdefault(level, []).append(field)
    cases = "".join(
        f"       case ({', '.join(repr(f) for f in by_level[lev])})\n"
        f"          ilev = {SUBGRID_LEVELS.index(lev) + 1}\n"
        for lev in SUBGRID_LEVELS
        if lev in by_level
    )
    is_index_field = textwrap.dedent("""
    integer function nc_index_level(varname) result(ilev)
       ! Subgrid level (1-5: g, t, l, c, p) a subgrid index field (<inst>_pp__<field>)
       ! refers to, 0 for other variables
       character(len=*), intent(in) :: varname
       integer :: i
       ilev = 0
       i = index(varname, '_pp__', back=.true.)
       if (i == 0) return
       select case (trim(varname(i+5:)))
    """) + cases + textwrap.dedent("""\
       end select
    end function nc_index_level

    logical function nc_is_index_field(varname)
       ! Subgrid index fields are offset per set by duplicateMod
       character(len=*), intent(in) :: varname
       nc_is_index_field = nc_index_level(varname) > 0
    end function nc_is_index_field
    """)

    return (
        is_index_field
        + textwrap.dedent("""
    integer function nc_subgrid_level(name) result(ilev)
       ! Subgrid level (1-5: g, t, l, c, p) of a dimension name, 0 if not subgrid
       character(len=*), intent(in) :: name
       select case (trim(name))
       case ('gridcell')
          ilev = 1
       case ('topo')
          ilev = 2
       case ('landunit')
          ilev = 3
       case ('column')
          ilev = 4
       case ('patch')
          ilev = 5
       case default
          ilev = 0
       end select
    end function nc_subgrid_level

    logical function nc_is_subgrid(name)
       character(len=*), intent(in) :: name
       nc_is_subgrid = nc_subgrid_level(name) > 0
    end function nc_is_subgrid

    integer function nc_first_subgrid_dim(ndim, dim_names) result(sd)
       ! First subgrid dimension of a variable, or 0 if it has none.
       integer, intent(in) :: ndim
       character(len=*), dimension(:), intent(in) :: dim_names
       integer :: i
       sd = 0
       do i = 1, min(ndim, size(dim_names))
          if (nc_is_subgrid(dim_names(i))) then
             sd = i
             return
          end if
       end do
    end function nc_first_subgrid_dim

    integer function nc_subgrid_dim(ndim, dim_names) result(sd)
       ! First subgrid dimension of a variable, or 0 when not duplicating.
       integer, intent(in) :: ndim
       character(len=*), dimension(:), intent(in) :: dim_names
       sd = 0
       if (spel_nsets <= 1) return
       sd = nc_first_subgrid_dim(ndim, dim_names)
    end function nc_subgrid_dim

    subroutine nc_file_subgrid(ncid, var_id, ndim, sd, nfile)
       ! First subgrid dimension (sd) of a file variable and its length in
       ! the file (nfile); sd = 0 when not duplicating.
       integer, intent(in) :: ncid, var_id, ndim
       integer, intent(out) :: sd, nfile
       integer :: nd, i
       integer, allocatable :: dimids(:)
       character(len=nf90_max_name) :: dname
       sd = 0; nfile = 0
       if (spel_nsets <= 1) return
       call check(nf90_inquire_variable(ncid, var_id, ndims=nd))
       allocate (dimids(nd))
       call check(nf90_inquire_variable(ncid, var_id, dimids=dimids))
       do i = 1, min(ndim, nd)
          call check(nf90_inquire_dimension(ncid, dimids(i), name=dname, len=nfile))
          if (nc_is_subgrid(dname)) then
             sd = i
             return
          end if
       end do
       nfile = 0
    end subroutine nc_file_subgrid

    integer function nc_set_ub(lb, ub) result(ub1)
       ! Upper bound of set 1 for a range duplicated spel_nsets times.
       integer, intent(in) :: lb, ub
       ub1 = ub
       if (spel_nsets <= 1) return
       if (mod(ub - lb + 1, spel_nsets) /= 0) return
       ub1 = lb - 1 + (ub - lb + 1)/spel_nsets
    end function nc_set_ub

    subroutine nc_put_bounds_att(ncid, varid, ndim, dim_names, lbs, ubs)
       ! Write l/ubounds attributes, as set 1's bounds when duplicating and
       ! as 1:n along the subgrid dimension of a gridcell subset.
       integer, intent(in) :: ncid, varid, ndim
       character(len=*), dimension(:), intent(in) :: dim_names
       integer, intent(in) :: lbs(ndim), ubs(ndim)
       integer :: lb1(ndim), ub1(ndim), sd
       lb1 = lbs
       ub1 = ubs
       sd = nc_subgrid_dim(ndim, dim_names)
       if (sd > 0) ub1(sd) = nc_set_ub(lbs(sd), ubs(sd))
       if (associated(spel_subset)) then
          sd = nc_first_subgrid_dim(ndim, dim_names)
          if (sd > 0) then
             lb1(sd) = 1
             ub1(sd) = nc_subset_n(dim_names(sd), ubs(sd) - lbs(sd) + 1, 'bounds attribute')
          end if
       end if
       call check(nf90_put_att(ncid, varid, 'lbounds', lb1))
       call check(nf90_put_att(ncid, varid, 'ubounds', ub1))
    end subroutine nc_put_bounds_att
    """)
        + select_set1
    )


def gen_subset_helpers() -> str:
    """
    Fortran helpers for writing a gridcell subset (spel_subset_type).

    The capture module (spel/scripts/subset.py) selects the gridcells and
    sets each level with nc_subset_set_level; while the subset is active,
    nc_subset_select reduces every write buffer to the kept positions along
    its subgrid dimension and renumbers the subgrid index fields.
    """
    select = ""
    for suffix, ftype, as_int in (("r8", "real(8)", "nint(buffer(i))"), ("int", "integer", "buffer(i)")):
        select += textwrap.dedent(f"""
        subroutine nc_subset_select_{suffix}(ndim, dims, dim_names, buffer, varname)
           ! Keep the subset's positions along the subgrid dimension of a flattened
           ! write buffer and renumber subgrid index fields.
           integer, intent(in) :: ndim
           integer, intent(inout) :: dims(ndim)
           character(len=*), dimension(:), intent(in) :: dim_names
           {ftype}, allocatable, intent(inout) :: buffer(:)
           character(len=*), intent(in) :: varname
           {ftype}, allocatable :: b3(:,:,:)
           integer :: sd, ilev, pre, post, n, i, k
           sd = nc_first_subgrid_dim(ndim, dim_names)
           if (sd > 0) then
              ilev = nc_subgrid_level(dim_names(sd))
              n = nc_subset_n(dim_names(sd), dims(sd), varname)
              pre = product(dims(1:sd-1))
              post = product(dims(sd+1:ndim))
              b3 = reshape(buffer, [pre, dims(sd), post])
              buffer = reshape(b3(:, spel_subset%lev(ilev)%pos, :), [pre*n*post])
              dims(sd) = n
           end if
           ilev = nc_index_level(varname)
           if (ilev == 0) return
           if (varname(index(varname, '_pp__', back=.true.)+5:) == 'gindex') then
              ! global index of the kept gridcells: the subset is its own domain
              buffer = [(i, i = 1, size(buffer))]
              return
           end if
           associate (lev => spel_subset%lev(ilev))
              do i = 1, size(buffer)
                 k = {as_int} - lev%beg + 1
                 if (k < 1 .or. k > lev%nfull) cycle  ! e.g. ispval: not an index
                 if (lev%newidx(k) > 0) then
                    buffer(i) = lev%newidx(k)
                 else
                    buffer(i) = fill_int  ! refers to a dropped gridcell
                 end if
              end do
           end associate
        end subroutine nc_subset_select_{suffix}
        """)

    return (
        textwrap.dedent("""
    subroutine nc_subset_set_level(sub, ilev, beg, keep)
       ! Level ilev (1-5: g, t, l, c, p) keeps the positions where keep is true;
       ! keep spans the full (process) arrays, which start at index beg.
       type(spel_subset_type), intent(inout) :: sub
       integer, intent(in) :: ilev, beg
       logical, intent(in) :: keep(:)
       integer :: i
       associate (lev => sub%lev(ilev))
          lev%beg = beg
          lev%nfull = size(keep)
          lev%pos = pack([(i, i = 1, size(keep))], keep)
          if (allocated(lev%newidx)) deallocate (lev%newidx)
          allocate (lev%newidx(size(keep)))
          lev%newidx = 0
          lev%newidx(lev%pos) = [(i, i = 1, size(lev%pos))]
       end associate
    end subroutine nc_subset_set_level

    subroutine nc_subset_activate(sub)
       ! Writes keep only sub's gridcells until nc_subset_deactivate
       type(spel_subset_type), target, intent(in) :: sub
       spel_subset => sub
    end subroutine nc_subset_activate

    subroutine nc_subset_deactivate()
       nullify (spel_subset)
    end subroutine nc_subset_deactivate

    integer function nc_subset_count(sub, ilev) result(n)
       type(spel_subset_type), intent(in) :: sub
       integer, intent(in) :: ilev
       n = 0
       if (allocated(sub%lev(ilev)%pos)) n = size(sub%lev(ilev)%pos)
    end function nc_subset_count

    integer function nc_subset_n(dim_name, nfull, varname) result(n)
       ! Length of a subgrid dimension in the active subset. A variable whose
       ! subgrid dimension doesn't span the process arrays can't be subset.
       character(len=*), intent(in) :: dim_name, varname
       integer, intent(in) :: nfull
       integer :: ilev
       ilev = nc_subgrid_level(dim_name)
       if (nfull /= spel_subset%lev(ilev)%nfull) then
          print *, 'ERROR (spel subset): ', trim(varname), ' has ', nfull, ' ', trim(dim_name), &
             ' entries, the process has ', spel_subset%lev(ilev)%nfull
          error stop 'spel subset: subgrid dimension does not match the process bounds'
       end if
       n = nc_subset_count(spel_subset, ilev)
    end function nc_subset_n

    subroutine nc_subset_put_atts(ncid)
       ! Record the subset in a new file's global attributes
       integer, intent(in) :: ncid
       call check(nf90_put_att(ncid, nf90_global, 'spel_subset_mask', trim(spel_subset%mask)))
       if (allocated(spel_subset%gindex)) &
          call check(nf90_put_att(ncid, nf90_global, 'spel_subset_gindex', spel_subset%gindex))
       if (allocated(spel_subset%latdeg)) &
          call check(nf90_put_att(ncid, nf90_global, 'spel_subset_latdeg', spel_subset%latdeg))
       if (allocated(spel_subset%londeg)) &
          call check(nf90_put_att(ncid, nf90_global, 'spel_subset_londeg', spel_subset%londeg))
    end subroutine nc_subset_put_atts
    """)
        + select
    )


INBUF_DIMS = 3


def gen_inbuf_helpers() -> str:
    """
    Fortran helpers for the buffered time-step input reads (read_elmtypes).

    nc_buf_load reads timesteps t0..t0+nt-1 of one file variable into a
    buffer shaped like set 1 with time as the last dimension. Once
    spel_buf_on_device is set (spel_inbuf_to_device) the buffers live on the
    device as well and every load updates the device copy.

    nc_buf_copy copies time slice `it` of a buffer into the variable,
    replicating it into the spel_nsets sets along the subgrid dimension sd.
    The copy runs in an OpenACC compute region when the variable is present
    on the device, so the inputs never pass through the host.
    """
    lines = [
        textwrap.dedent("""
        integer function nc_buf_chunk(dt_in_file) result(chunk)
           ! Timesteps per input buffer: the whole file unless SPEL_IO_CHUNK caps it
           integer, intent(in) :: dt_in_file
           character(len=32) :: env
           integer :: stat, n
           chunk = dt_in_file
           call get_environment_variable('SPEL_IO_CHUNK', env, status=stat)
           if (stat == 0) then
              read (env, *, iostat=stat) n
              if (stat == 0 .and. n > 0) chunk = min(n, dt_in_file)
           end if
           print *, 'SPEL input buffer: ', chunk, ' timesteps per read'
        end function nc_buf_chunk
        """)
    ]

    for t, ftype in (("double", "real(8)"), ("integer", "integer")):
        for dim in range(0, INBUF_DIMS + 1):
            lines.append(_gen_buf_load(dim, t, ftype))
            lines.append(_gen_buf_copy(dim, t, ftype, ftype))
    for dim in range(0, INBUF_DIMS + 1):
        lines.append(_gen_buf_copy(dim, "logical", "logical", "integer"))
    return "".join(lines)


def _gen_buf_load(dim: int, t: str, ftype: str) -> str:
    name = f"nc_buf_load_{t}_{dim}"
    buf_dims = make_dim_str(dim + 1, lambda i: ":")
    alloc_shape = ", ".join([f"fshape({i + 1})" for i in range(dim)] + ["chunk"])
    upd_sect = ", ".join([":"] * dim + ["1:nt"])
    if dim > 0:
        start = f"[{', '.join(['1'] * dim)}, t0]"
        count = "[fshape, nt]"
        subgrid = textwrap.indent(textwrap.dedent(f"""\
           call nc_file_subgrid(ncid, var_id, {dim}, sd, nfile)
           fshape = vshape
           if (sd > 0) then
              if (vshape(sd) /= spel_nsets*nfile) then
                 sd = 0
              else
                 fshape(sd) = nfile
              end if
           end if
        """), "   ")
        shape_check = "any(shape(buf) /= [fshape, chunk])"
    else:
        start, count = "[t0]", "[nt]"
        subgrid = "   sd = 0; nfile = 0\n"
        shape_check = "size(buf) /= chunk"
    fshape_decl = f"   integer :: fshape({dim})\n" if dim > 0 else ""
    first = ", ".join([":"] * dim + ["1"])
    each = ", ".join([":"] * dim + ["t"])
    return (
        f"\nsubroutine {name}(ncid, varname, vshape, buf, t0, nt, chunk, sd, nfile)\n"
        "   integer, intent(in) :: ncid, vshape(:), t0, nt, chunk\n"
        "   character(len=*), intent(in) :: varname\n"
        f"   {ftype}, allocatable, intent(inout) :: buf({buf_dims})\n"
        "   integer, intent(inout) :: sd, nfile\n"
        "   integer :: var_id, nd, t\n"
        f"{fshape_decl}"
        "   call check(nf90_inq_varid(ncid, trim(varname), var_id))\n"
        f"{subgrid}"
        "   if (allocated(buf)) then\n"
        f"      if ({shape_check}) then\n"
        "         print *, 'nc_buf_load: input shape changed for ', trim(varname)\n"
        "         error stop\n"
        "      end if\n"
        "   else\n"
        f"      allocate (buf({alloc_shape}))\n"
        "      if (spel_buf_on_device) then\n"
        "         !$acc enter data create(buf)\n"
        "      end if\n"
        "   end if\n"
        "   call check(nf90_inquire_variable(ncid, var_id, ndims=nd))\n"
        f"   if (nd > {dim}) then\n"
        f"      call check(nf90_get_var(ncid, var_id, buf, start={start}, count={count}))\n"
        "   else\n"
        "      ! no time dimension in the file (e.g. scalars): same value every step\n"
        f"      call check(nf90_get_var(ncid, var_id, buf({first})))\n"
        "      do t = 2, nt\n"
        f"         buf({each}) = buf({first})\n"
        "      end do\n"
        "   end if\n"
        "   if (spel_buf_on_device) then\n"
        f"      !$acc update device(buf({upd_sect}))\n"
        "   end if\n"
        f"end subroutine {name}\n"
    )


def _gen_buf_copy(dim: int, t: str, var_type: str, buf_type: str) -> str:
    name = f"nc_buf_copy_{t}_{dim}"
    conv = " == 1" if t == "logical" else ""
    head = (
        f"\nsubroutine {name}(var, buf, it, sd, nfile)\n"
        f"   {var_type}, intent(inout) :: var{'(' + make_dim_str(dim, lambda i: ':') + ')' if dim else ''}\n"
        f"   {buf_type}, intent(in) :: buf({make_dim_str(dim + 1, lambda i: ':')})\n"
        "   integer, intent(in) :: it, sd, nfile\n"
    )
    if dim == 0:
        return (
            head
            + f"   var = buf(it){conv}\n"
            + "#ifdef _OPENACC\n"
            + "   if (spel_buf_on_device) then\n"
            + "      if (acc_is_present(var, storage_size(var)/8)) then\n"
            + "         !$acc update device(var)\n"
            + "      end if\n"
            + "   end if\n"
            + "#endif\n"
            + f"end subroutine {name}\n"
        )

    idx = list(range(1, dim + 1))
    loop_vars = ", ".join(f"i{i}" for i in idx)
    lhs = ", ".join(f"i{i} + merge(o, 0, sd == {i})" for i in idx)
    rhs = f"buf({loop_vars}, it){conv}"

    def nest(directive: str) -> str:
        s = directive
        s += "      do k = 1, nk\n"
        for i in reversed(idx):
            s += f"      do i{i} = 1, size(buf, {i})\n"
        s += "         o = (k-1)*nfile\n"
        s += f"         var({lhs}) = {rhs}\n"
        s += "      end do\n" * (dim + 1)
        return s

    return (
        head
        + f"   integer :: k, nk, o, {loop_vars}\n"
        + "   logical :: dev\n"
        + "   nk = 1\n"
        + "   if (sd > 0) nk = spel_nsets\n"
        + "   dev = .false.\n"
        + "#ifdef _OPENACC\n"
        + "   if (spel_buf_on_device) dev = acc_is_present(var)\n"
        + "#endif\n"
        + "   if (dev) then\n"
        + nest(
            f"      !$acc parallel loop gang vector collapse({dim + 1}) "
            "private(o) present(var, buf)\n"
        )
        + "   else\n"
        + nest("")
        + "   end if\n"
        + f"end subroutine {name}\n"
    )


def gen_read_numeric(dim: int, t: str) -> str:
    tabs = hio.indent(Tab.reset)
    if t == "double":
        t_str = "real(8)"
    else:
        t_str = t
    subname = f"nc_read_{t}_{dim}"
    if dim > 0:
        dim_str = make_dim_str(dim, lambda i: ":")
        set_cases = "".join(
            f"{tabs}         case ({sd})\n"
            f"{tabs}            var({make_dim_str(dim, lambda i: 'lo:hi' if i == sd - 1 else ':')}) = tmp\n"
            for sd in range(1, dim + 1)
        )
        tmp_shape = make_dim_str(dim, lambda i: f"count({i+1})")

        return f"""
{tabs}subroutine {subname}(ncid, varname, ndim, var, timestep)
{tabs}   integer, intent(in) :: ncid
{tabs}   character(len=*), intent(in) :: varname
{tabs}   integer, intent(in) :: ndim 
{tabs}   {t_str}, intent(out) :: var({dim_str})
{tabs}   integer, intent(in) :: timestep
{tabs}   integer :: var_id
{tabs}   integer, allocatable :: start(:), count(:)
{tabs}   {t_str}, allocatable :: tmp({dim_str})
{tabs}   integer :: sd, nfile, k, lo, hi

{tabs}   if (timestep > 0) then
{tabs}     allocate(start(ndim+1), count(ndim+1))
{tabs}     start(1:ndim) = 1
{tabs}     start(ndim+1) = timestep
{tabs}     count(1:ndim) = shape(var)
{tabs}     count(ndim+1) = 1
{tabs}  else
{tabs}     allocate(start(ndim), count(ndim))
{tabs}     start(1:ndim) = 1
{tabs}     count(1:ndim) = shape(var)
{tabs}  end if

{tabs}   call check(nf90_inq_varid(ncid, trim(varname), var_id))
{tabs}   call nc_file_subgrid(ncid, var_id, ndim, sd, nfile)
{tabs}   if (sd > 0) then
{tabs}      if (size(var, sd) /= spel_nsets*nfile) sd = 0
{tabs}   end if
{tabs}   if (sd == 0) then
{tabs}      call check(nf90_get_var(ncid, var_id, var,start=start, count=count))
{tabs}   else
{tabs}      ! read set 1 from the file and replicate it into every set
{tabs}      count(sd) = nfile
{tabs}      allocate(tmp({tmp_shape}))
{tabs}      call check(nf90_get_var(ncid, var_id, tmp,start=start, count=count))
{tabs}      do k = 1, spel_nsets
{tabs}         lo = (k-1)*nfile + 1; hi = k*nfile
{tabs}         select case (sd)
{set_cases}{tabs}         end select
{tabs}      end do
{tabs}   end if
{tabs}end subroutine {subname}

"""
    else:  # dim == 0
        return textwrap.dedent(f"""
    {tabs}subroutine {subname}(ncid, varname, var, timestep)
    {tabs}   integer, intent(in) :: ncid
    {tabs}   character(len=*), intent(in) :: varname
    {tabs}   {t_str}, intent(out) :: var
    {tabs}   integer, intent(in) :: timestep
    {tabs}   {t_str} :: tmp(1)
    {tabs}   integer :: var_id, start(1), count(1)

    {tabs}   call check(nf90_inq_varid(ncid, trim(varname), var_id))
    {tabs}   if (timestep > 0) then 
    {tabs}       start = [timestep]
    {tabs}       call check(nf90_get_var(ncid, var_id, tmp, start=[timestep],count=[1]))
    {tabs}       var = tmp(1)
    {tabs}    else 
    {tabs}       call check(nf90_get_var(ncid, var_id, var))
    {tabs}    endif 
    {tabs}end subroutine {subname}

    """)


def gen_read_str() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(f"""
    {tabs}subroutine nc_read_string(ncid, varname, dim_name, var)
    {tabs}   integer, intent(in) :: ncid
    {tabs}   character(len=*), intent(in) :: varname
    {tabs}   character(len=*), intent(in) :: dim_name
    {tabs}   character(len=*), intent(out) :: var
    {tabs}   integer :: var_id, dimid, strlen,i
    {tabs}   character(:), allocatable :: buf

    {tabs}   call check(nf90_inq_varid(ncid, trim(varname), var_id))
    {tabs}   call check(nf90_inq_dimid(ncid, trim(dim_name), dimid))
    {tabs}   call check(nf90_inquire_dimension(ncid, dimid, len=strlen))
    {tabs}   allocate(character(strlen) :: buf); 
    {tabs}   call check(nf90_get_var(ncid, var_id, buf))
    {tabs}   var = ""
    {tabs}   ! netCDF pads with NULs, which trim() keeps; stop at the first one
    {tabs}   do i=1, min(strlen, len(var))
    {tabs}      if (buf(i:i) == achar(0)) exit
    {tabs}      var(i:i) = buf(i:i)
    {tabs}   end do
    {tabs}end subroutine

        """)


def make_dim_str(dim: int, f: Callable[[int], str]) -> str:
    dims = [f(i) for i in range(0, dim)]
    return ",".join(dims)


def gen_read_logical(dim: int) -> str:
    tabs = hio.indent(Tab.reset)
    subname = f"nc_read_logical_{dim}"
    if dim > 0:
        dim_str = make_dim_str(dim, lambda i: ":")
        temp_decl = f"integer, allocatable :: temp({dim_str})"
        bounds_str = make_dim_str(dim, lambda i: f"lbs({i+1}):ubs({i+1})")
        alloc_temp = f"allocate(temp({bounds_str}))"
        return textwrap.dedent(f"""
   {tabs}subroutine {subname}(ncid, varname, ndim, var, timestep)
   {tabs}   integer, intent(in) :: ncid
   {tabs}   character(len=*), intent(in) :: varname
   {tabs}   integer, intent(in) :: ndim
   {tabs}   logical, intent(out) :: var({dim_str})
   {tabs}   integer, intent(in) :: timestep
   {tabs}   {temp_decl}
   {tabs}   ! Locals:
   {tabs}   integer :: lbs(ndim), ubs(ndim)
   {tabs}   lbs = lbound(var); ubs = ubound(var)
   {tabs}   {alloc_temp}

   {tabs}   call nc_read_var(ncid,varname,ndim,temp, timestep)
   {tabs}   var({dim_str}) = .false.
   {tabs}   where(temp == 1) var = .true.

   {tabs}end subroutine {subname}

        """)
    else:
        return textwrap.dedent(f"""
   {tabs} subroutine {subname}(ncid, varname, var, timestep)
   {tabs}   integer, intent(in) :: ncid
   {tabs}   character(len=*), intent(in) :: varname
   {tabs}   logical, intent(out) :: var
   {tabs}   integer, intent(in) :: timestep
   {tabs}   integer :: temp

   {tabs}   call nc_read_var(ncid,varname, temp, timestep)
   {tabs}   if (temp == 1) then
   {tabs}      var = .true.
   {tabs}   else
   {tabs}      var = .false.
   {tabs}   end if

   {tabs} end subroutine {subname}

        """)


def gen_check() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(f"""
    {tabs}subroutine check(status)
    {tabs}   integer, intent(in) :: status
    {tabs}   if (status /= nf90_noerr) then
    {tabs}      print *, trim(nf90_strerror(status))
    {tabs}      stop 2
    {tabs}   end if
    {tabs}end subroutine check

    {tabs}logical function spel_streq(a, b)
    {tabs}   ! case-insensitive string equality, used by namelist guards in ReadWriteMod
    {tabs}   character(len=*), intent(in) :: a, b
    {tabs}   integer :: i, ca, cb
    {tabs}   spel_streq = .false.
    {tabs}   if (len_trim(a) /= len_trim(b)) return
    {tabs}   do i = 1, len_trim(a)
    {tabs}      ca = iachar(a(i:i))
    {tabs}      cb = iachar(b(i:i))
    {tabs}      if (ca >= iachar('A') .and. ca <= iachar('Z')) ca = ca + 32
    {tabs}      if (cb >= iachar('A') .and. cb <= iachar('Z')) cb = cb + 32
    {tabs}      if (ca /= cb) return
    {tabs}   end do
    {tabs}   spel_streq = .true.
    {tabs}end function spel_streq

    {tabs}logical function nc_has_var(ncid, varname)
    {tabs}   ! false for globals that were unallocated when the file was written
    {tabs}   integer, intent(in) :: ncid
    {tabs}   character(len=*), intent(in) :: varname
    {tabs}   integer :: varid
    {tabs}   nc_has_var = nf90_inq_varid(ncid, trim(varname), varid) == nf90_noerr
    {tabs}end function nc_has_var
    """)


def gen_nc_file() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(f"""
    {tabs}integer function nc_create_or_open_file(fn, mode) result(ncid)
    {tabs}   character(len=*), intent(in) :: fn
    {tabs}   integer, intent(in) :: mode
    {tabs}   if (mode == read_file) then
                block 
                   logical :: exists
                   inquire(file=fn,exist=exists)
                   if (.not. exists) stop "Error: " // trim(fn) // " Not Found"
                end block
    {tabs}      call check(nf90_open(trim(fn), nf90_nowrite + nf90_netcdf4, ncid))
    {tabs}   else if(mode == append_file) then
    {tabs}      call check(nf90_open(trim(fn), nf90_write + nf90_netcdf4, ncid))
    {tabs}   else
    {tabs}      call check(nf90_create(trim(fn), nf90_clobber + nf90_netcdf4, ncid))
    {tabs}      if (associated(spel_subset)) call nc_subset_put_atts(ncid)
    {tabs}   end if
    {tabs}end function nc_create_or_open_file

    """)


def gen_nc_define_var() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(f"""
    {tabs}subroutine nc_define_var(ncid, ndim, dims, dim_list, varname, xtype, var_id, time)
    {tabs}   integer, intent(in) :: ncid, ndim
    {tabs}   integer, intent(in) :: dims(ndim)
    {tabs}   character(len=32), dimension(ndim), intent(in) :: dim_list
    {tabs}   character(len=*), intent(in) :: varname
    {tabs}   integer, intent(in) :: xtype  ! e.g. NF90_DOUBLE, NF90_INT, NF90_CHAR, NF90_STRING
    {tabs}   integer, intent(out) :: var_id
    {tabs}   logical, intent(in) :: time
    {tabs}   ! Locals
    {tabs}   integer :: i, status, total_dims, sd
    {tabs}   integer :: ldims(ndim+1)
    {tabs}   integer, allocatable :: dim_ids(:)
    {tabs}   character(len=32), allocatable :: dim_names(:)

    {tabs}   allocate(dim_names(ndim))
    {tabs}   dim_names(:) = dim_list(:)
    {tabs}   ldims(1:ndim) = dims
    {tabs}   ldims(ndim+1) = 0
    {tabs}   sd = nc_subgrid_dim(ndim, dim_names)
    {tabs}   if (sd > 0) then
    {tabs}      if (mod(ldims(sd), spel_nsets) == 0) ldims(sd) = ldims(sd)/spel_nsets
    {tabs}   end if
    {tabs}   if (associated(spel_subset)) then
    {tabs}      sd = nc_first_subgrid_dim(ndim, dim_names)
    {tabs}      if (sd > 0) ldims(sd) = nc_subset_n(dim_names(sd), ldims(sd), varname)
    {tabs}   end if

    {tabs}   total_dims = ndim
    {tabs}   if (time) then 
    {tabs}      total_dims = ndim + 1
    {tabs}      dim_names = [dim_names, 'time']
    {tabs}   end if

    {tabs}   allocate (dim_ids(total_dims))
    {tabs}   do i = 1, total_dims
    {tabs}      status = nf90_inq_dimid(ncid, trim(dim_names(i)), dim_ids(i))
    {tabs}      if(status .ne. nf90_noerr) then
    {tabs}         call check(nf90_def_dim(ncid, trim(dim_names(i)), ldims(i), dim_ids(i)))
    {tabs}      end if
    {tabs}   end do

    {tabs}   if (ndim == 0) then
    {tabs}      ! scalar variable: pass zero-length dim_ids
    {tabs}      call check(nf90_def_var(ncid, trim(varname), xtype, dim_ids(1:0), var_id))
    {tabs}   else
    {tabs}      call check(nf90_def_var(ncid, trim(varname), xtype, dim_ids, var_id))
    {tabs}   end if

    {tabs}   select case (xtype)
    {tabs}   case (nf90_double)
    {tabs}      call check(nf90_put_att(ncid,var_id,"_FillValue", fill_double))
    {tabs}   case (nf90_int)
    {tabs}      call check(nf90_put_att(ncid,var_id,"_FillValue", fill_int))
    {tabs}   end select

    {tabs}   deallocate (dim_ids)
    {tabs}end subroutine
    """)


def gen_nc_write_numeric_scalar() -> list[str]:
    lines: list[str] = []
    n_types = ["integer", "double"]
    map_ftype = {"integer": "integer", "double": "real(8)"}
    tabs = hio.indent(Tab.reset)

    for t in n_types:
        ftype = map_ftype[t]
        lines.append(textwrap.dedent(f"""
       {tabs}subroutine nc_write_{t}_scalar(ncid, var, varname)
       {tabs}   integer, intent(in) :: ncid
       {tabs}   {ftype}, intent(in) :: var
       {tabs}   character(len=*), intent(in) :: varname
       {tabs}   !! locals:
       {tabs}   integer :: var_id

       {tabs}   call check(nf90_inq_varid(ncid, trim(varname), var_id))
       {tabs}   call check(nf90_put_var(ncid, var_id, var))

       {tabs}end subroutine nc_write_{t}_scalar
        """))
    return lines


def gen_nc_write_numeric_array() -> list[str]:
    lines: list[str] = []
    n_types = ["integer", "double"]
    map_ftype = {"integer": "integer", "double": "real(8)"}
    tabs = hio.indent(Tab.reset)

    for t in n_types:
        ftype = map_ftype[t]
        sub_lines = textwrap.dedent(f"""
    {tabs}subroutine nc_write_{t}(ncid, ndim, dims, dim_names, var, varname, timestep)
    {tabs}   integer, intent(in) :: ncid
    {tabs}   integer, intent(in) :: ndim
    {tabs}   integer, intent(in) :: dims(ndim)
    {tabs}   character(len=*), dimension(:), intent(in) :: dim_names
    {tabs}   {ftype}, intent(in) :: var(product(dims))
    {tabs}   character(len=*), intent(in) :: varname
    {tabs}   integer, intent(in) :: timestep
    {tabs}   !! locals:
    {tabs}   integer :: var_id, i
    {tabs}   logical :: unlim
    {tabs}   real(8), allocatable :: data_1d(:), data_2d(:, :), data_3d(:, :, :)
    {tabs}   real(8), allocatable :: buffer(:)
    {tabs}   integer, allocatable :: start(:), count(:)
    {tabs}   integer :: ldims(ndim)

    {tabs}   allocate (buffer(product(dims)))
    {tabs}   buffer = var
    {tabs}   ldims = dims
    {tabs}   if (spel_nsets > 1) call nc_select_set1(ndim, ldims, dim_names, buffer, varname)
    {tabs}   if (associated(spel_subset)) call nc_subset_select(ndim, ldims, dim_names, buffer, varname)
    {tabs}   if(timestep > 0) then 
    {tabs}      unlim = .true.
    {tabs}      allocate(start(ndim+1), count(ndim+1))
    {tabs}   else
    {tabs}      unlim = .false.
    {tabs}      allocate(start(ndim), count(ndim))
    {tabs}   endif

    {tabs}   do i = 1, ndim
    {tabs}      start(i) = 1
    {tabs}      count(i) = ldims(i)
    {tabs}   end do

    {tabs}   if (unlim) then 
    {tabs}      start(ndim+1) = timestep
    {tabs}      count(ndim+1) = 1
    {tabs}   endif

    {tabs}   call check(nf90_inq_varid(ncid, trim(varname), var_id))
    {tabs}   select case (ndim)
    {tabs}   case (1)
    {tabs}      allocate (data_1d(ldims(1))); data_1d = reshape(buffer, [ldims(1)])
    {tabs}      call check(nf90_put_var(ncid, var_id, data_1d,start=start,count=count))
    {tabs}   case (2)
    {tabs}      allocate (data_2d(ldims(1), ldims(2))); data_2d = reshape(buffer, [ldims(1), ldims(2)])
    {tabs}      call check(nf90_put_var(ncid, var_id, data_2d,start=start,count=count))
    {tabs}   case (3)
    {tabs}      allocate (data_3d(ldims(1), ldims(2), ldims(3))); data_3d = reshape(buffer, [ldims(1), ldims(2), ldims(3)])
    {tabs}      call check(nf90_put_var(ncid, var_id, data_3d,start=start,count=count))
    {tabs}   case default
    {tabs}      stop "(nc_write_{t}) doesn't support >3D doubles"
    {tabs}   end select
    {tabs}end subroutine nc_write_{t}

        """)
        lines.append(sub_lines)
    return lines


def gen_nc_write_string() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(f"""
    {tabs}subroutine nc_write_string(ncid, var, varname)
    {tabs}  integer, intent(in) :: ncid
    {tabs}   character(len=*), intent(in) :: varname
    {tabs}   character(len=*), intent(in) :: var
    {tabs}   integer :: var_id, i
    {tabs}   call check(nf90_inq_varid(ncid, trim(varname), var_id))
    {tabs}   call check(nf90_put_var(ncid, var_id, trim(var)))
    {tabs}end subroutine nc_write_string
    """)


def gen_nc_write_logical() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(f"""
    {tabs}subroutine nc_write_logical(ncid, ndim, dims, dim_names, var, varname, timestep)
    {tabs}   integer, intent(in) :: ncid, ndim
    {tabs}   integer, intent(in) :: dims(ndim)
    {tabs}   character(len=*), dimension(:), intent(in) :: dim_names
    {tabs}   logical, intent(in) :: var(product(dims))
    {tabs}   character(len=*), intent(in) :: varname
    {tabs}   integer, intent(in) :: timestep
    {tabs}   integer :: var_id, i
    {tabs}   integer, allocatable :: int_buf(:)
    {tabs}   integer, allocatable :: data_1d(:), data_2d(:, :), data_3d(:, :, :)
    {tabs}   logical :: unlim
    {tabs}   integer, allocatable :: start(:), count(:)

    {tabs}   integer :: ldims(ndim)

    {tabs}   allocate (int_buf(product(dims)))
    {tabs}   int_buf = merge(1, 0, var)  ! logical → int (1=true, 0=false)
    {tabs}   ldims = dims
    {tabs}   if (spel_nsets > 1) call nc_select_set1(ndim, ldims, dim_names, int_buf, varname)
    {tabs}   if (associated(spel_subset)) call nc_subset_select(ndim, ldims, dim_names, int_buf, varname)
    {tabs}   if(timestep > 0) then 
    {tabs}      unlim = .true.
    {tabs}      allocate(start(ndim+1), count(ndim+1))
    {tabs}   else
    {tabs}      unlim = .false.
    {tabs}      allocate(start(ndim), count(ndim))
    {tabs}   endif

    {tabs}   do i = 1, ndim
    {tabs}      start(i) = 1
    {tabs}      count(i) = ldims(i)
    {tabs}   end do

    {tabs}   if (unlim) then 
    {tabs}      start(ndim+1) = timestep
    {tabs}      count(ndim+1) = 1
    {tabs}   endif

    {tabs}   call check(nf90_inq_varid(ncid, trim(varname), var_id))
    {tabs}   select case (ndim)
    {tabs}   case (1)
    {tabs}      allocate (data_1d(ldims(1))); data_1d = reshape(int_buf, [ldims(1)])
    {tabs}      call check(nf90_put_var(ncid, var_id, data_1d,start=start,count=count))
    {tabs}   case (2)
    {tabs}      allocate (data_2d(ldims(1), ldims(2))); data_2d = reshape(int_buf, [ldims(1), ldims(2)])
    {tabs}      call check(nf90_put_var(ncid, var_id, data_2d,start=start,count=count))
    {tabs}   case (3)
    {tabs}      allocate (data_3d(ldims(1), ldims(2), ldims(3))); data_3d = reshape(int_buf, [ldims(1), ldims(2), ldims(3)])
    {tabs}      call check(nf90_put_var(ncid, var_id, data_3d,start=start,count=count))
    {tabs}   case default
    {tabs}      stop "(nc_write_logical) doesn't support >3D vars"
    {tabs}   end select
    {tabs}end subroutine nc_write_logical

    {tabs}subroutine nc_write_logical_scalar(ncid, var, varname)
    {tabs}   integer, intent(in) :: ncid
    {tabs}   logical, intent(in) :: var
    {tabs}   character(len=*), intent(in) :: varname
    {tabs}   !! locals:
    {tabs}   integer :: var_id, buf
    {tabs}   buf = 0
    {tabs}   if (var) buf = 1
    {tabs}   call check(nf90_inq_varid(ncid, trim(varname), var_id))
    {tabs}   call check(nf90_put_var(ncid, var_id, buf))
    {tabs}end subroutine nc_write_logical_scalar
    """)


def gen_nc_read_timeslices() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(f"""
    {tabs}integer function nc_read_timeslices(fn) result(time_len)
    {tabs}    character(len=*), intent(in) :: fn

    {tabs}    integer :: ncid         ! file ID
    {tabs}    integer :: time_dimid   ! dimension ID for 'time'
    {tabs}    integer :: ierr         ! error status

    {tabs}    ! Open file in read-only mode
    {tabs}    ierr = nf90_open(trim(fn), NF90_NOWRITE, ncid)
    {tabs}    if (ierr /= nf90_noerr) stop "Error opening file"

    {tabs}    ! Inquire about the 'time' dimension
    {tabs}    ierr = nf90_inq_dimid(ncid, "time", time_dimid)
    {tabs}    if (ierr /= nf90_noerr) stop "Error: 'time' dimension not found"

    {tabs}    ! Get the length of the 'time' dimension
    {tabs}    call check(nf90_inquire_dimension(ncid, time_dimid, len=time_len))
    {tabs}    if (ierr /= nf90_noerr) stop "Error inquiring length of 'time' dimension"


    {tabs}    ! Close file
    {tabs}    call check(nf90_close(ncid))

    {tabs}end function nc_read_timeslices
    """)


if __name__ == "__main__":
    generate_nc_io()
