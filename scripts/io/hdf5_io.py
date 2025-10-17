import re

import scripts.io.helper as hio
from scripts.DerivedType import DerivedType
from scripts.utilityFunctions import Variable


def match_h5_type(var_type: str) -> str:
    match var_type:
        case "real":
            return "H5T_NATIVE_DOUBLE"
        case "integer":
            return "H5T_NATIVE_INTEGER"
        case "logical":
            return "H5T_NATIVE_INTEGER"
        case "character":
            return "H5T_C_S1"
        case _:
            raise ValueError(f"Unsupported Fortran type: {var_type}")


def generate_elmtypes_io_hdf5(
    type_dict: dict[str, DerivedType],
    inst_to_dtype_map: dict[str, str],
    casedir: str,
):
    tabs = hio.indent(hio.Tab.reset)
    filename = "ReadWriteMod.F90"
    mod_name = filename.replace(".F90", "")

    active_instances = {
        inst_var.name: inst_var
        for dtype in type_dict.values()
        for inst_var in dtype.instances.values()
        if inst_var.active
    }

    lines: list[str] = []
    lines.append(f"module {mod_name}\n")
    lines.append(f"{tabs}!!! Auto-generated Fortran code for HDF5 I/O\n")
    lines.append(f"{tabs}use hdf5\n")
    use_statements: set[str] = set()
    for inst_var in active_instances.values():
        stmt = f"{tabs}use {inst_var.declaration}, only: {inst_var.name}\n"
        use_statements.add(stmt)
    lines.extend(list(use_statements))

    lines.extend(
        [
            f"{tabs}implicit none\n",
            f"{tabs}public :: read_elmtypes, write_elmtypes\n",
            "contains\n",
        ]
    )

    dtype_vars: dict[str, Variable] = {}

    for inst_var in active_instances.values():
        type_name = inst_to_dtype_map[inst_var.name]
        dtype = type_dict[type_name]
        for field_var in dtype.components.values():
            if field_var.active:
                new_var = field_var.copy()
                new_var.name = f"{inst_var.name}%{field_var.name.split('%')[-1]}"
                dtype_vars[new_var.name] = new_var

    print("Number of fields for i/o:", len(dtype_vars.keys()))
    sub_lines = create_h5io_routine(
        mode="r",
        sub_name="read_elmtypes",
        vars=dtype_vars,
        fn="spel_elmtypes.h5",
    )
    lines.extend(sub_lines)
    sub_lines = create_h5io_routine(
        mode="w",
        sub_name="write_elmtypes",
        vars=dtype_vars,
        fn="spel_elmtypes.h5",
    )
    lines.extend(sub_lines)

    lines.append(f"end module {mod_name}\n")

    with open(f"{casedir}/{filename}", "w") as ofile:
        ofile.writelines(lines)

    return


def generate_constants_io_hdf5(
    vars: dict[str, Variable],
    casedir: str,
):
    tabs = hio.indent(hio.Tab.reset)
    filename = "FUTConstantsMod.F90"
    mod_name = filename.replace(".F90", "")

    lines: list[str] = []
    lines.append(f"module {mod_name}\n")
    lines.append(f"{tabs}!!! Auto-generated Fortran code for HDF5 I/O\n")
    lines.append(f"{tabs}use hdf5\n")
    use_stmts, _ = hio.var_use_statements(vars)
    lines.extend(use_stmts)

    lines.extend(
        [
            f"{tabs}implicit none\n",
            f"{tabs}public :: read_constants, write_constants\n",
            "contains\n",
        ]
    )

    sub_lines = create_h5io_routine(
        mode="r",
        sub_name="read_constants",
        vars=vars,
        fn="spel_constants.h5",
    )
    lines.extend(sub_lines)
    sub_lines = create_h5io_routine(
        mode="w",
        sub_name="write_constants",
        vars=vars,
        fn="spel_constants.h5",
    )
    lines.extend(sub_lines)

    lines.append(f"end module {mod_name}")

    with open(f"{casedir}/{filename}", "w") as ofile:
        ofile.writelines(lines)

    return


def create_h5io_routine(
    mode: str,
    sub_name: str,
    vars: dict[str, Variable],
    fn: str,
):
    lines: list[str] = []
    tabs = hio.indent()

    max_var_dim = 5
    lines.append(f"{tabs}subroutine {sub_name}(nsets)\n")
    tabs = hio.indent(hio.Tab.shift)

    lines.extend(
        [
            f"{tabs}use H5InterfaceMod\n",
            f"{tabs}integer, intent(in) :: nsets\n",
            f"{tabs}integer, parameter :: maxdim={max_var_dim}\n",
            f"{tabs}integer, parameter :: mode_create = 0, mode_open=1\n",
            f"{tabs}integer :: errcode\n",
            f"{tabs}integer(hid_t) :: file_id\n",
            f"{tabs}integer :: lbounds(5), ubounds(5), delta, nsets_m_1, err\n",
            f"{tabs}character(len=256) :: fn='{fn}'\n\n",
            f"{tabs}nsets_m_1 = nsets - 1\n",
            f"{tabs}call h5open_f(err) ! initialize hdf5\n",
        ]
    )

    if mode == "r":
        lines.append(f"{tabs}call h5_open_create_file(fn, mode_open, file_id)\n")
    elif mode == "w":
        lines.append(f"{tabs}call h5_open_create_file(fn, mode_create, file_id)\n")

    for var in vars.values():
        # Open dataset
        if var.dim > 0:
            newlines = h5_array_read(var) if mode == "r" else h5_array_write(var)
        else:
            newlines = h5_scalar(mode, var)

        lines.extend(newlines)

    lines.append(f"{tabs}call h5fclose_f(file_id, errcode)\n")
    lines.append(f"{tabs}call h5close_f(errcode)\n")

    tabs = hio.indent(hio.Tab.unshift)
    lines.append(f"{tabs}end subroutine {sub_name}\n\n")
    return lines


def h5_array_write(var: Variable) -> list[str]:
    lines: list[str] = []
    tabs = hio.indent()

    # Write the bounds for each dimension
    lubounds = set_lubounds(var)

    lines.extend(lubounds)
    lines.extend(
        [
            # bounds dims
            f"{tabs}call h5_write_data(file_id, '{var.name}_lb', lbounds(1:{var.dim}))\n",
            f"{tabs}call h5_write_data(file_id, '{var.name}_ub', ubounds(1:{var.dim}))\n",
            # Write the actual array data
            f"{tabs}call h5_write_data(file_id, '{var.name}', {var.name})\n",
        ]
    )

    return lines


def set_lubounds(var: Variable) -> list[str]:
    tabs = hio.indent()
    lines: list[str] = []
    lines.append(f"{tabs}lbounds(1:{var.dim}) = lbound({var.name})\n")
    lines.append(f"{tabs}ubounds(1:{var.dim}) = ubound({var.name})\n")
    return lines


def h5_array_read(var: Variable) -> list[str]:
    lines: list[str] = []
    tabs = hio.indent()

    # Allocate array if necessary
    if var.allocatable:
        # Read the bounds for each dimension
        lines.extend(
            [
                f"{tabs}call h5_read_data(file_id,'{var.name}_lb', lbounds(1:{var.dim}))\n"
                f"{tabs}call h5_read_data(file_id,'{var.name}_ub', ubounds(1:{var.dim}))\n"
            ]
        )
        alloc_ = [f"lbounds({i}):ubounds({i})" for i in range(1, var.dim + 1)]
        if var.subgrid in ["g", "l", "t", "c", "p"]:
            lines.append(f"{tabs}delta = ubounds(1)-lbounds(1)+1\n")
            alloc_[0] = f"{alloc_[0]}+nsets_m_1*delta"
        alloc_str = ",".join(alloc_)
        lines.append(f"{tabs}allocate({var.name}({alloc_str}))\n")
    lines.append(f"{tabs}call h5_read_data(file_id, '{var.name}', {var.name})\n")

    return lines


def h5_scalar(mode: str, var: Variable) -> list[str]:
    lines: list[str] = []
    tabs = hio.indent()

    if mode == "r":
        lines.append(f"{tabs}call h5_read_data(file_id, '{var.name}', {var.name})\n")
    else:
        lines.append(f"{tabs}call h5_write_data(file_id, '{var.name}', {var.name})\n")

    return lines
