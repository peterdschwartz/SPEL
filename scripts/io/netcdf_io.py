import re
import sys
import textwrap
from collections.abc import Callable

import scripts.io.helper as hio
from scripts import logging_configs
from scripts.config import spel_mods_dir
from scripts.DerivedType import DerivedType
from scripts.utilityFunctions import Variable

Tab = hio.Tab


def create_nc_define_vars(
    vars: dict[str, Variable],
    time: bool,
    elminst_vars: list[tuple[str,Variable]]=[],
    bounds: bool = False,
) -> list[str]:
    """
    Create Subroutine for defining netcdf variables
    """
    tabs = hio.indent()
    arg_str = ",bounds" if bounds else ""

    decls: list[str] = []
    for type_mod, inst in elminst_vars:
        arg_str = f"{arg_str}, {inst.name}"
        var_type = inst.type
        decls.append(f"{tabs}{tabs}type({var_type}), intent(inout) :: {inst.name}")
    lines: list[str] = [f"{tabs}subroutine define_vars(ncid{arg_str})\n"]
    tabs = hio.indent(hio.Tab.shift)
    lines.append(f"{tabs}integer, intent(in) :: ncid\n")
    if bounds:
        lines.append(f"{tabs}type(bounds_type), intent(in) :: bounds\n")
    lines.extend([f"{decl}\n" for decl in decls])

    lines.append(f"{tabs}integer :: varid, time_id\n")
    lines.append(f"{tabs}character(len=32), dimension(5) :: dim_names\n")
    if time:
        lines.append(
            f"{tabs}call check(nf90_def_dim(ncid, 'time', NF90_UNLIMITED, time_id))\n"
        )
    nc_defns = create_nc_def(vars, time)
    lines.extend(nc_defns)
    tabs = hio.indent(hio.Tab.unshift)
    lines.append(f"{tabs}end subroutine define_vars\n")

    return lines


def generate_elmtypes_io_netcdf(
    type_dict: dict[str, DerivedType],
    inst_to_dtype_map: dict[str, str],
    casedir: str,
):
    tabs = hio.indent(hio.Tab.reset)
    filename = "ReadWriteMod.F90"
    mod_name = filename.replace(".F90", "")

    logger = logging_configs.get_logger("genNetCDF")

    lines: list[str] = []
    lines.extend(
        [
            f"module {mod_name}\n",
            f"{tabs}!!! Auto-generated Fortran code for netcdf-fortran I/O\n",
            f"{tabs}use netcdf\n",
            f"{tabs}use nc_io\n",
            f"{tabs}use nc_allocMod\n",
        ]
    )

    def active_mask(dtype: DerivedType, inst_name: str) -> bool:
        all_ptrs = bool(
            len([field for field in dtype.components.values() if not field.pointer])
            == 0
        )
        return (
            not all_ptrs
            and inst_name not in ["filter", "filter_inactive_and_active"]
            and not re.match("(c13|c14)", inst_name)
        )

    active_instances = {
        inst_var.name: inst_var
        for dtype in type_dict.values()
        for inst_var in dtype.instances.values()
        if inst_var.active and active_mask(dtype,inst_var.name)
    }
    use_statements, elminst_vars = hio.var_use_statements(active_instances, type_dict)
    lines.extend(list(use_statements))

    lines.extend(
        [
            f"{tabs}implicit none\n",
            f"{tabs}public :: read_elmtypes, write_elmtypes, define_vars\n",
            "contains\n",
        ]
    )

    dtype_vars: dict[str, Variable] = {}

    for inst_var in active_instances.values():
        type_name = inst_to_dtype_map[inst_var.name]
        dtype = type_dict[type_name]
        for field_var in dtype.components.values():
            if field_var.active and not field_var.pointer:
                new_var = field_var.copy()
                new_var.name = f"{inst_var.name}%{field_var.name.split('%')[-1]}"
                dtype_vars[new_var.name] = new_var

    sub_lines = create_nc_define_vars(dtype_vars, elminst_vars=elminst_vars,time=True, bounds=True)
    lines.extend(sub_lines)
    sub_lines = create_netcdf_io_routine(
        mode=hio.IOMode.read,
        sub_name="read_elmtypes",
        vars=dtype_vars,
        time=True,
        bounds=True,
        elminst_vars=elminst_vars,
    )
    lines.extend(sub_lines)
    sub_lines = create_netcdf_io_routine(
        mode=hio.IOMode.write,
        sub_name="write_elmtypes",
        vars=dtype_vars,
        time=True,
        bounds=True,
        elminst_vars=elminst_vars,
    )
    lines.extend(sub_lines)

    lines.append(f"end module {mod_name}\n")

    logger.info(f"Writing {filename}")
    with open(f"{casedir}/{filename}", "w") as ofile:
        ofile.writelines(lines)

    return


def generate_constants_io_netcdf(vars: dict[str, Variable], casedir: str):
    """
    Function generates fortran module for constants needed in unit-test
    """

    tabs = hio.indent(hio.Tab.reset)

    filename = "FUTConstantsMod.F90"
    mod_name = filename.replace(".F90", "")

    lines: list[str] = []
    lines.append(f"module {mod_name}\n")
    lines.append(f"{tabs}!!! Auto-generated Fortran code for netcdf-fortran I/O\n")
    lines.append(f"{tabs}use netcdf\n")
    lines.append(f"{tabs}use nc_io\n")
    lines.append(f"{tabs}use nc_allocMod\n")
    use_stmts, _ = hio.var_use_statements(vars)
    lines.extend(use_stmts)

    lines.extend(
        [
            f"{tabs}implicit none\n",
            f"{tabs}public :: read_constants, write_constants, define_vars\n",
            "contains\n",
        ]
    )
    sub_lines = create_nc_define_vars(vars, time=False)
    lines.extend(sub_lines)

    sub_lines = create_netcdf_io_routine(
        hio.IOMode.read,
        "read_constants",
        vars,
        time=False,
        elminst_vars=[],
    )

    lines.extend(sub_lines)

    sub_lines = create_netcdf_io_routine(
        hio.IOMode.write,
        "write_constants",
        vars,
        time=False,
        elminst_vars=[],
    )
    lines.extend(sub_lines)

    tabs = hio.indent(hio.Tab.unshift)
    lines.extend(f"{tabs}end module {mod_name}\n")

    with open(f"{casedir}/{filename}", "w") as ofile:
        ofile.writelines(lines)

    return


def create_netcdf_io_routine(
    mode: hio.IOMode,
    sub_name: str,
    vars: dict[str, Variable],
    time: bool,
    elminst_vars: list[tuple[str,Variable]],
    bounds: bool = False,
) -> list[str]:
    tabs = hio.indent()
    if bounds:
        arg_str: str = "io_inst, bounds"
        define_args: str = "ncid, bounds"
    else:
        arg_str: str = "io_inst"
        define_args: str = "ncid"

    decls: list[str] = []
    for type_mod, inst in elminst_vars:
        arg_str = f"{arg_str}, {inst.name}"
        define_args = f"{define_args}, {inst.name}"
        var_type = inst.type
        decls.append(f"{tabs}{tabs}type({var_type}), intent(inout) :: {inst.name}")

    lines: list[str] = [f"{tabs}subroutine {sub_name}({arg_str})\n"]
    tabs = hio.indent(hio.Tab.shift)

    mode_str = "create_file" if mode == hio.IOMode.write else "read_file"
    stmt = f"{tabs}type(bounds_type), intent(inout) :: bounds\n" if bounds else ""
    lines.extend([f"{decl}\n" for decl in decls])

    # Arguments + Locals:
    lines.extend(
        [
            f"{tabs}type(spel_io_type), intent(inout) :: io_inst\n" f"{stmt}",
            f"{tabs}integer :: ncid, timestep\n",
            f"{tabs}character(len=256) :: new_fn\n",
        ]
    )
    lines.append(f"{tabs}new_fn = trim(io_inst%get_fn())\n")

    if mode == hio.IOMode.write:
        if bounds:
            lines.extend(
                [
                    f"{tabs}if( io_inst%new_file) then\n",
                    f"{tabs}{tabs}print *, 'creating file', trim(new_fn)\n"
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
        else:
            lines.extend(
                [
                    f"{tabs}print *, 'creating file: ', trim(new_fn)\n"
                    f"{tabs}ncid = nc_create_or_open_file(trim(new_fn), create_file)\n",
                    f"{tabs}call define_vars({define_args})\n",
                    f"{tabs}call check(nf90_enddef(ncid))\n",
                ]
            )

    if mode == hio.IOMode.read:
        lines.append(f"{tabs}if(io_inst%end_run) return\n")
        if bounds:
            lines.append(
                f"{tabs}if(io_inst%dt_in_file == 99999) io_inst%dt_in_file = nc_read_timeslices(new_fn)\n"
            )
        lines.append(f"{tabs}timestep = io_inst%timestep\n")
        lines.append(f"{tabs}ncid = nc_create_or_open_file(trim(new_fn), {mode_str})\n")
        sub_lines = create_nc_read(vars, time)
    else:
        sub_lines = create_nc_write(vars, time=time)
    lines.extend(sub_lines)
    lines.append(f"{tabs}call check(nf90_close(ncid))\n")
    tabs = hio.indent(hio.Tab.unshift)
    lines.append(f"{tabs}end subroutine {sub_name}\n")

    return lines



def create_nc_def(vars: dict[str, Variable], time: bool) -> list[str]:
    """
    Function generates calls to 'nc_define_vars' calls for defining variables prior to writing
    """
    lines: list[str] = []
    tabs = hio.indent()

    for var in vars.values():
        dim_names_str = get_dim_names(var, time)
        varname = var.name.replace("%", "__")
        nc_type = match_nc_type(var.type)
        time_str = ".true." if time and var.dim > 0 else ".false."
        if nc_type == "nf90_char":
            dim_str = f"[len({var.name})]"
            dim = 1 + time
            stmt = f"call nc_define_var(ncid, 1, {dim_str}, dim_names, '{varname}', {nc_type}, varid, {time_str})\n"
        else:
            dim_str = f"shape({var.name})" if var.dim > 0 else "[0]"
            dim = var.dim + time
            stmt = f"call nc_define_var(ncid, {var.dim}, {dim_str}, dim_names, '{varname}', {nc_type}, varid, {time_str})\n"

        if var.dim > 0 or nc_type == "nf90_char":
            lines.append(f"{tabs}dim_names(1:{dim}) = {dim_names_str}\n")
        lines.append(f"{tabs}{stmt}")
        # if array store lbounds and ubounds:
        if var.dim > 0:
            stmt = f"call check(nf90_put_att(ncid, varid, 'lbounds', lbound({var.name}))); call check(nf90_put_att(ncid, varid, 'ubounds', ubound({var.name})));"
            lines.append(f"{tabs}{stmt}\n")

    return lines


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
    if time:
        dim_names.append("'time'")

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


def create_nc_write(vars: dict[str, Variable], time: bool) -> list[str]:
    """
    Function to create the
        call nc_write_var(ncid, dim, shape, dim_names, var, varname)
    or for characters:
        call nc_write_var(ncid, var, varname)
    """
    lines: list[str] = []
    tabs = hio.indent()

    scalars = [var for var in vars.values() if var.dim == 0]
    arrays = [var for var in vars.values() if var.dim > 0]

    timestep = ", timestep" if time else ", -1"

    for var in scalars:
        varname = var.name.replace("%", "__")
        if var.type == "character":
            stmt = f"call nc_write_var_array(ncid, {var.name}, '{varname}')\n"
        else:
            stmt = f"call nc_write_var_scalar(ncid, {var.name}, '{varname}')\n"
        lines.append(f"{tabs}{stmt}")

    for var in arrays:
        dim_names_str = get_dim_names(var, time)
        reshape_str = f"reshape({var.name}, [product(shape({var.name}))])"
        varname = var.name.replace("%", "__")
        stmt = f"call nc_write_var_array(ncid,{var.dim}, shape({var.name}), {dim_names_str}, {reshape_str}, '{varname}'{timestep})\n"
        # lines.append(f"{tabs}print *, 'Writing {var.name}'\n")
        lines.append(f"{tabs}{stmt}")

    return lines


def create_nc_read(vars: dict[str, Variable], time: bool) -> list[str]:
    """
    Function to create the
        call nc_read_var(ncid, dim, shape, dim_names, var, varname)
    or for characters:
        call nc_read_var(ncid, var, varname)
    """
    lines: list[str] = []
    tabs = hio.indent()
    scalars = [var for var in vars.values() if var.dim == 0]
    arrays = [var for var in vars.values() if var.dim > 0]

    time_str = ", timestep" if time else ", -1"

    for var in scalars:
        varname = var.name.replace("%", "__")
        if var.ptrscalar:
            lines.append(f"{tabs}allocate({var.name})\n")
        if var.type == "character":
            stmt = (
                f"call nc_read_var(ncid, '{varname}', '{var.name}_str', {var.name})\n"
            )
        else:
            stmt = f"call nc_read_var(ncid, '{varname}', {var.name}{time_str})\n"
        lines.append(f"{tabs}{stmt}")

    for var in arrays:
        assert (
            var.type != "character"
        ), f"Error - Need to implement array of characters nc write for {var.name}"

        varname = var.name.replace("%", "__")
        lines.append(f'{tabs}call nc_alloc(ncid, "{varname}", {var.dim}, {var.name})\n')
        stmt = f"call nc_read_var(ncid,'{varname}', {var.dim}, {var.name}{time_str})\n"
        lines.append(f"{tabs}{stmt}")

    return lines


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
    use_statements, _ = hio.var_use_statements(active_instances,type_dict)
    lines.extend(list(use_statements))

    return


def generate_nc_io():
    lines: list[str] = []

    type_list = ["double", "integer", "logical", "string"]
    read_dims = 3

    tabs = hio.indent(hio.Tab.reset)
    header: str = textwrap.dedent(
        f"""
    module nc_io
    {tabs}use netcdf
    {tabs}use iso_fortran_env
    {tabs}use iso_c_binding
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
    """
    )
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
    lines.append(f"{tabs}public :: nc_read_timeslices\n")

    lines.append("contains\n")

    nc_check = gen_check()
    lines.append(nc_check)
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

    lines.append("end module nc_io\n")

    with open(f"{spel_mods_dir}/nc_io.F90", "w") as ofile:
        ofile.writelines(lines)


def gen_spel_type_routines() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(
        f"""
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
    """
    )


def gen_nc_read_type(dim: int, t: str) -> str:
    tabs = hio.indent(Tab.reset)
    if t == "string":
        return gen_read_str()
    if t == "logical":
        return gen_read_logical(dim)
    else:
        return gen_read_numeric(dim, t)


def gen_read_numeric(dim: int, t: str) -> str:
    tabs = hio.indent(Tab.reset)
    if t == "double":
        t_str = "real(8)"
    else:
        t_str = t
    subname = f"nc_read_{t}_{dim}"
    if dim > 0:
        dim_str = make_dim_str(dim, lambda i: ":")

        return textwrap.dedent(
            f"""
    {tabs}subroutine {subname}(ncid, varname, ndim, var, timestep)
    {tabs}   integer, intent(in) :: ncid
    {tabs}   character(len=*), intent(in) :: varname
    {tabs}   integer, intent(in) :: ndim 
    {tabs}   {t_str}, intent(out) :: var({dim_str})
    {tabs}   integer, intent(in) :: timestep
    {tabs}   integer :: var_id
    {tabs}   integer, allocatable :: start(:), count(:)

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
    {tabs}   call check(nf90_get_var(ncid, var_id, var,start=start, count=count))
    {tabs}end subroutine {subname}

        """
        )
    else:  # dim == 0
        return textwrap.dedent(
            f"""
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

    """
        )


def gen_read_str() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(
        f"""
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
    {tabs}   do i=1, strlen 
    {tabs}      var(i:i) = buf(i:i)
    {tabs}   end do
    {tabs}end subroutine

        """
    )


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
        return textwrap.dedent(
            f"""
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

        """
        )
    else:
        return textwrap.dedent(
            f"""
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

        """
        )


def gen_check() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(
        f"""
    {tabs}subroutine check(status)
    {tabs}   integer, intent(in) :: status
    {tabs}   if (status /= nf90_noerr) then
    {tabs}      print *, trim(nf90_strerror(status))
    {tabs}      stop 2
    {tabs}   end if
    {tabs}end subroutine check
    """
    )


def gen_nc_file() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(
        f"""
    {tabs}integer function nc_create_or_open_file(fn, mode) result(ncid)
    {tabs}   character(len=*), intent(in) :: fn
    {tabs}   integer, intent(in) :: mode
    {tabs}   if (mode == read_file) then
    {tabs}      call check(nf90_open(trim(fn), nf90_nowrite + nf90_netcdf4, ncid))
    {tabs}   else if(mode == append_file) then
    {tabs}      call check(nf90_open(trim(fn), nf90_write + nf90_netcdf4, ncid))
    {tabs}   else
    {tabs}      call check(nf90_create(trim(fn), nf90_clobber + nf90_netcdf4, ncid))
    {tabs}   end if
    {tabs}end function nc_create_or_open_file

    """
    )


def gen_nc_define_var() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(
        f"""
    {tabs}subroutine nc_define_var(ncid, ndim, dims, dim_names, varname, xtype, var_id, time)
    {tabs}   integer, intent(in) :: ncid, ndim
    {tabs}   integer, intent(in) :: dims(ndim)
    {tabs}   character(len=32), dimension(ndim), intent(in) :: dim_names
    {tabs}   character(len=*), intent(in) :: varname
    {tabs}   integer, intent(in) :: xtype  ! e.g. NF90_DOUBLE, NF90_INT, NF90_CHAR, NF90_STRING
    {tabs}   integer, intent(out) :: var_id
    {tabs}   logical, intent(in) :: time
    {tabs}   ! Locals
    {tabs}   integer :: i, status, total_dims
    {tabs}   integer, allocatable :: dim_ids(:)

    {tabs}   if (time) then 
    {tabs}      total_dims = ndim + 1
    {tabs}   else
    {tabs}      total_dims = ndim
    {tabs}   end if

    {tabs}   allocate (dim_ids(total_dims))
    {tabs}   do i = 1, total_dims
    {tabs}      status = nf90_inq_dimid(ncid, trim(dim_names(i)), dim_ids(i))
    {tabs}      if(status .ne. nf90_noerr) then
    {tabs}         call check(nf90_def_dim(ncid, trim(dim_names(i)), dims(i), dim_ids(i)))
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
    """
    )


def gen_nc_write_numeric_scalar() -> list[str]:
    lines: list[str] = []
    n_types = ["integer", "double"]
    map_ftype = {"integer": "integer", "double": "real(8)"}
    tabs = hio.indent(Tab.reset)

    for t in n_types:
        ftype = map_ftype[t]
        lines.append(
            textwrap.dedent(
                f"""
       {tabs}subroutine nc_write_{t}_scalar(ncid, var, varname)
       {tabs}   integer, intent(in) :: ncid
       {tabs}   {ftype}, intent(in) :: var
       {tabs}   character(len=*), intent(in) :: varname
       {tabs}   !! locals:
       {tabs}   integer :: var_id

       {tabs}   call check(nf90_inq_varid(ncid, trim(varname), var_id))
       {tabs}   call check(nf90_put_var(ncid, var_id, var))

       {tabs}end subroutine nc_write_{t}_scalar
        """
            )
        )
    return lines


def gen_nc_write_numeric_array() -> list[str]:
    lines: list[str] = []
    n_types = ["integer", "double"]
    map_ftype = {"integer": "integer", "double": "real(8)"}
    tabs = hio.indent(Tab.reset)

    for t in n_types:
        ftype = map_ftype[t]
        sub_lines = textwrap.dedent(
            f"""
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

    {tabs}   allocate (buffer(product(dims)))
    {tabs}   buffer = var
    {tabs}   if(timestep > 0) then 
    {tabs}      unlim = .true.
    {tabs}      allocate(start(ndim+1), count(ndim+1))
    {tabs}   else
    {tabs}      unlim = .false.
    {tabs}      allocate(start(ndim), count(ndim))
    {tabs}   endif

    {tabs}   do i = 1, ndim
    {tabs}      start(i) = 1
    {tabs}      count(i) = dims(i)
    {tabs}   end do

    {tabs}   if (unlim) then 
    {tabs}      start(ndim+1) = timestep
    {tabs}      count(ndim+1) = 1
    {tabs}   endif

    {tabs}   call check(nf90_inq_varid(ncid, trim(varname), var_id))
    {tabs}   select case (ndim)
    {tabs}   case (1)
    {tabs}      allocate (data_1d(dims(1))); data_1d = reshape(buffer, [dims(1)])
    {tabs}      call check(nf90_put_var(ncid, var_id, data_1d,start=start,count=count))
    {tabs}   case (2)
    {tabs}      allocate (data_2d(dims(1), dims(2))); data_2d = reshape(buffer, [dims(1), dims(2)])
    {tabs}      call check(nf90_put_var(ncid, var_id, data_2d,start=start,count=count))
    {tabs}   case (3)
    {tabs}      allocate (data_3d(dims(1), dims(2), dims(3))); data_3d = reshape(buffer, [dims(1), dims(2), dims(3)])
    {tabs}      call check(nf90_put_var(ncid, var_id, data_3d,start=start,count=count))
    {tabs}   case default
    {tabs}      stop "(nc_write_{t}) doesn't support >3D doubles"
    {tabs}   end select
    {tabs}end subroutine nc_write_{t}

        """
        )
        lines.append(sub_lines)
    return lines


def gen_nc_write_string() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(
        f"""
    {tabs}subroutine nc_write_string(ncid, var, varname)
    {tabs}  integer, intent(in) :: ncid
    {tabs}   character(len=*), intent(in) :: varname
    {tabs}   character(len=*), intent(in) :: var
    {tabs}   integer :: var_id, i
    {tabs}   call check(nf90_inq_varid(ncid, trim(varname), var_id))
    {tabs}   call check(nf90_put_var(ncid, var_id, trim(var)))
    {tabs}end subroutine nc_write_string
    """
    )


def gen_nc_write_logical() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(
        f"""
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

    {tabs}   allocate (int_buf(product(dims)))
    {tabs}   int_buf = merge(1, 0, var)  ! logical → int (1=true, 0=false)
    {tabs}   if(timestep > 0) then 
    {tabs}      unlim = .true.
    {tabs}      allocate(start(ndim+1), count(ndim+1))
    {tabs}   else
    {tabs}      unlim = .false.
    {tabs}      allocate(start(ndim), count(ndim))
    {tabs}   endif

    {tabs}   do i = 1, ndim
    {tabs}      start(i) = 1
    {tabs}      count(i) = dims(i)
    {tabs}   end do

    {tabs}   if (unlim) then 
    {tabs}      start(ndim+1) = timestep
    {tabs}      count(ndim+1) = 1
    {tabs}   endif

    {tabs}   call check(nf90_inq_varid(ncid, trim(varname), var_id))
    {tabs}   select case (ndim)
    {tabs}   case (1)
    {tabs}      allocate (data_1d(dims(1))); data_1d = reshape(int_buf, [dims(1)])
    {tabs}      call check(nf90_put_var(ncid, var_id, data_1d,start=start,count=count))
    {tabs}   case (2)
    {tabs}      allocate (data_2d(dims(1), dims(2))); data_2d = reshape(int_buf, [dims(1), dims(2)])
    {tabs}      call check(nf90_put_var(ncid, var_id, data_2d,start=start,count=count))
    {tabs}   case (3)
    {tabs}      allocate (data_3d(dims(1), dims(2), dims(3))); data_3d = reshape(int_buf, [dims(1), dims(2), dims(3)])
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
    """
    )


def gen_nc_read_timeslices() -> str:
    tabs = hio.indent(Tab.reset)
    return textwrap.dedent(
        f"""
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
    """
    )
