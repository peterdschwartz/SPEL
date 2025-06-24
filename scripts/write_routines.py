from __future__ import annotations

import logging
import re
import subprocess as sp
import sys
import textwrap
from collections import namedtuple
from typing import TYPE_CHECKING, Dict, Iterable

import scripts.io.helper as hio
from scripts.analyze_subroutines import Subroutine
from scripts.edit_files import macros
from scripts.fortran_modules import get_module_name_from_file
# from scripts.io.hdf5_io import (generate_constants_io_hdf5,
#                                 generate_elmtypes_io_hdf5)
from scripts.io.netcdf_io import (generate_constants_io_netcdf,
                                  generate_elmtypes_io_netcdf, generate_verify)
from scripts.logging_configs import get_logger
from scripts.mod_config import (ELM_SRC, PHYSICAL_PROP_TYPE_LIST, _bc,
                                spel_mods_dir, spel_output_dir,
                                unit_test_files)
from scripts.types import LineTuple, SubInit
from scripts.utilityFunctions import (Variable, comment_line,
                                      find_file_for_subroutine, getArguments,
                                      line_unwrapper, unwrap_section)

if TYPE_CHECKING:
    from scripts.DerivedType import DerivedType
    TypeDict = dict[str,DerivedType]
InstToDTypeMap = dict[str,str]
SubDict = dict[str,Subroutine]

def adjust_bounds(bounds:str)->str:
    """
    Function that maps e.g., begx_all -> begx
    """
    subgrids = re.findall(r'(?<=beg|end)(g|c|p|t|l)', bounds)
    if not subgrids:
        return bounds
    sg_set: set[str] = set(subgrids)
    assert len(sg_set) == 1, "Variable is allocated over multiple subgrids!"
    s = sg_set.pop()
    return re.sub(rf'(beg|end){s}\w+\b',lambda m: f"{m.group(1)}{s}",bounds)


def get_delta_from_dim(dim, delta):
    """
    this function will parse the dimension string to find if it's
    a patch, col, land, topo, or grid index
    """
    newdim = []
    bool = False
    if not dim:
        newdim = ""
        return newdim

    dim = dim.replace("(", "")
    dim = dim.replace(")", "")
    dim_li = dim.split(",")

    if delta == "y":
        for el in dim_li:
            if "begp" in el and "endp" in el:
                newdim.append("begp:endp")  # newdim.append('nc*deltap+1:(nc+1)*deltap')
            elif "begc" in el and "endc" in el:
                newdim.append("begc:endc")
            elif "begl" in el and "endl" in el:
                newdim.append("begl:endl")
            elif "begg" in el and "endg" in el:
                newdim.append("begg:endg")
            elif "begt" in el and "endt" in el:
                newdim.append("begt:endt")
            else:
                newdim.append(":")

    if delta == "n":
        for el in dim_li:
            if "begp" in el and "endp" in el:
                newdim.append("begp_copy:endp_copy")
            elif "begc" in el and "endc" in el:
                newdim.append("begc_copy:endc_copy")
            elif "begl" in el and "endl" in el:
                newdim.append("begl_copy:endl_copy")
            elif "begg" in el and "endg" in el:
                newdim.append("begg_copy:endg_copy")
            elif "begt" in el and "endt" in el:
                newdim.append("begt_copy:endt_copy")
            else:
                newdim.append(":")

    newdim = ",".join(newdim)
    newdim = "(" + newdim + ")"

    return newdim

def generate_cmake(files: list[str], case_dir: str):
    """
    Generates a CMakeLists.txt file for compiling unit-test 
    using cmake
    """
    from scripts.edit_files import macros
    exe_name = "elmtest"

    cmake_script = textwrap.dedent(f"""
    cmake_minimum_required(VERSION 3.20)
    project(ELM-UnitTest LANGUAGES Fortran)

    option(DBG "Enable Debug mode" OFF)
    option(GPU "Enable OpenACC (GPU) support" OFF)

    enable_testing()
    set(CMAKE_VERBOSE_MAKEFILE ON)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    set(CMAKE_Fortran_FLAGS "" CACHE STRING "Fortran Compiler Flags" FORCE)
    if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
        message(STATUS "Using gfortran as the Fortran compiler")
        string(APPEND CMAKE_Fortran_FLAGS " -fconvert=big-endian -ffree-line-length-none -ffixed-line-length-none")
        string(APPEND CMAKE_Fortran_FLAGS " -fallow-argument-mismatch")
        if (DBG)
            message(STATUS "Debug + Coverage build enabled")
            string(APPEND CMAKE_Fortran_FLAGS " -g -O0 -fprofile-arcs -ftest-coverage")
        endif()
    else()
        if (DBG)
            set(CMAKE_Fortran_FLAGS "${{CMAKE_Fortran_FLAGS}} -g -O0 -Mchkptr -Mchkstk" CACHE STRING "Fortran Compiler Flags" FORCE)
        else()
            set(CMAKE_Fortran_FLAGS "${{CMAKE_Fortran_FLAGS}} -fast")
        endif()

        if (GPU)
            message(STATUS "Enabling OpenACC GPU support")
            set(OPENACC_FLAGS "-gpu -Minfo=accel -cuda")
            # Append to existing flags for each build type
            set(CMAKE_Fortran_FLAGS "${{CMAKE_Fortran_FLAGS}} ${{OPENACC_FLAGS}}" CACHE STRING "Fortran Compiler Flags" FORCE)
        endif()
    endif()

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(NetCDFF REQUIRED netcdf-fortran)

    file(GLOB SOURCES "*.F90")
    add_executable({exe_name} ${{SOURCES}})
    include_directories(${{NetCDFF_INCLUDE_DIRS}})
    target_link_directories({exe_name} PRIVATE ${{NetCDFF_LIBRARY_DIRS}})
    target_link_libraries({exe_name} PRIVATE ${{NetCDFF_LIBRARIES}})

    find_package(LAPACK REQUIRED)

    if (LAPACK_FOUND)
        message(STATUS "LAPACK found: ${{LAPACK_LIBRARIES}}")
        target_link_libraries(elmtest PRIVATE ${{LAPACK_LIBRARIES}})
    endif()


    target_compile_definitions({exe_name} PRIVATE {" ".join(macros)})
    message(STATUS "Final Fortran flags: ${{CMAKE_Fortran_FLAGS}}")

    # Add test
    add_test(NAME run_elmtest COMMAND elmtest)
    # === Coverage Helper Targets ===
    # Object directory
    set(OBJECT_DIR "${{CMAKE_BINARY_DIR}}/CMakeFiles/elmtest.dir")
    message(STATUS "Object directory: ${{OBJECT_DIR}}")

    # Create coverage directory
    file(MAKE_DIRECTORY "${{CMAKE_BINARY_DIR}}/coverage")
    # gcov target
    add_custom_target(gcov
        COMMENT "Generating GCOV report..."
        COMMAND ${{CMAKE_CTEST_COMMAND}} --output-on-failure
        COMMAND echo "=================== GCOV ===================="
        COMMAND ${{CMAKE_COMMAND}} -E make_directory coverage
        COMMAND gcov -abcf ${{OBJECT_DIR}}/*.o > coverage/gcov.log
        COMMAND grep -H "Lines executed" coverage/gcov.log || true
        WORKING_DIRECTORY ${{CMAKE_BINARY_DIR}}
        DEPENDS elmtest
    )

    # scrub target: cleans .gcda/.gcno and object files
    add_custom_target(scrub
        COMMAND ${{CMAKE_COMMAND}} --build . --target clean
        COMMAND find ${{OBJECT_DIR}} -name '*.gcda' -delete
        COMMAND find ${{OBJECT_DIR}} -name '*.gcno' -delete
        COMMAND rm -rf ${{CMAKE_BINARY_DIR}}/coverage
        COMMENT "Scrubbing coverage and build artifacts"
        WORKING_DIRECTORY ${{CMAKE_BINARY_DIR}}
    )

    """)

    with open(f"{case_dir}/CMakeLists.txt", "w") as cmake_file:
        cmake_file.writelines(cmake_script)


def generate_makefile(files: list[str], case_dir: str):
    """
    This function takes the list of needed files
    and generates a makefile and finally saves it in the case dir
    """

    noF90 = [f.split("/")[-1] for f in files]
    object_list = [f.replace(".F90", ".o") for f in noF90]

    FC = "mpifort"
    FC_FLAGS_ACC = " -gpu=deepcopy -Minfo=accel -acc -cuda\n"
    FC_FLAGS_DEBUG = " -g -O0 -Mbounds -Mchkptr -Mchkstk\n"
    MODEL_FLAGS = "-D" + " -D".join(macros)

    MODEL_FLAGS = MODEL_FLAGS + "\n"

    unit_test_objs = " ".join(unit_test_files)
    objs = " ".join(object_list)


    lines: list[str] = [
        # Extract a candidate directory from the PATH that contains 'hdf5'
        "HDF5_DIR := $(shell echo $$PATH | awk -F: '{for(i=1;i<=NF;i++){if($$i ~ /hdf5/){print $$i; exit}}}')\n",
        "HDF5_BASE := $(shell dirname $(HDF5_DIR))\n",
        "HDF5_INC := $(HDF5_BASE)/include\n",
        "HDF5_LIB := $(HDF5_BASE)/lib\n",
        "ifeq ($(HDF5_DIR),)\n",
        "$(info Couldn't find hdf5 in path)\n",
        "else\n",
        "$(info HDF5_DIR: $(HDF5_DIR))\n",
        "endif\n",
        "FC= " + FC + "\n",
        "FC_FLAGS_ACC= " + FC_FLAGS_ACC,
        "FC_FLAGS_DEBUG = " + FC_FLAGS_DEBUG,
        "MODEL_FLAGS= " + MODEL_FLAGS,
        'INCLUDE_DIR = "${CURDIR}"\n',
        "HDF5_LDFLAGS = -L$(HDF5_LIB) -lhdf5_fortran -lhdf5\n",
        "FC_FLAGS = $(FC_FLAGS_DEBUG) $(MODEL_FLAGS) -I$(HDF5_INC)\n",
        "TEST = $(findstring acc,$(FC_FLAGS))\n\n",
        # Create string of ordered objct files.
        f"objects = {objs} {unit_test_objs}\n",
        "elmtest.exe : $(objects)\n",
        "\t$(FC) $(FC_FLAGS) -o elmtest.exe $(objects) $(HDF5_LDFLAGS)\n\n",
        "#.SUFFIXES: .o .F90\n",
    ]


    # These files do not need to be compiled with ACC flags or optimizations
    # Can cause errors or very long compile times
    noopt_list = ["duplicateMod"]
    for f in noopt_list:
        lines.append(f"{f}.o : {f}.F90\n")
        lines.append("\t$(FC) -O0 -c $(MODEL_FLAGS) $<\n")

    lines.extend([
        "%.o : %.F90\n",
        "\t$(FC) $(FC_FLAGS) -c -I $(INCLUDE_DIR) $<\n",
        "ifeq (,$(TEST))\n",
        "verificationMod.o : verificationMod.F90\n",
        "\t$(FC) -O0 -c $<\n",
        "else\n",
        "verificationMod.o : verificationMod.F90\n",
        "\t$(FC) -O0 -gpu=deepcopy -acc -c $<\n",
        "endif\n\n",
         ".PHONY: clean\n" ,
         "clean:\n" ,
         "\trm -f *.mod *.o *.exe\n" ,
    ])

    with open(f"{case_dir}/Makefile", "w") as ofile:
        ofile.writelines(lines)


def write_elminstMod(typedict, case_dir):
    """
    Writes elm_instMod to contain only variable decalarations
    needed for this Unit Test
    """
    file = open(f"{case_dir}/elm_instMod.F90", "w")
    spaces = " " * 2
    file.write("module elm_instMod\n")
    # use statements
    for type_name, dtype in typedict.items():
        for gv in dtype.instances.values():
            if gv.declaration == "elm_instmod":
                file.write(spaces + f"use {dtype.declaration}, only : { type_name }\n")

    file.write(spaces + "implicit none\n")
    file.write(spaces + "save\n")
    file.write(spaces + "public\n")
    for type_name, dtype in typedict.items():
        for gv in dtype.instances.values():
            if gv.declaration == "elm_instmod":
                file.write(spaces + f"type({type_name}) :: {gv.name}\n")

    file.write("end module elm_instMod\n")
    file.close()


def insert_at_token(lines: list[str], token: str, lines_to_add: Iterable[str]):
    regex_token = re.compile(rf"^\s*({token})")
    ln = 0
    token_line = 0
    found_token = False
    ln = 0
    while not found_token:
        line = lines[ln]
        m_token = regex_token.search(line)
        if m_token:
            token_line = ln
            found_token = True
        else:
            ln += 1

    if token_line == 0:
        print(f"Error: could find {token} in main.F90")
        sys.exit(1)

    for i, line_add in enumerate(lines_to_add):
        lines.insert(token_line + 1 + i, line_add)

    return lines


def find_parent_subroutine_call(
    subroutines: Dict[str, Subroutine],
    type_dict: dict[str, DerivedType],
    inst_to_type: dict[str,str],
):
    """
    Function that for a list of Subroutines, finds a call signature to
    insert into main.F90

    Returns only one result, so if a function is used in several places,
    manual modification may be required
    """

    logger = get_logger("WriteRoutines",level=logging.INFO)
    search_file = ELM_SRC
    mods_to_add = []
    var_decl_to_add = []
    calls = []

    for sub in subroutines.values():
        name = sub.name
        cmd = f'grep -rin -E "^[[:space:]]*(call[[:space:]]* {name})\\b" {search_file} | head -1'
        output = sp.getoutput(cmd)
        output = output.split(":")
        filename = output[0]
        call_ln = int(output[1]) - 1

        ifile = open(filename, "r")
        raw_lines = ifile.readlines()
        ifile.close()
        mod_lines = unwrap_section(raw_lines,startln=0)

        _, mod_name = get_module_name_from_file(filename)

        idx, call_string = next(((i, el) for i,el in enumerate(mod_lines) if el.ln == call_ln), (None,None))
        if not call_string or not idx or call_ln != mod_lines[idx].ln:
            logger.error(f"Couldn't match call_string for {name}. Expected call_ln {call_ln} -- got {mod_lines[idx].ln}")
            sys.exit(1)
        calls.append(call_string.line)
        args = getArguments(call_string.line)

        # Search backwards from subroutine call to get the calling subroutine :
        subname = None
        for ln in range(idx, -1, -1):
            line = mod_lines[ln].line
            match_sub = re.search(r"^\s*(subroutine)\s+", line)
            if match_sub:
                split_str = match_sub.group().strip()
                subname = line.split(split_str)[1].split("(")[0].strip()
                logger.info(f"Found Subroutine: {subname}\n")
                break

        if not subname:
            logger.error(f"Error::Couldn't find calling subroutine for {sub.name}")
            sys.exit(1)
        fn, startl, endl = find_file_for_subroutine(name=subname, fn=filename)
        sub_init = SubInit(
            name=subname,
            mod_name=mod_name,
            file=fn,
            start=startl,
            end=endl,
            mod_lines=mod_lines,
            function=None,
            cpp_start=None,
            cpp_end=None,
            cpp_fn="",
        )
        parent_sub = Subroutine(init_obj=sub_init)

        args_as_vars = {}
        args_as_instances = {}
        for arg in args:
            argname = arg.split("%")[0]
            if argname in inst_to_type:
                type_name = inst_to_type[argname]
                dtype = type_dict[type_name]
                inst_var = dtype.instances[argname]
                args_as_instances[inst_var.name] = inst_var
                if not inst_var.active:
                    type_dict[type_name].instances[argname].active = True

        args_not_found = [
            arg for arg in args if arg.split("%")[0] not in args_as_instances
        ]
        args_to_search = args_not_found.copy()

        for arg in args_to_search:
            if arg in parent_sub.arguments:
                argvar = parent_sub.arguments[arg]
                if argvar.type in type_dict:
                    dtype = type_dict[argvar.type]
                    inst_var: Variable = list(dtype.instances.values())[0]
                    args_as_instances[inst_var.name] = inst_var
                else:
                    args_as_vars[arg] = parent_sub.arguments[arg]
            elif arg in parent_sub.local_variables:
                args_as_vars[arg] = parent_sub.local_variables[arg]
            else:
                # Assume it's a derived type removed for fut purposes (ie, alm_fates)
                logger.info(
                    _bc.WARNING + f"Can't find {arg} (non-udt global var?)" + _bc.ENDC
                )
                args_as_vars[arg] = Variable(
                    type="integer",
                    name=arg,
                    dim=0,
                    subgrid="?",
                    bounds="",
                    ln=-1,
                )

        # Variables to have declarations added to main.F90
        if args_as_vars:
            for argvar in args_as_vars.values():
                type_string = argvar.type
                if type_string not in { "real", "integer", "character", "logical" }:
                    type_string = f"type({type_string})"
                elif type_string == "real":
                    type_string = f"{type_string}(r8)"
                decl = f"{type_string} :: {argvar.name}{argvar.bounds}"
                if decl + "\n" not in var_decl_to_add:
                    var_decl_to_add.append(decl + "\n")

        # Global variables , access through module interface
        if args_as_instances:
            for inst in args_as_instances.values():
                use_statement = f"use {inst.declaration}, only : {inst.name}"
                if use_statement + "\n" not in mods_to_add:
                    mods_to_add.append(use_statement + "\n")

    Additions = namedtuple("Additions", ["calls", "mods", "vars"])

    return Additions(calls=calls, mods=mods_to_add, vars=var_decl_to_add)


def adjust_call_sig(args, calls, num_filter_members):
    """
    Function to replace typical elm variables (ie, filters)
    """
    regex_filter = re.compile(r"\b(filter_)\w+")
    filter_member_str = "|".join(num_filter_members)
    regex_numf = re.compile(r"\b({})\b".format(filter_member_str))

    adj_args = args[:]
    adj_calls = calls[:]
    for i, decl in enumerate(args):
        m_f = regex_filter.search(decl)
        m_numf = regex_numf.search(decl)
        if m_f:
            adj_args.remove(decl)
        elif m_numf:
            print(f"removing: {decl} matched {m_numf.group()}")
            adj_args.remove(decl)
        elif "bounds" in decl:
            adj_args.remove(decl)

    for i, el in enumerate(calls):
        if re.search("\b(bounds)\b",adj_calls[i]):
            adj_calls[i] = adj_calls[i].replace("bounds", "bounds_clump")
        if regex_filter.search(adj_calls[i]):
            adj_calls[i] = adj_calls[i].replace("filter_","filter(nc)")

    return adj_args, adj_calls


def get_filter_members(filter_type: DerivedType):
    member_list = []
    for member_var in filter_type.components.values():
        if "num_" in member_var.name:
            member_list.append(member_var.name)

    return member_list


def prepare_main(
    subroutines: dict[str,Subroutine],
    type_dict: dict[str,DerivedType],
    instance_to_type: dict[str,str],
    casedir:str,
):
    """
    Function to insert USE dependencies into main.F90 and subroutine calls for the FUT subs
    """
    iofile = open(f"{spel_mods_dir}main.F90", "r")
    lines = iofile.readlines()
    iofile.close()

    use_token = "!#USE_START"
    var_token = "!#VAR_DECL"
    call_token = "!#CALL_SUB"
    modules_to_add = []
    for s in subroutines.values():
        mod = s.module
        modules_to_add.append(f"use {mod}, only : {s.name}\n")

    additions = find_parent_subroutine_call(subroutines, type_dict, instance_to_type)

    num_filters = get_filter_members(type_dict["clumpfilter"])
    adj_args, adj_calls = adjust_call_sig(additions.vars, additions.calls, num_filters)

    modules_to_add.extend(additions.mods)

    lines = insert_at_token(lines=lines, token=use_token, lines_to_add=modules_to_add)
    lines = insert_at_token(lines=lines, token=var_token, lines_to_add=adj_args)
    lines = insert_at_token(
        lines=lines,
        token=call_token,
        lines_to_add=reversed(adj_calls),
    )

    active_inst: set[str] = set()
    for dtype in type_dict.values():
        for instance in dtype.instances.values():
            if instance.active:
                active_inst.add(instance.name)

    copyin_lines: list[str] = ["!$acc enter data copyin(& \n"]
    for el in sorted(active_inst):
        copyin_lines.append(f"!$acc& {el},&\n")
    copyin_lines.append("!$acc& )\n")
    lines = insert_at_token(lines=lines,token="!#ACC_COPYIN",lines_to_add=copyin_lines)

    with open(f"{casedir}/main.F90", "w") as iofile:
        iofile.writelines(lines)

    return None


def prepare_unit_test_files(
    type_dict: TypeDict,
    case_dir:str,
    global_vars: dict[str,Variable],
    subroutines:SubDict,
    instance_to_type: InstToDTypeMap,
):
    """
    This function will prepare the use headers of main, initializeParameters,
    and readConstants.  It will also clean the variable initializations and
    declarations in main and elm_instMod
    """
    non_param_vars = {v.name : v for v in global_vars.values() if not v.parameter }
    prepare_main(subroutines, type_dict, instance_to_type, case_dir)
    # Write DeepCopyMod for UnitTest
    create_deepcopy_module(type_dict, case_dir, "DeepCopyMod")
    generate_elmtypes_io_netcdf(type_dict, instance_to_type, case_dir)
    generate_constants_io_netcdf(vars=non_param_vars, casedir=case_dir)

    # generate_constants_io_hdf5(vars=non_param_vars,casedir=case_dir)
    # generate_elmtypes_io_hdf5(type_dict, instance_to_type, case_dir)
    create_update_mod(non_param_vars, case_dir)

    # OLD WAY
    # create_constants_io(mode="r", global_vars=global_vars, casedir=case_dir)
    # create_constants_io(mode="w", global_vars=global_vars, casedir=case_dir)
    # clean_use_statements(mod_list=mod_list, file="initializeParameters", case_dir=case_dir)
    # clean_use_statements(mod_list=mod_list, file="update_accMod", case_dir=case_dir)
    prep_elm_init(type_dict, case_dir)

    # create list of variables that should be used for verification.
    verify_set: set[str] = {
        key
        for sub in subroutines.values()
        for key, val in sub.elmtype_access_sum.items()
        if val in ["w", "rw"]
    }
    generate_verify(verify_set, type_dict)


    return

def prep_elm_init(type_dict: TypeDict,case_dir: str):
    """
    Modifies elm_initializeMod.
    """
    # allocations_to_add = create_type_allocators(type_dict,case_dir)

    ifile = open(f"{spel_mods_dir}/elm_initializeMod.F90", "r")
    lines = ifile.readlines()
    ifile.close()

    # type_init_token = "!#VAR_INIT_START"
    # lines = insert_at_token(
    #     lines=lines, token=type_init_token, lines_to_add=allocations_to_add,
    # )

    active_instances = {
        inst_var.name: inst_var
        for dtype in type_dict.values()
        for inst_var in dtype.instances.values()
        if inst_var.active
    }

    tabs = hio.indent(hio.Tab.reset)
    use_statements: set[str] = set()
    for inst_var in active_instances.values():
        if inst_var.type != "bounds_type":
            stmt = f"{tabs}use {inst_var.declaration}, only: {inst_var.name}\n"
            use_statements.add(stmt)

    lines = insert_at_token(lines=lines,
                            token="!#USE_START",
                            lines_to_add=use_statements,)

    # write adjusted main to file in case dir
    with open(f"{case_dir}/elm_initializeMod.F90", "w") as of:
        of.writelines(lines)

    return


def create_update_mod(vars: dict[str, Variable], casedir: str):
    tabs = hio.indent(hio.Tab.reset)
    mod_name = "UpdateParamsAccMod"
    fn = f"{mod_name}.F90"

    sub_name = "update_params_acc"

    lines: list[str]=[f"module {mod_name}\n"]
    for var in vars.values():
        stmt = f"{tabs}use {var.declaration}, only : {var.name}\n"
        lines.append(stmt)

    lines.extend([
        f"{tabs}implicit none\n",
        f"{tabs}public :: update_params_acc\n",
        "contains\n\n",
    ])

    lines.append(f"{tabs}subroutine {sub_name}()\n")
    tabs = hio.indent(hio.Tab.shift)
    lines.append(rf"{tabs}!$acc update device(&\n")

    for var in vars.values():
        dim_str = ','.join([":" for i in range(var.dim)])
        dim_str = f"({dim_str})" if dim_str else ""
        lines.append(rf"{tabs}!$acc&   {var.name}{dim_str},&\n")
    lines.append(f"{tabs}!$acc&  )\n\n")

    tabs = hio.indent(hio.Tab.unshift)
    lines.append(f"{tabs}end subroutine {sub_name}\n")
    lines.append(f"end module {mod_name}\n")

    with open(f"{casedir}/{fn}",'w') as ofile:
        ofile.writelines(lines)

    return

def duplicate_clumps(typedict: dict[str,DerivedType]):
    """
    Function that writes a Fortran module containing
    subroutines needed to duplicate the input data
    to an arbirtary number of gridcells
    """
    file = open(f"{spel_output_dir}duplicateMod.F90", "w")
    spaces = " " * 2
    file.write("module duplicateMod\n")
    file.write("contains\n")

    file.write("subroutine duplicate_weights(unique_sites,total_gridcells)\n")
    file.write(spaces + "use elm_varsur, only : wt_lunit, urban_valid\n")
    file.write(spaces + "implicit none\n")
    file.write(spaces + "integer, intent(in) :: unique_sites\n")
    file.write(spaces + "integer, intent(in) :: total_gridcells\n")
    file.write(spaces + "integer :: g, dim3, l, gcopy\n")
    file.write(spaces + "dim3 = size(wt_lunit,3)\n")
    file.write(spaces + "print *, 'dim3 = ',dim3\n")
    file.write(spaces + "do g = unique_sites+1, total_gridcells\n")
    file.write(spaces + spaces + "gcopy = mod(g-1,unique_sites)+1\n")
    file.write(spaces + spaces + "do l=1,dim3\n")
    file.write(spaces * 3 + "wt_lunit(g,1,l) = wt_lunit(gcopy,1,l)\n")
    file.write(spaces * 2 + "end do\n")
    file.write(spaces * 2 + "urban_valid(g,1) = urban_valid(gcopy,1)\n")
    file.write(spaces + "end do\n")
    file.write("end subroutine duplicate_weights\n")

    file.write("subroutine duplicate_clumps(mode,unique_sites,num_sites)\n")

    # Use statements
    for type_name, dtype in typedict.items():
        if dtype.active:
            for var in dtype.instances.values():
                if var.active:
                    mod = var.declaration
                    vname = var.name
                    file.write(spaces + "use {}, only : {}\n".format(mod, vname))

    file.write(
        spaces + "use decompMod, only : bounds_type, get_clump_bounds, procinfo\n"
    )
    file.write(spaces + "use elm_varcon\n")
    file.write(spaces + "use elm_varpar\n")
    file.write(spaces + "use elm_varctl\n")
    file.write(spaces + "use landunit_varcon\n")
    file.write(spaces + "implicit none\n")
    file.write(spaces + "integer, intent(in) :: mode, unique_sites, num_sites\n")
    file.write(spaces + "type(bounds_type)  :: bounds_copy, bounds\n")
    file.write(spaces + "integer :: errcode, nc,nc_copy, nclumps\n")
    file.write(spaces + "integer :: begp_copy, endp_copy, begp, endp\n")
    file.write(spaces + "integer :: begc_copy, endc_copy, begc, endc\n")
    file.write(spaces + "integer :: begg_copy, endg_copy, begg, endg\n")
    file.write(spaces + "integer :: begt_copy, endt_copy, begt, endt\n")
    file.write(spaces + "integer :: begl_copy, endl_copy, begl, endl\n")
    file.write(spaces + "nclumps = num_sites\n")
    #
    file.write(spaces + "if(mode == 0) then\n")
    file.write(spaces * 2 + "do nc=unique_sites+1, num_sites\n")
    file.write(spaces * 3 + "nc_copy = mod(nc-1,unique_sites)+1\n")
    file.write(spaces * 3 + "call get_clump_bounds(nc,bounds)\n")
    file.write(spaces * 3 + "call get_clump_bounds(nc_copy,bounds_copy)\n")

    file.write(spaces * 3 + "begg_copy=bounds_copy%begg; endg_copy=bounds_copy%endg\n")
    file.write(spaces * 3 + "begt_copy=bounds_copy%begt; endt_copy=bounds_copy%endt\n")
    file.write(spaces * 3 + "begl_copy=bounds_copy%begl; endl_copy=bounds_copy%endl\n")
    file.write(spaces * 3 + "begc_copy=bounds_copy%begc; endc_copy=bounds_copy%endc\n")
    file.write(spaces * 3 + "begp_copy=bounds_copy%begp; endp_copy=bounds_copy%endp\n")

    file.write(spaces * 3 + "begg=bounds%begg; endg=bounds%endg\n")
    file.write(spaces * 3 + "begt=bounds%begt; endt=bounds%endt\n")
    file.write(spaces * 3 + "begl=bounds%begl; endl=bounds%endl\n")
    file.write(spaces * 3 + "begc=bounds%begc; endc=bounds%endc\n")
    file.write(spaces * 3 + "begp=bounds%begp; endp=bounds%endp\n")

    # First create mode for duplicating decomp related data.
    ignore_list = ["is_veg", "is_bareground", "wt_ed"]
    for type_name, dtype in typedict.items():
        if dtype.active and type_name in PHYSICAL_PROP_TYPE_LIST:
            for var in dtype.instances.values():
                if var.active:
                    for field_var in dtype.components.values():
                        active = field_var.active
                        bounds = field_var.bounds
                        if not active:
                            continue
                        if "%" not in field_var.name:
                            fname = var.name + "%" + field_var.name
                            comp_name = field_var.name
                        else:
                            fname = field_var.name
                            comp_name = field_var.name.split("%")[1]
                        if comp_name in ignore_list:
                            continue

                        dim = bounds
                        newdim = get_delta_from_dim(dim, "y")
                        dim1 = get_delta_from_dim(dim, "n")
                        dim1 = dim1.replace("_all", "")
                        if newdim == "(:)" or newdim == "":
                            continue
                        file.write(spaces * 3 + fname + newdim + " &" + "\n")
                        file.write(spaces * 4 + "= " + fname + dim1 + "\n")

    file.write(spaces * 2 + "end do\n")

    file.write(spaces + "else\n")
    file.write(spaces * 2 + "do nc=unique_sites+1, num_sites\n")
    file.write(spaces * 3 + "nc_copy = mod(nc-1,unique_sites)+1\n")
    file.write(spaces * 3 + "call get_clump_bounds(nc,bounds)\n")
    file.write(spaces * 3 + "call get_clump_bounds(nc_copy,bounds_copy)\n")

    file.write(spaces * 3 + "begg_copy=bounds_copy%begg; endg_copy=bounds_copy%endg\n")
    file.write(spaces * 3 + "begt_copy=bounds_copy%begt; endt_copy=bounds_copy%endt\n")
    file.write(spaces * 3 + "begl_copy=bounds_copy%begl; endl_copy=bounds_copy%endl\n")
    file.write(spaces * 3 + "begc_copy=bounds_copy%begc; endc_copy=bounds_copy%endc\n")
    file.write(spaces * 3 + "begp_copy=bounds_copy%begp; endp_copy=bounds_copy%endp\n")

    file.write(spaces * 3 + "begg=bounds%begg; endg=bounds%endg\n")
    file.write(spaces * 3 + "begt=bounds%begt; endt=bounds%endt\n")
    file.write(spaces * 3 + "begl=bounds%begl; endl=bounds%endl\n")
    file.write(spaces * 3 + "begc=bounds%begc; endc=bounds%endc\n")
    file.write(spaces * 3 + "begp=bounds%begp; endp=bounds%endp\n")

    # Duplicate statements for unit test variables
    for type_name, dtype in typedict.items():
        if dtype.active and type_name not in PHYSICAL_PROP_TYPE_LIST:
            for var in dtype.instances.values():
                if not var.active:
                    continue
                for field_var in dtype.components.values():
                    active = field_var.active
                    bounds = field_var.bounds
                    if not active:
                        continue
                    if "%" not in field_var.name:
                        fname = var.name + "%" + field_var.name
                        comp_name = field_var.name
                    else:
                        fname = field_var.name
                        comp_name = field_var.name.split("%")[1]
                    dim = bounds
                    newdim = get_delta_from_dim(dim, "y")
                    dim1 = get_delta_from_dim(dim, "n")
                    dim1 = dim1.replace("_all", "")
                    if newdim == "(:)" or newdim == "":
                        continue
                    file.write(spaces * 3 + fname + newdim + " &" + "\n")
                    file.write(spaces * 4 + "= " + fname + dim1 + "\n")

    file.write(spaces + "end do\n")
    file.write(spaces + "end if\n")

    file.write("end subroutine duplicate_clumps\n")
    file.write("end module\n")
    file.close()

def create_init_params(global_vars: dict[str,Variable], casedir:str):
    """
    Function to write "InitializeParametersMod" to contain 
    subroutine to allocate global variables in the unit-test.
    """
    mod_name = "InitializeParametersMod"
    lines: list[str] = []
    lines.append(f"module {mod_name}\n")

    tabs = hio.indent(hio.Tab.reset)
    use_statements: set[str] = set()
    for gv in global_vars.values():
        stmt = f"{tabs}use {gv.declaration}, only: {gv.name}\n"
        use_statements.add(stmt)
    lines.extend(list(use_statements))
    lines.append(f"{tabs}implicit none\n")

    lines.append(f"{tabs}contains\n")
    sub_name = "read_dims_and_alloc_params"
    tabs = hio.indent(hio.Tab.shift)
    lines.append(f"{tabs}subroutine {sub_name}()\n")

    tabs = hio.indent(hio.Tab.shift)
    lines.append(f"{tabs}character(len=256) :: in_file=spel_constants.txt\n")
    alloc_arrays = [ v for v in global_vars.values() if v.allocatable]

    # Declare local variables for dimensions
    for var in alloc_arrays:
        dims = [ f"{var.name}_lb{i+1}" for i in range(var.dim)]
        dims.extend( [ f"{var.name}_ub{i+1}" for i in range(var.dim)])
        stmt = f"{tabs}integer :: {','.join(dims)}\n"
        lines.append(stmt)



    return


def create_constants_io(mode, global_vars, casedir):
    """
    Function to create either readConstantsMod or writeConstantsMod
    """
    filename = "ReadConstantsMod.F90" if mode == "r" else "WriteConstantsMod.F90"

    tab = " " * TAB_WIDTH
    indent = 1
    lines: list[str] = []
    mod_name = filename.replace(".F90", "")
    lines.append(f"module {mod_name}\n")
    lines.append("implicit None\n")
    lines.append("contains\n")
    sub_name = mod_name.replace("Mod", "")
    tabs = tab * indent
    lines.append(f"{tabs}subroutine {sub_name}()\n")

    indent += 1
    tabs += tab * indent
    for gv in global_vars.values():
        if gv.parameter or not gv.active:
            continue
        lines.append(f"{tabs}use {gv.declaration}, only : {gv.name}\n")
    lines.append(tabs + "use fileio_mod, only : fio_open, fio_close\n")
    if mode == "r":
        lines.append(tabs + "use fileio_mod, only : fio_read\n")
    lines.append(tabs + "integer :: fid = 18, errcode=0\n")
    lines.append(tabs + 'character(len=64) :: ofile = "E3SM_constants.txt"\n')
    access = "1" if mode == "r" else "2"
    lines.append(tabs + f"call fio_open(fid,ofile, {access})\n\n")
    for gv in global_vars.values():
        if gv.parameter or not gv.active:
            continue
        if mode != "r":
            lines.append(f'{tabs}write(fid,"(A)") "{gv.name}"\n')
            lines.append(f"{tabs}write(fid,*) {gv.name}\n")
        else:
            str1 = f"{tabs}call fio_read(fid,'{gv.name}', {gv.name}, errcode=errcode)\n"
            str2 = f"{tabs}if (errcode .ne. 0) stop\n"
            lines.append(str1)
            lines.append(str2)

    lines.append(f"{tabs}call fio_close(fid)\n")
    indent -= 1
    tabs = tab * indent
    lines.append(f"{tabs}end subroutine {sub_name}\n")
    lines.append(f"end module {mod_name}")

    with open(f"{casedir}/{filename}", "w") as ofile:
        ofile.writelines(lines)

    return None


def create_write_vars(typedict, subname, use_isotopes=False):
    """
    This function generates the subroutine write_vars that is to be called from
    E3SM prior to execution of the desired subroutine
        * typedict : dictionary of all derived types related to Unit Test with
                     needed ones toggled by dtype.active = True
        * subname  : used to name file written by `write_vars`
        * use_isotopes : flag toggles c13/c14 vars.
    """
    func_name = "create_write_vars"
    spaces = " " * 2  # holds tabs indentations without using \t
    ofile = open(f"{spel_output_dir}writeMod.F90", "w")
    ofile.write("module writeMod\n")
    ofile.write("contains\n")
    ofile.write("subroutine write_vars()\n")
    ofile.write(spaces + "use fileio_mod, only : fio_open, fio_close\n")
    ofile.write(spaces + "use elm_varsur, only : wt_lunit, urban_valid\n")

    # Use statements
    for type_name, dtype in typedict.items():
        if dtype.active:
            for var in dtype.instances.values():
                if var.active:
                    c13c14 = bool("c13" in var.name or "c14" in var.name)
                    if c13c14:
                        continue
                    mod = var.declaration
                    ofile.write(spaces + "use {}, only : {}\n".format(mod, var.name))

    ofile.write(spaces + "implicit none\n")
    ofile.write(spaces + "integer :: fid\n")
    ofile.write(spaces + f'character(len=64) :: ofile = "{subname}_vars.txt"\n')
    ofile.write(spaces + "fid = 23\n")

    for dtype in typedict.values():
        if dtype.active:
            create_write_read_functions(dtype,"w", ofile, gpu=True)

    for dtype in typedict.values():
        if dtype.active:
            create_write_read_functions(dtype,"w", ofile, gpu=True)

    ofile.write(spaces + "call fio_open(fid,ofile, 2)\n\n")
    ofile.write(spaces + 'write(fid,"(A)") "wt_lunit"\n')
    ofile.write(spaces + "write(fid,*) wt_lunit\n")
    ofile.write(spaces + 'write(fid,"(A)") "urban_valid"\n')
    ofile.write(spaces + "write(fid,*) urban_valid\n\n")

    # Add I/O statements for each component of needed derived types.
    #  the physical property types will be done first.
    for dtype in typedict.values():
        if dtype.active and dtype.type_name in PHYSICAL_PROP_TYPE_LIST:
            create_write_read_functions(dtype,"w", ofile)

    for dtype in typedict.values():
        if dtype.active and dtype.type_name not in PHYSICAL_PROP_TYPE_LIST:
            create_write_read_functions(dtype,"w", ofile)

    ofile.write(spaces + "call fio_close(fid)\n")
    ofile.write("end subroutine write_vars\n")
    ofile.write("end module writeMod\n")
    ofile.close()


def create_read_vars(typedict):
    """
    Function that creates the readMod.F90 file similar to writeMod.F90.
    """
    # TODO:Incorporate the ability to use I/O libraries like netcdf if desired.
    #     Original purpose of unit test is to be as light weight as possible, but
    #     netCDF may allow more flexibility for generating unit tests of larger scale
    #     such as for EcosystemDynNoLeaching2.
    spaces = " " * 2
    ofile = open(f"{spel_output_dir}readMod.F90", "w")
    ofile.write("module readMod\n")

    # Use Statements
    for key in typedict.keys():
        for var in typedict[key].instances.values():
            if var.active:
                mod = var.declaration
                vname = var.name
                c13c14 = bool("c13" in vname or "c14" in vname)
                if c13c14:
                    continue
                ofile.write("use {}, only : {}\n".format(mod, vname))

    # The remaining use statements are for decomp vars
    ofile.write("use decompMod, only : bounds_type\n")
    ofile.write("use elm_varcon\n")
    ofile.write("use elm_varpar\n")
    ofile.write("use elm_varctl\n")
    ofile.write("use landunit_varcon\n")
    ofile.write("use elm_instMod, only: glc2lnd_vars\n")

    ofile.write("contains\n")

    # Subroutine read_weights
    ofile.write("subroutine read_weights(in_file,numg)\n")
    ofile.write(spaces + "use fileio_mod, only : fio_open, fio_read, fio_close\n")
    ofile.write(spaces + "use elm_varsur, only : wt_lunit, urban_valid\n")
    ofile.write(spaces + "implicit none\n")
    ofile.write(spaces + "character(len=256), intent(in) :: in_file\n")
    ofile.write(spaces + "integer, intent(in) :: numg\n")
    ofile.write(spaces + "integer :: errcode = 0\n\n")
    ofile.write(spaces + "call fio_open(18,in_file,1)\n")
    ofile.write(
        spaces + "call fio_read(18,'wt_lunit',wt_lunit(1:numg,1,:),errcode=errcode)\n"
    )
    ofile.write(spaces + "if(errcode .ne. 0) stop\n")
    ofile.write(
        spaces
        + "call fio_read(18,'urban_valid',urban_valid(1:numg,1),errcode=errcode)\n"
    )
    ofile.write(spaces + "if(errcode .ne. 0) stop\n\n")
    ofile.write(spaces + "end subroutine read_weights\n\n")

    # Beginning of Subroutine `read_vars`
    ofile.write("subroutine read_vars(in_file,bounds,mode,nsets)\n")
    ofile.write(spaces + "use fileio_mod, only : fio_open, fio_read, fio_close\n")
    ofile.write(spaces + "implicit none\n")

    # Dummy Variables
    ofile.write(spaces + "character(len=256),intent(in) :: in_file\n")
    ofile.write(spaces + "type(bounds_type), intent(in) :: bounds\n")
    ofile.write(spaces + "integer, intent(in) :: mode\n")
    ofile.write(spaces + "integer, intent(in) :: nsets\n")

    # Local Variables
    ofile.write(spaces + "integer :: errcode = 0\n")
    ofile.write(spaces + "integer :: begp,  endp\n")
    ofile.write(spaces + "integer :: begc,  endc\n")
    ofile.write(spaces + "integer :: begg,  endg;\n")
    ofile.write(spaces + "integer :: begl,  endt;\n")
    ofile.write(spaces + "integer :: begt,  endl;\n")
    #
    ofile.write(spaces + "begp = bounds%begp; endp = bounds%endp/nsets\n")
    ofile.write(spaces + "begc = bounds%begc; endc = bounds%endc/nsets\n")
    ofile.write(spaces + "begl = bounds%begl; endl = bounds%endl/nsets\n")
    ofile.write(spaces + "begt = bounds%begt; endt = bounds%endt/nsets\n")
    ofile.write(spaces + "begg = bounds%begg; endg = bounds%endg/nsets\n")

    ofile.write(spaces + 'print *,"begp range :",begp, endp\n')
    ofile.write(spaces + 'print *,"begc range :",begc, endc\n')
    ofile.write(spaces + 'print *,"begl range :",begl, endl\n')
    ofile.write(spaces + 'print *,"begt range :",begt, endt\n')
    ofile.write(spaces + 'print *,"begg range :",begg, endg\n')

    ofile.write(spaces + "call fio_open(18,in_file, 1)\n")
    ofile.write(spaces + "if(mode == 1) then\n")
    ofile.write(spaces + "print *, 'reading in physical properties'\n")

    # Start with physical property types
    for dtype in typedict.values():
        if dtype.active and dtype.type_name in PHYSICAL_PROP_TYPE_LIST:
            create_write_read_functions(dtype,"r", ofile)
    ofile.write(spaces + "else\n")

    for dtype in typedict.values():
        if dtype.active and dtype.type_name not in PHYSICAL_PROP_TYPE_LIST:
            create_write_read_functions(dtype,"r", ofile)

    ofile.write(spaces + "end if\n")
    ofile.write(spaces + "call fio_close(18)\n")
    ofile.write("end subroutine read_vars\n")
    ofile.write("end module\n")
    ofile.close()


def create_pointer_type_sub(lines, dtype, all_active=False):
    tabs = hio.indent(hio.Tab.reset)

    lines.append(f"{tabs}subroutine set_pointers_{dtype.type_name}(this_type)\n")
    tabs = hio.indent(hio.Tab.shift)
    inst_dim = max([inst.dim for inst in dtype.instances.values()])
    dim_string = ""
    if inst_dim == 1:
        dim_string = "(:)"
    else:
        print(f"ERROR: {dtype.type_name} inst.dim > 1 may need extra care")
        sys.exit(1)
    lines.append(
        f"{tabs}type({dtype.type_name}), intent(inout) :: this_type{dim_string}\n"
    )

    # Allocate the instance based on the number of unique targets for the pointer fields:
    num_targets = 0
    for member_var in dtype.components.values():
        active = member_var.active
        num_targets_new = len(member_var.pointer)
        if num_targets == 0:
            num_targets = num_targets_new
        elif num_targets_new != num_targets:
            print(f"WARNING::{dtype} pointers with inconsistent targets")
            sys.exit(1)

def find_global_vars(dtype: DerivedType)->set[str]:

    bounds_to_search =[var.bounds for var in dtype.components.values() if var.active]
    if not dtype.init_sub_ptr:
        return []
    global_vars = dtype.init_sub_ptr.active_global_vars
    gv_str = "|".join(global_vars.keys())
    regex_gvs = re.compile(rf"\b({gv_str})\b")

    used_matches = [dim for dim in filter(lambda x: regex_gvs.search(x), bounds_to_search)]
    use_statements: set[str] = set()
    for m in used_matches:
        m_var_names = regex_gvs.findall(m)
        for varname in m_var_names:
            if not varname:
                continue
            var = global_vars[varname]
            stmt = f"use {var.declaration}, only: {varname}"
            use_statements.add(stmt)

    return use_statements

def create_type_sub(lines: list[str], dtype: DerivedType, all_active:bool=False,):
    """
    Functiion that creates a subroutine to allocate
    the active members of dtype
    Arguments
        * lines : lines corresponding to the Module to be written (ie. Read/WriteConstantsMod)
        * dtype : DerivedType obj of type to allocate
        * all_active : Allocate all members regardless of use in UnitTest
    """
    tabs = hio.indent(hio.Tab.reset)

    lines.append(f"{tabs}subroutine allocate_{dtype.type_name}(bounds,this_type)\n")
    tabs = hio.indent(hio.Tab.shift)

    use_stmts = find_global_vars(dtype)
    if use_stmts:
        for stmt in use_stmts:
            lines.append(f"{tabs}{stmt}\n")

    lines.append(f"{tabs}type(bounds_type), intent(in) :: bounds\n")
    inst_dim = max([inst.dim for inst in dtype.instances.values()])
    if inst_dim == 0:
        lines.append(f"{tabs}type({dtype.type_name}), intent(inout) :: this_type\n")
    elif inst_dim == 1:
        lines.append(f"{tabs}type({dtype.type_name}), intent(inout) :: this_type(:)\n")
    else:
        print(f"ERROR: {dtype.type_name} inst.dim > 1 may need extra care")
        sys.exit(1)
    lines.append(tabs + "integer :: begp, endp\n")
    lines.append(tabs + "integer :: begc, endc\n")
    lines.append(tabs + "integer :: begl, endl\n")
    lines.append(tabs + "integer :: begt, endt\n")
    lines.append(tabs + "integer :: begg, endg\n")
    if inst_dim == 1:
        lines.append(tabs + "integer :: i,N\n")
    lines.append(tabs + "!-------------------------------------\n")
    lines.append(tabs + "begp = bounds%begp; endp = bounds%endp\n")
    lines.append(tabs + "begc = bounds%begc; endc = bounds%endc\n")
    lines.append(tabs + "begl = bounds%begl; endl = bounds%endl\n")
    lines.append(tabs + "begg = bounds%begg; endg = bounds%endg\n")
    lines.append(tabs + "begt = bounds%begt; endt = bounds%endt\n")

    if inst_dim == 1:
        lines.append(f"{tabs}N=size(this_type)\n")
        lines.append(tabs + "do i = 1, N\n")
        tabs = hio.indent(hio.Tab.shift)
    inst_name = "this_type" if inst_dim == 0 else "this_type(i)"
    for member_var in dtype.components.values():
        active = member_var.active
        if active:
            dim_string = ""
            field_name = member_var.name.split("%")[1] if "%" in member_var.name else member_var.name
            if member_var.dim > 0:
                dim_li = [":" for i in range(0, member_var.dim)]
                dim_string = ",".join(dim_li)
                dim_string = f"({dim_string})"
                bounds = adjust_bounds(member_var.bounds)
                statement = f"{tabs}allocate({inst_name}%{field_name}{bounds});"
                lines.append(statement)
            elif member_var.ptrscalar:
                statement = f"{tabs}allocate({inst_name}%{field_name});"
                lines.append(statement)

            init_val = None
            match member_var.type:
                case "real":
                    init_val = "spval"
                case "integer":
                    init_val = "ispval"
                case "logical":
                    init_val = ".False."
                case "character":
                    init_val = "''"

            if not init_val:
                print(f"Error: No init_val for {member_var}")
                sys.exit(1)
            init_statement = f"{inst_name}%{field_name}{dim_string} = {init_val}\n"
            lines.append(init_statement)

    # End subroutine:
    tabs = hio.indent(hio.Tab.unshift)
    lines.append(f"{tabs}end subroutine allocate_{dtype.type_name}\n\n")
    return lines


def create_type_allocators(
        type_dict: TypeDict,
        casedir: str,
)-> set[str]:
    """
    function to creates a fortran module to allocate the active
    members of derived types
    """

    filename = "UnitTestAllocatorMod"
    lines = []
    indent = 0
    tab = " " * TAB_WIDTH
    indent += 1
    tabs = tab * indent

    skip_types = [
        "clumpfilter",
        # "vegetation_physical_properties",
        # "column_physical_properties",
        # "landunit_physical_properties",
        # "gridcell_physical_properties_type",
        # "topounit_physical_properties",
    ]
    lines.append(f"module {filename}\n")
    lines.append(f"{tabs}use elm_varcon\n")
    lines.append(f"{tabs}use elm_varpar\n")
    lines.append(f"{tabs}use decompMod ,only : bounds_type\n")

    active_instances = {
        inst_var.name: inst_var
        for dtype in type_dict.values()
        for inst_var in dtype.instances.values()
        if inst_var.active
    }
    active_types = {inst_var.type: True for inst_var in active_instances.values()}

    for type_name in active_types:
        if type_name in skip_types:
            continue
        dtype = type_dict[type_name]
        statement = f"{tabs}use {dtype.declaration},only: {type_name}\n"
        if statement not in lines:
            lines.append(statement)
    lines.append(f"{tabs}implicit None\n")

    for type_name in active_types:
        if type_name in skip_types:
            continue
        statement = f"{tabs} public :: allocate_{type_name}\n"
        lines.append(statement)

    lines.append(f"{tabs}contains\n")

    # Generate type allocator subroutines:
    for type_name in active_types:
        if type_name in skip_types:
            continue
        dtype = type_dict[type_name]
        lines = create_type_sub(lines, dtype)

    lines.append(f"end module {filename}")

    with open(f"{casedir}/{filename}.F90", "w") as ofile:
        ofile.writelines(lines)

    # Return list of subroutine calls to be inserted in elm_init
    allocations: set[str] = set()
    tabs = " " * 6
    for inst in active_instances.values():
        if inst.type in skip_types or "c13" in inst.name or "c14" in inst.name:
            continue
        statement = f"{tabs}call allocate_{inst.type}(bounds_proc,{inst.name})\n"
        allocations.add(statement)

    return allocations


def create_deepcopy_subroutine(lines, dtype: DerivedType, all_active=False):
    """
    Function to generate a subroutine for dtype that manually
    copies over the active members of dtype.
        if all_active, all members are considered active
    """
    tabs = hio.indent(hio.Tab.reset)
    array1d = False
    scalar = False
    for inst in dtype.instances.values():
        if not inst.active:
            continue
        if inst.dim == 0:
            scalar = True
        elif inst.dim == 1:
            array1d = True
        else:
            print(_bc.WARNING + f"WARNING:Instance dim > 1D: {inst}" + _bc.ENDC)

    members_to_copy = [
        field
        for field in dtype.components.values()
        if field.active or all_active
    ]

    if scalar:
        subname = f"deepcopy_{dtype.type_name}"
        lines.append(f"{tabs}subroutine {subname}(this_type)\n")
        tabs = hio.indent(hio.Tab.shift)
        lines.append(f"{tabs}type({dtype.type_name}), intent(inout) :: this_type\n")
        lines.append(f"{tabs}!$acc enter data copyin(this_type)\n")
        lines.append(f"{tabs}!$acc enter data copyin(&\n")
        for num, member in enumerate(members_to_copy):
            dim_string = ""
            if member.dim > 0:
                dim_li = [":" for i in range(0, member.dim)]
                dim_string = ",".join(dim_li)
                dim_string = f"({dim_string})"

            name = "this_type" + "%" + member.name + dim_string
            lines.append(f"{tabs}!$acc& {name}")
            final_num = bool(num == len(members_to_copy) - 1)
            if not final_num:
                lines.append(",&\n")
            else:
                lines.append(")")
        lines.append("\n")
        tabs = hio.indent(hio.Tab.unshift)
        lines.append(f"{tabs}end subroutine {subname}\n")

    if array1d:
        subname = f"deepcopy_{dtype.type_name}_array"
        lines.append(f"{tabs}subroutine {subname}(this_type)\n")
        tabs = hio.indent(hio.Tab.shift)
        lines.append(f"{tabs}type({dtype.type_name}), intent(inout) :: this_type(:)\n")
        lines.append(f"{tabs}integer :: i,N\n")
        lines.append(f"{tabs}N=size(this_type)\n")
        lines.append(f"{tabs}!$acc enter data copyin(this_type(:))\n")
        lines.append(tabs + "do i = 1, N\n")
        tabs = hio.indent(hio.Tab.shift)

        inst_name = "this_type(i)"
        lines.append(f"{tabs}!$acc enter data copyin(&\n")
        for num, member in enumerate(members_to_copy):
            dim_string = ""
            if member.dim > 0:
                dim_li = [":" for i in range(0, member.dim)]
                dim_string = ",".join(dim_li)
                dim_string = f"({dim_string})"

            name = inst_name + "%" + member.name + dim_string
            lines.append(f"{tabs}!$acc& {name}")
            final_num = bool(num == len(members_to_copy) - 1)
            if not final_num:
                lines.append(",&\n")
            else:
                lines.append(")\n")
        tabs = hio.indent(hio.Tab.unshift)
        lines.append("end do\n")
        tabs = hio.indent(hio.Tab.unshift)
        lines.append(f"{tabs}end subroutine {subname}\n")

    return lines


def create_deepcopy_module(
    type_dict: dict[str, DerivedType],
    casedir: str,
    modname: str,
    all_active=False,
):
    """
    Function to create subroutine(s) for performing a manual deepcopy
    """
    tabs = hio.indent("reset")
    lines = []
    interface_name = "deepcopy_type"

    lines.append(f"module {modname}\n")

    active_types = {dtype for dtype in type_dict.values() if dtype.active}
    for dtype in active_types:
        statement = f"{tabs}use {dtype.declaration},only: {dtype.type_name}\n"
        if statement not in lines:
            lines.append(statement)
    lines.append(f"{tabs}implicit none\n")

    # Create interface:
    lines.append(f"{tabs}interface {interface_name}\n")
    tabs = hio.indent("shift")
    for dtype in active_types:
        statement = f"{tabs}module procedure :: deepcopy_{dtype.type_name}\n"
        lines.append(statement)
    tabs = hio.indent("unshift")
    lines.append(f"{tabs}end interface deepcopy_type\n")
    lines.append("contains\n\n")

    for dtype in active_types:
        create_deepcopy_subroutine(
            lines=lines,
            dtype=dtype,
            all_active=all_active,
        )

    lines.append(f"end module {modname}\n")

    with open(f"{casedir}/{modname}.F90", "w") as ofile:
        ofile.writelines(lines)
    return None

def create_write_read_functions(dtype: DerivedType, rw, ofile, gpu=False):
    """
    This function will write two .F90 functions that write read and write statements for all
    components of the derived type
    rw is a variable that holds either read or write mode
    """
    tab = " " * 2

    fates_list = ["veg_pp%is_veg", "veg_pp%is_bareground", "veg_pp%wt_ed"]
    for var in dtype.instances.values():
        if not var.active:
            continue
        if rw.lower() == "write" or rw.lower() == "w":
            ofile.write(tab + "\n")
            ofile.write(
                tab
                + f"!====================== {var.name} ======================!\n"
            )
            ofile.write(tab + "\n")
            if gpu:
                ofile.write(tab + "!$acc update self(& \n")

            # Any component of the derived type accessed by the Unit Test should have been toggled active at this point.
            # Go through the instance of this derived type and write I/O for any active components.
            vars = []
            for field_var in dtype.components.values():
                active = field_var.active
                if not active:
                    continue

                # Filter out C13/C14 duplicates and fates only variables.
                c13c14 = bool("c13" in field_var.name or "c14" in field_var.name)
                if c13c14:
                    continue
                if "%" not in field_var.name:
                    fname = var.name + "%" + field_var.name
                else:
                    fname = field_var.name
                if fname in fates_list:
                    continue
                if gpu:
                    vars.append(fname)
                else:
                    str1 = f'write (fid, "(A)") "{fname}" \n'
                    str2 = f"write (fid, *) {fname}\n"
                    ofile.write(tab + str1)
                    ofile.write(tab + str2)
            if gpu:
                for n, v in enumerate(vars):
                    if n + 1 < len(vars):
                        ofile.write(tab + f"!$acc {v}, &\n")
                    else:
                        ofile.write(tab + f"!$acc {v} )\n")
        elif rw.lower() == "read" or rw.lower() == "r":
            ofile.write(tab + "\n")
            ofile.write(
                tab
                + "!====================== {} ======================!\n".format(
                    var.name
                )
            )
            ofile.write(tab + "\n")

            # Any component of the derived type accessed by the Unit Test should have been toggled active at this point.
            # Go through the instance of this derived type and write I/O for any active components.
            for field_var in dtype.components.values():
                active = field_var.active
                bounds = field_var.bounds
                if not active:
                    continue
                c13c14 = bool("c13" in field_var.name or "c14" in field_var.name)
                if c13c14:
                    continue
                if "%" not in field_var.name:
                    fname = var.name + "%" + field_var.name
                else:
                    fname = field_var.name
                if fname in fates_list:
                    continue
                dim = bounds
                dim1 = get_delta_from_dim(dim, "y")
                dim1 = dim1.replace("_all", "")
                str1 = "call fio_read(18,'{}', {}{}, errcode=errcode)\n".format(
                    fname, fname, dim1
                )
                str2 = "if (errcode .ne. 0) stop\n"
                ofile.write(tab + str1)
                ofile.write(tab + str2)
    return

def create_fortls(case_dir:str):
    """
    Function that creates placeholder .fortls for lsp
    """

    lines = textwrap.dedent("""
    {
    "nthreads" : 4,
       "sort_keywords": false,
       "debug_log": false,
       "lowercase_intrinsics": true,
    }
    """)

    with open(f"{case_dir}/.fortls.json",'w') as ofile:
        ofile.writelines(lines)
    return
