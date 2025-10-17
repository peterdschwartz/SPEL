from __future__ import annotations

import logging
import os
import re
import subprocess as sp
import sys
from logging import Logger
from pprint import pprint
from typing import Optional

from scripts.analyze_subroutines import Subroutine
from scripts.check_sections import check_function_start, create_init_obj
from scripts.config import E3SM_SRCROOT, spel_output_dir
from scripts.DerivedType import DerivedType, parse_derived_type_definition
from scripts.fortran_modules import (
    FortranModule,
    ModTree,
    build_module_tree,
    get_filename_from_module,
    get_module_name_from_file,
    parse_use_stmts,
)
from scripts.fortran_parser.spel_ast import GenericOperatorExpression
from scripts.logging_configs import get_logger, set_logger_level
from scripts.profiler_context import profile_ctx
from scripts.types import (
    LineTuple,
    LogicalLineIterator,
    ParseState,
    PassManager,
    PreProcTuple,
    SubInit,
    SubStart,
)
from scripts.utilityFunctions import (
    find_file_for_subroutine,
    find_variables,
    line_unwrapper,
    parse_variable_decl,
    unwrap_section,
)

# Define Return types
ModParseResult = tuple[dict[str, SubInit], list[str]]

# Compile list of lower-case module names to remove
# SPEL expects these all to be lower-case currently
bad_modules = {
    "abortutils",
    "shr_log_mod",
    "elm_time_manager",
    "shr_infnan_mod",
    "pio",
    "shr_sys_mod",
    "perf_mod",
    "shr_assert_mod",
    "spmdmod",
    "restutilmod",
    "vocemissionmod",
    "restfilemod",
    "histfilemod",
    "accumulmod",
    "ncdio_pio",
    "shr_strdata_mod",
    "fileutils",
    "elm_nlutilsmod",
    "shr_mpi_mod",
    "shr_nl_mod",
    "shr_str_mod",
    "controlmod",
    "getglobalvaluesmod",
    "organicfilemod",
    "elmfatesinterfacemod",
    "externalmodelconstants",
    "externalmodelinterfacemod",
    "waterstatetype",
    "seq_drydep_mod",
    "shr_file_mod",
    "mct_mod",
    "spmdgathscatmod",
    "perfmod_gpu",
    "dynpftfilemod",
    "dyncropfilemod",
    "dynharvestmod",
    "dynfateslandusechangemod",
    "cnallocationbetrmod",
    "dynedmod",
    "elm_interface_pflotranmod",
    "elm_interface_funcsmod",
}

fates_mod = ["elmfatesinterfacemod"]
betr_mods = ["betrsimulationalm"]

bad_subroutines = {
    "endrun",
    "restartvar",
    "hist_addfld1d",
    "hist_addfld2d",
    "init_accum_field",
    "extract_accum_field",
    "hist_addfld_decomp",
    "ncd_pio_openfile",
    "ncd_io",
    "ncd_pio_closefile",
    "alm_fates",
    "elm_fates",
    "ncd_inqdlen",
    "t_start_lnd",
    "t_stop_lnd",
    "ep_betr",
}

remove_subs = [
    "prepare_data_for_em_ptm_driver",
    "prepare_data_for_em_vsfm_driver",
    "decompinit_lnd_using_gp",
]

regex_if = re.compile(r"^if\s*\(.*\)\s*then", re.IGNORECASE)
regex_do_start = re.compile(r"^\s*(\w+:)?\s*do\b", re.IGNORECASE)
regex_do_end = re.compile(r"^\s*(end\s*do(\s+\w+)?)", re.IGNORECASE)
regex_ifthen = re.compile(r"^(if\b)(.+)(then)$", re.IGNORECASE)
regex_endif = re.compile(r"^(end\s*if)", re.IGNORECASE)

regex_include_assert = re.compile(r"^(#include)\s+[\"\'](shr_assert.h)[\'\"]")
regex_sub = re.compile(r"^\s*(subroutine)\s+")
regex_shr_assert = re.compile(r"^\s*call\s+(shr_assert_all|shr_assert)\b")
regex_end_sub = re.compile(r"^\s*(end subroutine)")

regex_func = re.compile(r"\bfunction\s+\w+\s*\(")
regex_result = re.compile(r"\bresult\s*\(\s*\w+\s*\)$")
regex_func_type = re.compile(r"(type\s*\(|integer|real|logical|character|complex)")

regex_end_func = re.compile(r"\s*(end\s*function)\b")
regex_gsmap = re.compile(r"(gsmap)")

# Set up PassManager for parsing file
#  Names for PassFns
parse_sub_start = "parse_sub_start"
parse_sub_end = "parse_sub_end"
parse_func_start = "parse_func_start"
parse_func_end = "parse_func_end"
parse_shr_assert = "parse_shr_assert"
parse_inc_shr_assert = "parse_inc_shr_assert"
host_program = "check_host_program"

# These are required to be re-compiled after an initial pass through file
parse_bad_inst = "parse_bad_inst"
parse_sub_call = "parse_sub_call"

#
# Macros that we want to process. Any not in the list will be
# skipped.
macros = ["MODAL_AER"]
#    gfortran -D<FLAGS> -I{E3SM_SRCROOT}/share/include -cpp -E <file> > <output>
# will generate the preprocessed file with Macros processed.
# But what to do about line numbers?
# The preprocessed file will have a '# <line_number>' indicating
# the line number immediately after the #endif in the original file.

ModDict = dict[str, FortranModule]
SubDict = dict[str, Subroutine]
TypeDict = dict[str, DerivedType]


def set_comment(state: ParseState, logger: Logger = None):
    state.line_it.comment_cont_block()
    return

def check_contains(state: ParseState, logger: Logger = None):
    if state.in_sub == 0 and state.in_func == 0: 
        return
    state.host_program = state.line_it.i

    if state.in_sub > 0:
        finalize_subroutine(state,logger)
    elif state.in_func > 0:
        finalize_function(state,logger)

def set_in_subroutine(state: ParseState, logger: Logger):
    """
    Function that parses the line of a subroutine for it's name, and
    f desired, will comment out the entire subroutine, else return it's name and starting lineno
    """
    func_name = "( set_in_subroutine )"
    parent = ""
    if state.in_sub > 0:
        logger.info("Sub module subroutine!")
        parent = state.sub_start[-1].subname

    assert state.curr_line
    state.in_sub += 1
    ct = state.get_start_index()

    full_line = state.curr_line.line
    start_ln = state.curr_line.ln

    subname = full_line.split()[1].split("(")[0].strip()
    if "_oacc" in subname:
        logger.info(f"{func_name} skipping {subname}")
        return

    cpp_ln = ct if state.cpp_file else None
    sub_start: Optional[SubStart] = SubStart(
        subname=subname,
        start_ln=start_ln,
        cpp_ln=cpp_ln,
        parent=parent,
    )

    # Check if subroutine needs to be removed before processing
    match_remove = bool(subname.lower() in remove_subs)
    test_init = bool("init" not in subname.lower().replace("initcold", "cold"))
    match_gsmap = regex_gsmap.search(subname)

    # Remove if it's an IO routine
    test_decompinit = bool(subname.lower() == "decompinit_lnd_using_gp")
    remove = bool((match_remove and test_init) or match_gsmap or test_decompinit)
    if remove:
        _, _ = state.line_it.consume_until(
            end_pattern=regex_end_sub, start_pattern=None
        )
        state.removed_subs.append(subname)
        state.in_sub -= 1
        state.line_it.comment_cont_block(index=ct)
        return

    state.sub_start.append(sub_start)
    return


def finalize_subroutine(state: ParseState, logger: Logger):
    func_name = "( finalize_subroutine )"
    if state.in_sub == 0:
        return
    sub_start = state.sub_start.pop()
    create_init_obj(state=state, logger=logger, sub_start=sub_start)
    # Reset subroutine state
    state.in_sub -= 1
    assert state.in_sub == len(state.sub_start)
    return


def set_in_function(state: ParseState, logger: Logger):
    """
    Parse beginning of function statement
    """
    func_name = "( set_in_function )"
    assert state.curr_line
    if regex_end_func.search(state.curr_line.line):
        return
    if (
        not regex_func_type.search(state.curr_line.line)
        and not regex_result.search(state.curr_line.line)
        and not re.match(r"^function\b", state.curr_line.line)
    ):
        logger.error(
            f"False Function statement for {state.curr_line}\n"
            f"type match: {find_variables.search(state.curr_line.line)}\n"
            f"result match: {regex_result.search(state.curr_line.line)}\n"
        )
        return
    parent = ""
    if state.in_sub > 0:
        assert len(state.sub_start) == 1
        parent = state.sub_start[0].subname

    state.in_func += 1
    cpp_ln = state.get_start_index() if state.cpp_file else None
    start_ln_pair = PreProcTuple(ln=state.curr_line.ln, cpp_ln=cpp_ln)
    func_init = check_function_start(state.curr_line.line, start_ln_pair, parent)
    state.func_init.append(func_init)
    return


def finalize_function(state: ParseState, logger: Logger):
    func_name = "( finalize_function )"
    if not state.in_func > 0 or not state.func_init:
        logger.error(
            f"{func_name}::ERROR: Matched function end without matching the start!\n{state.curr_line}"
        )
        sys.exit(1)
    func_init = state.func_init.pop()
    create_init_obj(state=state, logger=logger,func_init=func_init)
    # reset function state:
    state.in_func -= 1
    return


def handle_bad_inst(state: ParseState, logger: Logger):
    """
    Function to remove object from commented out module
    """
    assert state.curr_line
    l_cont = state.curr_line.line
    # Found an instance of something used by a "bad" module
    match_decl = find_variables.search(l_cont)
    if not match_decl:
        # Check if the bad element is in an if statement. If so, need to remove the entire if statement/block
        match_if = regex_if.search(l_cont)
        match_do = regex_do_start.search(l_cont)
        if match_if:
            start_index = state.line_it.start_index
            _, _ = state.line_it.consume_until(regex_endif, regex_if)
            state.line_it.comment_cont_block(index=start_index)
        elif match_do:
            start_index = state.get_start_index()
            _, _ = state.line_it.consume_until(regex_do_end, regex_do_start)
            state.line_it.comment_cont_block(index=start_index)
        else:
            # simple statement just remove
            set_comment(state, logger)
    elif match_decl and not state.in_sub > 0:
        # global variable, just remove
        set_comment(state, logger)
    return


def apply_comments(lines: list[LineTuple]) -> list[LineTuple]:
    comment_ = "!#py "

    def commentize(lt: LineTuple) -> LineTuple:
        if not lt.commented:
            return lt
        # find first non‑ws character and inject comment_
        new_line = re.sub(
            r"^(\s*)(\S)", lambda m: f"{m.group(1)}{comment_}{m.group(2)}", lt.line
        )
        return LineTuple(line=new_line, ln=lt.ln, commented=lt.commented)

    return [commentize(lt) for lt in lines]


def parse_bad_modules(
    state: ParseState,
    logger: Logger,
):
    """
    Comments out `use <bad_module>: ...` lines, updates bad_subroutines list.
    """
    global bad_subroutines
    global bad_modules
    # Build bad_modules pattern dynamically
    bad_mod_string = "|".join(bad_modules)
    module_pattern = re.compile(rf"\s*use\s+\b({bad_mod_string})\b", re.IGNORECASE)

    for fline in state.line_it:
        full_line = fline.line
        start_index = state.line_it.start_index
        orig_ln = state.line_it.lines[start_index].ln
        m = module_pattern.search(full_line)
        if not m:
            continue
        if ":" in full_line:
            # If there are explicit component lists after ':'
            comps = full_line.split(":", 1)[1].rstrip("\n")
            # remove Fortran assignment(=) syntax
            comps = re.sub(r"\bassignment\(=\)\b", "", comps, flags=re.IGNORECASE)
            for el in comps.split(","):
                el = el.strip()
                # handle => renaming
                if "=>" in el:
                    name, alias = [x.strip() for x in el.split("=>", 1)]
                    if "r8" in name:
                        continue
                    el = name if name.lower() == "nan" else f"{name}|{alias}"
                el = el.lower()
                if el and el not in bad_subroutines:
                    bad_subroutines.add(el)
            # comment out matched statement
            state.line_it.comment_cont_block(index=start_index)
        else:
            set_comment(state, logger)

    state.line_it.reset()

    return


def remove_subroutine(og_lines, cpp_lines, start):
    """Function to comment out an entire subroutine"""
    func_name = "remove_subroutine"
    end_sub = False

    og_ln = start.ln
    endline = 0
    while not end_sub:
        if og_ln > len(og_lines):
            print(f"{func_name}::ERROR didn't find end of subroutine")
            sys.exit(1)
        match_end = re.search(r"^(\s*end subroutine)", og_lines[og_ln].lower())
        if match_end:
            end_sub = True
            endline = og_ln
        og_lines[og_ln] = "!#py " + og_lines[og_ln]
        og_ln += 1

    if start.cpp_ln:
        # Manually find the end of the subroutine in the preprocessed file
        # Since the subroutine is not of interest, just find the end
        end_sub = False
        cpp_endline = 0
        cpp_ln = start.cpp_ln
        while not end_sub:
            match_end = re.search(r"^(\s*end subroutine)", cpp_lines[cpp_ln].lower())
            if match_end:
                end_sub = True
                cpp_endline = cpp_ln
            cpp_ln += 1
    else:
        cpp_endline = None

    out_ln_pair = PreProcTuple(cpp_ln=cpp_endline, ln=endline)

    return og_lines, out_ln_pair


def parse_local_mods(lines, start):
    """This function is called to determine if
    a subroutine uses ncdio_pio. and remove it if it does
    """
    past_mods = False
    remove = False
    ct = start
    while not past_mods and not remove and ct < len(lines):
        line = lines[ct]
        l = line.split("!")[0]
        if not l.strip():
            ct += 1
            continue
            # line is just a commment
        lline = line.strip().lower()
        if "ncdio_pio" in lline or "histfilemod" in lline or "spmdmod" in lline:
            remove = True
            break
        match_var = re.search(
            r"^(type|integer|real|logical|implicit)", l.strip().lower()
        )
        if match_var:
            past_mods = True
            break
        ct += 1

    return remove


def get_used_mods(
    ifile: str,  # fpath
    mods: list[str],  # list[fpath]
    singlefile: bool,
    mod_dict: dict[str, FortranModule],
    verbose: bool = False,
):
    """
    Checks to see what mods are needed to compile the file
    """
    # Keep track of nested level
    linenumber, module_name = get_module_name_from_file(fpath=ifile)
    fort_mod = FortranModule(fname=ifile, name=module_name, ln=linenumber)

    # Return if this module was aleady added for another subroutine
    if fort_mod.name in mod_dict:
        return mods, mod_dict

    lower_mods = [get_module_name_from_file(m)[1] for m in mods]
    # Read file
    needed_mods = []
    ct = 0
    fn = ifile
    file = open(fn, "r")
    lines = file.readlines()
    file.close()

    lpairs = [LineTuple(line=line, ln=i) for i, line in enumerate(lines)]
    line_it = LogicalLineIterator(lines=lpairs, log_name="get_used_mods")
    fort_mod.num_lines = len(lines)
    fort_mod.module_lines = [LineTuple(line=line, ln=i) for i, line in enumerate(lines)]

    regex_type_start = re.compile(r"^\s*type\s*(?!\()", re.IGNORECASE)
    regex_skip = re.compile(r"^type\s*\(")
    regex_type_end = re.compile(r"^\s*end\s*type")

    # Define regular expressions for catching variables declared in the module
    regex_contains = re.compile(r"^(contains)", re.IGNORECASE)
    in_type: bool = False

    match_use = re.compile(r"^(use)\s+")
    use_lines = line_it.get_lines(match_use)
    use_stmts = parse_use_stmts(use_lines)
    for stmt in use_stmts:
        mod = stmt.module
        if mod in bad_modules or re.search(r'betr',mod.lower()):
            continue
        needed_modfile = get_filename_from_module(mod, verbose=verbose)
        if needed_modfile is None:
            bad_modules.add(mod)
        else:
            fort_mod.add_dependency(stmt)
            if (
                mod not in needed_mods
                and mod not in lower_mods
                and mod not in ["cudafor", "verificationmod"]
            ):
                needed_mods.append(mod)

    var_decls_lines: list[LineTuple] = []
    module_head = True
    for fline in line_it:
        full_line = fline.line
        ct = line_it.get_start_ln()
        match_contains = regex_contains.search(full_line)
        if match_contains and not in_type:
            module_head = False
            fort_mod.end_of_head_ln = ct
            continue

        if module_head:
            if not in_type:
                m_start = regex_type_start.search(full_line)
                m_skip = regex_skip.search(full_line)
                in_type = bool(m_start and not m_skip)
            else:
                m_end = regex_type_end.search(full_line)
                if m_end:
                    in_type = False
            if not in_type and find_variables.search(full_line):
                var_decls_lines.append(LineTuple(ln=ct, line=full_line))

    if module_head:
        fort_mod.end_of_head_ln = len(lines) - 1
    head_lines = [
        LineTuple(line=lines[i], ln=i) for i in range(0, fort_mod.end_of_head_ln)
    ]
    parsed_type_dict = parse_derived_type_definition(head_lines, module_name, ifile)
    fort_mod.defined_types.update(parsed_type_dict)
    # Done with first pass through the file.
    # Check against already used Mods
    files_to_parse = []
    for m in needed_mods:
        needed_modfile = get_filename_from_module(m, verbose=verbose)
        if needed_modfile not in mods:
            files_to_parse.append(needed_modfile)
            mods.append(needed_modfile)

    if var_decls_lines:
        variable_list = parse_variable_decl(var_decls_lines, module_name)
        for v in variable_list:
            v.declaration = module_name
            fort_mod.global_vars[v.name] = v
    list_type_names = [key for key in fort_mod.defined_types]
    for gvar in fort_mod.global_vars.values():
        if gvar.type in list_type_names:
            if gvar.name not in fort_mod.defined_types[gvar.type].instances:
                fort_mod.defined_types[gvar.type].instances[gvar.name] = gvar

    mod_dict[fort_mod.name] = fort_mod
    if ifile not in mods:
        mods.append(ifile)

    # Recursive call to the mods that need to be processed
    if files_to_parse and not singlefile:
        for f in files_to_parse:
            mods, mod_dict = get_used_mods(
                ifile=f,
                mods=mods,
                verbose=verbose,
                singlefile=singlefile,
                mod_dict=mod_dict,
            )

    return mods, mod_dict


def remove_cpp_directives(
    cpp_lines: list[str], fn: str, logger: Logger
) -> list[LineTuple]:
    """
    Map cpp_lines back to original line numbers using # line directives.
    Skip mappings when directives point to included files or builtins.
        Returns map orig_ln -> cpp_ln, work lines.
    """

    mapping: dict[int, int] = {}
    work_lines: list[LineTuple] = []

    orig_ln: Optional[int] = None
    target = os.path.abspath(fn)
    remove_include = "shr_assert.h"
    match_assert = False
    for i, line in enumerate(cpp_lines):
        # Match GCC line directive: # lineno "filename" flags
        m = re.match(r"#\s*(\d+)\s+\"(.*)\"", line)
        if m:
            lineno = int(m.group(1)) - 1
            fname = m.group(2)
            # Normalize path
            # Only map when returning to lines in our source file
            # Only treat fname as file if it exists on disk
            abs = os.path.abspath(fname) if os.path.exists(fname) else None
            if abs and os.path.basename(abs) == remove_include:
                match_assert = True
            orig_ln = lineno if abs == target else None

            continue

        # If current_orig set, map this preprocessed line to that orig line
        if orig_ln is not None:
            # capture the first encountered preprocessed text for that orig line
            if match_assert:
                work_lines.append(
                    LineTuple(
                        line=f"#include '{remove_include}'",
                        ln=orig_ln - 1,
                        commented=True,
                    )
                )
                match_assert = False
            mapping[orig_ln] = i
            work_lines.append(LineTuple(line=cpp_lines[i], ln=orig_ln, commented=False))
            orig_ln += 1

    return work_lines


def apply_preprocessor(fn: str) -> list[str]:
    base_fn = fn.split("/")[-1]
    new_fn = f"{spel_output_dir}cpp_{base_fn}"

    # Set up cmd for preprocessing. Get macros used:
    macros_string = "-D" + " -D".join(macros)
    cmd = f"gfortran -I{E3SM_SRCROOT}/share/include {macros_string} -cpp -E {fn} > {new_fn}"
    _ = sp.getoutput(cmd)

    # read lines of preprocessed file
    file = open(new_fn, "r")
    cpp_lines = file.readlines()
    file.close()

    return cpp_lines


def modify_file(
    orig_lines: list[LineTuple],
    fort_mod: FortranModule,
    pass_manager: PassManager,
    case_dir: str,
    overwrite: bool,
) -> ModParseResult:
    """
    Function that modifies the source code of the file
    Occurs after parsing the file for subroutines and modules
    """
    func_name = "( modify_file )"
    fn: str = fort_mod.filepath
    mod_name: str = fort_mod.name

    logger = pass_manager.logger
    # Test if the file in question contains any ifdef statements:
    # cmd = f'grep -E "ifn?def"  {fn} | grep -v "_OPENACC"'
    # output = sp.getoutput(cmd)
    base_fn = fn.split("/")[-1]

    cpp_file = True
    cpp_lines = apply_preprocessor(fn)
    lines = [line.line for line in orig_lines]

    work_lines = remove_cpp_directives(cpp_lines, fn, logger)
    ### SANITY CHECK ####
    for lt in work_lines:
        if lt.line.rstrip("\n").strip():
            if not re.search(
                r"(__file__|__line__|include|assert)", lines[lt.ln].lower()
            ):
                assert (
                    lt.line.rstrip("\n").strip() == lines[lt.ln].rstrip("\n").strip()
                ), f"Couldn't map cpp lines for {base_fn}\n{lt.line}\n /=\n{lines[lt.ln]}"

    state = ParseState(
        module_name=mod_name,
        fort_mod=fort_mod,
        cpp_file=cpp_file,
        work_lines=work_lines,
        orig_lines=orig_lines,
        path=fn,
        curr_line=None,
        line_it=LogicalLineIterator(work_lines),
        logger=logger,
        sub_init_dict={},
        removed_subs=[],
        in_sub=0,
        in_func=0,
    )
    parse_bad_modules(state, logger)

    # Join bad subroutines into single string with logical OR for regex. Commented out if matched.
    # these two likely don't need to be separate regexes
    regex_hack = re.compile(r"\b(nan|spval|r8)\b")
    global bad_subroutines
    bad_subroutines = {el for el in bad_subroutines if not regex_hack.search(el)}

    bad_sub_string = "|".join(bad_subroutines)
    bad_sub_string = f"({bad_sub_string})"
    regex_call = re.compile(rf"\s*(call)[\s]+{bad_sub_string}", re.IGNORECASE)
    regex_bad_inst = re.compile(rf"\b({bad_sub_string})\b")

    pass_manager.remove_pass(name=parse_sub_call)
    pass_manager.remove_pass(name=parse_bad_inst)
    pass_manager.add_pass(pattern=regex_call, fn=set_comment, name=parse_sub_call)
    pass_manager.add_pass(
        pattern=regex_bad_inst,
        fn=handle_bad_inst,
        name=parse_bad_inst,
    )
    pass_manager.run(state)

    remove_reference_to_subroutine(state, fort_mod.end_of_head_ln)

    if cpp_file:
        # take the commented work_lines and comment corresponding orig_lines
        for work_line in state.work_lines:
            if work_line.commented:
                og_ln = work_line.ln
                state.orig_lines[og_ln].commented = True
        parsed_lts = apply_comments(state.work_lines)
        write_lts = apply_comments(state.orig_lines)
        write_lines = [line.line for line in write_lts]
    else:
        parsed_lts = apply_comments(state.work_lines)
        write_lines = [line.line for line in parsed_lts]

    parsed_lines = [line.line for line in parsed_lts]

    if overwrite:
        out_fn = f"{case_dir}/{fn.split('/')[-1]}"
        with open(out_fn, "w") as ofile:
            ofile.writelines(write_lines)

    return state.sub_init_dict, parsed_lines


def process_for_unit_test(
    case_dir: str,
    mod_dict: dict[str, FortranModule],
    sub_dict: dict[str, Subroutine],
    mods: list[str],
    required_mods: list[str],
    sub_name_list: list[str],
    overwrite=False,
    verbose=False,
    singlefile=False,
):
    """
    This function looks at the whole .F90 file.
    Comments out functions that are cumbersome
    for unit testing purposes.
    Gets module dependencies of the module and
    process them recursively.

    Arguments:
        fname -> File path for .F90 file that with needed subroutine
        case_dir -> label of Unit Test
        mods     -> list of already known (if any) files that were previously processed
        required_mods -> list of modules that are required for the unit test (see config.py)
        main_sub_dict -> dictionary of all subroutines encountered for the unit test.
        verbose  -> Print more info
        singlefile -> flag that disables recursive processing.
    """
    func_name = "( process_for_unit_test )"

    # First, get complete list of module to be processed and removed.
    # and then add processed file to list of mods:
    pass_manager = PassManager(logger=get_logger("PassManager"))
    pass_manager.add_pass(pattern=regex_sub, fn=set_in_subroutine, name=parse_sub_start)
    pass_manager.add_pass(
        pattern=regex_end_sub,
        fn=finalize_subroutine,
        name=parse_sub_end,
    )
    pass_manager.add_pass(
        pattern=regex_end_func,
        fn=finalize_function,
        name=parse_func_end,
    )
    pass_manager.add_pass(pattern=regex_func, fn=set_in_function, name=parse_func_start)
    pass_manager.add_pass(
        pattern=regex_shr_assert,
        fn=set_comment,
        name=parse_shr_assert,
    )
    pass_manager.add_pass(
        pattern=regex_include_assert,
        fn=set_comment,
        name=parse_inc_shr_assert,
    )

    pass_manager.add_pass(
        pattern=re.compile(r"^(contains)", re.IGNORECASE),
        fn=check_contains,
        name=host_program,
    )

    with profile_ctx(enabled=False, section="get_used_mods") as pc:
        # Find if this file has any not-processed mods
        for s in sub_name_list:
            if "::" in s:
                mod_name, sub_name = s.split("::")
                fname = get_filename_from_module(mod_name)
                assert fname, f"Error -- couldn't find file for {s}"
            else:
                fname, _, _ = find_file_for_subroutine(name=s)
            mods, mod_dict = get_used_mods(
                ifile=fname,
                mods=mods,
                verbose=verbose,
                singlefile=singlefile,
                mod_dict=mod_dict,
            )

    required_mod_paths = [get_filename_from_module(m) for m in required_mods]
    for rmod in required_mod_paths:
        if rmod not in mods:
            mods, mod_dict = get_used_mods(
                ifile=rmod,
                mods=mods,
                verbose=verbose,
                singlefile=singlefile,
                mod_dict=mod_dict,
            )

    for fort_mod in mod_dict.values():
        fort_mod.modules = fort_mod.sort_module_deps(
            startln=0,
            endln=fort_mod.num_lines,
        )
        fort_mod.head_modules = fort_mod.sort_module_deps(
            startln=0,
            endln=fort_mod.end_of_head_ln,
        )

    # Sort the file dependencies
    with profile_ctx(enabled=False, section="sort_file_dependency") as pc:
        ordered_mods = sort_file_dependency(mod_dict)

    # Next, each needed module is parsed for subroutines and removal of
    # any dependencies that are not needed for an ELM unit test (eg., I/O libs,...)
    # Note:
    #    Modules are parsed starting with leaf nodes so that all
    #    child subroutines will have been instantiated
    sub_init_dict: dict[str, SubInit] = {}
    with profile_ctx(enabled=False, section="modify_file") as pc:
        for mod_name in ordered_mods:
            fort_mod = mod_dict[mod_name]
            if mod_dict[mod_name].modified:
                continue
            lines = fort_mod.module_lines[:]

            temp_objs, parsed_lines = modify_file(
                lines,
                fort_mod,
                pass_manager,
                case_dir,
                overwrite=overwrite,
            )
            sub_init_dict.update(temp_objs)
            mod_dict[mod_name].subroutines = {sub for sub in temp_objs.keys()}
            mod_lines_unwrp = unwrap_section(lines=parsed_lines, startln=0)
            mod_dict[mod_name].module_lines = mod_lines_unwrp
            mod_dict[mod_name].modified = True

    if not sub_init_dict:
        print(func_name + "No Subroutines/Functions found! -- exiting")
        sys.exit(1)
    for sub_id, initobj in sub_init_dict.items():
        m_lines = mod_dict[initobj.mod_name].get_mod_lines()
        initobj.mod_lines = m_lines
        subname = initobj.name
        sub_dict[f"{initobj.mod_name}::{subname}"] = Subroutine(initobj)

    for fort_mod in mod_dict.values():
        fort_mod.find_allocations(sub_dict)

    return ordered_mods


def remove_reference_to_subroutine(state: ParseState, head_ln: int):
    """
    Given list of subroutine names, this function goes back and
    comments out declarations and other references
    """
    if not state.removed_subs:
        return
    sname_string = "|".join(state.removed_subs)

    regex_sub_decl = re.compile(rf"\b({sname_string})\b")
    state.line_it.reset()
    for fline in state.line_it:
        full_line = fline.line
        if state.line_it.get_orig_ln() > head_ln:
            break
        if regex_sub_decl.search(full_line):
            set_comment(state)


def sort_file_dependency(mod_dict: ModDict) -> list[str]:
    """
    Function that unravels a dictionary of all module files
    that were parsed in process_for_unit_test.

    Each element of the dictionary is a FortranModule object.
    """
    trees: list[ModTree] = build_module_tree(mod_dict)
    order: list[str] = []
    for tree in trees:
        for node in tree.traverse_postorder():
            if node.node not in order:
                order.append(node.node)

    return [m for m in order]
