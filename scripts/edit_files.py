from __future__ import annotations

import logging
import os
import pprint
import re
import subprocess as sp
import sys
from logging import Logger
from typing import Optional

import scripts.dynamic_globals as dg
from scripts.analyze_subroutines import Subroutine
from scripts.check_sections import check_function_start, create_init_obj
from scripts.DerivedType import DerivedType, get_derived_type_definition
from scripts.fortran_modules import (FortranModule, ModTree, build_module_tree,
                                     get_filename_from_module,
                                     get_module_name_from_file)
from scripts.logging_configs import get_logger, set_logger_level
from scripts.mod_config import E3SM_SRCROOT, spel_output_dir
from scripts.profiler_context import profile_ctx
from scripts.types import (LineTuple, LogicalLineIterator, ParseState,
                           PassManager, PreProcTuple, SubInit, SubStart)
from scripts.utilityFunctions import (comment_line, find_file_for_subroutine,
                                      find_variables, line_unwrapper,
                                      parse_line_for_variables, unwrap_section)

# Define Return types
ModParseResult = tuple[dict[str,SubInit],list[str]]

# Compile list of lower-case module names to remove
# SPEL expects these all to be lower-case currently
bad_modules = [
    "abortutils",
    "shr_log_mod",
    "elm_time_manager",
    "shr_infnan_mod",
    "clm_time_manager",
    "pio",
    "shr_sys_mod",
    "perf_mod",
    "shr_assert_mod",
    "spmdmod",
    "restutilmod",
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
    "temperaturetype",
    "waterfluxtype",
    "shr_file_mod",
    "mct_mod",
    "elm_instmod",
    "spmdgathscatmod",
    "perfmod_gpu",
]

fates_mod = ["elmfatesinterfacemod"]
betr_mods = ["betrsimulationalm"]

bad_subroutines = [
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
]

remove_subs = [
    "prepare_data_for_em_ptm_driver",
    "prepare_data_for_em_vsfm_driver",
    "decompinit_lnd_using_gp",
]

# regex_if = re.compile(r"^(if)[\s]*(?=\()", re.IGNORECASE)
regex_if = re.compile(r'^if\s*\(.*\)\s*then', re.IGNORECASE)
regex_do_start = re.compile(r"^\s*(\w+:)?\s*do\b", re.IGNORECASE)
regex_do_end = re.compile(r"^\s*(end\s*do(\s+\w+)?)", re.IGNORECASE)
regex_ifthen = re.compile(r"^(if\b)(.+)(then)$", re.IGNORECASE)
regex_endif = re.compile(r"^(end\s*if)", re.IGNORECASE)

regex_include_assert = re.compile(r"^(#include)\s+[\"\'](shr_assert.h)[\'\"]")
regex_sub = re.compile(r"^\s*(subroutine)\s+")
regex_shr_assert = re.compile(r"^\s*(shr_assert_all|shr_assert)\b")
regex_end_sub = re.compile(r"^\s*(end subroutine)")

regex_func = re.compile(r"\bfunction\s+\w+\s*\(")
regex_result = re.compile(r"\bresult\s*\(\s*\w+\s*\)$")
regex_func_type = re.compile(
    r"(type\s*\(|integer|real|logical|character|complex)", re.IGNORECASE
)

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

def set_comment(state: ParseState, logger: Logger):
    state.line_it.comment_cont_block()
    return

def set_in_subroutine(state: ParseState,logger: Logger):
    """
    Function that parses the line of a subroutine for it's name, and
    f desired, will comment out the entire subroutine, else return it's name and starting lineno
    """
    func_name = "( set_in_subroutine )"
    if state.in_sub:
        logger.info("Sub module subroutine!")

    assert state.curr_line
    state.in_sub = True
    ct = state.get_start_index()

    full_line = state.curr_line.line
    start_ln = state.curr_line.ln

    subname = full_line.split()[1].split("(")[0].strip()
    if  "_oacc" in subname:
        logger.info(f"{func_name} skipping {subname}")
        return
    logger.debug(f"{func_name}found subroutine {subname} at {ct+1}")

    cpp_ln = ct if state.cpp_file else None
    sub_start: Optional[SubStart] = SubStart(subname=subname,start_ln=start_ln,cpp_ln=cpp_ln)

    # Check if subroutine needs to be removed before processing
    match_remove = bool(subname.lower() in remove_subs)
    test_init = bool("init" not in subname.lower().replace("initcold", "cold"))
    match_gsmap = regex_gsmap.search(subname)

    # Remove if it's an IO routine
    test_decompinit = bool(subname.lower() == "decompinit_lnd_using_gp")
    remove = bool((match_remove and test_init) or match_gsmap or test_decompinit)
    logger.debug(f"Match gsmap {match_gsmap} with pattern: {regex_gsmap.pattern} on {subname}\n"
        f"remove: {remove}"
                    )
    if remove:
        _, _ = state.line_it.consume_until(end_pattern=regex_end_sub, start_pattern=None)
        state.removed_subs.append(subname)
        state.in_sub = False
        state.line_it.comment_cont_block(index=ct)
        sub_start = None
    state.sub_start = sub_start
    state.func_init = None

    return

def finalize_subroutine(state: ParseState, logger: Logger):
    func_name = "( finalize_subroutine )"
    if not state.in_sub:
        print(
            f"{func_name}::ERROR: Matched subroutine end without matching the start! {state.curr_line}"
        )
        sys.exit(1)
    create_init_obj(state=state, logger=logger)
    # Reset subroutine state
    state.in_sub = False
    state.sub_start = None
    return

def set_in_function(state: ParseState, logger: Logger):
    """
    Parse beginning of function statement
    """
    func_name = "( set_in_function )"
    assert state.curr_line
    if regex_end_func.search(state.curr_line.line):
        return
    if (not regex_func_type.search(state.curr_line.line) 
            and not regex_result.search(state.curr_line.line)
            and not re.match(r'^function\b',state.curr_line.line) ):
        logger.debug(f"False Function statement for {state.curr_line}\n"
            f"type match: {find_variables.search(state.curr_line.line)}\n"
            f"result match: {regex_result.search(state.curr_line.line)}\n")
        return
    if state.in_sub:
        logger.error(f"{func_name}::Error-Encountered function decl in subroutine!\n{state.curr_line}")
        sys.exit(1)

    state.in_func = True
    cpp_ln = state.get_start_index() if state.cpp_file else None
    start_ln_pair = PreProcTuple(ln=state.curr_line.ln,cpp_ln=cpp_ln)
    logger.debug(f"{func_name} curr line: {state.curr_line} cpp_ln = {cpp_ln}")
    func_init = check_function_start(state.curr_line.line, start_ln_pair, logger)
    state.func_init = func_init
    return

def finalize_function(state: ParseState, logger: Logger):
    func_name = "( finalize_function )"
    if not state.in_func or not state.func_init:
        logger.error(
            f"{func_name}::ERROR: Matched function end without matching the start!\n{state.curr_line}"
        )
        sys.exit(1)
    create_init_obj(state=state,logger=logger)
    # reset function state:
    state.in_func = False
    state.func_init = None
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
            _, _ = state.line_it.consume_until(regex_endif,regex_if)
            state.line_it.comment_cont_block(index=start_index)
        elif match_do:
            start_index = state.get_start_index()
            _, _ = state.line_it.consume_until(regex_do_end, regex_do_start)
            state.line_it.comment_cont_block(index=start_index)
        else:
            # simple statement just remove
            set_comment(state, logger)
    elif match_decl and not state.in_sub:
        # global variable, just remove
        # in sub requires further differentiation between arguments/locals -- removing arguments is foolhardy
        set_comment(state,logger)
    return


def apply_comments(lines: list[LineTuple]) -> list[LineTuple]:
    comment_ = "!#py "

    def commentize(lt: LineTuple) -> LineTuple:
        if not lt.commented:
            return lt
        # find first non‑ws character and inject comment_
        new_line = re.sub(r"^(\s*)(\S)",
                          lambda m: f"{m.group(1)}{comment_}{m.group(2)}",
                          lt.line)
        return LineTuple(line=new_line, ln=lt.ln, commented=lt.commented)

    return [ commentize(lt) for lt in lines ]

def parse_bad_modules(
    state: ParseState,
    logger: Logger,
) :
    """
    Comments out `use <bad_module>: ...` lines, updates bad_subroutines list.
    """
    global bad_subroutines
    global bad_modules
    # Build bad_modules pattern dynamically
    bad_mod_string = "|".join(bad_modules)
    module_pattern = re.compile(rf"\s*use\s+\b({bad_mod_string})\b", re.IGNORECASE)

    logger.debug(f"Module pattern: {module_pattern.pattern}")
    for full_line, _ in state.line_it:
        start_index = state.line_it.start_index
        orig_ln = state.line_it.lines[start_index].ln
        m = module_pattern.search(full_line)
        if not m:
            continue
        if ':' in full_line:
            # If there are explicit component lists after ':'
            logger.debug(f"removing dep:\n{full_line}")
            comps = full_line.split(':', 1)[1].rstrip("\n")
            # remove Fortran assignment(=) syntax
            comps = re.sub(r"\bassignment\(=\)\b", "", comps, flags=re.IGNORECASE)
            for el in comps.split(','):
                el = el.strip()
                # handle => renaming
                if '=>' in el:
                    # think this is backwards... 
                    name, alias = [x.strip() for x in el.split('=>',1)]
                    el = name if name.lower() == 'nan' else f"{name}|{alias}"
                el = el.lower()
                if el and el not in bad_subroutines:
                    bad_subroutines.append(el)
            # comment out matched statement
            state.line_it.comment_cont_block(index=start_index)
        else:
            set_comment(state,logger)

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
        ifile: str, # fpath
        mods: list[str], # list[fpath]
        singlefile: bool,
        mod_dict: dict[str, FortranModule],
        verbose: bool=False,
):
    """
    Checks to see what mods are needed to compile the file
    """
    func_name = "get_used_mods"

    # Keep track of nested level
    linenumber, module_name = get_module_name_from_file(fpath=ifile)
    fort_mod = FortranModule(fname=ifile, name=module_name, ln=linenumber)

    # Return if this module was aleady added for another subroutine
    if fort_mod.name in mod_dict:
        return mods, mod_dict

    # Read file
    needed_mods = []
    ct = 0
    fn = ifile
    file = open(fn, "r")
    lines = file.readlines()
    file.close()

    fort_mod.num_lines = len(lines)

    # Define regular expressions for catching variables declared in the module
    regex_contains = re.compile(r"^(contains)", re.IGNORECASE)
    user_defined_types = {}  # dictionary to store user-defined types in module

    module_head = True
    while ct < len(lines):
        l, ct = line_unwrapper(lines=lines, ct=ct)
        l = l.strip().lower()
        match_contains = regex_contains.search(l)
        if match_contains:
            module_head = False
            fort_mod.end_of_head_ln = ct
            # Nothing else to check in this line
            ct += 1
            continue

        if module_head:
            # First check if a user-derived type is being defined
            # create new l to simplify regex needed.
            lprime = l.replace("public", "").replace("private", "").replace(",", "")
            match_type_def1 = re.search(r"^(type\s*::)", lprime)  # type :: type_name
            match_type_def2 = re.search(r"^(type\s+)(?!\()", lprime)  # type type_name
            type_name = None
            if match_type_def1:
                # Get type name
                type_name = lprime.split("::")[1].strip()
            elif match_type_def2:
                # Get type name
                matched_expr = match_type_def2.group()
                type_name = lprime.replace(matched_expr, "").strip()
            if type_name:
                # Analyze the type definition
                user_dtype, ct = get_derived_type_definition(
                    ifile=ifile,
                    modname=module_name,
                    lines=lines,
                    ln=ct,
                    type_name=type_name,
                    verbose=verbose,
                )
                user_defined_types[type_name] = user_dtype

            # Check for variable declarations
            variable_list = parse_line_for_variables(
                ifile=ifile, l=l, ln=ct, verbose=verbose,
            )
            # Store variable as Variable Class object and add to Module object
            if variable_list:
                for v in variable_list:
                    v.declaration = module_name
                    fort_mod.global_vars[v.name] = v

        match_use = re.search(r"^(use)[\s]+", l)
        if match_use:
            # Get rid of comma if no space
            templ = l.replace(",", " ")
            mod = templ.split()[1]
            mod = mod.strip()
            mod = mod.lower()
            if mod not in bad_modules:
                fort_mod.add_dependency(mod=mod,line=l,ln=ct)
                lower_mods = [get_module_name_from_file(m)[1] for m in mods] 
                if (
                    mod not in needed_mods
                    and mod not in lower_mods
                    and mod not in ["elm_instmod", "cudafor", "verificationmod"]
                ):
                    needed_mods.append(mod)
        ct += 1

    # Done with first pass through the file.
    # Check against already used Mods
    files_to_parse = []
    for m in needed_mods:
        if m in bad_modules:
            continue
        needed_modfile = get_filename_from_module(m, verbose=verbose)
        if needed_modfile is None:
            if verbose:
                print(
                    f"Couldn't find {m} in ELM or shared source -- adding to removal list"
                )
            bad_modules.append(m)
            regex_modkey = re.compile(rf"{m}@\d+")
            keys_to_remove = [key for key in fort_mod.modules_by_ln if regex_modkey.search(key)]
            for key in keys_to_remove:
                fort_mod.modules_by_ln.pop(key)
        elif needed_modfile not in mods:
            files_to_parse.append(needed_modfile)
            mods.append(needed_modfile)

    # Store user-defined types in the module object
    # and find any global variables that have the user-defined type

    list_type_names = [key for key in user_defined_types.keys()]
    for gvar in fort_mod.global_vars.values():
        if gvar.type in list_type_names:
            if gvar.name not in user_defined_types[gvar.type].instances:
                user_defined_types[gvar.type].instances[gvar.name] = gvar

    fort_mod.defined_types = user_defined_types
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



def remove_cpp_directives(cpp_lines: list[str], fn: str, logger: Logger) -> list[LineTuple]:
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
                work_lines.append(LineTuple(line=f"#include '{remove_include}'", ln=orig_ln-1, commented=True))
                match_assert = False
            mapping[orig_ln] = i
            work_lines.append(LineTuple(line=cpp_lines[i],ln=orig_ln,commented=False))
            orig_ln += 1

    return work_lines


def modify_file(
    lines: list[str],
    fn: str,
    mod_name: str,
    pass_manager: PassManager,
    case_dir: str,
    overwrite: bool,
)->ModParseResult:
    """
    Function that modifies the source code of the file
    Occurs after parsing the file for subroutines and modules
    """
    func_name = "( modify_file )"
    iter_logger = get_logger("LineIter")
    set_logger_level(logger=iter_logger, level=logging.INFO)
    set_logger_level(logger=pass_manager.logger, level=logging.INFO)
    mod_debug = "???"
    if mod_name == mod_debug:
        set_logger_level(logger=pass_manager.logger, level=logging.DEBUG)
        set_logger_level(logger=iter_logger, level=logging.DEBUG)

    logger = pass_manager.logger
    # Test if the file in question contains any ifdef statements:
    cmd = f'grep -E "ifn?def"  {fn} | grep -v "_OPENACC"'
    output = sp.getoutput(cmd)
    base_fn = fn.split("/")[-1]

    if output:
        # For CPP files, regex operates on the cpp_lines, but only the original lines are commented
        cpp_file = True
        new_fn = f"{spel_output_dir}cpp_{base_fn}"

        # Set up cmd for preprocessing. Get macros used:
        macros_string = "-D" + " -D".join(macros)
        cmd = f"gfortran -I{E3SM_SRCROOT}/share/include {macros_string} -cpp -E {fn} > {new_fn}"
        _ = sp.getoutput(cmd)

        # read lines of preprocessed file
        file = open(new_fn, "r")
        cpp_lines = file.readlines()
        file.close()
    else:
        cpp_file = False
        cpp_lines = lines[:]

    orig_lines = [ LineTuple(line=text,ln=i) for i,text in enumerate(lines) ]
    if cpp_file:
        work_lines = remove_cpp_directives(cpp_lines, fn,logger)
        ### SANITY CHECK ####
        for lt in work_lines:
            if lt.line.rstrip("\n").strip():
                if not re.search(r'(__FILE__|__LINE__|include)', lines[lt.ln]):
                    assert lt.line == lines[lt.ln], f"Couldn't map cpp lines for {base_fn}\n{lt.line} /= {lines[lt.ln]}"
    else:
        work_lines = orig_lines

    state = ParseState(
        module_name=mod_name,
        cpp_file=cpp_file,
        work_lines=work_lines,
        orig_lines=orig_lines,
        path=fn,
        curr_line=None,
        line_it= LogicalLineIterator(work_lines),
        sub_init_dict={},
        removed_subs=[],
        in_sub=False,
        in_func=False,
        sub_start=None,
        func_init=None,
    )
    parse_bad_modules(state, logger)

    # Join bad subroutines into single string with logical OR for regex. Commented out if matched.
    # these two likely don't need to be separate regexes
    global bad_subroutines
    bad_subroutines = [el for el in bad_subroutines if el != 'nan']
    bad_sub_string = "|".join(bad_subroutines)
    bad_sub_string = f"({bad_sub_string})"
    regex_call = re.compile(rf"\s*(call)[\s]+{bad_sub_string}",  re.IGNORECASE)
    regex_bad_inst = re.compile(rf"\b({bad_sub_string})\b")

    pass_manager.remove_pass(name=parse_sub_call)
    pass_manager.remove_pass(name=parse_bad_inst)
    pass_manager.add_pass(pattern=regex_call,fn=set_comment,name=parse_sub_call)
    pass_manager.add_pass(pattern=regex_bad_inst,fn=handle_bad_inst,name=parse_bad_inst)
    pass_manager.run(state)

    if mod_name == mod_debug:
        pass_manager.logger.info("Commented out the following lines: ")
        for lt in state.work_lines:
            if lt.commented:
                pass_manager.logger.info(f"{lt}")

    if cpp_file:
        # take the commented work_lines and comment corresponding orig_lines
        for work_line in state.work_lines:
            if work_line.commented:
                og_ln = work_line.ln
                state.orig_lines[og_ln].commented = True
        parsed_lts = apply_comments(state.work_lines)
        write_lts = apply_comments(state.orig_lines)
        write_lines = [ line.line for line in write_lts]
    else: 
        parsed_lts = apply_comments(state.work_lines)
        write_lines = [ line.line for line in parsed_lts ]

    parsed_lines = [ line.line for line in parsed_lts]

    if overwrite:
        out_fn = f"{case_dir}/{fn.split('/')[-1]}"
        logger.debug("Writing to file: %s", out_fn)
        with open(out_fn, "w") as ofile:
            ofile.writelines(write_lines)

    return state.sub_init_dict, parsed_lines


def process_for_unit_test(
    case_dir: str,
    mod_dict: dict[str, FortranModule],
    sub_dict: dict[str,Subroutine],
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
        required_mods -> list of modules that are required for the unit test (see mod_config.py)
        main_sub_dict -> dictionary of all subroutines encountered for the unit test.
        verbose  -> Print more info
        singlefile -> flag that disables recursive processing.
    """
    func_name = "( process_for_unit_test )"

    # First, get complete list of module to be processed and removed.
    # and then add processed file to list of mods:
    pass_manager = PassManager(logger=get_logger("PassManager"))
    pass_manager.add_pass(pattern=regex_sub,fn=set_in_subroutine, name=parse_sub_start )
    pass_manager.add_pass(pattern=regex_end_sub,fn=finalize_subroutine,name=parse_sub_end)
    pass_manager.add_pass(pattern=regex_end_func,fn=finalize_function,name=parse_func_end)
    pass_manager.add_pass(pattern=regex_func,fn=set_in_function,name=parse_func_start)
    pass_manager.add_pass(pattern=regex_shr_assert,fn=set_comment,name=parse_shr_assert)
    pass_manager.add_pass(pattern=regex_include_assert,fn=set_comment,name=parse_inc_shr_assert)

    with profile_ctx(enabled=False, section="get_used_mods") as pc:
        # Find if this file has any not-processed mods
        for s in sub_name_list:
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
        fort_mod.modules = fort_mod.sort_module_deps(startln=0,endln=fort_mod.num_lines)
        fort_mod.head_modules = fort_mod.sort_module_deps(startln=0,endln=fort_mod.end_of_head_ln)

    # Sort the file dependencies
    with profile_ctx(enabled=False, section="sort_file_dependency") as pc:
        ordered_mods = sort_file_dependency(mod_dict)

    # Next, each needed module is parsed for subroutines and removal of
    # any dependencies that are not needed for an ELM unit test (eg., I/O libs,...)
    # Note:
    #    Modules are parsed starting with leaf nodes so that all
    #    child subroutines will have been instantiated
    sub_init_dict: dict[str, SubInit] = {} 
    with profile_ctx(enabled=False,section="modify_file") as pc:
        for mod_name in ordered_mods:
            mod_file = get_filename_from_module(mod_name)
            if not mod_file:
                sys.exit(f"Error -- couldn't find file for {mod_name}")
            if mod_dict[mod_name].modified:
                continue
            file = open(mod_file, "r")
            lines = file.readlines()
            file.close()
            temp_objs, parsed_lines = modify_file(
                lines,
                mod_file,
                mod_name,
                pass_manager,
                case_dir,
                overwrite=overwrite,
            )
            sub_init_dict.update(temp_objs)
            mod_dict[mod_name].subroutines = { sub.split("::")[-1] for sub in temp_objs.keys() }

            mod_lines_unwrp = unwrap_section(lines=parsed_lines, startln=0)
            mod_dict[mod_name].module_lines = mod_lines_unwrp

            mod_dict[mod_name].modified = True

    if not sub_init_dict:
        print(func_name+"No Subroutines/Functions found! -- exiting")
        sys.exit(1)
    with profile_ctx(enabled=False, section="Initialize Subroutines") as pc:
        for sub_id, initobj in sub_init_dict.items():
            m_lines = mod_dict[initobj.mod_name].get_mod_lines()
            initobj.mod_lines = m_lines
            subname = initobj.name
            sub_dict[subname] = Subroutine(initobj,)

    return ordered_mods


def remove_reference_to_subroutine(lines, subnames):
    """
    Given list of subroutine names, this function goes back and
    comments out declarations and other references
    """
    sname_string = "|".join(subnames)
    sname_string = f"({sname_string})"
    print(f"remove reference to {sname_string}")
    ct = 0
    while ct < len(lines):
        line = lines[ct]
        l = line.split("!")[0]  # don't search comments
        if not l.strip():
            ct += 1
            continue
        match = re.search(sname_string.lower(), l.lower())
        if match:
            lines, ct = comment_line(lines=lines, ct=ct)
        ct += 1
    return lines


def sort_file_dependency(mod_dict: ModDict)-> list[str]:
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
