from __future__ import annotations

import re
import subprocess as sp
import sys
from typing import TYPE_CHECKING, Optional

from scripts.types import ArgDesc, ArgType

if TYPE_CHECKING:
    from scripts.analyze_subroutines import Subroutine

from scripts.mod_config import ELM_SRC, _bc
from scripts.types import PointerAlias
from scripts.utilityFunctions import Variable


def match_input_arguments(l_args, sub: Subroutine, special, verbose=False):
    """
    function that matches the args to the corresponding dummy args
    of the subroutine.
    Arguments:
        l_args : list of arguments passed to child subroutine
        sub : subroutine object of candidate child subroutine
        special : special data type used for math expressions
    """
    func_name = "match_input_arguments"

    test_args = [v for v in sub.arguments.values()]

    num_input_args = len(l_args)

    # Get number of optional arguments
    num_optional_args = 0
    for arg in test_args:
        if arg.optional:
            num_optional_args += 1
    if verbose:
        print(f"{sub.name}:: {num_optional_args} Optional arguments found")

    # Keep track of which args can be matched.
    matched = [False] * num_input_args
    matching = True

    # Simple check regarding number of arguments first:
    num_dummy_args = len(test_args)
    if num_input_args > num_dummy_args:
        if verbose:
            print(f"Too many arguments for {sub.name}")
        return matched  # Too many arguments
    if num_input_args < num_dummy_args - num_optional_args:
        if verbose:
            print(f"Not enough arguments for {sub.name}")
            print(
                f"Num input args: {num_input_args} / Num dummy args: {num_dummy_args}"
            )
        return matched  # not enough arguments

    # get list of arg names for keyword comparsions
    test_arg_names = [k for k in sub.arguments.keys()]

    argn = 0  # argument number
    skip = 0  # keep track of skipped optional arguments

    # Go through each input arg and see if it can be matched to a dummy arg
    while matching and argn < num_input_args and argn + skip < num_dummy_args:
        input_arg = l_args[argn]

        if "=" in input_arg.name:
            # Note this is not sufficient for subroutines
            # that are differentiated only by dummy arg type
            # but have same names
            test = input_arg.name
            keyword, varname = test.split("=")
            keyword = keyword.strip()
            varname = varname.strip()
            if keyword in test_arg_names:
                if verbose:
                    print(f"{sub.name} accepts {keyword} as keyword")
                matched[argn] = True
                argn += 1
                continue
            else:
                if verbose:
                    print(f"{sub.name} doesn't support {keyword} as keyword")
                matching = False
                continue

        dummy_arg = test_args[argn]
        # check type and dimension:
        same_type = bool(input_arg.type.strip() == dummy_arg.type.strip())
        if not same_type and input_arg.type == special:
            if dummy_arg.type.strip() in ["real", "integer"]:
                same_type = True

        same_dim = bool(input_arg.dim == dummy_arg.dim)
        if same_type and same_dim:
            # This variable should correspond
            # to this dummy argument
            if verbose:
                print(f"{input_arg.name} matches {dummy_arg.name}")
            matched[argn] = True
            argn += 1  # go to next argument
        else:
            if verbose:
                print(f"{input_arg.name} {input_arg.type} {input_arg.dim}")
            if verbose:
                print(
                    f" does not match \n{dummy_arg.name} {dummy_arg.type} {dummy_arg.dim}"
                )
            # Check to see if dummy_arg is optional
            # If it is optional, then cycle through the
            # rest of the variables (which should all be optional right?)
            # Assuming no keywords are used
            if dummy_arg.optional:
                if verbose:
                    print(
                        f"{dummy_arg.name} argument is optional and did not match -- Skipping"
                    )
                skip += 1
            else:
                # This subroutine is not a match
                matching = False
    # return matched array
    return matched


def determine_arg_name(
    matched_vars: list[str],
    child_sub: Subroutine,
    args: list[str],
    verbose=False,
) -> list[PointerAlias]:
    """
    Function that takes a list of vars passed as arguments to subroutine and
    checks if any correspond to any in 'matched_vars'
        * matched_vars : list of strings for variable names to match
        * child_sub : Subroutine obj
        * args : arguments passed to child_sub

        returns list of PointerAlias(ptr=argname,obj=varname)
            where `argname` is the dummy arg name in the subroutine
            and `varname` is the variable passed to it.
    """
    func_name = "determine_arg_name::"

    var_string = "|".join(matched_vars)
    var_string = f"({var_string})"

    if verbose:
        print(func_name, args)

    arg_vars_list = []
    # Make lists of matched expressions and their location in args.
    matches = [arg for arg in args if re.search(r"\b{}\b".format(var_string), arg)]
    match_locs = [args.index(m) for m in matches]
    for i, locs in enumerate(match_locs):
        matched_arg = matches[i]
        if "=" in matched_arg:
            # Check if keyword:
            keyword, m_var_name = matched_arg.split("=")
            keyword = keyword.strip()
            m_var_name = m_var_name.strip()
            actual_arg = child_sub.arguments[keyword]
        else:
            # if not keyword we need to match by position.
            # Get argument position by converting to list of keys
            arg_key_list = [arg for arg in child_sub.arguments.keys()]
            arg_key = arg_key_list[locs]
            actual_arg = child_sub.arguments[arg_key]
            m_var_name = matched_arg

        arg_to_dtype = PointerAlias(actual_arg.name, m_var_name)
        if verbose:
            print(f"{func_name}{arg_to_dtype}")
        arg_vars_list.append(arg_to_dtype)

    return arg_vars_list


def resolve_interface2(
    iname: str,
    args_desc: list[ArgDesc],
    sub_dict: dict[str, Subroutine],
) -> Optional[str]:
    iprocs = get_interface_procedures(iname)

    for proc in iprocs:
        sub = sub_dict[proc]
        resolved = compare_args(args_desc, sub)
        if resolved:
            return proc

    return None


def compare_args(
    args_desc: list[ArgDesc],
    sub: Subroutine,
) -> bool:

    test_arg_list = list(sub.arguments.values())
    num_args = len(args_desc)
    argn = 0
    keyword = args_desc[0].keyword
    while not keyword:
        test_arg = test_arg_list[argn]
        parg_type = args_desc[argn].argtype
        neqv = bool(parg_type != ArgType(datatype=test_arg.type, dim=test_arg.dim))
        if neqv and not test_arg.optional:
            return False
        argn += 1
        if argn > num_args - 1:
            return True
        keyword = args_desc[argn].keyword

    if keyword:
        for i in range(argn, num_args):
            key_ident = args_desc[i].key_ident
            if key_ident not in sub.arguments:
                return False
            test_arg = sub.arguments[key_ident]
            neqv = bool(
                args_desc[i].argtype
                != ArgType(datatype=test_arg.type, dim=test_arg.dim)
            )
            if neqv:
                return False

    return True


def get_interface_procedures(iname: str, verbose: bool = False) -> list[str]:
    """
    Function that finds the potential procedures of an interface
    """
    if verbose:
        print(_bc.FAIL + f"Resolving interface for {iname}\n with args: {args}")
    cmd = f'grep -rin --exclude-dir={ELM_SRC}external_models/ -E "^[[:space:]]+(interface {iname})" {ELM_SRC}*'
    output = sp.getoutput(cmd)

    # Get file and line number for interface
    # list that goes:  [filename, linenumber, interface {iname}]
    output = output.split(":")
    if len(output) != 3:
        sys.exit(
            f"resolve_interface:: Couldn't find file with interface {iname}\n"
            f"cmd: {cmd}\noutput: {output}"
        )
    fn, ln, _ = output

    ln = int(ln)

    # Read file:
    ifile = open(fn, "r")
    lines = ifile.readlines()
    ifile.close()
    # Get list of possible procedures within the interface

    regex_end = re.compile(r"^\s*(end)\s+(interface)", re.IGNORECASE)
    regex_procedure = re.compile(r"^\s*(module)\s+(procedure)\s+", re.IGNORECASE)

    interface_sub_names = []  # list of subroutine names in the interface
    ct = ln - 1
    in_interface = True
    while in_interface:
        m_end = regex_end.search(lines[ct])
        if m_end:
            in_interface = False
        else:
            m_proc = regex_procedure.search(lines[ct])
            if m_proc:
                subname = lines[ct].replace(m_proc.group(), "").strip().lower()
                interface_sub_names.extend([s.strip() for s in subname.split(",")])
        ct += 1

    return interface_sub_names
