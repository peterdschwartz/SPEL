"""
Python Module that collects functions that
have broad utility for several modules in SPEL
"""

from __future__ import annotations

import copy
import re
import subprocess as sp
import sys
from collections import namedtuple
from pprint import pprint
from typing import TYPE_CHECKING, List, Pattern, Tuple

from scripts.fortran_parser.lexer import Lexer
from scripts.fortran_parser.spel_ast import VariableDecl
from scripts.fortran_parser.spel_parser import Parser
from scripts.types import LineTuple, LogicalLineIterator

if TYPE_CHECKING:
    from scripts.analyze_subroutines import Subroutine

from scripts.config import ELM_SRC, _bc
from scripts.fortran_parser.tracing import Trace

# Regular Expressions
find_type = re.compile(
    r"(?<=\()\s*\w+\s*(?=\))"
)  # Matches type(user-type) -> user-type
# Match any variable declarations
find_variables = re.compile(
    r"^(class\s*\(|type\s*\(|integer|real|logical|character|complex)", re.IGNORECASE
)
# Match variable names
regex_var = re.compile(r"\w+")
# Match subgrid index i.e., bounds%begc -> c
regex_subgrid_index = re.compile(r"(?<=bounds\%beg)[a-z]", re.IGNORECASE)
# Capture instrinsic types to separate from user-defined types
intrinsic_type = re.compile(r"^(integer|real|logical|character)", re.IGNORECASE)
# Capture user-defined types
user_type = re.compile(r"^(class\s*\(|type\s*\()", re.IGNORECASE)
# non-greedy capture for arrays
ng_regex_array = re.compile(r"\w+?\s*\(.+?\)")
# capture array bounds only:
regex_bounds = re.compile(r"(?<=(\w|\s))\(.+\)")

# Named tuple used to store line numbers
# for preprocessed and original files
PreProcTuple = namedtuple("PreProcTuple", ["cpp_ln", "ln"])


class Variable(object):
    """
    Class to hold information on the variable declarations in a subroutine
        * self.type : data type of variable
        * self.name : name of variable
        * self.subgrid : subgrid level used for allocation
        * self.ln : line number of declaration
        * self.dim : dimension of variable
        * self.declaration : module var is declared in
        * self.keyword  : argument alias
        * self.optional : optional arguments
        * self.subs : list of subroutines that use variable (not implemented)
    """

    def __init__(
        self,
        type: str,
        name: str,
        subgrid: str,
        ln: int,
        dim: int,
        parameter: bool = False,
        declaration="",
        optional=False,
        keyword="",
        active=False,
        pointer=[],
        private=False,
        bounds="",
        ptrscalar=False,
        intent="",
        value="",
    ):
        self.type: str = type
        self.name: str = name
        self.subgrid = subgrid
        self.ln: int = ln
        self.dim: int = dim
        self.parameter: bool = parameter
        # These are used for Argument variables
        self.optional: bool = optional
        self.keyword: str = keyword
        self.intent: str = intent
        # filter_used corresponds to adjusting memory allocations
        self.filter_used = ""

        # Module where variable is declared
        if declaration:
            self.declaration: str = declaration
        else:
            self.declaration: str = ""
        self.active: bool = active
        self.private: bool = private
        self.pointer: list[str] = pointer.copy()
        self.bounds: str = bounds
        self.ptrscalar = ptrscalar

        self.allocatable: bool = False
        if self.dim > 0 and not self.bounds:
            self.allocatable = True

        self.default_value = value

    def __eq__(self, other):
        if (
            self.name == other.name
            and self.type == other.type
            and self.dim == other.dim
        ):
            return True
        else:
            return False

    def __str__(self):
        return f"{self.type} {self.name} {self.dim}-D {self.bounds}"

    def __repr__(self):
        return f"Variable({self.type} {self.name} {self.bounds})"

    def copy(self):
        return copy.deepcopy(self)

    def printVariable(self, ofile=sys.stdout):
        ofile.write(f"{self}\n")


def split_func_line(line):
    """
    Function to split function definition line
    """
    func_name = "( split_func_line )"
    regex_func = re.compile(r"\b(function)\b")
    regex_remove = re.compile(r"\b(pure|elemental)\b")

    split_line = regex_func.split(line)
    split_line = [s.strip() for s in split_line]
    split_line[0] = regex_remove.sub("", split_line[0]).strip()

    if len(split_line) != 3 or split_line[1].strip() != "function":
        print(f"{func_name}Split Failed", split_line)
        sys.exit(1)

    return split_line


def removeBounds(line, verbose=False):
    """
    This function to remove any array accesses from
    subroutine calls so that the input arguments to subrtouines
    can be split by commas.

    NOTE: Since there are use cases with nested parenthesis,
    this may be better off using a tokenizer and parser for the
    parenthesis.
    """
    cc = r"[ a-zA-Z0-9%\-\+]*?:[ a-zA-Z0-9%\-\+]*?"
    non_greedy1D = re.compile(r"\({}\)".format(cc))
    non_greedy2D = re.compile(r"\({},{}\)".format(cc, cc))
    non_greedy3D = re.compile(r"\({},{},{}\)".format(cc, cc, cc))
    non_greedy4D = re.compile(r"\({},{},{},{}\)".format(cc, cc, cc, cc))
    regex_array_as_index = re.compile(r"\w+\s*\([,\w+\*-]+\)", re.IGNORECASE)
    # ng_array_ind = re.compile(r'(?<=\w)\s*(\(.+?\))')
    ng_array_ind = re.compile(r"(?<=\w)\s*\(([^()]*|(?:\([^()]*\))*)\)")

    newline = line

    newline = non_greedy1D.sub("", newline)
    newline = non_greedy2D.sub("", newline)
    newline = non_greedy3D.sub("", newline)

    max_count = 20
    loop_count = 0
    m_ = ng_array_ind.search(newline)
    while m_ and loop_count < max_count:
        newline = ng_array_ind.sub("", newline)
        m_ = ng_array_ind.search(newline)
        loop_count += 1

    if loop_count == max_count:
        print(_bc.FAIL + _bc.BOLD + f"MAX LOOP COUNT EXCEEDED for {line}")
        sys.exit(1)

    match_arrays = ng_regex_array.findall(newline)
    match_array_init = re.search(r"(\(/)(.+)(/\))", newline)
    if match_array_init:
        print(_bc.WARNING + "match array init::", match_array_init, _bc.ENDC)
    if match_arrays:
        newline = regex_array_as_index.sub(":", newline)
        # not all bounds could be removed.
        # Check if they are of the form arr(1,2) (no semicolon)
        for arr in match_arrays:
            expr = regex_bounds.search(arr).group()
            newline = newline.replace(expr, "")
    elif match_array_init:
        newline = newline.replace(match_array_init.group(), "")

    match_arrays = ng_regex_array.findall(newline)
    if match_arrays:
        print("removeBounds::Error - Couldn't remove array bounds completely")
        print("Remaining:", match_arrays)
        print("#-------------------------------------------------------------------")
        print(newline.split())
        print("#-------------------------------------------------------------------")
        match_arr_newline = regex_array_as_index.sub(":", line)
        print("subbed line:", match_arr_newline)
        sys.exit(1)

    return newline


@Trace.trace_decorator("getArguments")
def getArguments(full_line, verbose=False) -> List[str]:
    """
    Function that takes a string of the subroutine call
    as an argument and returns the variables
    passed as arguments.

    Will be used to compare with the subroutine's
    argument list.

    This is neccessary to change variable allocations,
    resolve ambiguities from interfaces, and
    track global variables passed as arguments.
    """

    if verbose:
        print(_bc.WARNING + f"getArguments:: Processing {full_line}" + _bc.ENDC)
    # Matches longest string between parentheses
    par = re.compile(r"(?<=\().+(?=\))")
    m = par.search(full_line)
    if not m:
        args = []
        return args
    args = m.group()

    newargs = removeBounds(args, verbose)
    args = newargs.split(",")
    args = [x.strip().lower() for x in args]
    args = [x for x in args if x != ""]

    if verbose:
        print("Args Found:", args, "\n")
    return args


def lineContinuationAdjustment(lines, ln, verbose=False):
    """
    This function returns takes a string that
    accounts for line continuations

    Could be simpler/easier to read without using enumerate
    in calling loop.
    """
    l = lines[ln]
    l = l.split("!")[0]
    l = l.rstrip("\n")
    l = l.strip().lower()

    lines_to_skip = 0
    while l.endswith("&"):
        lines_to_skip += 1
        ct = ln + lines_to_skip
        newline = lines[ct].split("!")[0]
        newline = newline.strip().lower()
        l = l[:-1] + newline.strip().rstrip("\n")

    if verbose and lines_to_skip > 0:
        print("Started with line: ")
        print(lines[ln])
        print("Returning line:")
        print(l)
        print(f"Skipped {lines_to_skip} lines")

    return l, lines_to_skip


def unwrap_section(lines: list[str], startln: int) -> list[LineTuple]:
    """
    lines: list of fortran lines to adjust for lineconinuation
    startln: line number for first line in lines in the file.
    """
    fline_list: list[LineTuple] = []
    line_it = LogicalLineIterator(
        lines=[LineTuple(line=line, ln=i) for i, line in enumerate(lines)]
    )
    for full_line, new_ln in line_it:
        ln = line_it.get_start_ln()
        if ln > startln and full_line:
            for stmt in full_line.split(';'):
                fline_list.append(LineTuple(line=stmt, ln=ln))
        ln = new_ln + 1

    return fline_list


def find_end_subroutine(fn, startline):
    """
    Function that will find next "end subroutine" statement starting
    from the subroutine starting line 'startline'.

    This may be needed in case the 'end subroutine <subroutine_name>'
    convention is not used.
    """
    func_name = "find_end_subroutine"
    ifile = open(fn, "r")
    lines = ifile.readlines()
    ifile.close()
    regex_end_subroutine = re.compile(r"\s*(end)\s+(subroutine)", re.IGNORECASE)
    for ln in range(startline, len(lines)):
        line = lines[ln]
        # get rid of comments
        line = line.split("!")[0]
        line = line.strip()
        if not line:
            continue
        match = regex_end_subroutine.search(line)
        if match:
            endline = ln
            break

    return endline


def find_file_for_subroutine(name, fn="", ignore_interface=False, verbose=False):
    """
    finds file, start and end line numbers for subroutines
    find file and start of interface block for interfaces
    """
    func_name = "find_file_for_subroutine"
    if not fn:
        search_file = f"{ELM_SRC}*"
    else:
        search_file = f"{fn}"

    interface_list = get_interface_list()
    if name not in interface_list or ignore_interface:
        cmd = f'grep -rin -E "^[[:space:]]*(subroutine[[:space:]]* {name})\\b" {search_file} | head -1'
        cmd_end = f'grep -rin -E "^[[:space:]]*(end subroutine[[:space:]]* {name})\\b" {search_file} | head -1'
    else:
        cmd = f'grep -rin -E "^[[:space:]]+(interface[[:space:]]* {name})" {search_file} | head -1'
        cmd_end = ""

    output = sp.getoutput(cmd)
    if verbose:
        print(f"{func_name}::output {output}")
    if not fn:
        file = output.split(":")[0]
        # Need to separate the file name and path
        # fpath = dir_regex.search(file).group()
        # file = dir_regex.sub('',file)
        output = output.split(":")
        if len(output) < 2:
            print(f"Error: Couldn't find info for {name}\n {output}")
            sys.exit()
        startline = int(output[1])
        if cmd_end != "":
            output = sp.getoutput(cmd_end)
            if not output:
                print(f"{func_name}::Didn't match end of subroutine\n output: {output}")
                endline = find_end_subroutine(file, startline)
                print(f"{func_name}::Endline found: {endline} for {name}")
            else:
                endline = int(output.split(":")[1])
        else:
            endline = 0
    else:
        file = fn
        output = output.split(":")
        if len(output) < 2:
            sys.exit(
                f"{func_name}::Didn't match subroutine\n cmd: {cmd}\n output: {output}"
            )
        startline = int(output[0])
        if cmd_end != "":
            output = sp.getoutput(cmd_end)
            if not output:
                print(f"{func_name}::Didn't match end of subroutine\n output: {output}")
                endline = find_end_subroutine(file, startline)
                print(f"{func_name}::Endline found: {endline} for {name}")
            else:
                output = output.split(":")
                endline = int(output[0])
        else:
            endline = 0

    if file == "grep":
        print(f"ERROR FILE FOR {name} NOT PRESENT")
        sys.exit(1)

    return file, startline, endline


def get_interface_list():
    """
    returns a list of all interfaces

    NOTE:  Should store it in config so it's only run once per
           Unit Test creation
    """

    cmd = f'grep -rin --exclude-dir={ELM_SRC}external_models/ -E "^[[:space:]]+(interface)" {ELM_SRC}*'
    output = sp.getoutput(cmd)
    output = output.split("\n")
    interface_list = []
    for el in output:
        el = el.split()
        interface = el[2]
        interface_list.append(interface.lower())

    return interface_list


def get_local_variables(sub: Subroutine):
    lines = sub.sub_lines[1:]
    local_var_lines = [
        lpair for lpair in filter(lambda x: find_variables.search(x.line), lines)
    ]
    local_variables = parse_variable_decl(local_var_lines, mod_name=sub.name)

    for var in local_variables:
        var.declaration = sub.module
        if var.name in sub.dummy_args_list:
            sub.arguments[var.name] = var
        else:
            sub.local_variables[var.name] = var


def determine_class_instance(sub, verbose=False):
    """
    Find out what the data structure 'this' corresponds to
    """

    func_name = "determine_class_instance"
    filename = sub.filepath

    subname = sub.name
    # Open file and  loop through subroutine lines
    file = open(filename, "r")
    lines = file.readlines()
    file.close()
    startline = sub.startline
    endline = sub.endline

    find_this = re.compile(r"^(type)", re.IGNORECASE)
    find_type = re.compile(r"(?<=\()\w+(?=\))")
    print(f"{func_name}::Determining variable for {subname}")
    found_this = False
    for ln in range(startline, endline):
        line = lines[ln].split("!")[0]
        line = line.strip()
        line = line.strip("\n")
        if not (line):
            continue

        m_this = find_this.search(line.lower())
        m_type = find_type.search(line.lower())

        if m_this and m_type:
            # the derived type should always be 1st
            var_type = m_type.group()
            found_this = True
            if verbose:
                print(f"{func_name}:: 'this' -> {var_type}")
            break

    if not found_this:
        print(
            f"Error: Couldn't find declaration for class variable in {sub.name} {sub.filepath}"
        )
        sys.exit()
    else:
        return var_type


def convertAssociateDict(associate_vars, varlist):
    """
    Need to eliminate this function and properly match arguments
    """
    dtypes = []
    replace_inst = []
    for vars in associate_vars.values():
        for v in vars:
            _type, field = v.split("%")
            if _type in replace_inst:
                _type = _type.replace("_inst", "vars")
            if _type not in dtypes:
                dtypes.append(_type)

    return


def adjust_array_access_and_allocation(local_arrs, sub, dargs=False, verbose=False):
    """
    Function edits ELM FORTRAN files to reduce memory.
    Replaces statements like "arr(bounds%begc:bounds%endc)" -> "arr(1:num_filterc)"
    and accesses like "arr(c)" -> "arr(fc)"

    * local_arrs    : list of local array Variables
    * sub           : Subroutine that calls this function and declares the local_arrs
    * sub.VariablesPassedToSub : dictionary that matches "arr" to subroutines that take them as arguments
    """
    import os.path
    import re

    from config import _bc, elm_files, spel_dir

    # Get lines of this file:
    ifile = open(elm_files + sub.filepath, "r")
    lines = ifile.readlines()  # read entire file
    ifile.close()

    track_changes = []
    arg_list = [v for v in sub.Arguments]
    scalar_list = [v.name for v in sub.LocalVariables["scalars"].values()]

    print(_bc.BOLD + _bc.FAIL + f"arguments for {sub.name} are", arg_list, _bc.ENDC)
    print(_bc.BOLD + _bc.FAIL + f"scalars for {sub.name} are", scalar_list, _bc.ENDC)

    # replace declarations first
    if not dargs:
        for arr in local_arrs:
            var = arr
            filter_used = arr.filter_used
            ln = var.ln
            # line number of declaration
            subgrid = var.subgrid  # subgrid index

            print(_bc.BOLD + _bc.WARNING + f"Adjusting {var.name}" + _bc.ENDC)
            filter_used = filter_used + subgrid

            # Check that the corresponding num_filter exists
            num_filter = "num_" + filter_used.replace("filter_", "")
            if num_filter in arg_list:
                print(num_filter, "is passed as an argument!")
            elif num_filter in scalar_list:
                print(f"{num_filter} is new local filter")
            else:
                sys.exit(f"utilityFunctions:: {num_filter} doesn't exit")
            lold = lines[ln]
            print(_bc.FAIL + lold.strip("\n") + _bc.ENDC)
            _str = f"bounds%beg{subgrid}:bounds%end{subgrid}"

            replace_str = f"1:{num_filter}"
            lnew = lines[ln].replace(_str, replace_str)
            print(_bc.OKGREEN + lnew.strip("\n") + _bc.ENDC)
            lines[ln] = lnew
            track_changes.append(lnew)

    # Go through all loops and make replacements for filter index
    ng_regex_array = re.compile(r"\w+\s*\([,\w+\*-]+\)", re.IGNORECASE)
    regex_var = re.compile(r"\w+")
    regex_indices = re.compile(r"(?<=\()(.+)(?=\))")
    print("Going through loops")
    # Make list for quick check if var should be adjusted.
    list_of_var_names = [v.name for v in local_arrs]

    for loop in sub.DoLoops:
        lstart = loop.start[0]
        lend = loop.end[0]
        if loop.subcall.name != sub.name:
            continue

        for n in range(lstart, lend):
            l = lines[n].split("!")[0]
            l = l.strip()
            if not l:
                continue
            m_arr = ng_regex_array.findall(lines[n])
            lold = lines[n]
            lnew = lines[n]

            replaced = False
            temp_line = lold

            removing = True
            while removing:
                # set removing to be False unless there is a match
                removing = False

                for arr in m_arr:
                    v = regex_var.search(arr).group()

                    # min and max functions are special cases since they take two argumnets (fix?)
                    if v in ["min", "max"]:
                        temp_line = temp_line.replace(arr, v)
                        removing = True
                        continue

                    # Check if var is
                    if v in list_of_var_names:
                        loc_ = list_of_var_names.index(v)
                        local_var = local_arrs[loc_]
                        var = local_var
                        subgrid = var.subgrid
                        filter_used = var.filter_used + var.subgrid

                        # Consistency check for filter.
                        # TODO: allow scripts to insert reverse filter (eg., "fc = col_to_filter(c)" )
                        loop_filter = loop.filter[0]
                        same_filter_type = bool(filter_used[:-1] == loop_filter[:-1])
                        if filter_used != loop_filter and not same_filter_type:
                            print(
                                f"Filter Mismatch: loops uses {loop.filter[0]}, {var.name} needs {filter_used}"
                            )
                            sys.exit()
                        elif same_filter_type and filter_used != loop_filter:
                            print(
                                _bc.WARNING
                                + _bc.BOLD
                                + f"{var.name} needs reverse filter!"
                            )

                        # Make replacement in line: (assumes subgrid is first index!!)
                        # lnew = lnew.replace(f"{v}({subgrid}",f"{v}(f{subgrid}")
                        lnew = re.sub(
                            r"{}\s*\({}".format(v, subgrid),
                            r"{}(f{}".format(v, subgrid),
                            lnew,
                        )
                        replaced = True

                    regex_check_index = re.compile(
                        r"\w+\([,a-z0-9*-]*(({})\(.\))[,a-z+0-9-]*\)".format(v),
                        re.IGNORECASE,
                    )
                    match = regex_check_index.search(temp_line)
                    if match:  # array {v} is being used as an index
                        # substitute from the entire line
                        i = regex_indices.search(arr).group()
                        i = r"\({}\)".format(i)
                        temp_line = re.sub(r"{}{}".format(v, i), v, temp_line)
                        removing = True
                m_arr = ng_regex_array.findall(temp_line)

            if replaced and verbose:
                print(_bc.FAIL + lold.strip("\n") + _bc.ENDC)
                print(_bc.OKGREEN + lnew.strip("\n") + _bc.ENDC)
                print("\n")
                lines[n] = lnew
                track_changes.append(lnew)

    # Check if subroutine calls need to be adjusted
    # First filter out sub arguments that aren't local
    # variables only accessed by a filter
    # Adjust Subroutine calls

    # Note that these subroutines have specific versions for
    # using filter or not using a filter.
    dont_adjust = ["c2g", "p2c", "p2g", "p2c", "c2l", "l2g"]
    dont_adjust_string = "|".join(dont_adjust)
    regex_skip_string = re.compile(f"({dont_adjust_string})", re.IGNORECASE)

    vars_to_check = {s: [] for s in sub.VariablesPassedToSubs}
    for subname, arg in sub.VariablesPassedToSubs.items():
        m_skip = regex_skip_string.search(subname)
        if m_skip:
            print(_bc.FAIL + f"{subname} must be manually altered !" + _bc.ENDC)
            continue
        for v in arg:
            if v.name in list_of_var_names:
                print(f"Need to check {v.name} for mem adjustment called in {subname}.")
                vars_to_check[subname].append(v)

    for sname, vars in vars_to_check.items():
        for v in vars:
            print(f"{sname} :: {v.name}")

    for subname, args in sub.VariablesPassedToSubs.items():
        m_skip = regex_skip_string.search(subname)
        if m_skip:
            print(_bc.FAIL + f"{subname} must be manually altered !" + _bc.ENDC)
            continue
        regex_subcall = re.compile(r"\s+(call)\s+({})".format(subname), re.IGNORECASE)

        for arg in args:
            # If index fails, then there is an inconsistency
            # in examineLoops
            if arg.name not in list_of_var_names:
                continue
            loc_ = list_of_var_names.index(arg.name)
            local_var = local_arrs[loc_]
            filter_used = local_var.filter_used + arg.subgrid
            # bounds string:
            bounds_string = r"(\s*bounds%beg{}\s*:\s*bounds%end{}\s*|\s*beg{}\s*:\s*end{}\s*)".format(
                arg.subgrid, arg.subgrid, arg.subgrid, arg.subgrid
            )
            # string to replace bounds with
            num_filter = "num_" + filter_used.replace("filter_", "")
            num_filter = f"1:{num_filter}"
            regex_array_arg = re.compile(r"{}\s*\({}".format(arg.name, bounds_string))

            for ln in range(sub.startline, sub.endline):
                line = lines[ln]
                match_call = regex_subcall.search(line)
                if match_call:
                    replaced = False
                    # create regex to match variables needed
                    l = line[:]
                    m_var = regex_array_arg.search(l)
                    if m_var:
                        lold = lines[ln].rstrip("\n")
                        lnew = regex_array_arg.sub(
                            f"{arg.name}({num_filter}", lines[ln]
                        )
                        lines[ln] = lnew
                        replaced = True
                        track_changes.append(lnew)

                    while l.rstrip("\n").strip().endswith("&") and not replaced:
                        ln += 1
                        l = lines[ln]
                        m_var = regex_array_arg.search(l)
                        if m_var:
                            lold = lines[ln].rstrip("\n")
                            lnew = regex_array_arg.sub(
                                f"{arg.name}({num_filter}", lines[ln]
                            )
                            lines[ln] = lnew
                            replaced = True
                            track_changes.append(lnew)

                    if replaced:
                        print(_bc.FAIL + lold.rstrip("\n") + _bc.ENDC)
                        print(_bc.OKGREEN + lnew.rstrip("\n") + _bc.ENDC)
                        print("\n")
                        break
                    else:
                        print(
                            _bc.FAIL
                            + f"Couldn't replace {arg.name} in {subname} subroutine call"
                            + _bc.ENDC
                        )
                        sys.exit()

    # Save changes:
    if track_changes:
        if not os.path.exists(spel_dir + f"modified-files/{sub.filepath}"):
            print(
                _bc.BOLD + _bc.WARNING + "Writing to file ",
                spel_dir + f"modified-files/{sub.filepath}" + _bc.ENDC,
            )
            ofile = open(spel_dir + f"modified-files/{sub.filepath}", "w")
            ofile.writelines(lines)
            ofile.close()
        else:
            print(
                _bc.BOLD + _bc.WARNING + "Writing to file ",
                elm_files + sub.filepath + _bc.ENDC,
            )
            ofile = open(elm_files + sub.filepath, "w")
            ofile.writelines(lines)
            ofile.close()

    for subname, args in sub.VariablesPassedToSubs.items():
        # Modify dummy arguments for child subs if needed
        # may be redundant to find file here?
        #
        m_skip = regex_skip_string.search(subname)
        if m_skip:
            print(_bc.FAIL + f"{subname} must be manually altered !" + _bc.ENDC)
            continue
        print(_bc.WARNING + f"Modifying dummy args for {subname}")
        file, startline, endline = find_file_for_subroutine(subname)
        childsub = sub.child_Subroutine[subname]
        adjust_child_sub_arguments(childsub, file, startline, endline, args)


def adjust_child_sub_arguments(sub, file, lstart, lend, args):
    """
    Function that checks and modifies bounds accesses
    for subroutine arguments
        * childsub : Subroutine instance for child subroutine
        * arg : Variable instance for dummy arg name
    """
    import os

    from config import _bc, spel_dir

    if os.path.exists(spel_dir + f"modified-files/{file}"):
        print(file, "has already been modified")
        file = f"../modified-files/{file}"

    print(f"Opening {sub.filepath}")
    ifile = open(sub.filepath, "r")
    lines = ifile.readlines()
    ifile.close()

    for arg in args:
        # Use keyword instead of arg name.
        # If keyword is missing then we need to determine the name!
        kw = arg.keyword
        if not kw:
            sys.exit(f"Error keyword for {arg.name} in {sub.name} is missing")

        regex_arg_type = re.compile(f"{arg.type}", re.IGNORECASE)
        regex_arg_name = re.compile(r"{}\s*\(".format(kw), re.IGNORECASE)
        regex_bounds_full = re.compile(
            r"({}\s*\()(bounds%beg{})\s*(:)\s*(bounds%end{})\s*(\))".format(
                kw, arg.subgrid, arg.subgrid
            ),
            re.IGNORECASE,
        )
        regex_bounds = re.compile(
            r"({}\s*\(\s*bounds%beg{}[\s:]+\))".format(kw, arg.subgrid), re.IGNORECASE
        )
        for ct in range(lstart - 1, lend):
            #
            line = lines[ct]
            lold = line
            line = line.split("!")[0]
            line = line.strip()

            replaced = False
            # Match type and name for arg
            match_type = regex_arg_type.search(line)
            match_name = regex_arg_name.search(line)

            if match_type and match_name:
                match_bounds = regex_bounds.search(line)
                match_full_bounds = regex_bounds_full.search(line)
                if match_bounds and not match_full_bounds:
                    lnew = regex_bounds.sub(f"{kw}(1:)", lold)
                    lines[ct] = lnew
                    print(_bc.BOLD + _bc.FAIL + lold.rstrip("\n") + _bc.ENDC)
                    print(_bc.BOLD + _bc.OKGREEN + lnew.rstrip("\n") + _bc.ENDC)
                    replaced = True
                else:
                    print(f"{line} -- No Action Needed")
                    sys.exit()

    # Write to file
    with open(sub.filepath, "w") as ofile:
        ofile.writelines(lines)

    # re-run access adjustment for the dummy args only!
    sub.filepath = file
    print(f"Adjust dargs for {sub.name} at {sub.filepath}")
    dargs = []
    for arg in args:
        v = arg
        v.name = v.keyword
        v.keyword = ""

    adjust_array_access_and_allocation(
        local_arrs=args, sub=sub, verbose=True, dargs=True
    )
    sys.exit("Why is this here?")


def determine_filter_access(sub, verbose=False):
    """
    Function that will go through all the loops for
    local variables that are bounds accessed.
    """

    print(_bc.BOLD + _bc.WARNING + f"Adjusting Allocation for {sub.name}" + _bc.ENDC)

    # TODO:
    # Insert function call to check if any local
    # variables are passed as subroutine arguments
    if verbose:
        print("Checking if variables can be allocated by filter")
    ok_to_replace = {}

    # List that accumulates all info necessary for memory adjustment
    local_vars = []  # [Variable]
    array_dict = sub.LocalVariables["arrays"]
    for vname, lcl_var in array_dict.items():
        if lcl_var.subgrid == "?":
            continue
        lcl_arr = lcl_var.name
        ok_to_replace[lcl_arr] = False
        filter_used = ""
        indx, ln = lcl_var.subgrid, lcl_var.ln
        for loop in sub.DoLoops:
            if loop.subcall.name != sub.name:
                continue
            if lcl_arr in loop.vars or lcl_arr in loop.reduce_vars:
                if not loop.filter:  # Loop doesn't use a filter?
                    filter_used = "None"
                    continue
                fvar, fidx, newidx = loop.filter
                # Case where no filter is used
                if not fvar:
                    if indx not in loop.index:
                        print(
                            f"{lcl_arr} {indx} used in {loop.subcall.name}:L{loop.start[0]} {fvar},{fidx}{loop.index}"
                        )
                if not filter_used:
                    # Get rid of the subgrid info
                    filter_used = fvar[:-1]
                elif filter_used != fvar[:-1] and filter_used != "Mixed":
                    print(
                        _bc.BOLD
                        + _bc.HEADER
                        + f"{lcl_arr} {indx} used in {loop.subcall.name}:L{loop.start[0]} {fvar},{fidx}{loop.index}"
                        + _bc.ENDC
                    )
                    filter_used = "Mixed"
                    break
        #
        if filter_used == "None":
            print(f"No filter being used -- won't adjust {lcl_arr}")
        elif filter_used == "Mixed":
            print(
                _bc.WARNING
                + f"{lcl_arr} has multiple filters being used -- won't adjust"
                + _bc.ENDC
            )
            ok_to_replace[lcl_arr] = False
        elif filter_used and filter_used != "Mixed":
            print(f"{lcl_arr} only uses {filter_used}{indx}")
            ok_to_replace[lcl_arr] = True
            lcl_var.filter_used = filter_used
            local_vars.append(lcl_var)
        else:
            print(_bc.WARNING + f"{lcl_arr} is not used by any loops!!" + _bc.ENDC)

    if local_vars:
        list_of_var_names = [v.name for v in local_vars]
        print(
            _bc.BOLD
            + _bc.HEADER
            + f"Adjusting {sub.name} vars:{list_of_var_names}"
            + _bc.ENDC
        )
        adjust_array_access_and_allocation(local_vars, sub=sub, verbose=True)
    else:
        print(
            _bc.BOLD
            + _bc.HEADER
            + f"No variables need to be adjusted for {sub.name}"
            + _bc.ENDC
        )


def line_unwrapper(lines: list[str], ct: int, verbose: bool = False) -> Tuple[str, int]:
    """
    Function that takes code segment that has line continuations
    and returns it all on one line.
    """
    beg_continuation = re.compile(r"^\s*(&)")
    simple_l = lines[ct].split("!")[0]  # remove comments
    # remove new line character
    simple_l = simple_l.rstrip("\n").strip()
    continuation = bool(simple_l.endswith("&"))
    full_line = simple_l.lower()
    newct = ct
    while continuation:
        newct += 1
        simple_l = lines[newct].split("!")[0].strip()  # in case of inline comments
        match_beg_cont = beg_continuation.search(simple_l)
        if match_beg_cont:
            simple_l = beg_continuation.sub("", simple_l)
            simple_l = simple_l.strip()
        simple_l = simple_l.rstrip("\n")
        # Fortran allow empty lines in between line continuations
        if simple_l.isspace() or not simple_l:
            continue
        full_line = full_line[:-1] + simple_l.strip().lower()
        continuation = bool(full_line.endswith("&"))

    return full_line, newct


def parse_variable_decl(lines: list[LineTuple], mod_name: str) -> list[Variable]:
    """
    Function
    """
    fn = "(parse_variable_decl)"
    variables = []
    if not lines:
        return variables
    line_it = LogicalLineIterator(lines=lines, log_name="parse_variable_decl")
    lexer = Lexer(line_it)
    parser = Parser(lexer, logger=mod_name)
    program = parser.parse_program()
    for stmt in program.statements:
        assert isinstance(stmt, VariableDecl), "Not variable decl"
        variables.extend(create_var_from_decl(stmt))
    for v in variables:
        v.declaration = mod_name
    return variables


def create_var_from_decl(stmt: VariableDecl) -> list[Variable]:
    """
    Function that traverses an VariableDecl AST Node and returns and
    list of Variable instances
    """
    variables: list[Variable] = []
    data_type = stmt.var_type.token_literal()
    attr_spec = stmt.get_attr_list()
    parameter = "parameter" in attr_spec
    private = "private" in attr_spec
    pointer = "pointer" in attr_spec
    dim_attr = "dimension" in attr_spec
    optional = "optional" in attr_spec
    intent = next((att for att in attr_spec if att.startswith("intent")), "")

    for ent in stmt.entities:
        bounds = ent.bounds
        bounds_str = ""
        init: str = str(ent.init)
        dim = 0
        subgrid = "?"
        if bounds:
            dim = len(bounds)
            bounds_str = ent.get_bounds_str()
            m_subgrid = regex_subgrid_index.search(bounds_str)
            if m_subgrid:
                subgrid = m_subgrid.group()
        elif dim_attr:
            dim = stmt.attrs.get_dim()
        variables.append(
            Variable(
                type=data_type,
                name=ent.token_literal(),
                subgrid=subgrid,
                ln=stmt.lineno,
                dim=dim,
                parameter=parameter,
                private=private,
                ptrscalar=pointer,
                value=init,
                optional=optional,
                bounds=bounds_str,
                intent=intent,
            )
        )
    return variables


def parse_line_for_variables(ifile, l, ln, ptr=False, verbose=False):
    """
    Function that takes a line of code and returns the variables
    that are declared in it.
    """
    # NOTE: This code is duplicated from getLocalVariables
    #       which should be refactored?
    func_name = "parse_line_for_variables"
    variable_list = []
    match_var = find_variables.search(l)
    if match_var:
        if "::" not in l:
            # regex to separate type from variable when there is no '::' separator
            ng_type = re.compile(r"^\w+?\s*\(.+?\)")
            match_type = ng_type.search(l)
            if match_type:
                # Means there exists a character(len=) or type(.) or real(r8)
                temp_decl = match_type.group()
                temp_vars = l.split(temp_decl)[1]
            else:  # means it's a simple decl such as 'integer'
                temp_decl = find_variables.search(l)
                temp_vars = find_variables.sub("", l).strip()
        else:
            temp = l.split("::")  # Is this a poor assumption?
            if len(temp) < 2:
                print(
                    f"{func_name}:: Error - Thought there was a variable decl at \n {ifile}::L{ln+1}\n{l} "
                )
                sys.exit(1)
            temp_decl = l.split("::")[0]
            temp_vars = l.split("::")[1]

        # Get data type of variable by checking for intrinsic types
        # and then user-defined types separately
        m_type = intrinsic_type.search(temp_decl)
        if m_type:
            data_type = m_type.group()
        else:  # User-Defined Type
            m_type = user_type.search(temp_decl)
            data_type = find_type.search(temp_decl).group()

        # Go through and replace all arrays first
        #
        # NOTE: It may be beneficial to store the initialized value
        # for any arrays, for now they are deleted and
        # only the variable name, dimension are stored
        regex_array_init = re.compile(r"=\s*(\(\/)(.)+?(\/\))")
        temp_vars = regex_array_init.sub("", temp_vars)

        # call function to remove any array bounds so that only
        # variable names are comma separated.
        remove_slices = removeBounds(temp_vars)

        # create list of only variable names
        var_list = remove_slices.split(",")
        var_list = [x.strip() for x in var_list if x.strip()]

        for var in var_list:
            # NOTE: similar to note above for arrays, this skips over
            #       initialization for other variables. May change if
            #       desired use case wants initial values.
            ptrscalar = False
            default_value: str = ""
            if "=" in var:
                default_value = var.split("=")[1].strip()
                var = var.split("=")[0].strip()
            # Check if current variable is an array
            ng_var_array = re.compile(rf"{var}?\s*\(.+?\)", re.IGNORECASE)
            match_array = ng_var_array.search(temp_vars)
            if match_array:
                arr = match_array.group()
                index = regex_subgrid_index.search(var)
                if index:
                    subgrid = index.group()
                else:
                    subgrid = "?"
                # Storing line number of declaration
                dim = arr.count(",") + 1
            else:
                subgrid = ""
                dim = 0
                if ptr:
                    ptrscalar = True
            parameter = bool("parameter" in temp_decl.lower())
            private = bool("private" in temp_decl.lower())

            variable_list.append(
                Variable(
                    data_type,
                    var,
                    subgrid,
                    ln,
                    dim,
                    parameter=parameter,
                    private=private,
                    ptrscalar=ptrscalar,
                    value=default_value,
                )
            )

    return variable_list


def check_cpp_line(base_fn, og_lines, cpp_lines, cpp_ln, og_ln, verbose=False):
    """Function to check if the compiler preprocessor (cpp) line
    is a cpp comment and adjust the line numbers accordingly.
    returns the adjusted line numbers
    """
    func_name = "check_cpp_line::"
    # regex to match if the proprocessor comments refer to mod_file
    regex_file = re.compile(r"({})".format(base_fn))
    # regex matches a preprocessor comment
    regex_cpp_comment = re.compile(r"(^# [0-9]+)")
    # Store current lines to check
    cpp_line = cpp_lines[cpp_ln]
    line = og_lines[og_ln]
    # line = line.split("!")[0].strip()

    # (NOTE: this is unnecessary and overly specific to elm use case?)
    regex_include_assert = re.compile(r"^(#include)\s+[\"\'](shr_assert.h)[\'\"]")
    if regex_include_assert.search(line):
        if verbose:
            print(f"{func_name}Found include statement to comment out:\n{line}")
            print(f"cpp line: {cpp_line}")
        newline = line.replace("#include", "!#py #include")
        og_lines[og_ln] = newline
        # include statements take up multiple lines:
        match_file = regex_file.search(cpp_line)
        while not match_file:
            cpp_ln += 1
            cpp_line = cpp_lines[cpp_ln]
            match_file = regex_file.search(cpp_line)

    # Check if the line is a preprocessor comment
    m_cpp = regex_cpp_comment.search(cpp_line)
    while m_cpp:
        # Check if the comment refers to the mod_file
        match_file = regex_file.search(cpp_line)
        if match_file:
            # Get the line for the original file, adjusted for 0-based indexing
            # NOTE: if not match_file, then the comment is for an include statement
            #       which will be handled in the main part of the code?
            og_ln = int(m_cpp.group().split()[1]) - 1
            if verbose:
                print(f"{func_name}:: Found CPP comment, new line number: {og_ln}")

        # Since it was a comment, go to the next line
        cpp_ln += 1
        m_cpp = regex_cpp_comment.search(cpp_lines[cpp_ln])

    return cpp_ln, og_ln, og_lines


def search_in_file_section(
    fpath: str,
    start_ln: int,
    end_ln: int,
    pattern: Pattern,
):
    """
    Function that iterates through the lines in a file using filter builtin.
    """
    with open(fpath, "r") as file:
        lines = enumerate(file, start=0)
        section = filter(lambda x: start_ln <= x[0] <= end_ln, lines)
        matches = [line[1] for line in filter(lambda x: pattern.search(x[1]), section)]

    return matches
