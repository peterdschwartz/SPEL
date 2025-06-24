"""
Module that holds the Loop Class
that is used to parse subroutines
"""

from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from scripts.fortran_parser.spel_ast import (DoLoop, ExpressionStatement,
                                             FuncExpression, InfixExpression)
from scripts.fortran_parser.spel_parser import Parser
from scripts.types import LineTuple, LogicalLineIterator, ReadWrite

if TYPE_CHECKING:
    from scripts.analyze_subroutines import Subroutine

from scripts.fortran_parser.lexer import Lexer
from scripts.logging_configs import get_logger
from scripts.mod_config import _bc
from scripts.utilityFunctions import Variable, lineContinuationAdjustment

regex_do_start = re.compile(r"^\s*(\w+\s*:)?\s*do\b", re.IGNORECASE)
regex_do_while = re.compile(r"do\s+while\b")
regex_do_end = re.compile(r"^\s*(end\s*do(\s+\w+)?)", re.IGNORECASE)

ReadWriteDict = dict[str, list[ReadWrite]]

Kind = Enum("Kind", ["doloop", "dowhile"])


@dataclass
class LoopStart:
    ln: int
    kind: Kind


def is_within_any_loop(line_no: int, loops: list[Loop]) -> bool:
    """
    Return True if `line_no` falls within any loop's start–end range
    """
    for loop in loops:
        if loop.start < line_no < loop.end:
            return True
    return False


def in_loop(line_no: int, loop: Loop) -> bool:
    return loop.start < line_no < loop.end


def get_loops(sub: Subroutine):
    """
    Function
    """
    lines = sub.sub_lines

    loops: list[Loop] = []
    line_it = LogicalLineIterator(lines, "LoopIter")
    nested: int = 0
    start_lns: list[LoopStart] = []
    end_lns: list[int] = []
    for full_line, _ in line_it:
        start_index = line_it.start_index
        orig_ln = line_it.lines[start_index].ln
        m_start = regex_do_start.search(full_line)
        m_dowhile = regex_do_while.search(full_line)
        if m_start:
            kind = Kind.dowhile if m_dowhile else Kind.doloop
            start_lns.append(LoopStart(ln=orig_ln, kind=kind))
            do_line = full_line
            nested += 1
        m_enddo = regex_do_end.search(full_line)
        if m_enddo:
            nested -= 1
            end_lns.append(orig_ln)

            end_ln = end_lns.pop()
            start_ln = start_lns.pop()
            loops.append(Loop(do_line, start_ln.ln, end_ln, start_ln.kind, sub, nested))

    input: list[str] = [l.line for l in loops]
    input_str = "\n".join(input)

    lex = Lexer(input=input_str)
    parser = Parser(lex=lex)
    program = parser.parse_program()
    for i, stmt in enumerate(program.statements):
        loop = loops[i]
        loop.node = stmt
        loop.index = loop.node.index

    aliases = check_filter_aliases(sub, loops)
    sub.logger.info(f"ALIASES: {aliases}")
    for loop in loops:
        if loop.kind == Kind.doloop:
            loop.get_loop_vars(aliases)

    sub.loops = loops
    return


def check_filter_aliases(sub: Subroutine, loops: list[Loop]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    regex_filter = re.compile(r"filter")
    filter_args = {
        var: rw for var, rw in sub.arg_access_by_ln.items() if regex_filter.match(var)
    }

    lines_to_parse: list[str] = []

    for f in filter_args.values():
        lines = [x.ltuple.line for x in f if not is_within_any_loop(x.ltuple.ln, loops)]
        lines_to_parse.extend(lines)

    input_str = "\n".join(lines_to_parse)
    lexer = Lexer(input_str)
    parser = Parser(lexer)
    program = parser.parse_program()

    for stmt in program.statements:
        assert isinstance(stmt, ExpressionStatement)
        expr = stmt.expression
        assert isinstance(expr, InfixExpression)
        left, op, right = expr.decompose()
        if op != "=":
            continue
        assert isinstance(left, FuncExpression)
        # we are looking for statements of the form:  ftemp(:) = filter_x(:)
        if not isinstance(right, FuncExpression):
            continue
        alias = str(left.function)
        filter_ = str(right.function)
        aliases[alias] = filter_

    return aliases


def print_tree(loops, indent=0):
    for loop in loops:
        parent = (
            f" ← {loop.outer_loop.start}-{loop.outer_loop.end}L"
            if loop.outer_loop
            else ""
        )
        print("  " * indent + f"{repr(loop)}{parent}")
        print_tree(loop.inner_loops, indent + 1)


def build_loop_hierarchy(loops: list[Loop]) -> list[Loop]:
    """ """
    loops = sorted(loops, key=lambda x: (x.nested, x.start))

    root_loops = []
    by_level = {0: []}  # nesting level → list of loops

    for loop in loops:
        level = loop.nested
        if level == 0:
            root_loops.append(loop)
            by_level.setdefault(0, []).append(loop)
        else:
            # Find most recent enclosing loop at level - 1
            for candidate in reversed(by_level.get(level - 1, [])):
                if candidate.start < loop.start and candidate.end > loop.end:
                    candidate.inner_loops.append(loop)
                    loop.outer_loop = candidate
                    break
            by_level.setdefault(level, []).append(loop)

    return root_loops


def accesses_in_range(
    lstart: int, lend: int, access_dict: ReadWriteDict
) -> ReadWriteDict:
    return {
        var: [rw for rw in accesses if lstart < rw.ln < lend]
        for var, accesses in access_dict.items()
        if any(lstart < rw.ln < lend for rw in accesses)
    }


class Loop(object):
    """
    Represents a loop construct in a program.

    Attributes:
        start : The starting line number of the loop.
        end : The ending line number of the loop.
        index : The loop indices.
        nested (int): The number of times the loop is nested.
        innerloops (list): The inner loops contained within this loop.
        vars (dict): A dictionary that holds the array variables modified by this loop.
        reduction (bool): Indicates if the loop contains reduction operations.
        reduce_vars (list): The variables involved in reduction operations.
        scalar_vars (dict): A dictionary that holds the scalar variables used in the loop.
        filter : A list of filters applied to the loop -- Currently ELM specific.

    Methods:
        printLoop(substartline=0, long=True): Prints information about the loop.
        removeArraysAsIndices(vdict, line, arrays, verbose): Removes arrays used as indices and sets them as read-only.
        parseVariablesinLoop(verbose=False): Parses the variables modified by the loop.
        addOpenACCFlags(lines_adjusted, subline_adjust, id, verbose=False): Adds OpenACC directives to the loop.
    """

    def __init__(
        self,
        line: str,
        start: int,
        end: int,
        kind: Kind,
        sub: Subroutine,
        nested: int,
    ):
        self.line: str = line
        self.start: int = start
        self.end: int = end
        self.kind: Kind = kind
        self.index: str = ""
        self.nested: int = nested
        self.inner_loops: list[Loop] = []
        self.outer_loop: Optional[Loop] = None
        self.node: DoLoop = None
        self.vars: dict[str, Variable] = {}
        self.reduction: bool = False
        self.reduce_vars: list[Variable] = []
        self.sub: Subroutine = sub
        self.local_vars: ReadWriteDict = {}
        self.global_vars: ReadWriteDict = {}
        self.arg_vars: ReadWriteDict = {}
        self.filter: list[str] = []
        self.index_map: dict[str, str] = {}
        self.index_remaps: dict[str, str] = {}
        self.new_indices: set[str] = set()
        self.new_mappings: set[str] = set()

    def __str__(self):
        if self.kind == Kind.doloop:
            return f"Loop({self.start}-{self.end}L nested={self.nested})"
        else:
            return f"While({self.start}-{self.end}L nested={self.nested})"

    def __repr__(self):
        if self.kind == Kind.doloop:
            return f"Loop({self.start}-{self.end}L nested={self.nested})"
        else:
            return f"While({self.start}-{self.end}L nested={self.nested})"

    def get_loop_vars(self, aliases: dict[str, str]):
        """
        Function that looks at variable usage of a looop and attempts
        to identify a filter
        """

        sub = self.sub
        logger = sub.logger
        self.local_vars = accesses_in_range(
            self.start,
            self.end,
            sub.local_vars_access_by_ln,
        )

        arg_vars = accesses_in_range(self.start, self.end, sub.arg_access_by_ln)
        self.local_vars.update(arg_vars)

        maybe_filters = ["filter"]
        maybe_filters.extend(aliases.keys())
        str_ = "|".join(maybe_filters)
        regex_filter = re.compile(rf"({str_})")
        self.filter = [x for x in self.local_vars if regex_filter.match(x)]

        filter_accesses: list[ReadWrite] = []
        for pfilter in self.filter:
            access = self.local_vars[pfilter]
            for rw in access:
                filter_accesses.append(rw)
        inputs = [x.ltuple.line for x in filter_accesses]
        input_str = "\n".join(inputs)
        self.infer_filter_index_map(input_str, regex_filter)

        self.filter = [aliases.get(f, f) for f in self.filter]

        return

    def infer_filter_index_map(self, input: str, pattern: re.Pattern):
        """
        Function to find global index mapped to loop index.
        """

        logger = self.sub.logger
        lexer = Lexer(input)
        parser = Parser(lexer)
        program = parser.parse_program()
        for stmt in program.statements:
            assert isinstance(stmt, ExpressionStatement)
            expr = stmt.expression
            # Only interested in assignments
            if not isinstance(expr, InfixExpression):
                continue
            left, op, right = expr.decompose()
            if op != "=":
                continue
            if isinstance(right, FuncExpression) and pattern.search(
                str(right.function)
            ):
                lhs_var = left.value
                rhs_array = str(right.function)
                arg = right.args[0]
                if len(right.args) > 1:
                    continue
                if arg.value == self.index:
                    self.index_map[lhs_var] = rhs_array
        return

    def parseVariablesinLoop(self, verbose=False):
        """
        Goes through loop line by line and
        returns the self.vars dictionary that
        holds the variables modified by this Loop
        """

        # non-greedy capture
        # ng_regex_array = re.compile(f'\w+?\({cc}+?\)')
        ng_regex_array = re.compile(r"\w+\s*\([,\w+\*-]+\)", re.IGNORECASE)

        regex_if = re.compile(r"^(if|else if|elseif)")
        regex_cond = re.compile(r"\((.+)\)")
        regex_subcall = re.compile(r"^(call)")
        #
        # regex to match code that should be ignored
        regex_skip = re.compile(r"^(write|print)")
        regex_dowhile = re.compile(r"\s*(do while)", re.IGNORECASE)

        # regex for scalar variables:
        # since SPEL already has the loop indices, no need to hardcode this?
        indices = [
            "i",
            "j",
            "k",
            "g",
            "l",
            "t",
            "c",
            "p",
            "fc",
            "fp",
            "fl",
            "ci",
            "pi",
            "n",
            "m",
            "s",
        ]
        if self.subcall.LocalVariables["scalars"]:
            list_of_scalars = [
                vname
                for vname in self.subcall.LocalVariables["scalars"].keys()
                if vname not in indices
            ]
            # print(_bc.OKBLUE+f"list of scalars for {self.subcall.name}\n {list_of_scalars}"+_bc.ENDC)
            str_ = "|".join(list_of_scalars)
            regex_scalars = re.compile(rf"(?<!\w)({str_})(?!\w)", re.IGNORECASE)
        else:
            list_of_scalars = []

        # Initialize dictionary that will
        # hold array variables used in the loop
        variable_dict = {}

        slice = self.lines[:]
        lines_to_skip = 0
        reprint = True
        for ln, line in enumerate(slice):
            if lines_to_skip > 0:
                lines_to_skip -= 1
                continue
            l, lines_to_skip = lineContinuationAdjustment(slice, ln, verbose)

            # match any functions or statements to be ignored
            match_skip = regex_skip.search(l)
            if match_skip:
                continue

            # match if statements
            match_if = regex_if.search(l)
            match_dowhile = regex_dowhile.search(l)
            if match_if:
                m = regex_cond.findall(l)
                if "then" not in l:
                    if verbose:
                        print("single line if statement")
                else:
                    if verbose:
                        print(ln, m)
            elif not match_dowhile:

                # Currently ignore subroutines called inside loops
                match_subcall = regex_subcall.search(l)
                if match_subcall:
                    continue

                # Find all array variables in the line
                m_arr = ng_regex_array.findall(l)
                if m_arr:
                    lnew = l
                    removing = True
                    while removing:
                        # temp, removing = self.removeArraysAsIndices(
                        #     vdict=variable_dict,
                        #     line=lnew,
                        #     arrays=m_arr,
                        #     verbose=verbose,
                        # )
                        lnew = temp
                        m_arr = ng_regex_array.findall(lnew)
                        if verbose:
                            print("New findall is: ", m_arr)

                    if m_arr:
                        variable_dict, reprint = self._getArrayVariables(
                            ln,
                            l,
                            m_arr,
                            variable_dict,
                            reprint=reprint,
                            verbose=verbose,
                        )

                # Find all local scalar variables
                if list_of_scalars:
                    m_scalars = regex_scalars.findall(l)
                    if m_scalars:
                        self._get_scalars(
                            ln, l, m_scalars, variable_dict, verbose=verbose
                        )

        if self.reduction:
            print(
                _bc.WARNING
                + "This Loop may contain a race-condition for the variables \n",
                f"{self.reduce_vars}" + _bc.ENDC,
            )

        # clarify read/write status of each variable in the loop
        for var, status in variable_dict.items():
            # remove duplicates
            status = list(set(status))
            # set read-only, write-only, rw status:
            if "r" in status and "w" not in status:
                variable_dict[var] = "ro"  # read only
            elif "w" in status and "r" not in status:
                variable_dict[var] = "wo"  # write only
            elif "w" in status and "r" in status:
                variable_dict[var] = "rw"  # read-write
        self.vars = variable_dict.copy()
        return

    def addOpenACCFlags(self, lines_adjusted, subline_adjust, id, verbose=False):
        """
        Function that will add openACC directives to loop

        lines_adjusted is a dictionary to keep track of line insertions into a subroutine
        to appropriately adjust where the loops start.
        """
        from mod_config import _bc

        total_loops = self.nested + 1
        ifile = open(f"{self.file}", "r")
        mod_lines = ifile.readlines()
        ifile.close()

        outer_lstart = (
            self.start[0] + subline_adjust + lines_adjusted[self.subcall.name]
        )
        # First check if OpenACC flags have already been added.
        if "!$acc" in mod_lines[outer_lstart - 2]:
            print("Loop already has OpenACC flags, Skipping")
            return
        if self.reduction:
            print(
                _bc.WARNING
                + f"Reduction in {self.file}::{id} for:\n {self.reduce_vars}"
                + _bc.ENDC
            )
            return

        tightly_nested = 1
        for loop in range(0, total_loops):
            # Loop through every loop that isn't this one and test to
            # see if they are nested.
            # Increment counter for the collapse clause
            if tightly_nested > 1:
                tightly_nested -= 1
                continue
            acc_parallel_directive = f" "
            for innerloops in range(loop, total_loops):
                diff = self.start[innerloops] - self.start[loop]
                if diff == tightly_nested:
                    tightly_nested += 1

            if loop == 0:
                # only put data clause on outermost loop
                acc_parallel_directive = (
                    "!$acc parallel loop independent gang vector default(present)"
                )
            else:
                # for inner loops just default to sequential for now
                # and allow the developer to increase parallelism based on profiling results
                acc_parallel_directive = "!$acc loop seq"
            if tightly_nested > 1:
                acc_parallel_directive = (
                    acc_parallel_directive + f" collapse({tightly_nested}) "
                )
            acc_parallel_directive = acc_parallel_directive + "\n"
            lstart = (
                self.start[loop]
                - 1
                + lines_adjusted[self.subcall.name]
                + subline_adjust
            )
            line = mod_lines[lstart]
            padding = " " * (len(line) - len(line.lstrip()))
            acc_parallel_directive = padding + acc_parallel_directive
            print(f"Inserting :\n {acc_parallel_directive}")
            mod_lines.insert(lstart, acc_parallel_directive)
            lines_adjusted[self.subcall.name] += 1

        lend = self.end[0] + lines_adjusted[self.subcall.name] + subline_adjust
        for ln in range(outer_lstart - 1, lend):
            print(ln, mod_lines[ln].strip("\n"))
        overwrite = input(f"Overwrite {self.file}?")
        if overwrite == "y":
            with open(f"{elm_files}{self.file}", "w") as ofile:
                ofile.writelines(mod_lines)
        else:
            sys.exit()

    def _getArrayVariables(
        self, ln, l, m_arr, variable_dict, reprint=True, verbose=False, interact=False
    ):
        """
        This function takes a given line in of a Loop
        and identifies the read/write status of each array
        variable present
        """
        # split the about the assignment
        if "=" not in l:
            print(f"Line does not contain an assignment -- Bug in code or regex?")
            print(l)
            sys.exit()

        assignment = l.split("=")

        if len(assignment) > 2:
            print(l)
            sys.exit("getArrayVariables::Too many equals in this case!")
        lhs = assignment[0]
        rhs = assignment[1]
        #
        regex_indices = re.compile(r"(?<=\()(.+)(?=\))")
        regex_var = re.compile(r"\w+")

        vars_already_examined = []
        for var in m_arr:
            # This means only the first instance of the variable is catalogued as
            # write, read or reduction.
            # get subgrid index from filter
            varname = regex_var.search(var).group()
            if varname in ["min", "max"]:
                continue
            if varname.lower() in vars_already_examined:
                continue
            vars_already_examined.append(varname.lower())

            # matches only exactly "varname"
            regex_varname = re.compile(rf"(?<!\w)({varname})(?!\w)", re.IGNORECASE)

            if "filter" in var:
                if "filter" in rhs:
                    subgrid_indx = lhs.strip()
                    fvar = regex_var.search(rhs).group()
                    filter_indx = regex_indices.search(rhs).group()
                    self.filter = [fvar, subgrid_indx, filter_indx]
                else:
                    # filter is being assigned -- no need to consider
                    subgrid_indx = rhs.strip()
                    fvar = regex_var.search(lhs).group()
                    filter_indx = regex_indices.search(lhs).group()
                    self.filter = [fvar, subgrid_indx, filter_indx]

            in_lhs = regex_varname.search(lhs)
            in_rhs = regex_varname.search(rhs)
            if varname == "tx":
                print(f"in LHS:", bool(in_lhs))
                print(f"in RHS: {bool(in_rhs)}\n Line: {l}")
            # check if var is only on assigned
            if in_lhs and not in_rhs:
                variable_dict.setdefault(varname, []).append("w")

            elif in_rhs and not in_lhs:
                variable_dict.setdefault(varname, []).append("r")

            # Now check if variable appears on both sides
            # Requires additional checking for race conditions/reduction operation
            elif in_lhs and in_rhs:
                m_indices = regex_indices.search(var).group()
                indices = m_indices.split(",")

                # variable appears on both sides and is missing at least one loop index
                # Need to ask if the variable is being reduced
                if verbose:
                    print(indices, self.index)

                # Since inner loops may not be tightly nested, we need to
                # find out how many loops this line of code is inside.
                loopcount = 0
                for n, loopstart in enumerate(self.start):
                    loopend = self.end[n]
                    linecount = ln + self.start[0]
                    if linecount > loopstart and linecount < loopend:
                        loopcount += 1
                if verbose:
                    print(f"This line {ln} is inside {loopcount} Loops!")

                if loopcount < 1:
                    for n, loopstart in enumerate(self.start):
                        print(loopstart, self.end[n])
                    sys.exit("Error: Not in a Loop!?")

                if len(indices) < loopcount:
                    if reprint and interact:
                        self.printLoop()
                        reprint = False
                    if var in self.reduce_vars:
                        continue
                    if interact:
                        reduction = input(
                            f"is {var} at line {ln} being reduced in this loop?"
                        )
                    else:
                        reduction = "y"
                    if reduction == "y":
                        self.reduction = True
                        self.reduce_vars.append(var)
                elif len(indices) == loopcount:
                    # Check if it's a lower-level subgrid to a higher level
                    loopindex = False
                    for i in self.index:
                        if indices[0] in i:
                            loopindex = True

                    if not loopindex:
                        print(f"reduction occuring between subgrids", l)
                        self.reduction = True
                        self.reduce_vars.append(var)
                    else:
                        # Check that both indices are the same
                        # catch scenarios like (c,j) = (c,j-1)
                        # where the order matters
                        same_indices = True
                        print(f"checking for same indices for :", l)
                        # using varname here didn't work for some reason
                        ng_regex_array = re.compile(
                            r"\w+\s*\([,\w+\*-]+\)", re.IGNORECASE
                        )

                        # Get lhs indices:
                        lhs_var = ng_regex_array.search(lhs).group()
                        lhs_indx = regex_indices.search(lhs_var).group()
                        lhs_indx = [m.replace(" ", "").lower() for m in lhs_indx]
                        # Get indices for all rhs instances:
                        rhs_vars = ng_regex_array.findall(rhs)
                        for rv in rhs_vars:
                            rvname = regex_var.search(rv).group()
                            if rvname != varname:
                                continue

                            rhs_indx = regex_indices.search(rv).group()
                            if len(rhs_indx) != len(lhs_indx):
                                same_indices = False
                                break
                            # compare the lists. Does this need regex for robustness?
                            rhs_indx = [m.replace(" ", "").lower() for m in rhs_indx]
                            if rhs_indx != lhs_indx:
                                same_indices = False
                                break
                        # May be an order dependent calculation. Flag as reduction
                        if not same_indices:
                            print(f"Order dependent calculation", l)
                            self.reduction = True
                            self.reduce_vars.append(var)

                # variable_dict.setdefault(varname,[]).append('rw')
                variable_dict.setdefault(varname, []).extend(["r", "w"])

        return variable_dict, reprint

    def _get_scalars(self, ln, l, m_scalars, variable_dict, verbose=True):
        """
        function that takes the local scalars matched in m_scalars
        and determines the write/read status and reduction status
        """
        ltemp = l.split("=")
        if len(ltemp) < 2:
            print(f"Error no assignment at {l}")
            sys.exit()
        lhs = l.split("=")[0]
        rhs = l.split("=")[1]

        m_scalars = list(
            dict.fromkeys(m_scalars).keys()
        )  # remove duplicate but keep order
        for lcl_var in m_scalars:
            regex = re.compile(rf"(?<![\w]){lcl_var}(?![\w])")
            m_lhs = regex.search(lhs)
            m_rhs = regex.search(rhs)
            if m_lhs and not m_rhs:  # only assigned
                variable_dict.setdefault(lcl_var, []).append("w")
            elif m_rhs and not m_lhs:
                variable_dict.setdefault(lcl_var, []).append("r")
            elif m_lhs and m_rhs:
                self.reduction = True
                self.reduce_vars.append(lcl_var)
                variable_dict.setdefault(lcl_var, []).extend(["r", "w"])
                print(f"{lcl_var} needs reduction operation in this Loop")
        return
