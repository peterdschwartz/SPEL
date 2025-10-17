"""
Module that holds the Loop Class
that is used to parse subroutines
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from scripts.fortran_parser.spel_ast import (
    DoLoop,
    ExpressionStatement,
    FuncExpression,
    InfixExpression,
)
from scripts.fortran_parser.spel_parser import Parser
from scripts.types import LineTuple, LogicalLineIterator, ReadWrite

if TYPE_CHECKING:
    from scripts.analyze_subroutines import Subroutine

from scripts.config import _bc
from scripts.fortran_parser.lexer import Lexer
from scripts.utilityFunctions import Variable

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


def get_do_blocks(sub: Subroutine):
    """
    Collects and groups the if-blocks (if, else if, else) within a Fortran subroutine into chains.

    Parameters:
    sub (Subroutine): The subroutine object containing the lines of code.

    """
    lines = sub.sub_lines
    loops: list[Loop] = []
    line_it = LogicalLineIterator(lines, "DoIter")

    for fline in line_it:
        full_line = fline.line
        start_index = line_it.start_index
        orig_ln = line_it.lines[start_index].ln

        # Check for possible if constructs
        m_start = regex_do_start.search(full_line)

        if m_start:
            if m_start.group(2):
                _, cur_idx = line_it.consume_until(regex_do_end, regex_do_end)
                block = [lpair for lpair in line_it.lines[start_index : cur_idx + 1]]
                line_it = LogicalLineIterator(lines=block, log_name="get_do_loops")
                lexer = Lexer(line_it)
                parser = Parser(lexer)
                program = parser.parse_program()

                block = [
                    f"{lpair.ln}  {lpair.line}"
                    for lpair in line_it.lines[start_index : cur_idx + 1]
                ]
                block_txt = "\n".join(block)
                sub.logger.info(f"\n{block_txt}")
                for stmt in program.statements:
                    print(stmt)

    sub.loops = loops
    return loops


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
    for fline in line_it:
        full_line = fline.line
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
        inputs = [x.ltuple for x in filter_accesses]
        self.infer_filter_index_map(inputs, regex_filter)

        self.filter = [aliases.get(f, f) for f in self.filter]

        return

    def infer_filter_index_map(self, input: list[LineTuple], pattern: re.Pattern):
        """
        Function to find global index mapped to loop index.
        """

        logger = self.sub.logger
        line_it = LogicalLineIterator(lines=input)
        lexer = Lexer(line_it)
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

    def addOpenACCFlags(self, lines_adjusted, subline_adjust, id, verbose=False):
        """
        Function that will add openACC directives to loop

        lines_adjusted is a dictionary to keep track of line insertions into a subroutine
        to appropriately adjust where the loops start.
        """
        from config import _bc

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
