from __future__ import annotations

import re
import sys
from copy import deepcopy
from pprint import pformat
from typing import TYPE_CHECKING, Optional

from spel.scripts.fortran_parser.sections import parse_blocks
from spel.scripts.fortran_parser.spel_ast import Expression, IfConstruct, InfixExpression

if TYPE_CHECKING:
    from spel.scripts.analyze_subroutines import Subroutine

from spel.scripts.fortran_parser.tokens import Token, TokenTypes
from spel.scripts.types import FlatIfs, IfType


def AND(a, b):
    if a is None:
        return b
    # clone if your nodes are mutable; drop deepcopy if they’re immutable
    return InfixExpression(
        tok=Token(TokenTypes.AND, ".and."),
        left=deepcopy(a),
        op=".and.",
        right=deepcopy(b),
    )


def flatten_if(
    if_node: IfConstruct,
    context_guard: Optional[Expression] = None,
) -> list[FlatIfs]:
    """
    Takes an if node turns it into a flattened list of conditions and start/end line numbers
    """

    flat_blocks: list[FlatIfs] = []

    guards, else_guard = if_node.build_branch_guards()
    g_if = AND(context_guard, guards[0])
    # First, the main IF
    start_ln = if_node.lineno
    end_ln = if_node.end_ln
    flat_blocks.append(FlatIfs(start=start_ln, end=end_ln, cond=g_if, kind=IfType.IF))

    for stmt in if_node.consequence.statements:
        if isinstance(stmt, IfConstruct):
            flat_blocks.extend(flatten_if(stmt, g_if))

    # ELSEIFs (use the guarded conditions)
    for idx, elif_node in enumerate(if_node.else_ifs, start=1):
        g_elif = AND(context_guard, guards[idx])
        flat_blocks.append(
            FlatIfs(start=elif_node.lineno, end=elif_node.end_ln, cond=g_elif, kind=IfType.ELSEIF)
        )
        for stmt in elif_node.consequence.statements:
            if isinstance(stmt, IfConstruct):
                flat_blocks.extend(flatten_if(stmt, g_elif))
    # ELSE
    if if_node.else_ and else_guard is not None:
        g_else = AND(context_guard, else_guard)
        flat_blocks.append(
            FlatIfs(start=if_node.else_.lineno, end=if_node.else_.end_ln, cond=g_else, kind=IfType.ELSE)
        )
        for stmt in if_node.else_.alternative.statements:
            if isinstance(stmt, IfConstruct):
                flat_blocks.extend(flatten_if(stmt, g_else))

    return flat_blocks


def get_if_blocks(sub: Subroutine):
    """
    Collects and groups the if-blocks (if, else if, else) within a Fortran subroutine
    Parameters:
    sub (Subroutine): The subroutine object containing the lines of code.
    """
    lines = sub.sub_lines

    debug_sub = "xxxx"
    verbose = True if sub.name == debug_sub else False

    regex_if_start = re.compile(r"^\s*if\s*\((.*?)\)\s*(then)?")
    regex_if_end = re.compile(r"^\s*end\s*if")
    regex_check_block = re.compile(r"^\s*if\s*\((.*?)\)\s*(then)")

    if_statements = parse_blocks(
        lines,
        regex_if_start,
        regex_if_end,
        regex_check=regex_check_block,
        verbose=verbose,
        tag=sub.name,
    )
    if if_statements:
        sub.if_blocks = if_statements
        flat_ifs: list[FlatIfs] = []
        for ifnode in if_statements:
            assert isinstance(ifnode, IfConstruct)
            flat_ifs.extend(flatten_if(ifnode))
        sub.flat_ifs = flat_ifs
    sub.ifs_analyzed = True
    return
