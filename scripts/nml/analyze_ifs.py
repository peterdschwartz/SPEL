from __future__ import annotations

import re
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

from scripts.fortran_parser.sections import parse_blocks
from scripts.fortran_parser.spel_ast import (Expression, IfConstruct,
                                             InfixExpression)

if TYPE_CHECKING:
    from scripts.analyze_subroutines import Subroutine

from scripts.fortran_parser.tokens import Token, TokenTypes
from scripts.types import FlatIfs, IfType


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


def flatten_if(if_node: IfConstruct, context_guard: Optional[Expression]=None) -> list[FlatIfs]:
    """
    Takes an if node turns it into a flattened list of conditions and start/end line numbers
    """

    flat_blocks: list[FlatIfs] = []

    guards, else_guard = if_node.build_branch_guards()
    g_if = AND(context_guard, guards[0])
    # First, the main IF
    start_ln = if_node.lineno
    end_ln = if_node.end_ln
    flat_blocks.append(FlatIfs(start_ln, end_ln, g_if, IfType.IF))

    for stmt in if_node.consequence.statements:
        if isinstance(stmt, IfConstruct):
            flat_blocks.extend(flatten_if(stmt, g_if))

    # ELSEIFs (use the guarded conditions)
    for idx, elif_node in enumerate(if_node.else_ifs, start=1):
        g_elif = AND(context_guard, guards[idx])
        flat_blocks.append(
            FlatIfs(elif_node.lineno, elif_node.end_ln, g_elif, IfType.ELSEIF)
        )
        for stmt in elif_node.consequence.statements:
            if isinstance(stmt, IfConstruct):
                flat_blocks.extend(flatten_if(stmt, g_elif))
    # ELSE
    if if_node.else_ and else_guard is not None:
        g_else = AND(context_guard, else_guard)
        flat_blocks.append(
            FlatIfs(if_node.else_.lineno, if_node.else_.end_ln, g_else, IfType.ELSE)
        )
        for stmt in if_node.else_.alternative.statements:
            if isinstance(stmt, IfConstruct):
                flat_blocks.extend(flatten_if(stmt, g_else))

    return flat_blocks


class IfBlock:
    """
    Class to hold infomation on if_blocks:
    * self.start : starting line number of particular block
    * self.end : absolute ending line number of particular block
    * self.relative_end : relative ending line number to entire block
        *  if (1) then       <- start of 1
        *     if (2) then    <- start of 2
        *         code
        *     else if (3)    <- start of 3, relative_end of 2
        *         code
        *     end if         <- end of 2 and 3, relative_end of 3
        *  else (4)          <- start of 4, relative_end of 1
        *     code
        *  end if            <- end of 1 and 4, relative_end of 4
    * self.condition : clause of if_block
    * self.default : True/False of condition based on default namelist values
                     Defaults to True if condition could not be evaluated
    * self.parent : pointer to parent if_block
    * self.children : list of nested if_blocks that are not within else_if/else blocks
    * self.elseif : list of else_if blocks within current block
    * self.elses : pointer to else_block
    * self.depth : layer of nested-ness
    * self.calls : list if subroutine calls in current if_block
    * self.kind : 1 for if, 2 for else_if, 3 for else
    * self.assigned_as_child : bookkeeping info
    * self.child_conditions : bookkeeping info
    """

    def __init__(
        self,
        start: int,
        condition: str,
        sub: Subroutine,
        kind: IfType,
    ):
        self.start: int = start
        self.end: int = -1
        self.relative_end: int = 0
        self.condition: str = condition
        self.default: bool = False
        self.sub = sub
        self.parent: Optional[IfBlock] = None
        self.children = []
        self.elseif = []
        self.elses: Optional[IfBlock] = None
        self.depth: int = 0
        self.calls = []
        self.kind: IfType = kind
        self.assigned_as_child: bool = False
        self.child_conditions = set()

    def add_child(self, child: IfBlock):
        """Add a child to this block."""
        self.children.append(child)

    def print_if_structure(self, indent=0):
        idx = "->|" * indent
        print(
            f"{idx}start: {self.start}, end: {self.end}, depth: {self.depth}, relative: {self.relative_end}, type: {self.kind}"
        )
        if self.condition:
            print(f"{idx}condition: {self.condition}")
        if self.default != -1:
            print(f"{idx}default: {self.default}")
        if self.calls:
            print(f"{idx}calls: {self.calls}")
        if self.elseif:
            print(f"{idx}else if:")
        for elf in self.elseif:
            print_if_structure(elf, indent + 1)
        if self.elses:
            print(f"{idx}else:")
        print_if_structure(self.elses, indent + 1)
        if self.parent:
            print(f"{idx}parent: {self.parent.start}")

        if self.children:
            print(f"{idx}children: ")
        for child in self.children:
            print_if_structure(child, indent + 2)

    def evaluate(self, namelist_dict):
        """
        Evaluate the if-condition
        """
        string = self.condition
        res = ""
        string = re.sub(r"(\.\w*\.)", r" \1 ", string)
        isString = bool(re.search(r"'", string))
        string = string.strip(" ()")
        res = string.split()
        left = True
        for w in range(len(res)):
            left = False
            res[w] = res[w].strip()
            if res[w] in namelist_dict:
                n = namelist_dict[res[w]]
                if n.variable:
                    res[w] = f"{n.variable.default_value}"
                    left = True

            if "'" not in res[w]:
                res[w] = f"{operands(res[w])}"

            """ 
            bit sketchy
            testing if res[w] is a "true" string, 
            like a variable name and not bools/keywords
            """

        res = " ".join(res)
        p = None
        try:
            p = eval(res)
        except:
            p = f"Error: {res}"
            return True
        return p


def get_if_condition(string):
    """
    Returns the clause in if_condition
    """
    condition = re.findall(r"if\s*\(\s*(.*)\s*\)", string)
    return condition[0] if condition else ""


def operands(op):
    """
    Mapping of Fortran operators to Python
    """
    op = op.strip()
    match op:
        case ".true.":
            return "True"
        case ".false.":
            return "False"
        case "+":
            return "+"
        case "-":
            return "-"
        case "*":
            return "*"
        case "/":
            return "//"
        case "**":
            return "**"
        case "==" | ".eq." | ".eqv.":
            return "=="
        case "/=" | ".ne." | ".neqv":
            return "!="
        case ">" | ".gt.":
            return ">"
        case "<" | ".lt.":
            return "<"
        case ">=" | ".ge.":
            return ">="
        case "<=" | ".le.":
            return "<="
        case ".and.":
            return "and"
        case ".or.":
            return "or"
        case ".not.":
            return "not"
        case _:
            return f"'{op}'"


def set_default_helper(node, namelist_dict, stack):
    if not node:
        return

    match (node.kind):
        case IfType.IF:  # `if` block
            node.default = node.evaluate(namelist_dict)
            if node.default:
                stack.append(1)
                for child in node.children:
                    set_default_helper(child, namelist_dict, stack)
                stack.pop()
            else:
                stack.append(-1)
                for elseif in node.elseif:
                    set_default_helper(elseif, namelist_dict, stack)

                    if stack and stack[-1] > 0:
                        return
                if node.elses and stack[-1] == -1:
                    set_default_helper(node.elses, namelist_dict, stack)
                    stack.pop()

        case IfType.ELSEIF:
            node.default = node.evaluate(namelist_dict)
            stack.append(2)
            if node.default:
                for child in node.children:
                    set_default_helper(child, namelist_dict, stack)
            else:
                stack.pop()

        case IfType.ELSE:
            node.default = True
            for child in node.children:
                set_default_helper(child, namelist_dict, stack)


def set_default(ifs, namelist_dict):
    """
    set default T/F of based on the default values
    of namelist_variables
    """
    flat = flatten(ifs)
    for i in flat:
        i.default = False
    for node in ifs:
        set_default_helper(node, namelist_dict, [])


def flatten_helper(node, depth, visited, res):
    if node:
        if node.start not in visited:
            res.append(node)
            visited.append(node.start)

        for i in node.elseif:
            flatten_helper(i, depth, visited, res)
        for i in node.children:
            flatten_helper(i, depth + 1, visited, res)

        if node.elses:
            flatten_helper(node.elses, depth, visited, res)

    return res


def flatten(blocks):
    """
    Returns all if/else_if/else blocks
    """
    total = []
    for block in blocks:
        total.extend(flatten_helper(node=block, depth=0, visited=[], res=[]))
    flattened_list = sorted(total, key=lambda x: x.start)
    return flattened_list


def print_if_structure(if_block, indent=0):
    if if_block:
        idx = "->|" * indent
        print(
            f"{idx}start: {if_block.start}, end: {if_block.end}, depth: {if_block.depth}, relative: {if_block.relative_end}, type: {if_block.kind}"
        )
        if if_block.condition:
            print(f"{idx}condition: {if_block.condition}")
        if if_block.default != -1:
            print(f"{idx}default: {if_block.default}")
        if if_block.calls:
            print(f"{idx}calls: {if_block.calls}")
        if if_block.elseif:
            print(f"{idx}else if:")
        for elf in if_block.elseif:
            print_if_structure(elf, indent + 1)
        if if_block.elses:
            print(f"{idx}else:")
        print_if_structure(if_block.elses, indent + 1)
        if if_block.parent:
            print(f"{idx}parent: {if_block.parent.start}")

        if if_block.children:
            print(f"{idx}children: ")
        for child in if_block.children:
            print_if_structure(child, indent + 2)


def get_if_blocks(sub: Subroutine):
    """
    Collects and groups the if-blocks (if, else if, else) within a Fortran subroutine
    Parameters:
    sub (Subroutine): The subroutine object containing the lines of code.
    """
    lines = sub.sub_lines

    debug_sub = 'xxxx'
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
            flat_ifs.extend(flatten_if(ifnode))
        sub.flat_ifs = flat_ifs
    sub.ifs_analyzed = True
    return


def set_parent_helper(node, parent):
    if node:
        node.parent = parent

        for child in node.children:
            set_parent_helper(child, node)
        for elseif in node.elseif:
            set_parent_helper(elseif, node)
        if node.elses:
            set_parent_helper(node.elses, node)


def set_parent(if_block_list):
    for node in if_block_list:
        set_parent_helper(node, None)


def run(filename):
    f = open(filename, "r")
    r = f.readlines()
    f.close()

    blocks = find_if_blocks(r)
    set_parent(blocks)
    flat = flatten(blocks)
    for i in flat:
        i.default = False

    for i in blocks:
        x = [i.end]

        if i.elses:
            x.append(i.elses.start)
        if i.elseif:
            x.append(i.elseif[0].start)
        i.relative_end = min(x)
    return blocks
