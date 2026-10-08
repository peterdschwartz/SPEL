from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from spel.scripts.fortran_modules import FortranModule

if TYPE_CHECKING:
    from spel.scripts.analyze_subroutines import Subroutine

import spel.scripts.dynamic_globals as dg
from spel.scripts.DerivedType import DerivedType
from spel.scripts.fortran_parser.evaluate import parse_subroutine_call
from spel.scripts.types import CallTag, CallTree, CallTuple, ReadWrite  # noqa: F401


def determine_level_in_tree(branch, tree_to_write):
    """
    Will be called recursively
    branch is a list containing names of subroutines
    ordered by level in call_tree
    """
    for j in range(0, len(branch)):
        sub_el = branch[j]
        islist = bool(type(sub_el) is list)
        if not islist:
            if j + 1 == len(branch):
                tree_to_write.append([sub_el, j - 1])
            elif type(branch[j + 1]) is list:
                tree_to_write.append([sub_el, j - 1])
        if islist:
            tree_to_write = determine_level_in_tree(sub_el, tree_to_write)
    return tree_to_write


def add_acc_routine_info(sub):
    """
    This function will add the !$acc routine directive to subroutine
    """
    filename = sub.filepath

    file = open(filename, "r")
    lines = file.readlines()  # read entire file
    file.close()

    first_use = 0
    ct = sub.startline
    while ct < sub.endline:
        line = lines[ct]
        l = line.split("!")[0]
        if not l.strip():
            ct += 1
            continue
            # line is just a commment

        if first_use == 0:
            m = re.search(r"[\s]+(use)", line)
            if m:
                first_use = ct

            match_implicit_none = re.search(r"[\s]+(implicit none)", line)
            if match_implicit_none:
                first_use = ct
            match_type = re.search(r"[\s]+(type|real|integer|logical|character)", line)
            if match_type:
                first_use = ct

        ct += 1
    print(f"first_use = {first_use}")
    lines.insert(first_use, "      !$acc routine seq\n")
    print(f"Added !$acc to {sub.name} in {filename}")
    with open(filename, "w") as ofile:
        ofile.writelines(lines)

def combine_many_statuses(statuses: list[str]) -> str:
    """
    Combine read/write statuses in program order.

    Rules:
    - If the first access is a pure write ('w'), overall status is 'w'
      regardless of later reads.
    - Otherwise, overall status is the union of all accesses.
    """

    if not statuses:
        return ""

    # First access determines input-ness
    first = statuses[0]

    if first == "w":
        return "w"

    # Otherwise, fall back to union logic
    perms = set()
    for s in statuses:
        perms |= set(s)

    return "".join(sorted(perms))

def find_child_subroutines(
    sub: Subroutine,
    sub_dict: dict[str, Subroutine],
    type_dict: dict[str, DerivedType],
) -> None:
    """
    """
    lines = sub.replace_associate_in_lines()

    regex_call = re.compile(r"^\s*(call)\b")
    matches = [line for line in filter(lambda x: regex_call.search(x.line), lines)]

    for call_line in matches:
        call_desc = parse_subroutine_call(
            sub=sub,
            sub_dict=sub_dict,
            input=call_line,
            ilist=dg.interface_list,
            type_dict=type_dict,
        )
        if call_desc:
            call_desc.aggregate_vars(sub)
            sub.sub_call_desc[call_desc.lpair.ln] = call_desc

    return


def construct_call_tree(
    sub: Subroutine,
    sub_dict: dict[str, Subroutine],
    dtype_dict: dict[str, DerivedType],
    mod_dict: dict[str, FortranModule],
    nested: int,
    failures: Optional[dict[str, str]] = None,
) -> list[CallTuple]:
    """
    Function that constructs a CallTree for the input subroutine.
    failures: if given, a child whose call info can't be collected is
    recorded there and kept as a leaf instead of raising.
    """

    for childsub in sub.child_subroutines.values():
        if childsub.preprocessed or childsub.library:
            continue
        if failures is not None and childsub.id in failures:
            continue
        try:
            childsub.collect_var_and_call_info(sub_dict, dtype_dict, mod_dict)
        except (Exception, SystemExit) as err:
            if failures is None:
                raise
            failures[childsub.id] = f"{type(err).__name__}: {err}"
            childsub.logger.error(f"Analysis failed for {childsub.id}: {failures[childsub.id]}")

    flat_call_list: list[CallTuple] = [
        CallTuple(
            nested=nested,
            subname=sub.id,
        )
    ]

    for childsub in sub.child_subroutines.values():
        if childsub.library:
            continue
        if failures is not None and childsub.id in failures:
            flat_call_list.append(CallTuple(nested=nested + 1, subname=childsub.id))
            continue
        child_list = construct_call_tree(
            childsub,
            sub_dict,
            dtype_dict,
            mod_dict,
            nested + 1,
            failures,
        )
        flat_call_list.extend(child_list)
    sub.abstract_call_tree = make_call_tree(flat_call_list)

    return flat_call_list


def make_call_tree(flat_calls: list[CallTuple]) -> CallTree:
    """
    Build a subroutine call tree from a flat list of CallTuple
    Assumes the first tuple is the root.
    """
    root = CallTree(flat_calls[0])
    stack = [(flat_calls[0].nested, root)]

    for call in flat_calls[1:]:
        node_tree = CallTree(call)
        # Pop from stack until we find the parent
        while stack and stack[-1][0] >= call.nested:
            stack.pop()
        if stack:
            parent_tree = stack[-1][1]
            parent_tree.add_child(node_tree)
        stack.append((call.nested, node_tree))

    return root
