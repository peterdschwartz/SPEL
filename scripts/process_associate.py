from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from analyze_subroutines import Subroutine

import re
import sys

from scripts.types import LineTuple


def getAssociateClauseVars(sub: Subroutine, verbose=False):
    """
    Funtion to extract the associated variables in a subroutine
    """
    func_name = "getAssociateClauseVars::"

    subroutine_name = sub.name
    lines = sub.sub_lines

    associate_vars = {}
    associate_start = 0
    associate_end = 0
    regex_associate = re.compile(r"\b(associate)\s*(?=\()")
    matches: list[LineTuple] = [
        line for line in filter(lambda x: regex_associate.search(x.line), lines)
    ]

    if len(matches) > 1:
        print(f"{func_name}:: multiple associate clauses founds! {sub.name}")
        print(matches)
        sys.exit(1)
    if not matches:
        return associate_vars, associate_start, associate_end

    if verbose:
        sub.logger.info(f"associate matches {matches}")
        sub.logger.info(
            f"{func_name}::associate start {associate_start} {sub.cpp_startline}"
        )

    associate_line = matches[0]
    associate_start = associate_line.ln

    associate_end = next((l.ln for l in lines if l.ln > associate_start), None)
    regex_str = re.compile(r"(?<=\()(.+)(?=\))")
    associate_string = regex_str.search(associate_line.line).group(0)

    if verbose:
        sub.logger.info(f"{func_name}::{associate_string}")

    regex_remove_index = re.compile(r"(\()(.+)(\))")
    for pair in associate_string.split(","):
        parsed = pair.split("=>")
        # find_ex = the ptr name to find in routine
        # repl_ex = the variable to replace it with
        find_ex = parsed[0].strip()
        repl_ex = parsed[1].strip()
        repl_ex = regex_remove_index.sub("", repl_ex)
        find_ex = find_ex.lower().strip()
        repl_ex = repl_ex.lower().strip()
        if find_ex in associate_vars:
            print(
                f"{func_name}::Error! Multiple associations for {find_ex} in {subroutine_name}"
            )
            sys.exit(1)
        associate_vars[find_ex] = repl_ex

    if verbose:
        sub.logger.info(
            f"{func_name}:: Found associate variables for {subroutine_name}"
        )
        for key in associate_vars:
            sub.logger.info(f"{key} => {associate_vars[key]}")

    return associate_vars, associate_start, associate_end
