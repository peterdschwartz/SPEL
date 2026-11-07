"""
Fortran text emitters shared by the transforms: directives, wrapped statements, conditions, small
constructs. Each produces lines (no newline) at a given indentation; long lines are continued with
`&` the way the transforms always wrote them, so generated code keeps its layout.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

GENERATED_TAG = "! spel"


def directive_list(indent: str, head: str, names: Sequence[str], width: int = 100,
                   sentinel: str = "!$acc", tail: str = ")") -> list[str]:
    """`!$acc head(a, b, ...)<tail rest>` with `!$acc&` continuations past `width`:

        !$acc update self(a, b, &
        !$acc&   c) if_present

    `tail` closes the list (`)`, or `) if_present if(cond)` ...)."""
    out, cur = [], f"{indent}{sentinel} {head}("
    for i, n in enumerate(names):
        piece = n + (", " if i + 1 < len(names) else tail)
        if len(cur) + len(piece) > width and not cur.endswith("("):
            out.append(cur.rstrip() + " &")
            cur = f"{indent}{sentinel}&   "
        cur += piece
    return out + [cur]


def directive_clauses(indent: str, head: str, clauses: Iterable[str], width: int,
                      sentinel: str = "!$omp") -> list[str]:
    """`!$omp head c1 c2 ...`; a clause that does not fit starts a `!$omp   c` continuation."""
    lines = [f"{indent}{sentinel} {head}"]
    for c in clauses:
        if len(lines[-1]) + 1 + len(c) <= width:
            lines[-1] += f" {c}"
        else:
            lines[-1] += " &"
            lines.append(f"{indent}{sentinel}   {c}")
    return lines


def wrap_items(head: str, items: Sequence[str], width: int = 100) -> list[str]:
    """`head` followed by comma-separated items (`use m, only: a, b`, `public :: a, b`),
    continued with `&` past `width`, continuation lines indented 4 past `head`."""
    pad = " " * (len(head) - len(head.lstrip()) + 4)
    out, line = [], head
    for i, item in enumerate(items):
        piece = item + ("," if i < len(items) - 1 else "")
        if len(line) + len(piece) + 1 > width and line.strip() != head.strip():
            out.append(line.rstrip() + " &")
            line = pad
        line += piece if line.endswith((" ", ":")) and line == pad else (" " + piece if not line.endswith(" ") else piece)
    out.append(line.rstrip())
    return out


def wrap_statement(line: str, width: int = 120) -> list[str]:
    """A long statement split after `, ` with `&` continuations indented 5 past the statement."""
    if len(line) <= width:
        return [line]
    pad = " " * (len(line) - len(line.lstrip()) + 5)
    out, cur = [], ""
    for piece in re.split(r"(?<=, )", line):
        if cur and len(cur) + len(piece) > width - 2:
            out.append(cur.rstrip() + " &")
            cur = pad + piece
        else:
            cur += piece
    out.append(cur)
    return out


def if_then(head: str, cond: str, width: int = 120) -> list[str]:
    """`<head>if (cond) then`, continued before ` .and. `/` .or. ` when longer than `width`."""
    line = f"{head}if ({cond}) then"
    if len(line) <= width:
        return [line]
    pieces = re.split(r"(?=\s\.(?:and|or)\.\s)", cond)
    out, cur = [], f"{head}if ("
    for p in pieces:
        if len(cur) + len(p) > width - 2 and cur.strip() != "if (":
            out.append(cur.rstrip() + " &")
            cur = head + "     " + p.lstrip()
        else:
            cur += p
    out.append(cur + ") then")
    return out


def if_block(indent: str, cond: str, body: Sequence[str], step: str = "   ") -> list[str]:
    """`if (cond) then / body / end if`, the body indented by `step` more than `indent`
    (body lines are given without the block's indentation)."""
    return [f"{indent}if ({cond}) then", *(f"{indent}{step}{b}" for b in body), f"{indent}end if"]


def do_loop(indent: str, index: str, lo: str, hi: str, body: Sequence[str] = (),
            step: Optional[str] = None, comment: str = "", body_step: str = "   ") -> list[str]:
    """`do index = lo, hi[, step]` / body / `end do`."""
    head = f"{indent}do {index} = {lo}, {hi}" + (f", {step}" if step else "") + (f"  {comment}" if comment else "")
    return [head, *(f"{indent}{body_step}{b}" for b in body), f"{indent}end do"]


def do_header(line: str, index: str, lo: str, hi: str, step: Optional[str] = None, comment: str = "") -> str:
    """A `do` header replacing `line`, keeping its indentation (and `comment`, when given)."""
    pad = line[: len(line) - len(line.lstrip())]
    text = f"{pad}do {index} = {lo}, {hi}" + (f", {step}" if step else "")
    return text + (f"  {comment}" if comment else "")


def use_stmt(indent: str, module: str, names: Sequence[str] = (), width: int = 100) -> list[str]:
    """`use module` or `use module, only : a, b` (wrapped)."""
    if not names:
        return [f"{indent}use {module}"]
    return wrap_items(f"{indent}use {module}, only :", list(names), width)


def declaration(indent: str, type_spec: str, names: Sequence[str], attrs: Sequence[str] = (),
                comment: str = "", width: int = 100) -> list[str]:
    """`type_spec[, attrs] :: a, b` (wrapped), with an optional trailing comment."""
    head = ", ".join([type_spec, *attrs])
    out = wrap_items(f"{indent}{head} ::", list(names), width)
    if comment:
        out[-1] += f"  {comment}"
    return out


def fresh_name(base: str, taken: Iterable[str]) -> str:
    """`base`, or `base_2`, `base_3`, ... : the first not in `taken` (case-insensitive)."""
    low = {t.lower() for t in taken}
    if base.lower() not in low:
        return base
    k = 2
    while f"{base}_{k}".lower() in low:
        k += 1
    return f"{base}_{k}"
