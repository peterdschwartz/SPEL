"""
Blocked filter loops: the SetValues idiom

    do fi = 1, num, block
       associate (idx => filter(fi:min(fi+block-1, num)))
       x(idx, ...) = value
       end associate
    end do

assigns array sections of `block` elements. `spel compact` turns them into one element
per iteration (`x(filter(fi), ...)`) for nvfortran's sake; the GPU port does the same so
the planner sees ordinary filter loops (one device thread per element) instead of
array-section assignments.

Loops come from the record (`all_loops`, the loop's `DoLoop` node); only the associate
name, which the parser has identified, is renamed in the text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TAG = "! spel unblock"


@dataclass
class BlockedLoop:
    do_ln: int  # `do fi = start, end, block`
    end_ln: int  # its `end do`
    index: str  # fi
    start: str
    end: str
    name: str  # associate name (idx)
    filter: str  # filter array the name selects from
    assoc_lines: list[int] = field(default_factory=list)  # associate header (with continuations) and `end associate`
    uses: list[int] = field(default_factory=list)  # body lines that use the name


def find_blocked_loops(rec, lines: list[str]) -> list[BlockedLoop]:
    """The blocked loops of a routine (record + parse tree), outermost first in source order."""
    from spel.scripts.fortran_parser.spel_ast import AssociateConstruct, BoundsExpression, FuncExpression
    from spel.transforms.compact.select import statement_end

    out: list[BlockedLoop] = []
    for loop in rec.all_loops():
        node = loop.node
        body = node.body.statements
        if node.step is None or len(body) != 1 or not isinstance(body[0], AssociateConstruct):
            continue
        assoc = body[0]
        if len(assoc.associations) != 1:
            continue
        (name, selector), = assoc.associations.items()
        if not isinstance(selector, FuncExpression) or len(selector.args) != 1:
            continue
        bounds = selector.args[0]
        if not isinstance(bounds, BoundsExpression) or str(bounds.start).lower() != loop.index:
            continue
        if statement_end(lines, loop.start_ln) != loop.start_ln or node.lineno != loop.start_ln \
                or node.end_ln != loop.end_ln:
            raise ValueError(f"line {loop.start_ln + 1}: unexpected blocked loop: {lines[loop.start_ln].strip()}")
        assoc_end = statement_end(lines, assoc.lineno)
        use = name_pattern(name)
        out.append(BlockedLoop(
            do_ln=loop.start_ln, end_ln=loop.end_ln, index=loop.index,
            start=str(node.start), end=str(node.end), name=name, filter=str(selector.function),
            assoc_lines=[*range(assoc.lineno, assoc_end + 1), assoc.end_ln],
            uses=[ln for ln in range(assoc_end + 1, assoc.end_ln) if use.search(lines[ln].split("!")[0])],
        ))
    return sorted(out, key=lambda b: b.do_ln)


def name_pattern(name: str) -> re.Pattern:
    """The associate name as a token (not a component `%idx`, not part of a longer name)."""
    return re.compile(rf"(?<![%\w]){re.escape(name)}\b", re.IGNORECASE)


def unblock(lines: list[str], loops: list[BlockedLoop]) -> int:
    """Rewrite the loops in place. Removed lines become None, so line numbers stay valid. Returns the count."""
    for b in loops:
        indent = lines[b.do_ln][: len(lines[b.do_ln]) - len(lines[b.do_ln].lstrip())]
        lines[b.do_ln] = f"{indent}do {b.index} = {b.start}, {b.end}  {TAG}"
        for ln in b.assoc_lines:
            lines[ln] = None
        use = name_pattern(b.name)
        for ln in b.uses:
            code, bang, comment = lines[ln].partition("!")
            lines[ln] = use.sub(f"{b.filter}({b.index})", code) + bang + comment
    return len(loops)


def apply(fut, routine_ids, dry_run: bool = False) -> list[str]:
    """Unblock every blocked loop of the given routines. Returns a log."""
    from spel.transforms.acc_apply import _path

    by_file: dict[Path, list] = {}
    for sid in sorted(routine_ids):
        sub = fut.subroutine_dict[sid]
        if sub.record is not None:
            by_file.setdefault(_path(sub), []).append(sub)
    log = []
    for path, subs in sorted(by_file.items()):
        lines = path.read_text().split("\n")
        n = 0
        for sub in subs:
            loops = find_blocked_loops(sub.record, lines)
            n += unblock(lines, loops)
            if loops:
                log.append(f"{sub.id}: {len(loops)} blocked loops")
        if n and not dry_run:
            path.write_text("\n".join(l for l in lines if l is not None))
    return log
