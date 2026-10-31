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

from spel.transforms.core.edits import Edit, delete, replace
from spel.transforms.core.registry import transform
from spel.transforms.core.source import indent
from spel.transforms.core.transform import Change, Transform

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


def unblock_edits(lines: list[str], loops: list[BlockedLoop]) -> list[Edit]:
    """The edits unblocking `loops`: the header loses its step, the associate goes, the uses of
    the associate name become `filter(index)`."""
    out: list[Edit] = []
    for b in loops:
        out.append(replace(b.do_ln, f"{indent(lines[b.do_ln])}do {b.index} = {b.start}, {b.end}  {TAG}"))
        out += [delete(ln) for ln in b.assoc_lines]
        use = name_pattern(b.name)
        for ln in b.uses:
            code, bang, comment = lines[ln].partition("!")
            out.append(replace(ln, use.sub(f"{b.filter}({b.index})", code) + bang + comment))
    return out


def unblock(lines: list[str], loops: list[BlockedLoop]) -> int:
    """Rewrite the loops in place. Removed lines become None, so line numbers stay valid. Returns the count."""
    for e in unblock_edits(lines, loops):
        lines[e.ln] = e.new[0] if e.new else None
    return len(loops)


@dataclass
class Options:
    routines: list[str] = field(default_factory=list, metadata={"help": "routine ids (default: every analyzed routine)"})


@transform
class Unblock(Transform):
    """
    Matches: `do fi = a, b, block` loops whose body is only `associate (idx => filter(fi:...))`.
    Changes: one element per iteration: `do fi = a, b`, the associate removed, `idx` -> `filter(fi)`.
    Refuses: (raises on) a blocked loop whose header the parser placed elsewhere.
    """

    name = "unblock"
    summary = "rewrite blocked filter loops (SetValues idiom) as one element per iteration"
    Options = Options

    def select(self, ctx, opts):
        yield from ctx.routines(opts.routines or None)

    def plan(self, ctx, opts, sub):
        lines = ctx.lines(sub)
        loops = find_blocked_loops(sub.record, lines)
        if loops:
            return Change(ctx.path(sub), unblock_edits(lines, loops), where=sub.id,
                          why=f"{len(loops)} blocked loops")


def apply(fut, routine_ids, dry_run: bool = False) -> list[str]:
    """Unblock every blocked loop of the given routines. Returns a log."""
    from spel.transforms.core.driver import InPlace, run
    from spel.transforms.core.transform import Context

    res = run(Unblock(), Context(fut), Options(routines=sorted(routine_ids)), sink=InPlace(), dry_run=dry_run)
    return [f"{c.where}: {c.why}" for c in res.plan.changes]
