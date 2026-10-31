"""
Inquiry functions in collapsed loop headers (`spel collapse-bounds`).

nvfortran 25.9 (cc80) runs a `parallel loop collapse(n)` whose `do` headers take their
bounds from `lbound`/`ubound` of a pointer (a derived-type component such as
`top_af%solad_pp`, an associate name of one, or a module pointer) as an EMPTY kernel on the
GPU: nothing is written, no error. A 1-D loop with the same bounds runs, a collapsed nest
with the bounds in host scalars runs, and the device descriptor holds the right bounds
(reproducer: spelval/assoc_test/t_assoc2.F90, 2026-10-30; it cost a day of bisecting as
"garbage t_soisno in urban columns": UrbanRadiation's section loops never filled
`top_af%solad_pp`).

This pass hoists every inquiry bound of a collapsed nest into an integer scalar assigned on
the host right before the directive:

    spel_lb1 = lbound(forc_solad, 1)  ! spel collapse-bounds
    spel_ub1 = ubound(forc_solad, 1)  ! spel collapse-bounds
    !$acc parallel loop gang vector collapse(2) default(present) if(elm_device_ready)
    do spel_i2 = 1, numrad
       do spel_i1 = spel_lb1, spel_ub1

The headers are rebuilt from the parsed loop nodes (`LoopInfo.node`: index, start, end,
step), so the pass needs the analysis of the sources it edits (`spel analyze` after
`acc-apply` added the directives); the scalars are declared after the routine's last
declaration and reused by every nest of the routine. Directives whose nest the record does
not know, or whose step uses an inquiry, are left alone and reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from spel.scripts.fortran_parser import spel_ast as A
from spel.transforms.acc_directives import DirectiveError, find_directives, next_code_line, strip_comment
from spel.transforms.core.edits import Edit, EditSet, insert, replace
from spel.transforms.core.fortran import declaration
from spel.transforms.core.registry import transform
from spel.transforms.core.source import indent
from spel.transforms.core.transform import Change, Refusal, Transform

TAG = "! spel collapse-bounds"
INQUIRY = {"lbound", "ubound", "size"}
CONSTRUCTS = {"parallel loop", "kernels loop"}


def has_inquiry(expr) -> bool:
    """True when the expression calls lbound/ubound/size (anywhere inside)."""

    def walk(node) -> bool:
        if isinstance(node, A.FuncExpression):
            fn = node.function
            name = fn.value if isinstance(fn, A.Identifier) else str(fn)
            if name.lower() in INQUIRY:
                return True
        for v in vars(node).values():
            if isinstance(v, A.Node) and walk(v):
                return True
            if isinstance(v, (list, tuple)) and any(isinstance(x, A.Node) and walk(x) for x in v):
                return True
        return False

    return expr is not None and walk(expr)


def _comment(line: str) -> str:
    code = strip_comment(line)
    return line[len(code):] if len(code) < len(line) else ""


def _header(line: str, index: str, lo: str, hi: str, step: Optional[str]) -> str:
    indent = line[: len(line) - len(line.lstrip())]
    text = f"{indent}do {index} = {lo}, {hi}" + (f", {step}" if step else "")
    cm = _comment(line).strip()
    return text + (f"  {cm}" if cm else "")


def plan_routine(sub, lines: list[str], notes: Optional[list[str]] = None) -> list[Edit]:
    """The edits hoisting the inquiry bounds of the collapsed nests of one routine (none: nothing
    to do). Nests the pass can't rewrite are reported in `notes` and left whole."""
    from spel.transforms.core.query import first_executable, routine_tree
    from spel.transforms.core.source import statement_end

    rec = sub.record
    loops = {}
    for top in rec.loop_nest:
        for lp in top.walk():
            loops[lp.start_ln] = lp
    try:
        directives = find_directives(lines, sub.startline, sub.endline)
    except DirectiveError as err:
        if notes is not None:
            notes.append(f"{sub.id}: {err}; routine left alone")
        return []
    edits: list[Edit] = []
    nmax = 0
    for d in directives:
        if d.construct not in CONSTRUCTS:
            continue
        try:
            n = d.collapse()
        except DirectiveError:
            continue
        if n < 2:
            continue
        ln = next_code_line(lines, d.end_ln)
        loop = loops.get(ln) if ln is not None else None
        if loop is None or loop.node is None:
            if notes is not None:
                notes.append(f"{sub.id}: line {d.start_ln + 1}: no analyzed loop follows the collapse directive; left alone")
            continue
        try:
            nest = loop.perfect_nest(n)
        except ValueError as err:
            if notes is not None:
                notes.append(f"{sub.id}: line {d.start_ln + 1}: {err}; left alone")
            continue
        if any(has_inquiry(lp.node.step) for lp in nest):
            if notes is not None:
                notes.append(f"{sub.id}: line {d.start_ln + 1}: inquiry in a loop step; left alone")
            continue
        assigns: list[str] = []
        headers: list[Edit] = []
        for k, lp in enumerate(nest, 1):
            node = lp.node
            lo, hi = str(node.start), str(node.end)
            if has_inquiry(node.start):
                assigns.append(f"{d.indent}spel_lb{k} = {lo}  {TAG}")
                lo = f"spel_lb{k}"
            if has_inquiry(node.end):
                assigns.append(f"{d.indent}spel_ub{k} = {hi}  {TAG}")
                hi = f"spel_ub{k}"
            if lo != str(node.start) or hi != str(node.end):
                if not re.match(r"\s*do\s+\w+\s*=", lines[lp.start_ln], re.I):
                    if notes is not None:
                        notes.append(f"{sub.id}: line {lp.start_ln + 1}: header not a plain `do index =`; left alone")
                    assigns = []
                    break
                headers.append(replace(lp.start_ln, _header(lines[lp.start_ln], node.index, lo, hi,
                                                            str(node.step) if node.step is not None else None)))
                nmax = max(nmax, k)
        if assigns:  # the whole nest, or nothing of it
            edits += headers
            edits.append(insert(d.start_ln, assigns, why=f"collapse({n}) nest at line {d.start_ln + 1}"))
    if not edits:
        return []
    names = [f"spel_lb{k}" for k in range(1, nmax + 1)] + [f"spel_ub{k}" for k in range(1, nmax + 1)]
    last = getattr(rec, "last_decl_ln", -1)
    if last >= 0:
        at = statement_end(lines, last) + 1
        ref = lines[last]
    else:
        at = first_executable(routine_tree(sub), rec)
        ref = lines[at]
    # above the assignments when the first nest starts right after the declarations
    decl = declaration(indent(ref) if ref else "    ", "integer", names,
                       comment=f"{TAG}: host copies of collapsed-loop inquiry bounds", width=10**6)
    edits.append(insert(at, decl, order=-1))
    return edits


def rewrite_routine(sub, lines: list[str], notes: Optional[list[str]] = None) -> int:
    """`plan_routine` applied to `lines` in place. Returns the number of directives changed."""
    edits = plan_routine(sub, lines, notes)
    lines[:] = EditSet().extend(edits).apply(lines)
    return sum(1 for e in edits if not e.n and e.order == 0)


@dataclass
class Options:
    routines: list[str] = field(default_factory=list, metadata={"help": "routine ids (default: every analyzed routine)"})


@transform
class CollapseBounds(Transform):
    """
    Matches: `parallel loop`/`kernels loop` directives with collapse(n>=2) whose nest takes a
             bound from lbound/ubound/size.
    Changes: assigns each such bound to spel_lb<k>/spel_ub<k> before the directive, uses the
             scalar in the `do` header, declares the scalars once per routine.
    Refuses: directives with no analyzed loop after them, imperfect nests, inquiries in a step,
             headers that aren't `do i = ...`.
    """

    name = "collapse-bounds"
    summary = "hoist inquiry-function bounds of collapsed loop nests into host scalars (nvfortran bug)"
    Options = Options

    def select(self, ctx, opts):
        yield from ctx.routines(opts.routines or None)

    def plan(self, ctx, opts, sub):
        notes: list[str] = []
        edits = plan_routine(sub, ctx.lines(ctx.path(sub)), notes)
        out = [Refusal(sub.id, n.split(": ", 1)[1]) if n.startswith(sub.id) else n for n in notes]
        if edits:
            k = sum(1 for e in edits if not e.n and e.order == 0)
            out.append(Change(ctx.path(sub), edits, where=sub.id,
                              why=f"{k} collapsed nests with inquiry bounds hoisted"))
        return out


def apply(fut, routine_ids, dry_run: bool = False) -> list[str]:
    """Rewrite the collapsed nests with inquiry bounds of the given routines. Returns a log."""
    from spel.transforms.core.driver import InPlace, run
    from spel.transforms.core.transform import Context

    ctx = Context(fut)
    res = run(CollapseBounds(), ctx, Options(routines=sorted(set(routine_ids))), sink=InPlace(), dry_run=dry_run)
    log = [f"{r.where}: {r.reason}" for r in res.plan.refusals] + res.plan.notes
    log += [f"{c.where}: {c.why}" for c in res.plan.changes]
    return log
