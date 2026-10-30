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
from pathlib import Path
from typing import Optional

from spel.scripts.fortran_parser import spel_ast as A
from spel.transforms.acc_directives import (AccDirective, DirectiveError, find_directives,
                                            next_code_line, strip_comment)

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


def rewrite_routine(sub, lines: list[str], notes: Optional[list[str]] = None) -> int:
    """Hoist the inquiry bounds of the collapsed nests of one routine; entries of `lines` become
    multi-line strings. Returns the number of directives changed."""
    from spel.transforms.compact.select import statement_end

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
        return 0
    edits: dict[int, str] = {}       # header line -> new text
    before: dict[int, list[str]] = {}  # directive start line -> assignments
    nmax, done = 0, 0
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
        for k, lp in enumerate(nest, 1):
            node = lp.node
            lo, hi = str(node.start), str(node.end)
            changed = False
            if has_inquiry(node.start):
                assigns.append(f"{d.indent}spel_lb{k} = {lo}  {TAG}")
                lo, changed = f"spel_lb{k}", True
            if has_inquiry(node.end):
                assigns.append(f"{d.indent}spel_ub{k} = {hi}  {TAG}")
                hi, changed = f"spel_ub{k}", True
            if changed:
                if not re.match(r"\s*do\s+\w+\s*=", lines[lp.start_ln], re.I):
                    if notes is not None:
                        notes.append(f"{sub.id}: line {lp.start_ln + 1}: header not a plain `do index =`; left alone")
                    assigns = []
                    break
                edits[lp.start_ln] = _header(lines[lp.start_ln], node.index, lo, hi,
                                             str(node.step) if node.step is not None else None)
                nmax = max(nmax, k)
        if assigns:
            before[d.start_ln] = assigns
            done += 1
    if not done:
        return 0
    for ln, text in edits.items():
        lines[ln] = text
    for ln, ass in before.items():
        lines[ln] = "\n".join(ass + [lines[ln]])
    names = [f"spel_lb{k}" for k in range(1, nmax + 1)] + [f"spel_ub{k}" for k in range(1, nmax + 1)]
    last = getattr(rec, "last_decl_ln", -1)
    if last >= 0:
        at = statement_end(lines, last) + 1
        ref = lines[last]
    else:
        from spel.transforms.acc_locals import first_executable
        from spel.transforms.acc_apply import routine_tree
        at = first_executable(routine_tree(sub), rec)
        ref = lines[at]
    indent = ref[: len(ref) - len(ref.lstrip())] if ref else "    "
    lines[at] = f"{indent}integer :: {', '.join(names)}  {TAG}: host copies of collapsed-loop inquiry bounds\n" + (lines[at] or "")
    return done


def apply(fut, routine_ids, dry_run: bool = False) -> list[str]:
    """Rewrite the collapsed nests with inquiry bounds of the given routines. Returns a log."""
    from spel.transforms.acc_apply import _path

    by_file: dict[Path, list] = {}
    for sid in sorted(set(routine_ids)):
        sub = fut.subroutine_dict.get(sid)
        if sub is not None and sub.record is not None:
            by_file.setdefault(_path(sub), []).append(sub)
    log: list[str] = []
    for path, subs in sorted(by_file.items()):
        lines = path.read_text().split("\n")
        n = 0
        # bottom-up: a routine's edits must not shift the lines of the routines above it
        for sub in sorted(subs, key=lambda s: -s.startline):
            k = rewrite_routine(sub, lines, log)
            if k:
                log.append(f"{sub.id}: {k} collapsed nests with inquiry bounds hoisted")
                n += k
        if n and not dry_run:
            path.write_text("\n".join(l for l in lines if l is not None))
    return log
