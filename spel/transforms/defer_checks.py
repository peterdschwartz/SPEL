"""
Defer host-only checks out of loops, so the loops can run on the device.

A loop that otherwise ports can hold an `if` block doing work a device cannot do or must
not do per element: an error report (`write`, `call endrun`), or a search for the last
element that fails a test (`found = .true.; indexc = c`, read after the loop). Such a
block B is replaced by a record of the loop iteration

    if (cond) spel_dc_k = max(spel_dc_k, i - (start) + 1)

(an integer max reduction: the planner ports the loop with `reduction(max:spel_dc_k)`), and
B's body runs once on the host after the loop, for the last iteration where B ran:

    if (spel_dc_k > 0) then
       i = start + spel_dc_k - 1
       <the loop's index mappings B uses, e.g. c = filter_soilc(fc)>
       <B's body>
    end if

A search keeps the last element that fails its test, as before. A check that aborts still
aborts (after the loop instead of at the first failing element). acc-sync puts the host
copies the body reads inside the `if (spel_dc_k > 0)` branch (a rare branch).

B is deferred only when the result is the same: B is a plain `if` (no else), not inside an
inner loop of the loop; it writes only local scalars the loop does not read elsewhere, calls
nothing but the abort routines; the loop-local scalars it reads are index mappings assigned
at the top of the loop body from the loop index and data the loop does not write. Anything
else is left alone and reported. Structure comes from the parse tree; the text edits only
place lines at the positions the tree gives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from spel.scripts.fortran_parser import spel_ast as A

PREFIX = "spel_dc_"
TAG = "! spel defer-checks"


@dataclass
class Deferral:
    routine: str
    path: Path
    loop_ln: int  # the `do` line (0-based)
    loop_end: int  # its `end do` line
    if_ln: int  # first line of the deferred if
    if_end: int  # its last line
    var: str  # spel_dc_<k>
    cond: str  # the if condition (source text)
    body: list[str]  # the if body (source lines)
    replay: list[str]  # index mappings to recompute (source lines)
    index: str
    start: str


@dataclass
class Plan:
    deferrals: list[Deferral] = field(default_factory=list)
    refused: list[tuple[str, int, str]] = field(default_factory=list)  # routine, line, reason


def _names(node) -> set[str]:
    from spel.scripts.db_export.trace import expr_names

    return expr_names(node)


def _assigned(st) -> Optional[tuple[object, object]]:
    """(lhs, rhs) of an assignment statement (two parse forms), else None."""
    if isinstance(st, A.AssignmentStatement):
        return st.left, st.right
    e = getattr(st, "expression", None)
    if isinstance(st, A.ExpressionStatement) and isinstance(e, A.InfixExpression) and e.operator.strip() == "=":
        return e.left_expr, e.right_expr
    return None


def _base(lhs) -> str:
    """The variable an assignment writes (`x`, `x(c)`, `x%y(c)` -> 'x')."""
    return re.split(r"[(%]", str(lhs).strip(), maxsplit=1)[0].strip().lower()


def _own_reads(st) -> set[str]:
    """Names a statement itself reads (an if: its condition; a loop: its bounds; not the
    statements nested in it)."""
    asg = _assigned(st)
    if asg is not None:
        sub = _names(asg[0]) - {_base(asg[0])}  # subscripts of the target
        return _names(asg[1]) | sub
    if isinstance(st, A.IfConstruct):
        return _names(st.condition)
    if isinstance(st, (A.DoLoop, A.DoWhile)):
        return set().union(*[_names(x) for x in (getattr(st, "start", None), getattr(st, "end", None)) if x is not None])
    if isinstance(st, (A.AssociateConstruct, A.BlockStatement)):
        return set()
    return _names(st)


def _walk(stmts, in_inner_loop=False):
    """(statement, inside an inner loop) for every statement of a block, recursively."""
    for st in stmts:
        yield st, in_inner_loop
        if isinstance(st, A.IfConstruct):
            yield from _walk(st.consequence.statements, in_inner_loop)
            for e in st.else_ifs:
                yield from _walk(e.consequence.statements, in_inner_loop)
            if st.else_ is not None:
                yield from _walk(st.else_.alternative.statements, in_inner_loop)
        elif isinstance(st, (A.DoLoop, A.DoWhile)):
            yield from _walk(st.body.statements, True)
        elif isinstance(st, A.AssociateConstruct):
            yield from _walk(st.body.statements, in_inner_loop)
        elif isinstance(st, A.BlockStatement):
            yield from _walk(st.statements, in_inner_loop)


def _outer_loops(stmts):
    """The loops not nested in another loop."""
    for st in stmts:
        if isinstance(st, A.DoLoop):
            yield st
        elif isinstance(st, A.IfConstruct):
            yield from _outer_loops(st.consequence.statements)
            for e in st.else_ifs:
                yield from _outer_loops(e.consequence.statements)
            if st.else_ is not None:
                yield from _outer_loops(st.else_.alternative.statements)
        elif isinstance(st, A.AssociateConstruct):
            yield from _outer_loops(st.body.statements)
        elif isinstance(st, A.BlockStatement):
            yield from _outer_loops(st.statements)


def _host_work(st) -> bool:
    """write/print or an abort call anywhere in the statement."""
    from spel.transforms.acc_apply import abort_calls

    blk = A.BlockStatement(None)
    blk.statements = [st]
    if abort_calls(blk):
        return True
    return any(isinstance(s, (A.WriteStatement, A.PrintStatement)) for s, _ in _walk([st]))


def _preceding(body, target) -> list:
    """The statements that run before `target` in each block enclosing it (from the loop body
    inwards; the enclosing if constructs are guards, not candidates for replay)."""
    out = []
    stmts = body
    while True:
        enclosing = None
        for s in stmts:
            end = getattr(s, "construct_end_ln", -1)
            if isinstance(s, A.IfConstruct) and s is not target and end >= 0 and s.lineno < target.lineno <= end:
                enclosing = s
                break
            if s.lineno >= target.lineno:
                break
            out.append(s)
        if enclosing is None or enclosing.else_ifs or enclosing.else_ is not None:
            return out
        stmts = enclosing.consequence.statements


def _aborts_unconditionally(st) -> bool:
    """A top-level statement of the if's branch is an abort call."""
    from spel.transforms.acc_apply import ABORTS

    for s in st.consequence.statements:
        if isinstance(s, A.SubCallStatement):
            f = s.function.function if isinstance(s.function, A.FuncExpression) else s.function
            if str(f).split("%")[-1].lower() in ABORTS:
                return True
    return False


def _if_header(lines: list[str], st) -> tuple[str, int]:
    """(condition text, last line of the header) of a block if; for a one-line if, the action
    statement starts after the condition on the same line."""
    i, text = st.lineno, ""
    while True:
        code = re.sub(r"!.*$", "", lines[i]).rstrip()
        cont = code.endswith("&")
        text += " " + code.rstrip("&").strip().lstrip("&")
        if not cont:
            break
        i += 1
    m = re.match(r"\s*if\s*\(", text, re.I)
    depth, k = 1, m.end()
    while depth:
        depth += {"(": 1, ")": -1}.get(text[k], 0)
        k += 1
    return text[m.end():k - 1].strip(), i


def plan_routine(sub, lines: list[str], tree, rec, start_k: int = 1) -> Plan:
    from spel.transforms.acc_directives import find_directives

    out = Plan()
    name = sub.id
    compute = [(d.start_ln, d.end_ln) for d in find_directives(lines, sub.startline, sub.endline)
               if d.is_compute_loop or d.is_compute_region]
    decl = (lambda v: rec.declaration(v)) if rec is not None else (lambda v: None)
    # a second run continues the numbering of the deferrals already in the routine
    done = [int(m.group(1)) for l in lines[sub.startline:sub.endline + 1] for m in [re.search(rf"\b{PREFIX}(\d+)\b", l)] if m]
    k = max([start_k - 1] + done) + 1
    for loop in _outer_loops(tree.body.statements):
        if any(a <= loop.lineno - 1 <= b for a, b in compute) or loop.step is not None or not loop.index:
            continue
        body = loop.body.statements
        stmts = list(_walk(body))
        writes_in_loop: set[str] = set()
        reads_in_loop: dict[int, set[str]] = {}
        for st, _ in stmts:
            asg = _assigned(st)
            if asg is not None:
                writes_in_loop.add(_base(asg[0]))
        cands = [st for st, inner in stmts if isinstance(st, A.IfConstruct) and not inner
                 and not st.else_ifs and st.else_ is None]
        # outermost candidate blocks only
        cand_lines = []
        for st in cands:
            end = st.construct_end_ln if st.construct_end_ln >= 0 else st.lineno
            if any(a <= st.lineno <= b for a, b in cand_lines):
                continue
            inner_stmts = [x for x, _ in _walk([st])][1:]
            scal_writes = []
            ok, why = True, ""
            for s in inner_stmts:
                asg = _assigned(s)
                if asg is not None:
                    lhs = asg[0]
                    if not isinstance(lhs, A.Identifier):
                        ok, why = False, "writes an array element or component"
                        break
                    d = decl(lhs.value.lower())
                    if d is None or d.dim:
                        ok, why = False, f"writes non-local or array '{lhs.value}'"
                        break
                    scal_writes.append(lhs.value.lower())
                elif isinstance(s, A.SubCallStatement):
                    from spel.transforms.acc_apply import ABORTS
                    f = s.function.function if isinstance(s.function, A.FuncExpression) else s.function
                    if str(f).split("%")[-1].lower() not in ABORTS:
                        ok, why = False, f"calls {f}"
                        break
                elif isinstance(s, (A.DoLoop, A.DoWhile)):
                    ok, why = False, "holds a loop"
                    break
                elif isinstance(s, (A.ExitStatement, A.ReturnStatement, A.StopStatement, A.GotoStatement)):
                    # exit/cycle: a first-match search or a jump, not a check that can run after the loop
                    ok, why = False, f"holds {type(s).__name__}"
                    break
            if any(v.startswith(PREFIX) for v in scal_writes):
                continue  # our own record of a deferred check (a second run)
            host = _host_work(st)
            # a search: B's scalars are its result, read after the loop
            after = {a.path.split("%")[0].lower() for a in rec.accesses()
                     if a.ln > loop.end_ln and "r" in a.status} if rec is not None else set()
            # (or escapes through an out/inout dummy argument)
            after |= {v for v in scal_writes if (decl(v) is not None and (decl(v).intent or "") in ("out", "inout"))}
            if not host and not (set(scal_writes) & after):
                continue
            # the iteration is recorded when B's condition holds: B's own work must then be
            # unconditional, or end in an abort (nested ifs only shape the report). Else the
            # nested ifs are the candidates.
            nested_if = any(isinstance(x, A.IfConstruct) for x in inner_stmts)
            if nested_if and not _aborts_unconditionally(st):
                continue
            if not ok:
                out.refused.append((name, st.lineno, why))
                continue
            # the scalars B writes: not read in the loop outside B
            outside = [s for s, _ in stmts if not (st.lineno <= s.lineno <= end)]
            read_outside = set()
            for s in outside:
                read_outside |= _own_reads(s)
            if set(scal_writes) & read_outside:
                out.refused.append((name, st.lineno, f"{sorted(set(scal_writes) & read_outside)} also read in the loop"))
                continue
            # loop-local scalars B reads: index mappings at the top of the loop body
            b_reads = set()
            for s in [x for x, _ in _walk([st])]:
                b_reads |= _own_reads(s)

            def local_scalar(v: str) -> bool:
                d = decl(v)
                return d is not None and not d.dim and v in writes_in_loop

            local_scalars = {v for v in b_reads if local_scalar(v) and v not in scal_writes and v != loop.index.lower()}
            replay, known, bad = [], {loop.index.lower()}, None
            for s in _preceding(body, st):  # statements before B in B's enclosing blocks, outermost first
                if s.lineno >= st.lineno:
                    break
                if isinstance(s, A.IfConstruct) and s.construct_end_ln >= 0 and not (s.lineno <= st.lineno <= s.construct_end_ln):
                    # a choice of a needed scalar (`if (is_fates(p)) then th = a else th = b`):
                    # replayable when it only assigns needed scalars from data the loop does not write
                    inner = [x for x, _ in _walk([s])][1:]
                    asgs = [_assigned(x) for x in inner]
                    if inner and all(a is not None and isinstance(a[0], A.Identifier)
                                     and a[0].value.lower() in local_scalars for a in asgs):
                        reads = _names(s.condition).union(*[_names(a[1]) for a in asgs])
                        if not ({n for n in reads if local_scalar(n)} - known) and not \
                                {n for n in reads if n in writes_in_loop and not local_scalar(n)}:
                            known |= {a[0].value.lower() for a in asgs}
                            replay.append((s.lineno, s.construct_end_ln))
                    continue
                asg = _assigned(s)
                if asg is None or not isinstance(asg[0], A.Identifier):
                    continue
                v = asg[0].value.lower()
                if v not in local_scalars:
                    continue
                rhs = _names(asg[1])
                locals_rhs = {n for n in rhs if local_scalar(n)}
                arrays_written = {n for n in rhs if n in writes_in_loop and n not in locals_rhs}
                if locals_rhs - known or arrays_written:
                    bad = v
                    break
                known.add(v)
                replay.append((s.lineno, s.lineno))
            missing = local_scalars - known
            if bad or missing:
                out.refused.append((name, st.lineno, f"reads loop-local {sorted(missing) or bad} that is no index mapping"))
                continue
            cond, hdr_end = _if_header(lines, st)
            if st.construct_end_ln >= 0:
                body_lines = lines[hdr_end + 1:st.construct_end_ln]
            else:
                text = re.sub(r"!.*$", "", lines[st.lineno]).strip()
                m = re.match(r"if\s*\(", text, re.I)
                depth, j = 1, m.end()
                while depth:
                    depth += {"(": 1, ")": -1}.get(text[j], 0)
                    j += 1
                body_lines = [lines[st.lineno][:len(lines[st.lineno]) - len(lines[st.lineno].lstrip())] + text[j:].strip()]
            cpp = [l for l in body_lines if l.lstrip().startswith("#")]
            if sum(1 for l in cpp if l.lstrip().startswith("#if")) != sum(1 for l in cpp if l.lstrip().startswith("#endif")):
                out.refused.append((name, st.lineno, "unbalanced preprocessor lines"))
                continue
            start = str(loop.start)
            out.deferrals.append(Deferral(
                routine=name, path=Path(sub.filepath), loop_ln=loop.lineno, loop_end=loop.end_ln,
                if_ln=st.lineno, if_end=end, var=f"{PREFIX}{k}", cond=cond, body=body_lines,
                replay=[l for a, b in replay for l in lines[a:b + 1]], index=loop.index, start=start))
            cand_lines.append((st.lineno, end))
            k += 1
    return out


def plan(fut, roots: Iterable[str]) -> Plan:
    from spel.transforms.acc_apply import _Facts, call_tree

    facts = _Facts(fut)
    out = Plan()
    for sid, sub in sorted(call_tree(fut, roots).items()):
        if sub.record is None:
            continue
        try:
            tree = facts.tree(sub)
        except BaseException:
            continue
        p = plan_routine(sub, facts.lines(sub), tree, sub.record)
        out.deferrals += p.deferrals
        out.refused += p.refused
    return out


def edits_for(lines: list[str], defs: list[Deferral], first_exec: dict[str, int]) -> list[tuple[int, int, int, list[str]]]:
    """(line, rank, lines to delete, lines to insert) for the deferrals of one file; of the
    insertions at one line, a lower rank ends up above a higher one."""
    ind = lambda s: s[:len(s) - len(s.lstrip())]
    out = []
    for d in defs:
        pad = ind(lines[d.if_ln])
        out.append((d.if_ln, 9, d.if_end - d.if_ln + 1,
                    [f"{pad}if ({d.cond}) {d.var} = max({d.var}, {d.index} - ({d.start}) + 1)  {TAG}"]))
        lp = ind(lines[d.loop_ln])
        out.append((d.loop_end + 1, 2 + d.if_ln * 1e-9, 0,
                    [f"{lp}if ({d.var} > 0) then  {TAG}: the last element where the check held",
                     f"{lp}   {d.index} = {d.start} + {d.var} - 1"]
                    + [f"{lp}   {l.strip()}" for l in d.replay] + d.body + [f"{lp}end if"]))
        out.append((d.loop_ln, 3 + d.if_ln * 1e-9, 0, [f"{lp}{d.var} = 0  {TAG}"]))
    by_routine: dict[str, list[str]] = {}
    for d in defs:
        by_routine.setdefault(d.routine, []).append(d.var)
    for r, vs in by_routine.items():
        ln = first_exec[r]
        out.append((ln, 1, 0, [f"    integer :: {', '.join(vs)}  {TAG}: elements recorded by deferred checks"]))
    return out


def apply(fut, p: Plan, dry_run: bool = False) -> list[str]:
    from spel.transforms.acc_apply import _Facts
    from spel.transforms.acc_directives import find_directives
    from spel.transforms.acc_locals import _above_directives, first_executable

    facts = _Facts(fut)
    log = []
    by_file: dict[Path, list[Deferral]] = {}
    for d in p.deferrals:
        by_file.setdefault(d.path, []).append(d)
    for path, defs in by_file.items():
        lines = path.read_text().split("\n")
        fe = {}
        for d in defs:
            sub = fut.subroutine_dict[d.routine]
            fe[d.routine] = _above_directives(first_executable(facts.tree(sub), sub.record),
                                              find_directives(lines, sub.startline, sub.endline))
        eds = edits_for(lines, defs, fe)
        # bottom-up; at one line: replacements before insertions
        for ln, _, n, new in sorted(eds, key=lambda e: (-e[0], -e[1])):
            lines[ln:ln + n] = new
        log.append(f"{path.name}: {len(defs)} checks deferred")
        if not dry_run:
            path.write_text("\n".join(lines))
    return log
