"""
Routine-local arrays that OpenACC device code uses without device storage.

A compute construct (`parallel`, `kernels`, `serial`, and their `loop` forms) that
references a local array needs the array on the device: through the construct's own
data clauses, an enclosing `data` region, an earlier `enter data`, or a `declare`.
Without one, a `default(present)` construct aborts on a GPU ("data in PRESENT clause
was not found") and any other construct copies the array in and out at every launch.
Multicore builds share host memory and never show it.

The same holds for a local array passed to a routine whose call tree runs kernels (the
callee's dummy is `present` there): NitrogenFert's `manure_col` passed to
`p2c_1d_filter`.

Everything comes from the analysis (locals, accesses, calls and loops from
`Subroutine.record`) and from the parsed directives (`acc_directives`); nothing is read
from the source text by pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from spel.transforms.acc_directives import AccDirective, find_directives

# clauses that give a variable device storage for the construct
DATA_CLAUSES = {
    "copy", "copyin", "copyout", "create", "present", "deviceptr", "attach", "no_create",
    "present_or_copy", "present_or_copyin", "present_or_copyout", "present_or_create",
    "pcopy", "pcopyin", "pcopyout", "pcreate",
}
# clauses that make a variable private to the construct (no device storage needed)
PRIVATE_CLAUSES = {"private", "firstprivate", "reduction"}
END_OF = {"parallel": "end parallel", "kernels": "end kernels", "serial": "end serial", "data": "end data"}


@dataclass
class Finding:
    routine: str
    ln: int  # 0-based line of the construct or call
    variable: str
    kind: str  # "present": default(present) construct (aborts) | "implicit": copied per launch | "call"
    detail: str = ""

    def __str__(self) -> str:
        what = {"present": "missing (default(present): aborts on a GPU)",
                "implicit": "implicit copy at every launch",
                "call": "passed to a device routine without device storage",
                "leak": "entered (enter data) but never deleted in the routine"}[self.kind]
        return f"{self.routine} line {self.ln + 1}: {self.variable}: {what}{f' ({self.detail})' if self.detail else ''}"


@dataclass
class RoutineReport:
    routine: str
    filepath: str
    findings: list[Finding] = field(default_factory=list)
    # local arrays needing device storage, in declaration order (the input of a fix)
    locals_needed: list[str] = field(default_factory=list)
    # of those, the ones the host also reads or writes (outside compute constructs and
    # device calls): their values must move, `create` alone would hand the device garbage
    host_accessed: list[str] = field(default_factory=list)
    # device-only locals that error checks print on the host: first line of each such branch
    # (an if that can abort), where an `update self` brings the values over first
    cold_reads: dict[str, list[int]] = field(default_factory=dict)
    # (call line, callee, local arrays passed) of device calls with uncovered locals
    device_calls: list[tuple[int, str, list[str]]] = field(default_factory=list)
    # locals the routine enters (`enter data`) and never deletes: their mapping outlives the
    # call, and a later array at the reused address is "partially present"
    leaked: list[str] = field(default_factory=list)


def _clause_vars(d: AccDirective, names: set[str]) -> set[str]:
    out: set[str] = set()
    for c in d.clauses:
        if c.name in names:
            if c.name == "reduction":
                out.update(c.reduction()[1])
            else:
                out.update(v.lower() for v in c.variables())
    return out


def _loop_at(loops, ln: int):
    for lp in loops:
        if lp.start_ln == ln:
            return lp
        hit = _loop_at(lp.children, ln)
        if hit is not None:
            return hit
    return None


def _compute_ranges(directives: list[AccDirective], rec) -> Iterator[tuple[AccDirective, int, int]]:
    """(construct, first line, last line) of every compute construct."""
    stack: list[AccDirective] = []
    for i, d in enumerate(directives):
        if d.is_compute_loop:
            # the loop follows the directive: the first analyzed loop starting after it
            nxt = min((lp.start_ln for lp in _all_loops(rec.loop_nest) if lp.start_ln > d.end_ln), default=None)
            lp = _loop_at(rec.loop_nest, nxt) if nxt is not None else None
            if lp is not None:
                yield d, d.start_ln, lp.end_ln
        elif d.is_compute_region:
            stack.append(d)
        elif stack and d.construct == END_OF.get(stack[-1].construct):
            start = stack.pop()
            yield start, start.start_ln, d.end_ln


def _all_loops(loops):
    for lp in loops:
        yield lp
        yield from _all_loops(lp.children)


def _data_regions(directives: list[AccDirective]) -> list[tuple[int, int, set[str]]]:
    """Structured `data` regions: (first line, last line, variables with device storage)."""
    out, stack = [], []
    for d in directives:
        if d.construct == "data":
            stack.append(d)
        elif d.construct == "end data" and stack:
            start = stack.pop()
            out.append((start.start_ln, d.end_ln, _clause_vars(start, DATA_CLAUSES)))
    return out


def check_routine(sub, device_routines: set[str], lines: list[str]) -> RoutineReport:
    """The local arrays of one analyzed routine that device code uses without storage."""
    rec = sub.record
    report = RoutineReport(sub.id, str(sub.filepath))
    if rec is None:
        return report
    directives = find_directives(lines, sub.startline, sub.endline)
    arrays = {n.lower(): d for n, d in rec.local_variables.items()
              if d.dim > 0 and "parameter" not in {a.lower() for a in d.attrs}}
    # leaks: any local (scalars too: their stack slots are reused by later arrays)
    local_names = {n.lower() for n, d in rec.local_variables.items() if "parameter" not in {a.lower() for a in d.attrs}}
    exited = set().union(*[_clause_vars(d, {"delete", "copyout", "detach"}) for d in directives
                           if d.construct == "exit data"]) if any(d.construct == "exit data" for d in directives) else set()
    for d in directives:
        if d.construct == "enter data":
            for name in sorted(_clause_vars(d, DATA_CLAUSES) & local_names - exited):
                if name not in report.leaked:
                    report.leaked.append(name)
                    report.findings.append(Finding(sub.id, d.start_ln, name, "leak", f"!$acc {d.construct}"))
    if not arrays and not report.leaked:
        return report
    regions = _data_regions(directives)
    entered = [(d.start_ln, _clause_vars(d, DATA_CLAUSES)) for d in directives if d.construct == "enter data"]
    declared = set().union(*[_clause_vars(d, DATA_CLAUSES) for d in directives if d.construct == "declare"]) \
        if any(d.construct == "declare" for d in directives) else set()

    def covered(name: str, ln: int, extra: set[str] = frozenset()) -> bool:
        if name in extra or name in declared:
            return True
        if any(a <= ln <= b and name in v for a, b, v in regions):
            return True
        return any(e_ln < ln and name in v for e_ln, v in entered)

    accesses = rec.accesses()
    needed: dict[str, None] = {}
    device_lines: list[tuple[int, int]] = []
    for d, lo, hi in _compute_ranges(directives, rec):
        device_lines.append((lo, hi))
        own = _clause_vars(d, DATA_CLAUSES)
        private = _clause_vars(d, PRIVATE_CLAUSES)
        default_present = any(c.name == "default" and (c.args or "").strip().lower() == "present" for c in d.clauses)
        used = {a.base.lower() for a in accesses if lo <= a.ln <= hi}
        for name in sorted(used & set(arrays)):
            if name in private or covered(name, d.start_ln, own):
                continue
            report.findings.append(Finding(sub.id, d.start_ln, name, "present" if default_present else "implicit",
                                           f"!$acc {d.construct}"))
            needed[name] = None
    for call in rec.calls:
        callee = str(call.name).split("%")[-1].lower()
        if callee not in device_routines:
            continue
        device_lines.append((call.ln, call.ln))
        passed = []
        for arg in call.args:
            base = (arg.path or "").split("%")[0].lower()
            if base in arrays and not covered(base, call.ln):
                report.findings.append(Finding(sub.id, call.ln, base, "call", f"call {call.name}"))
                needed[base] = None
                passed.append(base)
        if passed:
            report.device_calls.append((call.ln, str(call.name), sorted(set(passed))))
    order = list(arrays)
    report.locals_needed = sorted(needed, key=order.index)
    # host accesses; reads in an error check (a branch that can abort) only need an
    # `update self` there, not a copy around every kernel
    from spel.transforms.acc_apply import cold_branches, routine_tree

    try:
        cold = cold_branches(routine_tree(sub).body)
    except BaseException:  # the parser exits on what it cannot read: no exemption
        cold = []
    host: set[str] = set()
    cold_reads: dict[str, set[int]] = {}
    for a in accesses:
        name = a.base.lower()
        if name not in needed or any(lo <= a.ln <= hi for lo, hi in device_lines):
            continue
        branch = next((b for b in cold if b[0] <= a.ln <= b[1]), None)
        if branch is not None and "w" not in a.status:
            cold_reads.setdefault(name, set()).add(branch[2])
        else:
            host.add(name)
    report.host_accessed = [n for n in report.locals_needed if n in host]
    report.cold_reads = {n: sorted(v) for n, v in cold_reads.items() if n not in host}
    return report


def check(fut, routines: Optional[set[str]] = None) -> list[RoutineReport]:
    """Every analyzed routine (or the named ones) with OpenACC directives or device calls."""
    from spel.transforms.device.generate import device_routines

    device = device_routines(fut)
    texts: dict[str, list[str]] = {}
    out = []
    for sid, sub in sorted(fut.subroutine_dict.items()):
        if routines and sid.split("::")[-1].lower() not in routines and sid not in routines:
            continue
        path = Path(sub.filepath)
        if not path.is_absolute():
            from spel.scripts.config import E3SM_SRCROOT

            path = Path(E3SM_SRCROOT) / path
        if str(path) not in texts:
            try:
                texts[str(path)] = path.read_text(errors="replace").split("\n")
            except OSError:
                continue
        r = check_routine(sub, device, texts[str(path)])
        if r.findings:
            out.append(r)
    return out


# --- fixing -----------------------------------------------------------------------------

TAG = "! spel acc-locals"


@dataclass
class Insertion:
    ln: int  # insert before this 0-based line
    lines: list[str]
    order: int = 0  # among insertions at one line: lower first


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _wrap_directive(indent: str, head: str, names: list[str], width: int = 100) -> list[str]:
    """`!$acc head(a, b, ...)` with `!$acc&` continuations (text the fix writes)."""
    out, cur = [], f"{indent}!$acc {head}("
    for i, n in enumerate(names):
        piece = n + (", " if i + 1 < len(names) else ")")
        if len(cur) + len(piece) > width and not cur.endswith("("):
            out.append(cur.rstrip() + " &")
            cur = f"{indent}!$acc&   "
        cur += piece
    return out + [cur]


def first_executable(tree, rec) -> int:
    """
    Line of the routine's first executable statement (its own body, not nested ones).
    Statement-function definitions (`f(x) = ...` with `f` declared scalar) belong to the
    specification part although the parser reads them as assignments.
    """
    from spel.scripts.fortran_parser import spel_ast as A

    spec_kinds = (A.UseStatement, A.ImplicitNoneStatement, A.VariableDecl, A.TypeDef, A.AccessStatement,
                  A.ProcedureStatement)

    def statement_function(st) -> bool:
        e = getattr(st, "expression", None)
        if not (isinstance(st, A.ExpressionStatement) and isinstance(e, A.InfixExpression) and e.operator.strip() == "="):
            return False
        lhs = e.left_expr
        if not isinstance(lhs, A.FuncExpression) or not isinstance(lhs.function, A.Identifier):
            return False
        decl = rec.declaration(lhs.function.value.lower()) if rec is not None else None
        return decl is not None and decl.dim == 0

    for st in tree.body.statements:
        if not isinstance(st, spec_kinds) and not statement_function(st):
            return st.lineno
    return tree.end_ln


def plan_fix(sub, report: RoutineReport, lines: list[str]) -> list[Insertion]:
    """
    Device storage for the routine's uncovered local arrays:
    - device-only locals: `enter data create` after the declarations and `exit data
      delete` before every `return` and at the end (their values never leave the device);
    - locals the host also uses: a `data copy` region around every compute construct and
      device call that uses them (values move both ways each time: correct, not fast).
    Positions come from the parse tree and the record (spel_ast, module_spec).
    """
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.scripts.fortran_parser.module_spec import parse_modules, statements_with_next

    rec = sub.record
    path = Path(report.filepath)
    if not path.is_absolute():
        from spel.scripts.config import E3SM_SRCROOT

        path = Path(E3SM_SRCROOT) / path
    name = sub.id.split("::")[-1].lower()
    tree = next((p for m in parse_modules(path) for p in m.subprograms if p.name.lower() == name), None)
    if tree is None:
        raise ValueError(f"{path.name}: routine {name} not found by the parser")
    stmts = list(statements_with_next(tree.body, tree.end_ln))
    first_exec = first_executable(tree, rec)
    ind = _indent(lines[first_exec])
    out: list[Insertion] = []
    device_only = [n for n in report.locals_needed if n not in report.host_accessed]
    if device_only:
        out.append(Insertion(first_exec, [f"{ind}{TAG}: device storage of the local work arrays"]
                             + _wrap_directive(ind, "enter data create", device_only)))
    at: dict[int, list[str]] = {}
    for name in device_only:
        for ln in report.cold_reads.get(name, []):
            at.setdefault(ln, []).append(name)
    for ln, names in sorted(at.items()):
        out.append(Insertion(ln, _wrap_directive(_indent(lines[ln]), "update self", names), order=1))
    to_delete = device_only + [n for n in report.leaked if n not in device_only]
    if to_delete:
        ends = [st.lineno for st, _ in stmts if isinstance(st, A.ReturnStatement)] + [tree.end_ln]
        for ln in ends:
            out.append(Insertion(ln, _wrap_directive(_indent(lines[ln]) if ln != tree.end_ln else ind,
                                                     "exit data delete", to_delete), order=-1))
    host = set(report.host_accessed)
    if host:
        # the constructs and calls of the findings that use host-accessed locals
        spans: dict[tuple[int, int], set[str]] = {}
        directives = find_directives(lines, sub.startline, sub.endline)
        ranges = {d.start_ln: (lo, hi) for d, lo, hi in _compute_ranges(directives, rec)}
        next_of = {st.lineno: nxt for st, nxt in stmts}
        for f in report.findings:
            if f.variable not in host:
                continue
            if f.kind == "call":
                span = (f.ln, next_of.get(f.ln, f.ln + 1))
            else:
                lo, hi = ranges[f.ln]
                span = (lo, hi + 1)
            spans.setdefault(span, set()).add(f.variable)
        for (lo, after), names in sorted(spans.items()):
            i2 = _indent(lines[lo])
            out.append(Insertion(lo, _wrap_directive(i2, f"data copy", sorted(names)), order=1))
            out.append(Insertion(after, [f"{i2}!$acc end data"], order=-2))
    return out


def apply_fix(fut, reports: list[RoutineReport], dry_run: bool = False) -> list[str]:
    """Insert the data directives into the sources (bottom-up per file). Returns a report."""
    from spel.scripts.config import E3SM_SRCROOT

    by_file: dict[Path, list[tuple[object, RoutineReport]]] = {}
    for r in reports:
        sub = fut.subroutine_dict[r.routine]
        p = Path(r.filepath)
        p = p if p.is_absolute() else Path(E3SM_SRCROOT) / p
        by_file.setdefault(p, []).append((sub, r))
    log: list[str] = []
    for path, items in by_file.items():
        lines = path.read_text().split("\n")
        ins: list[Insertion] = []
        for sub, r in items:
            ins += plan_fix(sub, r, lines)
            log.append(f"{path.name}: {r.routine}: create {[n for n in r.locals_needed if n not in r.host_accessed]}"
                       f", copy around device use {r.host_accessed}")
        for x in sorted(ins, key=lambda x: (-x.ln, -x.order)):
            lines[x.ln:x.ln] = x.lines
        if not dry_run:
            path.write_text("\n".join(lines))
    return log
