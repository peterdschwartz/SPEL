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

from spel.transforms.core.source import statement_end

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from spel.transforms.acc_directives import AccDirective, find_directives
from spel.transforms.core.edits import Edit, stacked
from spel.transforms.core.fortran import directive_list
from spel.transforms.core.registry import transform
from spel.transforms.core.source import indent
from spel.transforms.core.transform import Change, Plan, Transform
from spel.transforms.core.query import above_directives as _above_directives  # noqa: F401
from spel.transforms.core.query import compute_ranges as _compute_ranges, first_executable

if TYPE_CHECKING:
    from spel.scripts.analyze_subroutines import Subroutine
    from spel.scripts.functional_unit_test import FunctionalUnitTest

# clauses that give a variable device storage for the construct
DATA_CLAUSES = {
    "copy", "copyin", "copyout", "create", "present", "deviceptr", "attach", "no_create",
    "present_or_copy", "present_or_copyin", "present_or_copyout", "present_or_create",
    "pcopy", "pcopyin", "pcopyout", "pcreate",
}
# clauses that make a variable private to the construct (no device storage needed)
PRIVATE_CLAUSES = {"private", "firstprivate", "reduction"}

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
    # of the device-only ones, those the host initializes in the specification part (an
    # initializer or a DATA statement) and never touches again: `enter data copyin`, since
    # `create` would hand the device an unfilled copy (SnowHydrology's dzmin, 2026-10-30)
    host_initialized: list[str] = field(default_factory=list)
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


def check_routine(sub: Subroutine, device_routines: set[str], lines: list[str],
                  internal_uses: Optional[dict[str, set[str]]] = None) -> RoutineReport:
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
    call_desc = sub.sub_call_desc or {}
    for call in rec.calls:
        callee = str(call.name).split("%")[-1].lower()
        # a generic name (`c2g`) runs kernels when the specific procedure it resolves to does
        desc = call_desc.get(call.ln)
        specific = str(desc.fn or "").split("::")[-1].lower() if desc is not None else ""
        if callee not in device_routines and specific not in device_routines:
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
    from spel.transforms.core.query import cold_branches, routine_tree

    try:
        cold = cold_branches(routine_tree(sub).body)
    except BaseException:  # the parser exits on what it cannot read: no exemption
        cold = []
    host: set[str] = set()
    initialized: set[str] = set()
    cold_reads: dict[str, set[int]] = {}
    for a in accesses:
        name = a.base.lower()
        if name not in needed or any(lo <= a.ln <= hi for lo, hi in device_lines):
            continue
        if a.context in ("init", "data"):
            initialized.add(name)  # the specification part: the value the device copy must start with
            continue
        branch = next((b for b in cold if b[0] <= a.ln <= b[1]), None)
        if branch is not None and "w" not in a.status:
            cold_reads.setdefault(name, set()).add(branch[2])
        else:
            host.add(name)
    # internal subprograms (internal_uses: name -> host variables they use) run on the host:
    # their accesses happen at the host-side call
    for call in rec.calls:
        uses = (internal_uses or {}).get(str(call.name).lower())
        if uses and not any(lo <= call.ln <= hi for lo, hi in device_lines):
            host |= uses & set(needed)
    report.host_accessed = [n for n in report.locals_needed if n in host]
    report.host_initialized = [n for n in report.locals_needed if n in initialized and n not in host]
    report.cold_reads = {n: sorted(v) for n, v in cold_reads.items() if n not in host}
    return report


def _internal_uses(fut: FunctionalUnitTest, sub: Subroutine) -> dict[str, set[str]]:
    """The host variables each internal subprogram of `sub` uses (from its record)."""
    from spel.scripts.fortran_parser.symbols import Origin

    out: dict[str, set[str]] = {}
    for isub in fut.subroutine_dict.values():
        if isub.is_internal and isub.host is sub and isub.record is not None:
            out[isub.name.split("::")[-1].lower()] = {
                a.base.lower() for a in isub.record.accesses() if a.origin is Origin.HOST}
    return out


def check(fut: FunctionalUnitTest, routines: Optional[set[str]] = None) -> list[RoutineReport]:
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
        r = check_routine(sub, device, texts[str(path)], _internal_uses(fut, sub))
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


_indent = indent
_wrap_directive = directive_list  # `!$acc head(a, b, ...)` with `!$acc&` continuations


def plan_fix(sub: Subroutine, report: RoutineReport, lines: list[str], structured: bool = False,
             notes: Optional[list[str]] = None) -> list[Insertion]:
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
    first_exec = _above_directives(first_executable(tree, rec), find_directives(lines, sub.startline, sub.endline))
    ind = _indent(lines[first_exec])
    out: list[Insertion] = []
    def allocatable(n: str) -> bool:
        d = rec.declaration(n)
        return d is not None and "allocatable" in {a.lower() for a in d.attrs}

    if structured:
        resident_shared: list[str] = []
    else:
        # no structured data regions (user rule 2026-11-03): locals the host also uses are resident
        # too; acc-sync adds the update device/self at the host<->device transitions. Allocatables
        # cannot be created before their allocate: they keep a copy region (reported)
        resident_shared = [n for n in report.host_accessed if n in report.locals_needed and not allocatable(n)]
        kept = [n for n in report.host_accessed if n in report.locals_needed and allocatable(n)]
        if kept and notes is not None:
            notes.append(f"{sub.id}: allocatable host-shared locals keep a data copy region: {', '.join(kept)}")
    device_only = [n for n in report.locals_needed if n not in report.host_accessed] + resident_shared
    created = [n for n in device_only if n not in report.host_initialized]
    initialized = [n for n in device_only if n in report.host_initialized]
    if device_only:
        block = [f"{ind}{TAG}: device storage of the local work arrays"]
        if created:
            block += _wrap_directive(ind, "enter data create", created)
        if initialized:
            block += [f"{ind}{TAG}: initialized in the specification part: the device copy starts from the host values"]
            block += _wrap_directive(ind, "enter data copyin", initialized)
        out.append(Insertion(first_exec, block))
    at: dict[int, list[str]] = {}
    for name in [n for n in device_only if n not in resident_shared]:
        for ln in report.cold_reads.get(name, []):
            at.setdefault(ln, []).append(name)
    sub_directives = find_directives(lines, sub.startline, sub.endline)
    for ln, names in sorted(at.items()):
        # above the directives attached to the statement (a `parallel loop` and its loop stay together)
        at_ln = _above_directives(ln, sub_directives)
        out.append(Insertion(at_ln, _wrap_directive(_indent(lines[ln]), "update self", names), order=1))
    to_delete = device_only + [n for n in report.leaked if n not in device_only]
    if to_delete:
        # the end of the executable part: the routine's `contains` when it has internal subprograms
        last = tree.contains_ln if tree.contains_ln >= 0 else tree.end_ln
        ends = [st.lineno for st, _ in stmts if isinstance(st, A.ReturnStatement)] + [last]
        for ln in ends:
            out.append(Insertion(ln, _wrap_directive(_indent(lines[ln]) if ln != last else ind,
                                                     "exit data delete", to_delete), order=-1))
    host = set(report.host_accessed) - set(resident_shared)
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
                # right after the call statement (its continuation lines included): the next
                # statement may be in another block (the next `case`)
                span = (f.ln, statement_end(lines, f.ln) + 1)
            else:
                lo, hi = ranges[f.ln]
                span = (lo, hi + 1)
            spans.setdefault(span, set()).add(f.variable)
        for (lo, after), names in sorted(spans.items()):
            i2 = _indent(lines[lo])
            out.append(Insertion(lo, _wrap_directive(i2, f"data copy", sorted(names)), order=1))
            out.append(Insertion(after, [f"{i2}!$acc end data"], order=-2))
    return out


def fix_edits(ins: list[Insertion]) -> list[Edit]:
    """The insertions as edits. Of insertions at one line with the same order, a later one ends
    up above an earlier one (as the fix has always placed them)."""
    return stacked((x.ln, x.lines, x.order) for x in ins)


@dataclass
class Options:
    routines: list[str] = field(default_factory=list,
                                metadata={"help": "routine names or ids (default: every routine with device code)"})
    structured: bool = field(default=False, metadata={
        "help": "locals the host also uses get `data copy` regions around device uses (default: resident, "
                "enter/exit data, the transitions synced by acc-sync)"})


@transform
class AccLocals(Transform):
    """
    Matches: routines whose compute constructs or device calls use local arrays without device
             storage (found by `check`).
    Changes: device-only locals get `enter data create/copyin` after the declarations and
             `exit data delete` before every return; locals the host also uses get a `data copy`
             region around each device use; host reads of device-only locals get `update self`.
             Device-only locals that `data copy` regions move at every kernel (no host use in the
             parsed source but allocate/deallocate and DEVICE_CONSUMERS calls) get `enter data
             create` after their allocate and `exit data delete` before their deallocate, and lose
             the copy regions and updates that name only them.
    """

    name = "acc-locals"
    summary = "give routine-local arrays used by device code device storage"
    Options = Options

    def build(self, ctx, opts):
        routines = set(opts.routines) or None
        if routines is None and ctx.config is not None:
            # only the routines the time step runs in the configuration (not PHS under use_hydrstress = .false.)
            routines = {s.id for s in ctx.routines()}
        plan = fix_plan(ctx, check(ctx.fut, routines), structured=opts.structured)
        # device-only work arrays that copy regions move every launch: create once instead
        for sid, sub in sorted(ctx.fut.subroutine_dict.items()):
            if routines and sid not in routines and sid.split("::")[-1].lower() not in routines:
                continue
            if sub.record is None or sub.library:
                continue
            lines = ctx.lines(sub)
            co = copy_only_locals(sub, lines)
            if co is None:
                continue
            plan.add(Change(ctx.path(sub), copy_only_edits(sub, co, lines), where=sid,
                            why=f"device-only {', '.join(co.names)}: enter data create once, no copy regions"))
        return plan


def fix_plan(ctx, reports: list[RoutineReport], structured: bool = False) -> Plan:
    """The insertions fixing `reports` (from `check`), per file."""
    plan = Plan(AccLocals.name)
    by_file: dict[Path, list] = {}
    for r in reports:
        sub = ctx.fut.subroutine_dict[r.routine]
        by_file.setdefault(ctx.path(sub), []).append((sub, r))
    for path, items in by_file.items():
        lines = ctx.lines(path)
        ins: list[Insertion] = []
        for sub, r in items:
            ins += plan_fix(sub, r, lines, structured=structured, notes=plan.notes)
            plan.notes.append(f"{path.name}: {r.routine}: create "
                              f"{[n for n in r.locals_needed if n not in r.host_accessed and n not in r.host_initialized]}"
                              f", copyin {r.host_initialized}, copy around device use {r.host_accessed}")
        plan.add(Change(path, fix_edits(ins), where=path.name))
    return plan


def apply_fix(fut: FunctionalUnitTest, reports: list[RoutineReport], dry_run: bool = False) -> list[str]:
    """Insert the data directives of `reports` into the sources. Returns a report."""
    from spel.transforms.core.driver import InPlace, apply_plan
    from spel.transforms.core.transform import Context

    ctx = Context(fut)
    plan = fix_plan(ctx, reports)
    apply_plan(plan, ctx, InPlace(), dry_run=dry_run)
    return plan.notes


# --- device-only work arrays held in copy regions ---------------------------------------------
#
# A local work array that only device code reads and writes, but that `data copy` regions put on
# the device around each kernel, moves host -> device -> host at every region (the accumulator
# updates' rbufslp: 200 MB per step, 2026-11-01). It needs one `enter data create` (after its
# allocate) and one `exit data delete` (before its deallocate) instead. The analysis cannot
# decide "device-only" alone: statements using excluded modules (accumulMod) are not in the
# record. So the routine's real source is parsed and every host statement checked.

# Routines that read/write an array argument only on the device when the device is ready (their
# data clauses are present_or_*; the SPEL_ACCUM_CHECK debug check compares with the host copy):
DEVICE_CONSUMERS = {"update_accum_field", "extract_accum_field"}


@dataclass
class CopyOnly:
    routine: str
    names: list[str]
    allocs: dict[str, list[int]]      # name -> lines after which it is allocated (statement ends)
    deallocs: dict[str, list[int]]    # name -> lines of its deallocate statements
    regions: list[tuple[int, int, int]]  # (start, start directive end, end data line) to remove
    updates: list[tuple[int, int]]    # (start, end) of update directives to remove


def copy_only_locals(sub: Subroutine, lines: list[str]) -> Optional[CopyOnly]:
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.scripts.registrations import iter_statements
    from spel.transforms.compact.rewrite import iter_expressions
    from spel.transforms.core.query import routine_tree
    from spel.transforms.core.source import statement_end

    rec = sub.record
    if rec is None:
        return None
    arrays = {n.lower() for n, d in rec.local_variables.items()
              if d.dim > 0 and "parameter" not in {a.lower() for a in d.attrs}}
    directives = find_directives(lines, sub.startline, sub.endline)
    stack, regions = [], []
    for d in directives:
        if d.construct == "data":
            stack.append(d)
        elif d.construct == "end data" and stack:
            regions.append((stack.pop(), d))
    copy_vars: set[str] = set()
    for start, _ in regions:
        copy_vars |= _clause_vars(start, {"copy", "copyin", "copyout", "present_or_copy", "pcopy"})
    other = set()
    for d in directives:
        if d.construct in ("enter data", "declare"):
            other |= _clause_vars(d, DATA_CLAUSES)
    cands = (copy_vars & arrays) - other
    if not cands:
        return None
    device = [(lo, hi) for _, lo, hi in _compute_ranges(directives, rec)]
    try:
        tree = routine_tree(sub)
    except BaseException:  # the parser exits on what it cannot read
        return None

    def refs(node) -> set[str]:
        return {e.value.lower() for e in iter_expressions(node) if isinstance(e, A.Identifier)}

    allocs: dict[str, list[int]] = {}
    deallocs: dict[str, list[int]] = {}
    host_use: set[str] = set()
    containers = (A.DoLoop, A.DoWhile, A.AssociateConstruct, A.BlockStatement, A.MacroIf,
                  A.WhereConstruct, A.SelectCaseConstruct)
    for st, _ in iter_statements(tree.body):
        ln = st.lineno
        if any(lo <= ln <= hi for lo, hi in device):
            continue
        if isinstance(st, A.IfConstruct):
            host_use |= refs(st.condition) & cands
            continue
        if isinstance(st, containers):
            continue
        if isinstance(st, A.AllocateStatement):
            objs = set()
            for o in st.objects:
                base = o.function if isinstance(o, A.FuncExpression) else o
                if isinstance(base, A.Identifier):
                    objs.add(base.value.lower())
            for nm in objs & cands:
                (deallocs if st.is_deallocate else allocs).setdefault(nm, []).append(
                    ln if st.is_deallocate else statement_end(lines, ln))
            host_use |= (refs(st) - objs) & cands  # in a source=/bounds expression
            continue
        if isinstance(st, A.SubCallStatement) and str(st.function.function).lower() in DEVICE_CONSUMERS:
            plain = {a.value.lower() for a in st.function.args if isinstance(a, A.Identifier)}
            host_use |= (refs(st) - plain) & cands
            continue
        host_use |= refs(st) & cands
    names = sorted(cands - host_use)
    if not names:
        return None
    keep = set(names)
    drop_regions = [(s.start_ln, s.end_ln, e.start_ln) for s, e in regions
                    if _clause_vars(s, DATA_CLAUSES) and _clause_vars(s, DATA_CLAUSES) <= keep
                    and all(c.name in ("copy", "copyin", "copyout", "present_or_copy", "pcopy", "if")
                            for c in s.clauses)]
    drop_updates = [(d.start_ln, d.end_ln) for d in directives if d.construct == "update"
                    and _clause_vars(d, {"self", "host", "device"}) and _clause_vars(d, {"self", "host", "device"}) <= keep]
    return CopyOnly(sub.id, names, {n: allocs.get(n, []) for n in names},
                    {n: deallocs.get(n, []) for n in names}, drop_regions, drop_updates)


def copy_only_edits(sub: Subroutine, co: CopyOnly, lines: list[str]) -> list[Edit]:
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.transforms.core.edits import replace
    from spel.transforms.core.query import routine_tree

    ins: list[Insertion] = []
    tree = None
    for n in co.names:
        if co.allocs[n]:
            for ln in co.allocs[n]:
                ins.append(Insertion(ln + 1, _wrap_directive(_indent(lines[ln]), "enter data create", [n]), order=2))
        else:
            tree = tree or routine_tree(sub)
            first = _above_directives(first_executable(tree, sub.record), find_directives(lines, sub.startline, sub.endline))
            ins.append(Insertion(first, _wrap_directive(_indent(lines[first]), "enter data create", [n]), order=2))
        if co.deallocs[n]:
            for ln in co.deallocs[n]:
                ins.append(Insertion(ln, _wrap_directive(_indent(lines[ln]), "exit data delete", [n]), order=-1))
        else:
            from spel.transforms.core.query import routine_tree as rt
            from spel.scripts.registrations import iter_statements

            tree = tree or rt(sub)
            last = tree.contains_ln if tree.contains_ln >= 0 else tree.end_ln
            ends = [st.lineno for st, _ in iter_statements(tree.body) if isinstance(st, A.ReturnStatement)] + [last]
            for ln in ends:
                ins.append(Insertion(ln, _wrap_directive(_indent(lines[ln]), "exit data delete", [n]), order=-1))
    edits = fix_edits(ins)
    for s0, s1, e in co.regions:
        edits.append(replace(s0, [], n=s1 - s0 + 1, expect=lines[s0:s1 + 1], why="copy region of a device-only local"))
        edits.append(replace(e, [], expect=[lines[e]], why="copy region of a device-only local"))
    for s0, s1 in co.updates:
        edits.append(replace(s0, [], n=s1 - s0 + 1, expect=lines[s0:s1 + 1], why="update of a device-only local"))
    return edits
