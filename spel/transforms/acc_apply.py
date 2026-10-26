"""
Port loops to OpenACC: insert the directives `spel accplan` proposes.

For the routines of a call tree (`roots`), every outermost loop the planner
(`acc_plan.propose`) classifies as elementwise or column-parallel, with a clause plan
and no warning, becomes a compute loop:

- `!$acc parallel loop gang vector [collapse(n)] default(present) private(...)` on the
  loop (the planner's clauses, the same rules as `spel acc2omp`), `!$acc loop seq` on
  every loop inside it (column-parallel: all inner loops; elementwise: the loops inside
  the collapsed nest). Each element keeps its order of operations: bit-for-bit.
- `!$acc routine seq` in every routine called (directly or not) from inside a new
  compute loop;
- `!$acc declare create(...)` in the modules of the module variables read there (the
  device data layer then updates them).

A loop stays on the host, and is reported, when it holds a statement a device cannot
run (`write`, `print`, `endrun`/`shr_sys_abort`) or an assignment through a vector
subscript (`x(idx) = ` with an index array: nvfortran writes only part of such a section
in a kernel), or when a routine it calls (directly
or not) already runs kernels, is ported itself, holds such a statement, or cannot be
resolved: a `routine seq` routine must not contain compute loops. Routines that are
both called from a candidate loop and have candidate loops of their own keep theirs;
the caller's loop stays on the host (iterated to a fixpoint).

An interchange loop (`do j` over levels outside `do fc` over columns, the columns
independent) whose two loops are perfectly nested, with inner bounds that don't use the
outer index, is swapped: the column loop becomes the compute loop and the level loop runs
`seq` inside it, so each column keeps its order of operations. Carried and scatter loops,
and interchange nests that are not perfect, need a rewrite and are only reported.
Everything comes from the analysis (records, loops, calls, accesses), the parse tree
and the parsed directives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from spel.transforms.acc_directives import find_directives
from spel.transforms.acc_plan import LoopProposal, propose

AUTO_KINDS = {"elementwise", "column-parallel"}
ABORTS = {"endrun", "shr_sys_abort", "abort", "mpi_abort"}
TAG = "! spel acc-apply"


@dataclass
class Port:
    routine: str  # Subroutine id
    proposal: LoopProposal
    callees: set[str] = field(default_factory=set)  # routine ids called inside the loop, transitively
    blockers: list[str] = field(default_factory=list)
    interchange: bool = False  # swap the loop with its tight inner loop (see _swappable)

    @property
    def lines(self) -> tuple[int, int]:
        return self.proposal.loop.start_ln, self.proposal.loop.end_ln


@dataclass
class PortPlan:
    ports: list[Port] = field(default_factory=list)  # loops that become compute loops
    blocked: list[Port] = field(default_factory=list)  # candidates kept on the host, with the reasons
    manual: list[Port] = field(default_factory=list)  # carried / scatter / interchange loops
    routine_seq: set[str] = field(default_factory=set)  # routine ids that get `!$acc routine seq`
    declares: dict[str, set[str]] = field(default_factory=dict)  # module -> module variables to declare
    private_globals: set[str] = field(default_factory=set)  # module variables a generated module cannot update


def _path(sub) -> Path:
    from spel.scripts.config import E3SM_SRCROOT

    p = Path(sub.filepath)
    return p if p.is_absolute() else Path(E3SM_SRCROOT) / p


def call_tree(fut, roots: Iterable[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    stack = [r for r in roots]
    while stack:
        sid = stack.pop()
        if sid in out or sid not in fut.subroutine_dict:
            continue
        sub = fut.subroutine_dict[sid]
        out[sid] = sub
        stack.extend(ch.id for ch in getattr(sub, "child_subroutines", {}).values())
    return out


def abort_calls(stmts) -> list[int]:
    """Lines of `call endrun(...)` & co. in a statement list (parse tree). The analysis drops
    these calls from its edited sources, so a routine's record does not list them."""
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.scripts.registrations import iter_statements

    out = []
    for st, _ in iter_statements(stmts):
        if isinstance(st, A.SubCallStatement):
            f = st.function.function if isinstance(st.function, A.FuncExpression) else st.function
            if str(f).split("%")[-1].lower() in ABORTS:
                out.append(st.lineno)
    return out


def cold_branches(stmts) -> list[tuple[int, int, int]]:
    """
    (first line, last line, line of the first statement) of each branch of the outermost
    if constructs that can abort (error checks): host code there runs only when a check fails.
    """
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.scripts.registrations import iter_statements

    out: list[tuple[int, int, int]] = []
    done_to = -1
    for st, _ in iter_statements(stmts):
        if st.lineno <= done_to or not isinstance(st, A.IfConstruct):
            continue
        branches = [st.consequence] + [e.consequence for e in st.else_ifs]
        headers = [e.lineno for e in st.else_ifs]
        if st.else_ is not None:
            branches.append(st.else_.alternative)
            headers.append(st.else_.lineno)
        if not abort_calls(_Block(branches)):
            continue
        done_to = st.construct_end_ln
        for b, nxt in zip(branches, headers + [st.construct_end_ln]):
            if b.statements:
                out.append((b.statements[0].lineno, nxt - 1, b.statements[0].lineno))
    return out


class _Block:
    """Several blocks as one statement list (for iter_statements)."""

    def __init__(self, blocks):
        self.statements = [x for b in blocks for x in b.statements]


def routine_tree(sub):
    """
    A routine's SubroutineDefinitionConstruct with original line numbers: from the
    preprocessed module (module_spec.parse_modules, cached), so cpp blocks inside a
    statement parse, else parsed from the raw lines (raises when that fails too).
    """
    from spel.scripts.fortran_parser.module_spec import ModuleSpecError, parse_modules
    from spel.scripts.registrations import parse_routine

    name = sub.id.split("::")[-1].lower()
    try:
        found = [p for m in parse_modules(_path(sub)) for p in m.subprograms
                 if p.name.lower() == name and p.lineno == sub.startline]
    except (ModuleSpecError, OSError):
        found = []
    return found[0] if found else parse_routine(_path(sub), sub.startline, sub.endline)


class _Facts:
    """Per-routine facts from the analysis and the parse tree (cached)."""

    def __init__(self, fut):
        self.fut = fut
        self._lines: dict[str, list[str]] = {}
        self._unsafe: dict[str, list[int]] = {}
        self._compute: dict[str, bool] = {}
        self._trees: dict[str, object] = {}

    def lines(self, sub) -> list[str]:
        key = str(_path(sub))
        if key not in self._lines:
            self._lines[key] = _path(sub).read_text(errors="replace").split("\n")
        return self._lines[key]

    def tree(self, sub):
        """The routine's parse tree, from the preprocessed file (cpp blocks inside statements,
        e.g. an `#ifdef` in an associate list, parse there), else from the raw lines."""
        if sub.id not in self._trees:
            self._trees[sub.id] = routine_tree(sub)
        return self._trees[sub.id]

    def has_compute(self, sub) -> bool:
        if sub.id not in self._compute:
            ds = find_directives(self.lines(sub), sub.startline, sub.endline)
            self._compute[sub.id] = any(d.is_compute_region or d.is_compute_loop for d in ds)
        return self._compute[sub.id]

    def unsafe_lines(self, sub) -> list[int]:
        """Lines of statements a device cannot run: I/O and aborts (parse tree), aborts (record calls)."""
        if sub.id in self._unsafe:
            return self._unsafe[sub.id]
        from spel.scripts.fortran_parser import spel_ast as A
        from spel.scripts.registrations import iter_statements

        out: list[int] = []
        try:
            tree = self.tree(sub)
            out += [st.lineno for st, _ in iter_statements(tree.body)
                    if isinstance(st, (A.WriteStatement, A.PrintStatement))]
            out += abort_calls(tree.body)
        except BaseException:  # the parser exits on what it cannot read: then every line is suspect
            out = list(range(sub.startline, sub.endline + 1))
        rec = sub.record
        if rec is not None:
            out += [c.ln for c in rec.calls if str(c.name).split("%")[-1].lower() in ABORTS]
        self._unsafe[sub.id] = sorted(set(out))
        return self._unsafe[sub.id]

    def vector_subscript_lines(self, sub) -> list[int]:
        """
        Lines of assignments through a vector subscript (`a(idx) = ` with an array `idx`,
        an associate name of a section, or a section `filter(1:n)` as the subscript):
        nvfortran writes only part of such a section in a kernel (the blocked SetValues
        loops). Plain sections (`a(c,:) = `, `a(c,1:n) = `) are fine: the iteration's
        thread runs the whole statement, and the planner treats the `:` as not covering
        the loop indices.
        """
        from spel.scripts.fortran_parser import spel_ast as A
        from spel.scripts.registrations import iter_statements

        try:
            tree = self.tree(sub)
        except BaseException:
            return list(range(sub.startline, sub.endline + 1))
        rec = sub.record

        def is_section(e) -> bool:
            return isinstance(e, A.FuncExpression) and any(isinstance(a, A.BoundsExpression) for a in e.args)

        arrays = {n for st, _ in iter_statements(tree.body) if isinstance(st, A.AssociateConstruct)
                  for n, e in st.associations.items() if is_section(e)}
        arrays |= {n.lower() for n, d in rec.declarations.items() if d.dim > 0} if rec is not None else set()

        def refs(e):
            if isinstance(e, A.FieldAccessExpression):
                yield from refs(e.left)
                yield from refs(e.field)
            elif isinstance(e, A.FuncExpression):
                yield e
                yield from refs(e.function)

        def vector(arg) -> bool:
            if isinstance(arg, A.Identifier):
                return arg.value.lower() in arrays
            return is_section(arg)

        out = []
        for st, _ in iter_statements(tree.body):
            lhs = None
            if isinstance(st, A.AssignmentStatement):
                lhs = st.left
            elif isinstance(st, A.ExpressionStatement) and isinstance(st.expression, A.InfixExpression) \
                    and st.expression.operator.strip() == "=":
                lhs = st.expression.left_expr
            if lhs is not None and any(vector(a) for f in refs(lhs) for a in f.args):
                out.append(st.lineno)
        return out

    def callees_in(self, sub, lo: int, hi: int) -> tuple[set[str], list[str]]:
        """Routine ids called on lines lo..hi (calls and non-intrinsic function references), unresolved names."""
        rec = sub.record
        names = [(str(c.name), c.ln) for c in rec.calls] + [(f.name, f.ln) for f in rec.function_calls if not f.intrinsic]
        children = getattr(sub, "child_subroutines", {})
        by_base: dict[str, list[str]] = {}
        for key, ch in children.items():
            for n in {key.split("::")[-1].lower(), ch.name.lower()}:
                by_base.setdefault(n, []).append(ch.id)
        found, missing = set(), []
        for name, ln in names:
            if not lo <= ln <= hi:
                continue
            base = name.split("%")[-1].lower()
            if base in ABORTS:
                continue
            ids = by_base.get(base)
            if not ids and "%" not in name:
                # module procedures (functions are not in child_subroutines): the scope knows them
                sym = rec.lookup(base)
                target = getattr(sym, "target", None) if sym is not None else None
                if target and target in self.fut.subroutine_dict:
                    ids = [target]
            if ids:
                found.update(ids)
            else:
                missing.append(name)
        return found, missing


def _closure(facts: _Facts, start: set[str]) -> tuple[set[str], list[str]]:
    """Every routine reached from `start`, and the unresolved calls on the way."""
    seen: set[str] = set()
    missing: list[str] = []
    stack = list(start)
    subs = facts.fut.subroutine_dict
    while stack:
        sid = stack.pop()
        if sid in seen or sid not in subs:
            continue
        seen.add(sid)
        s = subs[sid]
        if s.record is None:
            missing.append(f"{sid} (not analyzed)")
            continue
        ids, miss = facts.callees_in(s, s.startline, s.endline)
        missing += [f"{m} (in {sid})" for m in miss]
        stack.extend(ids)
    return seen, missing


def _swappable(loop) -> bool:
    """A perfect two-loop nest whose inner bounds don't use the outer index (one-line headers)."""
    inner = loop.tight_child
    if inner is None or loop.index is None or inner.index is None:
        return False
    if loop.index in {b.split("%")[0].lower() for b in inner.bound_paths}:
        return False
    return loop.node.lineno == loop.start_ln and inner.node.lineno == inner.start_ln


def plan(fut, roots: Iterable[str]) -> PortPlan:
    from spel.scripts.record_access import AccessMapper

    facts = _Facts(fut)
    mapper = AccessMapper(fut.subroutine_dict)
    tree = call_tree(fut, roots)
    out = PortPlan()
    candidates: list[Port] = []

    def visit(sub, props: list[LoopProposal]) -> None:
        for p in props:
            if p.kind == "has-directive":
                continue
            if p.kind == "carried" and p.children:
                out.manual.append(Port(sub.id, p, blockers=["carried (inner loops proposed separately)"]))
                visit(sub, p.children)
                continue
            port = Port(sub.id, p)
            if p.kind == "interchange" and _swappable(p.loop):
                port.interchange = True
            elif p.kind not in AUTO_KINDS:
                port.blockers.append(f"{p.kind} {p.reason}".strip())
                out.manual.append(port)
                continue
            if p.plan is None or any(n.level == "warn" for n in p.notes):
                port.blockers += [f"planner: {n.msg}" for n in p.notes if n.level == "warn"] or ["no clause plan"]
            lo, hi = port.lines
            if any(lo <= ln <= hi for ln in facts.unsafe_lines(sub)):
                port.blockers.append("device-unsafe statement (write/print/abort) in the loop")
            if any(lo <= ln <= hi for ln in facts.vector_subscript_lines(sub)):
                port.blockers.append("assignment through a vector subscript in the loop")
            direct, missing = facts.callees_in(sub, lo, hi)
            port.blockers += [f"unresolved call {m}" for m in missing]
            port.callees, deep_missing = _closure(facts, direct)
            port.blockers += [f"unresolved call {m}" for m in deep_missing]
            for cid in sorted(port.callees):
                c = fut.subroutine_dict.get(cid)
                if c is None:
                    continue
                if facts.has_compute(c):
                    port.blockers.append(f"calls {cid}, which runs kernels")
                if facts.unsafe_lines(c):
                    port.blockers.append(f"calls {cid}, which has device-unsafe statements")
            candidates.append(port)

    for sid, sub in sorted(tree.items()):
        if sub.record is None or not getattr(sub.record, "loop_nest", None):
            continue
        visit(sub, propose(sub, mapper, source=facts.lines(sub), min_lines=1))

    # a routine called from inside a compute loop must not contain compute loops itself
    changed = True
    while changed:
        changed = False
        ported_routines = {p.routine for p in candidates if not p.blockers}
        for p in candidates:
            if p.blockers:
                continue
            clash = sorted(p.callees & ported_routines)
            if clash:
                p.blockers.append(f"calls {', '.join(clash)}, whose own loops are ported")
                changed = True
    out.ports = [p for p in candidates if not p.blockers]
    out.blocked = [p for p in candidates if p.blockers]
    out.routine_seq = set().union(*[p.callees for p in out.ports]) if out.ports else set()

    # module variables read in the new compute loops and the routine-seq routines
    from spel.scripts.fortran_parser.module_spec import ModuleSpecError, module_spec

    def globals_in(sub, lo: int, hi: int) -> set[tuple[str, str]]:
        from spel.scripts.fortran_parser.symbols import Origin

        found = set()
        agv = {k.lower(): v for k, v in getattr(sub, "active_global_vars", {}).items()}
        for a in sub.record.accesses(origin=Origin.GLOBAL):
            if lo <= a.ln <= hi:
                v = agv.get(a.base.lower())
                if v is not None and not getattr(v, "parameter", False) and str(v.type).lower() not in fut.type_dict:
                    found.add((str(v.declaration).lower(), a.base.lower()))
        return found

    wanted: set[tuple[str, str]] = set()
    for p in out.ports:
        wanted |= globals_in(fut.subroutine_dict[p.routine], *p.lines)
    for cid in out.routine_seq:
        c = fut.subroutine_dict[cid]
        wanted |= globals_in(c, c.startline, c.endline)
    for mod, name in sorted(wanted):
        m = fut.module_dict.get(mod)
        if m is None:
            continue
        try:
            spec = module_spec(_path(m))
        except ModuleSpecError:
            continue
        if name in spec.declared:
            continue
        info = spec.variables.get(name)
        if info is None or "parameter" in info.attrs:
            continue
        out.declares.setdefault(mod, set()).add(name)
        if info.access != "public":
            out.private_globals.add(f"{mod}::{name}")
    return out


# --- applying ---------------------------------------------------------------------------


@dataclass
class Insertion:
    ln: int
    lines: list[str]
    order: int = 0


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _descendants(loop):
    for ch in loop.children:
        yield ch
        yield from _descendants(ch)


def _swap(p: Port, lines: list[str]) -> None:
    """Interchange the loop with its tight inner loop: exchange the `do` lines and the `end do` lines."""
    outer = p.proposal.loop
    inner = outer.tight_child
    for a, b in ((outer.start_ln, inner.start_ln), (outer.end_ln, inner.end_ln)):
        lines[a], lines[b] = _indent(lines[a]) + lines[b].lstrip(), _indent(lines[b]) + lines[a].lstrip()


def _loop_edits(p: Port, lines: list[str]) -> list[Insertion]:
    prop = p.proposal
    directive = [l for l in prop.directive_lines() if "loop seq" not in l]
    out = [Insertion(prop.loop.start_ln, directive)]
    if p.interchange:  # the inner loop (now the level loop) and everything in it run seq
        inner = prop.loop.tight_child
        return out + [Insertion(lp.start_ln, [f"{_indent(lines[lp.start_ln])}!$acc loop seq"])
                      for lp in inner.walk()]
    inner = _descendants(prop.nest[-1]) if prop.kind == "elementwise" else _descendants(prop.loop)
    for lp in inner:
        out.append(Insertion(lp.start_ln, [f"{_indent(lines[lp.start_ln])}!$acc loop seq"]))
    return out


def _routine_seq_edit(sub, lines: list[str]) -> Optional[Insertion]:
    """`!$acc routine seq` before the first executable statement (after the declarations)."""
    from spel.scripts.fortran_parser import spel_ast as A
    ds = find_directives(lines, sub.startline, sub.endline)
    if any(d.construct == "routine" for d in ds):
        return None
    from spel.transforms.acc_locals import first_executable

    tree = routine_tree(sub)
    first = first_executable(tree, sub.record)
    return Insertion(first, [f"{_indent(lines[first])}!$acc routine seq  {TAG}"], order=-5)


def _declare_edit(path: Path, names: set[str], lines: list[str]) -> Insertion:
    from spel.scripts.fortran_parser.module_spec import module_spec

    spec = module_spec(path)
    at = spec.contains_ln if spec.contains_ln is not None else spec.end_ln
    body = ", ".join(sorted(names))
    return Insertion(at, [f"  {TAG}: module data read in device code (updated by the device data layer)",
                          f"  !$acc declare create({body})"])


def apply(fut, pp: PortPlan, dry_run: bool = False) -> list[str]:
    """Insert the directives of a PortPlan into the sources (bottom-up per file)."""
    by_file: dict[Path, list[Insertion]] = {}
    texts: dict[Path, list[str]] = {}

    def lines_of(path: Path) -> list[str]:
        if path not in texts:
            texts[path] = path.read_text().split("\n")
        return texts[path]

    log: list[str] = []
    for p in pp.ports:
        sub = fut.subroutine_dict[p.routine]
        path = _path(sub)
        if p.interchange:  # same line count: the insertion lines stay valid
            _swap(p, lines_of(path))
        by_file.setdefault(path, []).extend(_loop_edits(p, lines_of(path)))
    for cid in sorted(pp.routine_seq):
        sub = fut.subroutine_dict[cid]
        path = _path(sub)
        e = _routine_seq_edit(sub, lines_of(path))
        if e is not None:
            by_file.setdefault(path, []).append(e)
    for mod, names in sorted(pp.declares.items()):
        path = _path(fut.module_dict[mod])
        by_file.setdefault(path, []).append(_declare_edit(path, names, lines_of(path)))
    for path, ins in by_file.items():
        lines = lines_of(path)
        for x in sorted(ins, key=lambda x: (-x.ln, -x.order)):
            lines[x.ln:x.ln] = x.lines
        log.append(f"{path.name}: {len(ins)} insertions")
        if not dry_run:
            path.write_text("\n".join(lines))
    return log


def summary(pp: PortPlan) -> list[str]:
    from collections import Counter

    out = [f"ported loops: {len(pp.ports)} ({dict(Counter(p.proposal.kind for p in pp.ports))}, "
           f"{sum(p.interchange for p in pp.ports)} interchanged) in "
           f"{len({p.routine for p in pp.ports})} routines",
           f"routine seq: {len(pp.routine_seq)} routines",
           f"declares: {sum(len(v) for v in pp.declares.values())} module variables in {len(pp.declares)} modules"
           + (f" ({len(pp.private_globals)} private: {', '.join(sorted(pp.private_globals))})" if pp.private_globals else ""),
           f"blocked (stay on the host): {len(pp.blocked)}",
           f"manual (carried/scatter/interchange): {len(pp.manual)}"]
    reasons = Counter(b.split(",")[0].split(" (")[0] for p in pp.blocked for b in p.blockers)
    out += [f"  {n:4d} x {r}" for r, n in reasons.most_common(12)]
    return out
