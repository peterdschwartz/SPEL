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
from typing import Callable, Iterable, Optional

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
    notes: list[str] = field(default_factory=list)  # what the plan could not determine
    declares: dict[str, set[str]] = field(default_factory=dict)  # module -> module variables to declare
    private_globals: set[str] = field(default_factory=set)  # module variables a generated module cannot update


def _path(sub) -> Path:
    from spel.scripts.config import E3SM_SRCROOT

    p = Path(sub.filepath)
    return p if p.is_absolute() else Path(E3SM_SRCROOT) / p


# character intrinsics with an allocated result: no device version in nvfortran
STRING_INTRINSICS = {"trim", "adjustl", "adjustr", "repeat"}
# edit_files' comment on an assignment whose right-hand side used an excluded module
EXCLUDED_MARKER = "the right-hand side used an excluded module"
# names the edit pass keeps for the unit tests although their module is excluded, and that a device
# cannot run: shr_infnan's nan/inf are derived-type constants assigned through an overloaded `=`
# (set_nan_0d_double; UpdateDaylength's `daylength = nan` failed to compile as a device routine)
HOST_ONLY_NAMES = {"nan", "inf", "shr_infnan_nan", "shr_infnan_posinf", "shr_infnan_neginf"}


def call_tree(fut, roots: Iterable[str],
              exclude: Optional[Callable[[str], bool]] = None) -> dict[str, object]:
    """The routines reachable from `roots` through `child_subroutines`.

    `exclude(routine_id)` cuts the tree: an excluded routine is left out together with
    everything reachable only through it. A routine kept on the host keeps its callees on
    the host: a kernel in such a callee finds the host routine's data absent (2026-10-29:
    template_col_from_natveg_array, called only by the host set_old_column_weights)."""
    out: dict[str, object] = {}
    stack = [r for r in roots]
    while stack:
        sid = stack.pop()
        if sid in out or sid not in fut.subroutine_dict:
            continue
        if exclude is not None and exclude(sid):
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
        self._device: dict[str, bool] = {}
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

    def is_device_routine(self, sub) -> bool:
        """`!$acc routine ...`: its loops run inside its callers' kernels (gang/vector/seq)."""
        if sub.id not in self._device:
            ds = find_directives(self.lines(sub), sub.startline, sub.endline)
            self._device[sub.id] = any(d.construct == "routine" for d in ds)
        return self._device[sub.id]

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
        out += self.excluded_lines(sub)
        if rec is not None:  # shr_infnan's `x = nan`: an assignment(=) overload with no device version
            out += [a.ln for a in rec.accesses() if a.path.lower() in HOST_ONLY_NAMES]
        self._unsafe[sub.id] = sorted(set(out))
        return self._unsafe[sub.id]

    def excluded_lines(self, sub) -> list[int]:
        """
        Executable lines the analysis replaced because they use a module it excludes (`!#py`
        statements, `lhs = lhs` stand-ins of edit_files): the source runs code the device has
        no version of (shr_infnan's `nan` assignment, a stream advance, a time query).
        UpdateDaylength's `daylength = nan` was ported as a kernel and failed to compile.
        """
        from spel.scripts.fortran_parser.scope_walk import AssignEvent
        from spel.transforms.acc_locals import first_executable

        lines = getattr(sub, "sub_lines", None) or []
        try:
            first = first_executable(self.tree(sub), sub.record)
        except BaseException:
            first = sub.startline
        out = {lt.ln for lt in lines
               if lt.ln >= first and ("!#py" in lt.line or EXCLUDED_MARKER in lt.line)}
        # the record knows the stand-ins (`sub_lines` drops the marker comment: pdep_interp's
        # stream read was ported as a kernel, with its running index as a sum reduction)
        if sub.record is not None:
            out |= {ev.ln for ev in sub.record.events
                    if isinstance(ev, AssignEvent) and getattr(ev, "stand_in", False)}
        return sorted(out)

    def string_intrinsic_lines(self, sub) -> list[int]:
        """
        Lines of statements that call a character intrinsic that allocates its result
        (`trim(isotope) == 'bulk'`): nvfortran has no device version of them (S-1058
        pgf90_trima). Each statement's own expressions are searched (conditions included),
        not the statements nested in it.
        """
        from spel.scripts.fortran_parser import spel_ast as A
        from spel.scripts.registrations import iter_statements

        try:
            tree = self.tree(sub)
        except BaseException:
            return list(range(sub.startline, sub.endline + 1))

        def calls(node, top: bool) -> bool:
            if isinstance(node, A.Statement) and not top:
                return False  # a nested statement: reported at its own line
            if isinstance(node, A.FuncExpression) and isinstance(node.function, A.Identifier) \
                    and str(node.function.value).lower() in STRING_INTRINSICS:
                return True
            if isinstance(node, A.Node):
                return any(calls(v, False) for v in vars(node).values())
            if isinstance(node, (list, tuple)):
                return any(calls(v, False) for v in node)
            return False

        return sorted({st.lineno for st, _ in iter_statements(tree.body) if calls(st, True)})

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


def reach_conditions(fut, root: str = "elm_driver::elm_drv") -> dict[str, object]:
    """Routine id -> namelist condition under which `root`'s call tree reaches it: the or over
    its call sites of (the caller's condition and the namelist if enclosing the call), as the
    device data layer's guards are composed (`nml_guards`). Routines the tree does not reach
    are absent (callers treat that as "always")."""
    from spel.scripts.fortran_parser.boolen_expression import AllOf, AnyOf, simplify

    always = AllOf(())
    if root not in fut.subroutine_dict:
        return {}
    callers: dict[str, list[tuple[str, int]]] = {}
    seen: set[str] = set()
    stack = [root]
    while stack:
        sid = stack.pop()
        if sid in seen:
            continue
        seen.add(sid)
        sub = fut.subroutine_dict[sid]
        for ln, desc in getattr(sub, "sub_call_desc", {}).items():
            child = getattr(sub, "child_subroutines", {}).get(desc.fn)
            if child is None or child.id not in fut.subroutine_dict:
                continue
            callers.setdefault(child.id, []).append((sid, ln))
            stack.append(child.id)
    memo: dict[str, object] = {root: always}

    def reach(sid: str, active: frozenset) -> object:
        if sid in memo:
            return memo[sid]
        conds = []
        for caller, ln in callers.get(sid, ()):
            if caller in active:
                continue
            if_ = fut.subroutine_dict[caller].namelist_if_at(ln)
            local = if_.condtional_expectation if if_ is not None else always
            conds.append(AllOf((reach(caller, active | {sid}), local)))
        out = simplify(AnyOf(tuple(dict.fromkeys(conds)))) if conds else always
        memo[sid] = out
        return out

    for sid in seen:
        reach(sid, frozenset())
    return memo


def implies(premise, conclusion, max_atoms: int = 14) -> bool:
    """premise -> conclusion for every assignment of their namelist atoms (truth table);
    False, i.e. "not known to hold", when there are too many atoms."""
    from itertools import product

    from spel.scripts.fortran_parser.boolen_expression import _atoms, _evaluate

    atoms = list(_atoms(conclusion, _atoms(premise, {})))
    if len(atoms) > max_atoms:
        return False
    for vals in product((True, False), repeat=len(atoms)):
        v = dict(zip(atoms, vals))
        if _evaluate(premise, v) and not _evaluate(conclusion, v):
            return False
    return True


def kernel_condition(sub, ln: int, reach: dict[str, object]):
    """The namelist condition under which the loop at line `ln` (0-based) of `sub` runs: how
    elm_drv's tree reaches the routine, and the namelist if enclosing the loop."""
    from spel.scripts.fortran_parser.boolen_expression import AllOf

    always = AllOf(())
    if_at = getattr(sub, "namelist_if_at", None)
    if_ = if_at(ln) if if_at is not None else None
    local = if_.condtional_expectation if if_ is not None else always
    return AllOf((reach.get(sub.id, always), local))


def guarded_instances(sub, lo: int, hi: int, instance_conds: dict[str, object], kernel_cond) -> list[str]:
    """Global derived-type instances accessed in lines lo..hi (0-based) whose device copy exists
    only under a namelist condition (`instance_conds`, the device data layer's guards: c14_grc_cs
    under use_c14) that `kernel_cond` (see `kernel_condition`) does not imply. A `default(present)`
    kernel that names such an instance aborts at launch when the copy is absent, even if the
    access sits in an `if (use_c14)` inside the loop (2026-10-29: CarbonStateUpdateDynPatch).
    They get `no_create`: the host address when absent, never dereferenced because the same flag
    guards the statements."""
    from spel.scripts.fortran_parser.symbols import Origin

    out: list[str] = []
    for a in sub.record.accesses(origin=Origin.GLOBAL):
        if not lo <= a.ln <= hi:
            continue
        inst = a.base.lower()
        cond = instance_conds.get(inst)
        if cond is None or inst in out or implies(kernel_cond, cond):
            continue
        out.append(inst)
    return sorted(out)


def _swappable(loop) -> bool:
    """A perfect two-loop nest whose inner bounds don't use the outer index (one-line headers)."""
    inner = loop.tight_child
    if inner is None or loop.index is None or inner.index is None:
        return False
    if loop.index in {b.split("%")[0].lower() for b in inner.bound_paths}:
        return False
    return loop.node.lineno == loop.start_ln and inner.node.lineno == inner.start_ln


def _nest(prop):
    """The perfect nest an interchange reorders (as many loops as `order` names), or None."""
    try:
        nest = prop.loop.perfect_nest(len(prop.order))
    except ValueError:
        return None
    if sorted(lp.index for lp in nest) != sorted(prop.order):
        return None
    return nest


def _reorderable(prop) -> bool:
    """The planner's interchange can be applied: a perfect nest with one-line `do` headers
    (the planner checked the bounds: the two-loop swap here, `_parallel_first` for a reorder)."""
    nest = _nest(prop)
    if nest is None:
        return False
    if len(nest) == 2 and not _swappable(prop.loop):
        return False
    return all(lp.node.lineno == lp.start_ln for lp in nest)


def plan(fut, roots: Iterable[str], exclude: Optional[Callable[[str], bool]] = None) -> PortPlan:
    """Compute-loop ports for the call trees of `roots`. `exclude` keeps routines, and
    whatever is reachable only through them, off the device (see `call_tree`)."""
    from spel.scripts.record_access import AccessMapper

    facts = _Facts(fut)
    mapper = AccessMapper(fut.subroutine_dict)
    tree = call_tree(fut, roots, exclude)
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
            if p.kind == "interchange" and _reorderable(p):
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
            if any(lo <= ln <= hi for ln in facts.string_intrinsic_lines(sub)):
                port.blockers.append("character intrinsic (trim/adjustl/...) in the loop: hoist it")
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
        if facts.is_device_routine(sub):
            continue  # already a device routine (e.g. `routine vector`): no compute constructs inside
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

    # instances the device data layer copies only under a namelist flag: `no_create` on the kernels
    # that name them outside that flag's ifs
    try:
        from spel.transforms.device.generate import access_guards

        guards = access_guards(fut, fut.type_dict)
        out.notes += guards.notes
    except Exception as exc:  # the stub analyses of tests have no elm_drv
        guards = None
        out.notes.append(f"no device-layer guards ({exc}): no_create clauses not planned")
    if guards is not None and guards.instance_conds:
        reach = reach_conditions(fut)
        for p in out.ports:
            sub = fut.subroutine_dict[p.routine]
            p.proposal.no_create = guarded_instances(sub, *p.lines, guards.instance_conds,
                                                     kernel_condition(sub, p.lines[0], reach))

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
    """Reorder the perfect nest to `order`: position k gets the `do` line (and the `end do`
    line) of the loop over order[k], with position k's indentation."""
    nest = _nest(p.proposal)
    by_index = {lp.index: lp for lp in nest}
    heads = {lp.index: lines[lp.start_ln].lstrip() for lp in nest}
    ends = {lp.index: lines[lp.end_ln].lstrip() for lp in nest}
    for pos, idx in zip(nest, p.proposal.order):
        lines[pos.start_ln] = _indent(lines[pos.start_ln]) + heads[idx]
        lines[pos.end_ln] = _indent(lines[pos.end_ln]) + ends[idx]
    assert len(by_index) == len(nest)


def _loop_edits(p: Port, lines: list[str]) -> list[Insertion]:
    prop = p.proposal
    directive = [l for l in prop.directive_lines() if "loop seq" not in l]
    out = [Insertion(prop.loop.start_ln, directive)]
    if p.interchange:  # the positions after the parallel ones, and every loop inside the nest, run seq
        nest = _nest(prop)
        seq = nest[prop.npar:] + [lp for ch in nest[-1].children for lp in ch.walk()]
        return out + [Insertion(lp.start_ln, [f"{_indent(lines[lp.start_ln])}!$acc loop seq"]) for lp in seq]
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
           f"no_create (namelist-guarded instances): {sum(1 for p in pp.ports if p.proposal.no_create)} loops",
           f"declares: {sum(len(v) for v in pp.declares.values())} module variables in {len(pp.declares)} modules"
           + (f" ({len(pp.private_globals)} private: {', '.join(sorted(pp.private_globals))})" if pp.private_globals else ""),
           f"blocked (stay on the host): {len(pp.blocked)}",
           f"manual (carried/scatter/interchange): {len(pp.manual)}"]
    reasons = Counter(b.split(",")[0].split(" (")[0] for p in pp.blocked for b in p.blockers)
    out += [f"  {n:4d} x {r}" for r, n in reasons.most_common(12)]
    return out
