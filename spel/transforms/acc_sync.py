"""
Host/device coherence for routines that mix host code and device kernels.

The device data layer keeps the device copy authoritative inside a device section of
elm_drv and copies it back at the end. Host statements left inside such a section
(loops not ported yet, setup code, first-step code) read stale host data and write
values the device never sees, and the end-of-step `update self` then overwrites them.

For the routines of a call tree (`roots`) this inserts, around every *host segment* (a
maximal run of consecutive statements of one block that run no kernels and call no
routine that does):

- `!$acc update self(...) if_present` before it: the arrays and scalar members it reads
  or writes that device code in the tree writes;
- `!$acc update device(...) if_present` after it, and before every `return` inside it:
  what it writes.

The headers of constructs that hold kernels (if conditions, do bounds) are evaluated on
the host too: their reads get an `update self` before the construct. An if construct
that can abort (`endrun` in a branch: an error check) is treated the same way even when
it runs no kernels, so its syncs run only when the error branch is taken.

Only intrinsic-typed data is updated (a whole derived-type object would overwrite the
host's pointer components with device addresses): members reached through derived-type
globals and dummies (`col_cf%x`, associate names expanded), acc-declared module
variables, and dummy arrays. Routines called from kernels (`routine seq`) run on the
device and get nothing. Correct, not fast: every port that removes a host segment
removes its syncs. Everything comes from the analysis (records, loops, calls,
accesses), the parse tree and the parsed directives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from spel.transforms.acc_apply import _Facts, _path, abort_calls, call_tree
from spel.transforms.acc_directives import find_directives

TAG = "! spel acc-sync"

# Configuration switches whose branches the GPU target configurations mostly skip (FATES,
# FAN, erosion, VIC, extra snow layers, plant hydraulic stress, PFLOTRAN, the ELM interface,
# carbon isotopes, the ECA/RD nutrient competition). An if construct testing one of them is
# treated like an error check: the syncs of its host code go inside its branches, so they run
# only when a branch is taken (always correct; it only matters for the cost).
OFF_BY_DEFAULT = frozenset({
    "use_fates", "use_fan", "use_erosion", "ero_ccycle", "use_vichydro", "use_extrasnowlayers",
    "use_hydrstress", "use_pflotran", "use_elm_interface", "use_c13", "use_c14", "nu_com",
})


@dataclass
class Segment:
    first: int  # 0-based line of the first statement
    after: int  # insert the update device before this line
    reads: set[str] = field(default_factory=set)
    writes: set[str] = field(default_factory=set)
    stale: set[str] = field(default_factory=set)  # of those, what device code writes: update self first
    returns: list[int] = field(default_factory=list)  # lines of return statements inside
    header_only: bool = False  # the header reads of a construct holding kernels
    skipped: set[str] = field(default_factory=set)  # paths it touches that cannot be updated


@dataclass
class RoutineSync:
    routine: str
    path: Path
    segments: list[Segment] = field(default_factory=list)
    skipped: set[str] = field(default_factory=set)  # paths that cannot be updated (array of derived types, ...)
    uses: dict[str, set[str]] = field(default_factory=dict)  # module -> globals to use-associate (callees' data)
    use_ln: int = -1  # where the added use statements go
    first_exec: int = -1  # the routine's first executable statement (pushes of specification-part writes go after
    #                       the acc-locals `enter data` there, where the device copies exist)


@dataclass
class SyncPlan:
    routines: list[RoutineSync] = field(default_factory=list)
    device_written: set[tuple[str, str]] = field(default_factory=set)
    seq: set[str] = field(default_factory=set)
    # calls, inside routines that run device work, of routines the analysis excludes
    # (edit_files.bad_modules): their effects are unknown, so nothing is synced around them
    notes: list[str] = field(default_factory=list)


class _Types:
    """Resolve a designator path to the type and rank of what it names."""

    def __init__(self, fut):
        self.types = {k.lower(): v for k, v in fut.type_dict.items()}
        # module variables by name, for the accesses callees make on the caller's call
        # lines (the caller need not use those modules itself); ambiguous names left out
        seen: dict[str, list] = {}
        for m in fut.module_dict.values():
            for n, v in getattr(m, "global_vars", {}).items():
                seen.setdefault(n.lower(), []).append(v)
        self.globals = {n: vs[0] for n, vs in seen.items() if len(vs) == 1}
        self.modules = fut.module_dict

    def expand(self, sub, path: str) -> list[str]:
        """
        The variables `path` names: through a pointer component with recorded targets
        (`transport_ptr_list(i)%conc_ptr` -> `col_cs%core%decomp_pools_vr`, ..., from the
        module's `=>` assignments), else `path` itself.
        """
        parts = path.lower().split("%")
        b = self.base(sub, parts[0])
        typ = b[0] if b else None
        for i, p in enumerate(parts[1:], start=1):
            t = self.types.get(typ)
            comp = t.components.get(p) if t is not None else None
            if comp is None:
                break
            targets = getattr(comp, "pointer", None) or []
            if targets:
                rest = "%".join(parts[i + 1:])
                return [x for tgt in targets for x in self.expand(sub, f"{tgt}%{rest}" if rest else tgt)]
            typ = str(comp.type).lower()
        return [path]

    def visible(self, sub, path: str) -> bool:
        """Whether every component of `path` is accessible in `sub` (private components
        only inside their type's module)."""
        from spel.scripts.fortran_parser.module_spec import ModuleSpecError, module_spec

        parts = path.lower().split("%")
        b = self.base(sub, parts[0])
        typ = b[0] if b else None
        for p in parts[1:]:
            t = self.types.get(typ)
            if t is None:
                return False
            mod = str(getattr(t, "declaration", "") or "").lower()
            if mod != str(sub.module).lower():
                m = self.modules.get(mod)
                try:
                    info = module_spec(_path(m)).types.get(typ) if m is not None else None
                except ModuleSpecError:
                    info = None
                if info is None or info.components.get(p, "private") != "public":
                    return False
            typ = str(t.components[p].type).lower()
        return True

    def public(self, module: str, name: str) -> bool:
        """Whether a module variable can be use-associated (from the parsed module)."""
        from spel.scripts.fortran_parser.module_spec import ModuleSpecError, module_spec

        m = self.modules.get(module)
        if m is None:
            return False
        try:
            info = module_spec(_path(m)).variables.get(name)
        except ModuleSpecError:
            return False
        return info is not None and info.access == "public"

    def base(self, sub, name: str):
        """(type, rank, origin kind) of a routine's name: dummy/local declaration or global."""
        rec = sub.record
        d = rec.declarations.get(name)
        if d is not None:
            return d.type.lower(), d.dim, ("dummy" if any(x.name == name for x in rec.dummy_args) else "local")
        v = {k.lower(): v for k, v in getattr(sub, "active_global_vars", {}).items()}.get(name) or self.globals.get(name)
        if v is not None:
            return str(v.type).lower(), v.dim, "global"
        return None

    def resolve(self, sub, path: str) -> Optional[tuple[tuple[str, ...], int, str, bool]]:
        """
        (key, rank, origin kind, updatable) for `path`; None when the path names a
        derived-type object or cannot be resolved. key: (owner type, member) for a member,
        ('#global', name) for a module variable, ('#dummy', routine id, name) for a dummy.
        """
        parts = path.lower().split("%")
        b = self.base(sub, parts[0])
        if b is None:
            return None
        typ, dim, kind = b
        owner = "#global" if kind == "global" else "#" + kind
        ok = True
        for p in parts[1:]:
            t = self.types.get(typ)
            if t is None:
                return None
            if dim > 0:
                ok = False  # `a(i)%x`: element of an array of derived types
            comp = t.components.get(p)
            if comp is None:
                return None
            owner, typ, dim = typ, str(comp.type).lower(), comp.dim
        if typ in self.types:
            return None
        key = (owner, parts[-1]) if len(parts) > 1 or kind == "global" else (owner, sub.id, parts[0])
        return key, dim, kind, ok


def _spans(directives, rec) -> list[tuple[int, int]]:
    from spel.transforms.acc_locals import _compute_ranges

    return [(lo, hi) for _, lo, hi in _compute_ranges(directives, rec)]


def _entries(maps) -> list[tuple[str, str, int, str]]:
    """(path, status, line, origin kind) of a routine's AccessMaps: its own accesses and,
    on call lines, its callees' effects (global accesses, argument statuses)."""
    out = []
    for kind, dicts in (("global", (maps.elmtype, maps.globals, maps.child_elmtype, maps.child_globals)),
                        ("dummy", (maps.args,)), ("local", (maps.locals,))):
        for d in dicts:
            for path, rws in d.items():
                out += [(path, rw.status, rw.ln, kind) for rw in rws]
    return out


def _data_entries(sub, maps) -> list[tuple[str, str, int, str]]:
    """
    `_entries` without the association contexts (`p => x`, nullify, allocate, deallocate):
    they change a descriptor, not data. Counted as writes they made acc-sync pull a pointer
    before `conc_o2 => ch4_vars%conc_o2_unsat_col` and push the stale host target after it
    (ch4_tran, AllocationMod's *_ptr, CanopyFluxes' bsun/bsha, LakeFluxes' z0*: the GPU's O2
    concentrations were zero, 2026-10-30), and pull/push around every deallocate.
    """
    from spel.scripts.record_access import ASSOCIATION_CONTEXTS

    assoc = {(a.path.lower(), a.ln) for a in sub.record.accesses() if a.context in ASSOCIATION_CONTEXTS}
    if not assoc:
        return _entries(maps)
    return [e for e in _entries(maps) if (e[0].lower(), e[2]) not in assoc]


def plan(fut, roots: Iterable[str], extra_seq: Iterable[str] = (),
         device_spans: Optional[dict[str, list[tuple[int, int]]]] = None,
         boundary: Optional[dict[str, tuple[list[tuple[str, str]], list[tuple[str, str]]]]] = None) -> SyncPlan:
    """
    The host segments of the call tree's routines and what each must sync.

    device_spans: routine id -> the line spans that run on the device, in place of the
    ones derived from the routine's kernels and calls (elm_drv: the calls its device
    transitions put on the device, which leave some kernel routines on the host).
    boundary: root id -> (accesses before, accesses after) the root by code outside the
    analysis, as (path, status) in the root's scope: what is written before is pushed at
    the root's first statement, what is read after and written by device code is pulled
    at its end (elm_drv: the coupler import and export around it).
    """
    from spel.scripts.fortran_parser.module_spec import parse_modules
    from spel.scripts.record_access import AccessMapper

    facts = _Facts(fut)
    types = _Types(fut)
    mapper = AccessMapper(fut.subroutine_dict)
    roots = list(roots)
    tree = {sid: s for sid, s in call_tree(fut, roots).items() if s.record is not None}
    out = SyncPlan()
    directives = {sid: find_directives(facts.lines(s), s.startline, s.endline) for sid, s in tree.items()}
    out.seq = {sid for sid, ds in directives.items() if any(d.construct == "routine" for d in ds)} | set(extra_seq)
    maps = {sid: mapper.maps(s) for sid, s in tree.items()}

    # routines whose call tree runs kernels: calls to them are device work for the caller
    runs_kernels: dict[str, bool] = {}

    def callees(s, ln: Optional[int] = None) -> set[str]:
        """Routines called by `s` (on line `ln`): the scope's resolution, generic
        interfaces through the call descriptions (AccessMapper.callee), plus acc-apply's."""
        events = [e for e in s.record.calls + s.record.function_calls
                  if (ln is None or e.ln == ln) and not getattr(e, "intrinsic", False)]
        ids = {c.id for c in (mapper.callee(s, s.record, e) for e in events) if c is not None}
        lo, hi = (s.startline, s.endline) if ln is None else (ln, ln)
        return ids | facts.callees_in(s, lo, hi)[0]

    def kernels(sid: str, stack=()) -> bool:
        if sid in runs_kernels:
            return runs_kernels[sid]
        s = fut.subroutine_dict.get(sid)
        if s is None or s.record is None or sid in stack or sid in out.seq:
            return False
        runs_kernels[sid] = facts.has_compute(s) or any(kernels(c, stack + (sid,)) for c in callees(s))
        return runs_kernels[sid]

    kernel_lines: dict[str, list[tuple[int, int]]] = {}
    device_lines: dict[str, list[tuple[int, int]]] = {}
    for sid, s in tree.items():
        if not kernels(sid):
            continue
        kernel_lines[sid] = _spans(directives[sid], s.record)
        calls = []
        src = facts.lines(s)
        for c in s.record.calls:
            if any(kernels(i) for i in callees(s, c.ln)):
                # the whole statement: the analysis records argument effects on its continuation lines
                end = c.ln
                while end + 1 < len(src) and src[end].split("!")[0].rstrip().endswith("&"):
                    end += 1
                calls.append((c.ln, end))
        device_lines[sid] = kernel_lines[sid] + calls

    # a root that runs no device work (an error search keeps it on the host) still sits
    # between device routines in elm_drv: its whole body is a host segment
    for sid in roots:
        if sid in tree and sid not in device_lines and sid not in out.seq:
            device_lines[sid] = []
    for sid, spans in (device_spans or {}).items():
        if sid in tree:
            device_lines[sid] = list(spans)

    # what device code writes (keys of _Types.resolve): in kernels (with the routine-seq
    # callees' effects on their call lines), then through the argument bindings: a dummy
    # device code writes makes its actuals stale, a stale actual its dummy
    for sid, spans in kernel_lines.items():
        for path0, status, ln, _ in _entries(maps[sid]):
            if "w" not in status or not any(lo <= ln <= hi for lo, hi in spans):
                continue
            for path in types.expand(tree[sid], path0):
                r = types.resolve(tree[sid], path)
                if r is not None:
                    out.device_written.add(r[0])
    pairs = []
    for sid, s in tree.items():
        for b in maps[sid].bindings:
            callee = tree.get(b.callee)
            if callee is None:
                continue
            d, a = types.resolve(callee, b.dummy), types.resolve(s, b.actual)
            if d is not None and a is not None:
                pairs.append((d[0], a[0]))
    changed = True
    while changed:
        changed = False
        for d, a in pairs:
            if (d in out.device_written) != (a in out.device_written):
                out.device_written |= {d, a}
                changed = True

    # host segments, in the routines that run device work; host-only callees are covered
    # by their callers' segments (their effects are on the call lines)
    for sid in sorted(device_lines):
        s = tree[sid]
        path = _path(s)
        name = sid.split("::")[-1].lower()
        node = next((p for m in parse_modules(path) for p in m.subprograms if p.name.lower() == name), None)
        if node is None:
            continue
        rs = RoutineSync(sid, path)
        _segments(rs, s, node, device_lines[sid], _data_entries(s, maps[sid]), facts.lines(s), types, out,
                  (boundary or {}).get(sid))
        if rs.segments:
            out.routines.append(rs)
        out.notes += excluded_calls_in(sid, path, node)
    return out


def excluded_calls_in(sid: str, path: Path, node) -> list[str]:
    """
    Notes on the calls `node` (a routine's parse tree) makes to routines of modules the
    analysis excludes (edit_files.bad_modules), through its own or its module's use
    statements. The analysis has no events for them, so the host writes of such a call
    are never pushed: dynSubgrid_driver's `dyncrop_interp` (dyncropFileMod, excluded for
    its genf90 dynVar modules) left the crop landunit/column weights and fertilizer
    stale on the device (2026-10-30). Until the callee is analyzable, the caller needs
    whole-state transitions around the call.
    """
    from spel.scripts.edit_files import bad_modules
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.scripts.fortran_parser.module_spec import callee_name, parse_modules, statements_with_next

    name = sid.split("::")[-1].lower()
    mod = next((m for m in parse_modules(path) if any(p.name.lower() == name for p in m.subprograms)), None)
    if mod is None:
        return []
    names: set[str] = set()
    for st in list(getattr(mod.spec, "statements", [])) + list(node.body.statements):
        if isinstance(st, A.UseStatement) and st.module.lower() in bad_modules and st.has_only:
            names |= {str(o).split("=>")[0].strip().lower() for o in st.objs}
    names -= {"t_startf", "t_stopf", "endrun", "shr_sys_flush", "shr_sys_abort", "get_curr_date",
              "get_curr_time_string", "get_nstep", "get_step_size", "is_first_step", "is_first_restart_step"}
    out = []
    for st, _ in statements_with_next(node.body, node.end_ln):
        c = callee_name(st)
        if c is not None and c.lower() in names:
            out.append(f"{sid}: line {st.lineno + 1}: call {c} into an excluded module; its effects are not synced")
    return out


def _end_of(st, lines: list[str]) -> int:
    """Last line of a statement: a construct's end, else the last continuation line."""
    end = getattr(st, "construct_end_ln", -1)  # an if construct's `end if` (its end_ln ends the first branch)
    if end is None or end < 0:
        end = getattr(st, "end_ln", -1)
    if end is not None and end >= 0:
        return end
    ln = st.lineno
    while ln + 1 < len(lines) and lines[ln].split("!")[0].rstrip().endswith("&"):
        ln += 1
    return ln


def _children(st) -> list:
    from spel.scripts.fortran_parser import spel_ast as A

    if isinstance(st, A.IfConstruct):
        out = [st.consequence] + [e.consequence for e in st.else_ifs]
        return out + ([st.else_.alternative] if st.else_ is not None else [])
    if isinstance(st, A.SelectCaseConstruct):
        return [c.body for c in st.cases]
    if isinstance(st, (A.DoLoop, A.DoWhile, A.AssociateConstruct)):
        return [st.body]
    if isinstance(st, A.BlockStatement):
        return [st]
    return []


def _condition_vars(expr) -> list[tuple[str, bool]]:
    """(base name, referenced with parentheses) of every name an expression uses: identifiers
    and the names of array references and function calls (the parser cannot tell an array
    element from a function result; the caller resolves the name)."""
    from spel.scripts.fortran_parser import spel_ast as A

    out: list[tuple[str, bool]] = []

    def walk(node):
        if isinstance(node, A.Identifier):
            out.append((str(node.value).split("%")[0].lower(), False))
            return
        if isinstance(node, A.FuncExpression):
            fn = node.function
            if isinstance(fn, A.Identifier):
                out.append((str(fn.value).split("%")[0].lower(), True))
            else:
                walk(fn)
            for a in node.args or []:
                walk(a)
            return
        if isinstance(node, A.Node):
            for v in vars(node).values():
                if isinstance(v, A.Node):
                    walk(v)
                elif isinstance(v, (list, tuple)):
                    for x in v:
                        if isinstance(x, A.Node):
                            walk(x)

    walk(expr)
    return out


def _header_lines(st) -> list[int]:
    from spel.scripts.fortran_parser import spel_ast as A

    if isinstance(st, A.IfConstruct):
        return [st.lineno] + [e.lineno for e in st.else_ifs]
    return [st.lineno]


def _segments(rs: RoutineSync, sub, node, device: list[tuple[int, int]], entries, lines: list[str],
              types: _Types, out: SyncPlan, boundary=None) -> None:
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.transforms.acc_locals import first_executable

    from spel.scripts.fortran_parser.symbols import Origin

    first_exec = first_executable(node, sub.record)
    rs.first_exec = first_exec
    use_stmts = [st for st in node.body.statements if isinstance(st, A.UseStatement)]
    rs.use_ln = (_end_of(use_stmts[-1], lines) if use_stmts else node.lineno) + 1

    def in_scope(path: str, kind: str) -> bool:
        """A callee's global the routine does not see gets a use statement; a name
        clash with the routine's own names makes the path unusable here."""
        base = path.split("%")[0]
        if not types.visible(sub, path):
            return False
        if kind != "global":
            return True
        sym = sub.record.lookup(base)
        v = types.globals.get(base)
        if sym is not None and not (sym.origin is Origin.EXTERNAL and v is not None):
            return sym.origin in (Origin.GLOBAL, Origin.EXTERNAL)
        # unknown here, or only through the external fallback (a whole `use` of a module outside
        # the analysis): an analyzed module's variable is not visible without a use statement
        mod = str(getattr(v, "declaration", "") or "").lower()
        if v is None or not mod or not types.public(mod, base):
            return False
        rs.uses.setdefault(mod, set()).add(base)
        return True

    def touches_device(lo: int, hi: int) -> bool:
        return any(not (b < lo or a > hi) for a, b in device)

    def on_device(ln: int) -> bool:  # a statement that starts in a kernel or is a device call
        return any(a <= ln <= b for a, b in device)

    def sync_names(lo: int, hi: int, seg: Segment) -> None:
        for path, status, ln, kind in [(x, st, ln, k) for p0, st, ln, k in entries for x in types.expand(sub, p0)]:
            if not lo <= ln <= hi:
                continue
            r = types.resolve(sub, path)
            if r is None:
                continue
            key, dim, kind, ok = r
            if kind in ("dummy", "local") and "%" not in path and dim == 0:
                continue  # scalar dummy or local: a host value (firstprivate/private in a kernel)
            if kind == "global" and "%" not in path and key not in out.device_written and "w" not in status:
                continue  # module variable the host only reads and no device code writes
            stale, written = key in out.device_written, "w" in status
            if not (stale or written):
                continue
            if not ok or not in_scope(path, kind):
                rs.skipped.add(path)
                seg.skipped.add(path)
                continue
            if written:
                seg.writes.add(path)
            if "r" in status:
                seg.reads.add(path)
            if stale:
                seg.stale.add(path)

    aborts = abort_calls(node.body)

    def flag_condition(cond) -> bool:
        """Every variable of the condition is a scalar no device code writes: a control flag
        (`weights_updated`, `create_glacier_mec_landunit`, `do_budgets`), a loop count, a
        function result (`get_do_transient_pfts()` has no variable at all). The header needs
        no sync, and the branches' host code syncs inside them, so a branch not taken costs
        nothing. An element or a section of an array, or a derived-type object, is not one."""
        for n, is_ref in _condition_vars(cond):
            r = types.resolve(sub, n)
            if r is None:
                if is_ref:
                    continue  # a function (intrinsic or not): no variable
                return False
            key, dim, kind, _ = r
            if dim != 0:
                return False
            if kind == "local":
                continue  # a local scalar holds a host value after a kernel (a reduction count: `nbad > 0`)
            if key in out.device_written:
                return False
        return True

    def cold(st, hi: int) -> bool:
        """An if construct that can abort (an error check), that replays a check spel
        defer-checks moved out of a loop (`if (spel_dc_k > 0)`), that tests a switch that is
        off by default (OFF_BY_DEFAULT), or whose condition reads only host flags
        (flag_condition): its syncs go inside its branches."""
        from spel.scripts.db_export.trace import expr_names
        from spel.transforms.defer_checks import PREFIX

        if not isinstance(st, A.IfConstruct):
            return False
        names = {n.lower() for n in expr_names(st.condition)}
        return (any(st.lineno <= ln <= hi for ln in aborts) or any(n.startswith(PREFIX) for n in names)
                or bool(names & OFF_BY_DEFAULT)
                or (st.construct_end_ln != -1 and flag_condition(st.condition)))  # a block if, not `if (c) return`

    def returns_in(lo: int, hi: int) -> list[int]:
        return [st.lineno for st, _ in _walk(node.body) if isinstance(st, A.ReturnStatement) and lo <= st.lineno <= hi]

    def block(stmts) -> None:
        run: list = []

        def flush():
            if not run:
                return
            lo, hi = run[0].lineno, _end_of(run[-1], lines)
            seg = Segment(lo, hi + 1, returns=returns_in(lo, hi))
            sync_names(lo, hi, seg)
            if seg.stale or seg.writes:
                rs.segments.append(seg)
            run.clear()

        for st in stmts:
            if st.lineno < first_exec:
                continue
            hi = _end_of(st, lines)
            if not touches_device(st.lineno, hi) and not cold(st, hi):
                run.append(st)
                continue
            flush()
            kids = [] if on_device(st.lineno) else _children(st)
            if kids:
                seg = Segment(st.lineno, st.lineno, header_only=True)
                for h in _header_lines(st):
                    sync_names(h, h, seg)
                seg.writes.clear()
                seg.stale &= seg.reads
                if seg.stale:
                    rs.segments.append(seg)
                for k in kids:
                    block(k.statements)
        flush()

    block(node.body.statements)

    if boundary is not None:
        # accesses outside the routine, on lines no entry uses: before -> push at the first
        # statement, after -> pull (what device code wrote) before `end subroutine`
        BEFORE, AFTER = -2, -3
        before, after = boundary
        entries += [(p, st, BEFORE, "global") for p, st in before if "w" in st]
        entries += [(p, st, AFTER, "global") for p, st in after if "r" in st]
        seg = Segment(first_exec, first_exec)
        sync_names(BEFORE, BEFORE, seg)
        seg.stale.clear()
        seg.reads.clear()
        if seg.writes:
            rs.segments.insert(0, seg)
        end = node.end_ln
        seg = Segment(end, end, header_only=True)
        sync_names(AFTER, AFTER, seg)
        seg.writes.clear()
        if seg.stale:
            rs.segments.append(seg)


def _walk(blk):
    from spel.scripts.registrations import iter_statements

    yield from iter_statements(blk)


# --- applying ---------------------------------------------------------------------------


def _directive(indent: str, head: str, names: list[str], width: int = 110, cond: Optional[str] = None) -> list[str]:
    if cond is None and head == "update self":
        # a pull while the device does not hold the state (initialization, host segments of
        # elm_drv) overwrites host results with uninitialized or stale device data (2026-10-28
        # UpdateAccVars; 2026-10-29 garbage t_soisno at step 0 after a re-placement without it)
        from spel.transforms.acc_plan import DEVICE_FLAG

        cond = DEVICE_FLAG
    out, cur = [], f"{indent}!$acc {head}("
    tail = ") if_present" + (f" if({cond})" if cond else "")
    for i, n in enumerate(names):
        piece = n + (", " if i + 1 < len(names) else tail)
        if len(cur) + len(piece) > width and not cur.endswith("("):
            out.append(cur.rstrip() + " &")
            cur = f"{indent}!$acc&   "
        cur += piece
    return out + [cur]


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def after_enter_data(lines: list[str], directives: list, first_exec: int) -> int:
    """
    The line after the `enter data` block (acc-locals' directives and their tag comments) of a
    routine whose first executable statement is `first_exec`: a push of a local written in the
    specification part (an initializer, a DATA statement) goes there, where its device copy
    exists; before the block it is a no-op (`if_present`): SnowHydrology's dzmin, 2026-10-30.
    The block is the first `enter data` at or after `first_exec` and before the first compute
    construct (it may follow an `associate` header); without one, `first_exec`.
    """
    starts = {d.start_ln: d for d in directives if d.construct == "enter data"}
    first_compute = min((d.start_ln for d in directives
                         if d.start_ln >= first_exec and d.construct.split()[0] in ("parallel", "kernels", "serial")),
                        default=len(lines))
    block = [ln for ln in starts if first_exec <= ln < first_compute]
    if not block:
        return first_exec
    ln = min(block)
    while ln < len(lines):
        if ln in starts:
            ln = starts[ln].end_ln + 1
            continue
        t = lines[ln].strip()
        if t.startswith("!") and not t.lower().startswith("!$acc"):
            ln += 1  # a tag comment of the block
            continue
        break
    return ln


def edits(plan_: SyncPlan, texts: dict[Path, list[str]],
          cond: Optional[str] = None) -> dict[Path, list[tuple[int, int, list[str]]]]:
    """(line, order, lines to insert before it) per file; `cond` adds an if clause to every update."""
    from spel.transforms.acc_locals import _above_directives

    by_file: dict[Path, list[tuple[int, int, list[str]]]] = {}
    directives: dict[Path, list] = {}
    for rs in plan_.routines:
        lines = texts[rs.path]
        ins = by_file.setdefault(rs.path, [])
        if rs.path not in directives:
            directives[rs.path] = find_directives(lines, 0, len(lines) - 1)
        if rs.uses:
            ind = _indent(lines[rs.use_ln - 1])
            ins.append((rs.use_ln, 0, [f"{ind}use {m}, only : {', '.join(sorted(n))}  {TAG}"
                                       for m, n in sorted(rs.uses.items())]))
        for seg in rs.segments:
            ind = _indent(lines[seg.first])
            before = sorted(seg.stale)
            if before:
                # above a directive that belongs to the segment's first statement, not between them
                ins.append((_above_directives(seg.first, directives[rs.path]), 1,
                            [f"{ind}{TAG}: device data this host code uses"]
                            + _directive(ind, "update self", before, cond=cond)))
            if seg.writes and not seg.header_only:
                w = sorted(seg.writes)
                # an empty segment (the boundary push at the first statement) goes above that
                # statement's directives, like an update self
                at = _above_directives(seg.after, directives[rs.path]) if seg.after == seg.first else seg.after
                if rs.first_exec >= 0 and seg.first < rs.first_exec:
                    # written in the specification part: after the locals' `enter data`
                    at = after_enter_data(lines, directives[rs.path], rs.first_exec)
                ins.append((at, -1, _directive(ind, "update device", w, cond=cond)))
                for r in seg.returns:
                    # after the segment's update self when the return is its first statement
                    # (`if (lake) return`): before it, the update would push stale host data
                    ins.append((r, 2, _directive(_indent(lines[r]), "update device", w, cond=cond)))
    return by_file


def dedupe_sync_blocks(lines: list[str]) -> list[str]:
    """
    In every run of consecutive acc-sync lines (TAG comments and `!$acc update ...
    if_present` statements with their `!$acc&` continuations, the text `edits` writes),
    keep each statement once, in order, and the TAG comment once before the first
    `update self`. Repeated runs of the sync (the analysis reads directives as comments,
    so it plans them again) then leave the source as one run does.
    """
    out: list[str] = []
    i = 0
    n = len(lines)

    def statement(j: int) -> tuple[int, list[str]] | None:
        """(end, lines) of the acc-sync update statement starting at j."""
        head = lines[j].lstrip().lower()
        if not head.startswith("!$acc update"):
            return None
        k = j
        while lines[k].rstrip().endswith("&") and k + 1 < n and lines[k + 1].lstrip().lower().startswith("!$acc&"):
            k += 1
        last = lines[k].rstrip().lower()
        if not (last.endswith("if_present") or ") if_present if(" in last):
            return None
        return k, lines[j:k + 1]

    while i < n:
        if lines[i].strip() != f"{TAG}: device data this host code uses" and statement(i) is None:
            out.append(lines[i])
            i += 1
            continue
        block: list[tuple[str, list[str]]] = []
        while i < n:
            if lines[i].strip() == f"{TAG}: device data this host code uses":
                block.append(("tag", [lines[i]]))
                i += 1
            elif (st := statement(i)) is not None:
                block.append(("upd", st[1]))
                i = st[0] + 1
            else:
                break
        seen: set[str] = set()
        tag = next((b[1] for b in block if b[0] == "tag"), None)
        tagged = False
        for kind, text in block:
            if kind == "tag":
                continue
            key = " ".join(t.strip() for t in text)
            if key in seen:
                continue
            seen.add(key)
            if tag is not None and not tagged and text[0].lstrip().lower().startswith("!$acc update self"):
                out.extend(tag)
                tagged = True
            out.extend(text)
    return out


def apply(fut, plan_: SyncPlan, dry_run: bool = False) -> list[str]:
    texts: dict[Path, list[str]] = {}
    for rs in plan_.routines:
        if rs.path not in texts:
            texts[rs.path] = rs.path.read_text().split("\n")
    log = []
    for path, ins in edits(plan_, texts).items():
        lines = texts[path]
        done = 0
        for ln, _, new in sorted(ins, key=lambda x: (-x[0], -x[1])):
            if lines[max(ln - len(new), 0):ln] == new or lines[ln:ln + len(new)] == new:
                continue  # inserted by an earlier run (the analysis reads directives as comments)
            lines[ln:ln] = new
            done += 1
        lines[:] = dedupe_sync_blocks(lines)
        log.append(f"{path.name}: {done} insertions")
        if not dry_run:
            path.write_text("\n".join(lines))
    return log


@dataclass
class StaleUpdate:
    routine: str
    directive: object      # the AccDirective
    name: str              # normalized name in the clause
    kind: str              # "push" (update device after a kernel write) | "pull" (update self after a host write)
    write_ln: int          # the last write before the directive

    def __str__(self) -> str:
        what = ("update device(%s) but its last write (line %d) is in a compute construct: stale push"
                if self.kind == "push" else
                "update self(%s) but its last write (line %d) is host code: possibly stale pull")
        return f"{self.routine}: line {self.directive.start_ln + 1}: " + what % (self.name, self.write_ln + 1)


def stale_updates(sub, lines: list[str]) -> list[StaleUpdate]:
    """
    Update directives of one routine that contradict the record's last write of a name:

    - `update device(x)` where the last write of `x` before the directive is inside a compute
      construct and no `update self(x)` follows that write: the device already holds the value
      and the host copy may never have been set; the push overwrites device results with
      garbage (CombineSnowLayers' dzminloc and SurfaceAlbedo's case_aer, 2026-10-30: pushes
      from before a pass moved the assignment into a kernel).
    - `update self(x)` where the last write of `x` before the directive is host code and no
      `update device(x)` follows that write: the pull overwrites a fresh host value with stale
      device data (UpdateAccVars, 2026-10-28). Lower confidence (a routine run with the device
      flag off makes it harmless).
    Hand-written directives are invisible to the planner; this check sees them.
    """
    import re

    from spel.transforms.acc_directives import split_top_level

    rec = sub.record
    if rec is None:
        return []
    directives = find_directives(lines, sub.startline, sub.endline)
    spans = _spans(directives, rec)

    def on_device(ln: int) -> bool:
        return any(a <= ln <= b for a, b in spans)

    def norm(x: str) -> str:
        return re.sub(r"\(.*?\)", "", x).strip().lower()

    from spel.scripts.record_access import ASSOCIATION_CONTEXTS

    writes: dict[str, list[tuple[int, bool]]] = {}
    deallocs: dict[str, list[int]] = {}
    for a in rec.accesses():
        if a.context == "deallocate":
            deallocs.setdefault(norm(a.path), []).append(a.ln)
        elif "w" in a.status and a.context != "inquiry" and a.context not in ASSOCIATION_CONTEXTS:
            writes.setdefault(norm(a.path), []).append((a.ln, on_device(a.ln)))
    updates: list[tuple[int, str, str]] = []  # (line, device|self, name)
    for d in directives:
        if d.construct != "update":
            continue
        for c in d.clauses:
            if c.name in ("device", "self", "host") and c.args:
                for arg in split_top_level(c.args):
                    updates.append((d.start_ln, "self" if c.name == "host" else c.name, norm(arg)))
    out: list[StaleUpdate] = []
    for d in directives:
        if d.construct != "update":
            continue
        for c in d.clauses:
            if c.name not in ("device", "self", "host") or not c.args:
                continue
            kind = "self" if c.name == "host" else c.name
            for arg in split_top_level(c.args):
                name = norm(arg)
                prior = [w for w in writes.get(name, []) if w[0] < d.start_ln]
                if not prior:
                    continue
                last_ln, last_dev = max(prior)
                other = "self" if kind == "device" else "device"
                if any(last_ln < ln < d.start_ln and k == other and n == name for ln, k, n in updates):
                    continue  # a round trip in between: consistent
                if any(last_ln < ln < d.start_ln for ln in deallocs.get(name, [])):
                    continue  # acc-sync's pull/push around a deallocate: the push is a no-op (not present)
                if kind == "device" and last_dev:
                    out.append(StaleUpdate(sub.id, d, name, "push", last_ln))
                if kind == "self" and not last_dev:
                    out.append(StaleUpdate(sub.id, d, name, "pull", last_ln))
    return out


def fix_stale_pushes(sub, lines: list[str], stale: list[StaleUpdate]) -> int:
    """
    Drop the stale names from their `update device` directives in `lines` (entries become
    multi-line strings or None; a directive left without names is removed). Returns the
    number of directives changed.
    """
    import re

    from spel.transforms.acc_directives import split_top_level

    by_dir: dict[int, set[str]] = {}
    dirs = {}
    for st in stale:
        if st.kind == "push":
            by_dir.setdefault(st.directive.start_ln, set()).add(st.name)
            dirs[st.directive.start_ln] = st.directive
    for ln, names in by_dir.items():
        d = dirs[ln]
        clauses = []
        for c in d.clauses:
            if c.name == "device" and c.args:
                keep = [a.strip() for a in split_top_level(c.args)
                        if re.sub(r"\(.*?\)", "", a).strip().lower() not in names]
                if keep:
                    clauses.append(f"device({', '.join(keep)})")
            elif c.args is not None:
                clauses.append(f"{c.name}({c.args})")
            else:
                clauses.append(c.name)
        text = None
        if any(cl.startswith("device(") for cl in clauses):
            text = f"{d.indent}!$acc update " + " ".join(clauses)
        lines[ln] = text if text is not None else None
        for k in range(ln + 1, d.end_ln + 1):
            lines[k] = None
    return len(by_dir)
