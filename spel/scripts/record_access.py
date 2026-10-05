"""
Legacy-shaped access maps (`*_access_by_ln`) derived from a SubroutineRecord.

Keys are designator paths: subscripts dropped, associate names expanded.
Each key maps to one ReadWrite per line; reads and writes of a key on the
same line merge to "rw".

Actual arguments get the callee's status for the matching dummy (and its
components, e.g. dummy `b` -> `b%f` becomes actual `t` -> `t%f`) at the call
line. Accesses through a pointer also apply to every target it is associated
with in the routine. A callee's global accesses (transitively) are kept in
`child_elmtype`/`child_globals` at the call line; an internal callee's
host-variable accesses bind at the call line like arguments. Callees are
mapped on demand, so the order in which
routines are mapped doesn't matter. Callees that can't be analyzed (library,
external, generic without a legacy call description, recursion) get
conservative statuses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

from spel.scripts.fortran_parser.scope_walk import (
    Access,
    CallArg,
    CallEvent,
    FunctionRef,
    SubroutineRecord,
)
from spel.scripts.fortran_parser.spel_ast import SemanticError
from spel.scripts.fortran_parser.symbols import Origin, SymbolKind
from spel.scripts.helper_functions import combine_many_statuses
from spel.scripts.types import (
    Annotate,
    ArgUsage,
    CallBinding,
    CallTag,
    LineTuple,
    PropagatedAccess,
    ReadWrite,
    Scope,
)

if TYPE_CHECKING:
    from spel.scripts.analyze_subroutines import Subroutine
    from spel.scripts.module_resolver import ModuleScopes

AccessDict = dict[str, list[ReadWrite]]

# status of an actual argument when the callee can't be analyzed
UNKNOWN_SUBROUTINE_ARG = "rw"
UNKNOWN_FUNCTION_ARG = "r"
# status of a dummy the callee never accesses
INTENT_STATUS = {"in": "r", "out": "w", "inout": "rw"}
# accesses that change a pointer's association, not its target
ASSOCIATION_CONTEXTS = {"ptr-assign", "nullify", "allocate", "deallocate"}


def merge_status(a: str, b: str) -> str:
    return "".join(sorted(set(a) | set(b)))


def summarize(rws: list[ReadWrite]) -> str:
    return combine_many_statuses([rw.status for rw in sorted(rws, key=lambda x: x.ln)])


@dataclass(frozen=True)
class ArgBinding:
    """
    `actual` (a designator path in `caller`) is passed to `dummy`, the
    `argn`-th dummy of `callee`, at line `ln`. A pointer actual is bound
    through each of its targets.
    """

    caller: str
    callee: str
    ln: int
    argn: int
    dummy: str
    actual: str
    origin: Origin


@dataclass
class AccessMaps:
    """
    elmtype: components of global (or external) derived-type variables
    globals: other global/external variables
    args:    dummy arguments
    locals:  local variables and function result
    host:    host-associated variables (internal subprograms)
    child_elmtype/child_globals: global accesses made by callees (transitively),
        one entry per call line with the callee's overall status
    bindings: actual -> dummy at each call to an analyzable callee
    """

    dummies: list[str] = field(default_factory=list)
    intents: dict[str, Optional[str]] = field(default_factory=dict)
    elmtype: AccessDict = field(default_factory=dict)
    globals: AccessDict = field(default_factory=dict)
    args: AccessDict = field(default_factory=dict)
    locals: AccessDict = field(default_factory=dict)
    host: AccessDict = field(default_factory=dict)
    child_elmtype: AccessDict = field(default_factory=dict)
    child_globals: AccessDict = field(default_factory=dict)
    bindings: list[ArgBinding] = field(default_factory=list)

    def all_elmtype(self) -> AccessDict:
        """Direct and callee accesses to global derived-type components."""
        return _union(self.elmtype, self.child_elmtype)

    def all_globals(self) -> AccessDict:
        """Direct and callee accesses to other global variables."""
        return _union(self.globals, self.child_globals)

    def dummy_status(self, dummy: str) -> dict[str, str]:
        """Overall status of `dummy` and its components, keyed by path."""
        out = {
            k: summarize(rws)
            for k, rws in self.args.items()
            if k == dummy or k.startswith(f"{dummy}%")
        }
        if not out and (status := INTENT_STATUS.get(self.intents.get(dummy) or "")):
            out[dummy] = status
        return out

    def add(
        self,
        origin: Origin,
        path: str,
        status: str,
        ln: int,
        line: Optional[LineTuple] = None,
    ) -> None:
        match origin:
            case Origin.GLOBAL | Origin.EXTERNAL:
                target = self.elmtype if "%" in path else self.globals
            case Origin.DUMMY:
                target = self.args
            case Origin.LOCAL | Origin.RESULT:
                target = self.locals
            case Origin.HOST:
                target = self.host
            case _:  # expression associate names, intrinsics
                return
        _put(target, path, status, ln, line)

    def add_child(
        self, path: str, status: str, ln: int, line: Optional[LineTuple] = None
    ) -> None:
        target = self.child_elmtype if "%" in path else self.child_globals
        _put(target, path, status, ln, line)


def _put(
    target: AccessDict, path: str, status: str, ln: int, line: Optional[LineTuple]
) -> None:
    rws = target.setdefault(path, [])
    for rw in reversed(rws):
        if rw.ln == ln:
            rw.status = merge_status(rw.status, status)
            return
    rws.append(ReadWrite(status, ln, line))


def _union(a: AccessDict, b: AccessDict) -> AccessDict:
    out = {k: list(v) for k, v in a.items()}
    for k, rws in b.items():
        out[k] = sorted(out.get(k, []) + rws, key=lambda rw: rw.ln)
    return out


def callsite_globals(caller: AccessMaps, lines: set[int]) -> AccessDict:
    """
    Global accesses of `caller` (its own and its callees') on `lines`. For the
    call lines of a routine, these are the globals bound to its dummies plus
    the routine's own global accesses, all at the call lines.
    """
    out: AccessDict = {}
    for path, rws in (caller.all_elmtype() | caller.all_globals()).items():
        hits = [rw for rw in rws if rw.ln in lines]
        if hits:
            out[path] = hits
    return out


class _PointerFanout:
    """
    Adds an access to `maps`, and to every target the accessed pointer is
    ever associated with in the routine (flow-insensitive).
    """

    def __init__(
        self, rec: SubroutineRecord, maps: AccessMaps, lines: dict[int, LineTuple]
    ):
        self.rec = rec
        self.maps = maps
        self.lines = lines
        self.targets = rec.pointer_targets()

    def __call__(self, origin: Origin, path: str, status: str, ln: int) -> None:
        line = self.lines.get(ln)
        self.maps.add(origin, path, status, ln, line)
        for target, t_origin in self._through_pointers(path):
            self.maps.add(t_origin, target, status, ln, line)

    def _through_pointers(self, path: str) -> list[tuple[str, Origin]]:
        out = []
        for ptr, targets in self.targets.items():
            if path != ptr and not path.startswith(f"{ptr}%"):
                continue
            for target in targets:
                base = target.split("%")[0]
                sym = self.rec.lookup(base)
                if sym is None:
                    raise SemanticError(f"{self.rec.name}: unknown pointer target {target}")
                out.append((target + path[len(ptr) :], sym.origin))
        return out

    def resolve(self, origin: Origin, path: str) -> list[tuple[str, Origin]]:
        """The variables `path` designates: a pointer's targets, else itself."""
        return self._through_pointers(path) or [(path, origin)]


class AccessMapper:
    """
    Builds (and caches on `sub.record_access`) the AccessMaps of routines.
    Unwalked routines are walked with `scopes` on demand.
    """

    def __init__(
        self,
        sub_dict: dict[str, Subroutine],
        scopes: Optional[ModuleScopes] = None,
    ):
        self.sub_dict = sub_dict
        self.scopes = scopes
        self._in_progress: set[str] = set()

    def maps(self, sub: Subroutine) -> Optional[AccessMaps]:
        """None while `sub` is being mapped (recursive call)."""
        if sub.record_access is not None:
            return sub.record_access
        if sub.id in self._in_progress:
            return None
        self._in_progress.add(sub.id)
        try:
            rec = sub.walk_syntax_tree(self.scopes)
            sub.record_access = self._build(sub, rec)
        finally:
            self._in_progress.discard(sub.id)
        return sub.record_access

    def _build(self, sub: Subroutine, rec: SubroutineRecord) -> AccessMaps:
        lines = {lt.ln: lt for lt in sub.sub_lines}
        maps = AccessMaps(
            dummies=[d.name for d in rec.dummy_args],
            intents={d.name: d.intent for d in rec.dummy_args},
        )
        # origin of each designator passed as an actual argument, by (ln, path)
        arg_origin: dict[tuple[int, str], Origin] = {}
        add = _PointerFanout(rec, maps, lines)
        for e in rec.events:
            match e:
                case Access(status="arg"):
                    arg_origin[(e.ln, e.path)] = e.origin
                case Access(context=context) if context in ASSOCIATION_CONTEXTS:
                    maps.add(e.origin, e.path, e.status, e.ln, lines.get(e.ln))
                case Access():
                    add(e.origin, e.path, e.status, e.ln)
                case CallEvent():
                    self._bind(sub, rec, add, e, arg_origin)
                case FunctionRef(intrinsic=False):
                    self._bind(sub, rec, add, e, arg_origin)
        return maps

    def _bind(
        self,
        sub: Subroutine,
        rec: SubroutineRecord,
        add: _PointerFanout,
        event: Union[CallEvent, FunctionRef],
        arg_origin: dict[tuple[int, str], Origin],
    ) -> None:
        is_call = isinstance(event, CallEvent)
        callee = self.callee(sub, rec, event)
        cmaps = None
        if callee is not None and not callee.library:
            cmaps = self.maps(callee)
        actuals = list(event.args)
        if is_call and event.object_path is not None:
            actuals.insert(0, CallArg(None, event.object_path, None))
        unknown = UNKNOWN_SUBROUTINE_ARG if is_call else UNKNOWN_FUNCTION_ARG

        for i, arg in enumerate(actuals):
            if arg.path is None:
                continue  # expression: its designators were walked as reads
            origin = arg_origin[(event.ln, arg.path)]
            if cmaps is None:
                statuses = {arg.path: unknown}
            else:
                dummy = arg.keyword or (
                    cmaps.dummies[i] if i < len(cmaps.dummies) else None
                )
                if dummy is None:
                    raise SemanticError(
                        f"{sub.id}: too many arguments to {event.name} @{event.ln}"
                    )
                statuses = {
                    arg.path + key[len(dummy) :]: status
                    for key, status in cmaps.dummy_status(dummy).items()
                }
                argn = cmaps.dummies.index(dummy) if dummy in cmaps.dummies else i
                for actual, a_origin in add.resolve(origin, arg.path):
                    add.maps.bindings.append(
                        ArgBinding(sub.id, callee.id, event.ln, argn, dummy, actual, a_origin)
                    )
            for path, status in statuses.items():
                add(origin, path, status, event.ln)

        if cmaps is not None:
            self._merge_callee(sub, rec, add, callee, cmaps, event.ln)

    def _merge_callee(
        self,
        sub: Subroutine,
        rec: SubroutineRecord,
        add: _PointerFanout,
        callee: Subroutine,
        cmaps: AccessMaps,
        ln: int,
    ) -> None:
        """
        The callee's global accesses (its own and its callees') and, for an
        internal callee, its host-variable accesses land on the call line.
        """
        line = add.lines.get(ln)
        for path, rws in (cmaps.all_elmtype() | cmaps.all_globals()).items():
            add.maps.add_child(path, summarize(rws), ln, line)
        for path, rws in cmaps.host.items():
            if callee.host_id == sub.id:
                base = path.split("%")[0]
                sym = rec.lookup(base)
                if sym is None:
                    raise SemanticError(f"{sub.id}: unknown host variable {path}")
                origin = sym.origin
            else:  # sibling internal subprogram: still the host's variable
                origin = Origin.HOST
            add(origin, path, summarize(rws), ln)

    def callee(
        self, sub: Subroutine, rec: SubroutineRecord, event: Union[CallEvent, FunctionRef]
    ) -> Optional[Subroutine]:
        """
        Resolved through the routine's scope; type-bound and generic calls
        fall back to the legacy call description at that line.
        """
        if "%" not in event.name:
            sym = rec.lookup(event.name)
            if sym is not None and sym.kind is SymbolKind.PROCEDURE:
                if sym.origin in (Origin.LOCAL, Origin.HOST):
                    sub_id = f"{sub.module}::{event.name}"
                else:
                    sub_id = sym.target
                if sub_id in self.sub_dict:
                    return self.sub_dict[sub_id]
        if isinstance(event, CallEvent):
            desc = sub.sub_call_desc.get(event.ln)
            if desc is not None and desc.fn in self.sub_dict:
                return self.sub_dict[desc.fn]
        return None


# ---------------------------------------------------------------------------
# legacy-named views (Subroutine.elmtype_access_by_ln & co.)
# ---------------------------------------------------------------------------


def _inst_field(path: str) -> str:
    return "%".join(path.split("%")[:2])


def elmtype_view(
    maps: AccessMaps,
    dummy_actuals: dict[str, set[str]],
    pointer_components: dict[str, list[str]],
) -> AccessDict:
    """
    Accesses to global derived-type components at `inst%field` depth: the
    routine's own, its callees' (at the call lines), and those of dummies
    bound to globals by `dummy_actuals` (at the routine's lines). An access
    to a pointer component also accesses each of its targets.
    """
    out: AccessDict = {}

    def put(path: str, rws: list[ReadWrite]) -> None:
        key = _inst_field(path)
        if "%" not in key:
            return
        for k in (key, *pointer_components.get(key, ())):
            for rw in rws:
                _put(out, k, rw.status, rw.ln, rw.ltuple)

    for path, rws in maps.all_elmtype().items():
        put(path, rws)
    for path, rws in maps.args.items():
        dummy = path.split("%")[0]
        for actual in sorted(dummy_actuals.get(dummy, ())):
            put(actual + path[len(dummy) :], rws)
    return {k: sorted(v, key=lambda rw: rw.ln) for k, v in out.items()}


def access_summary(view: AccessDict) -> dict[str, ReadWrite]:
    return {k: ReadWrite(summarize(rws), -1, None) for k, rws in view.items()}


def dummy_actuals(bindings: list[ArgBinding], callee: str) -> dict[str, set[str]]:
    """Globals passed to each dummy of `callee`."""
    out: dict[str, set[str]] = {}
    for b in bindings:
        if b.callee == callee and b.origin in (Origin.GLOBAL, Origin.EXTERNAL):
            out.setdefault(b.dummy, set()).add(b.actual)
    return out


def single_instance_actuals(
    arg_types: dict[str, str], instances: dict[str, list[str]]
) -> dict[str, set[str]]:
    """Bind each derived-type dummy to its type's instance, if it has only one."""
    return {
        dummy: {insts[0]}
        for dummy, type_name in arg_types.items()
        if len(insts := instances.get(type_name, [])) == 1
    }


_PROPAGATED_SCOPE = {
    Origin.GLOBAL: Scope.GLOBAL,
    Origin.EXTERNAL: Scope.GLOBAL,
    Origin.DUMMY: Scope.ARG,
    Origin.LOCAL: Scope.LOCAL,
    Origin.RESULT: Scope.LOCAL,
}


def propagated_access(sub_dict: dict) -> dict[str, dict[str, list[PropagatedAccess]]]:
    """
    For each callee: the callee's dummy accesses (at its own lines), keyed
    by the caller variable bound to the dummy, one entry per call site.
    """
    out: dict[str, dict[str, list[PropagatedAccess]]] = {}
    for caller in sub_dict.values():
        if caller.record_access is None:
            continue
        for b in caller.record_access.bindings:
            callee = sub_dict.get(b.callee)
            scope = _PROPAGATED_SCOPE.get(b.origin)
            if callee is None or callee.record_access is None or scope is None:
                continue
            base, _, member = b.actual.partition("%")
            binding = CallBinding(
                var_name=base,
                kind=Annotate.COMP if member else Annotate.VAR,
                scope=scope,
                argn=b.argn,
                member_path=member,
                nested_level=0,
                callee=b.callee,
                arg_usage=ArgUsage.DIRECT,
            )
            tag = CallTag(b.caller, b.callee, b.ln)
            for key, rws in callee.record_access.args.items():
                if key != b.dummy and not key.startswith(f"{b.dummy}%"):
                    continue
                new_key = b.actual + key[len(b.dummy) :]
                k_scope = (
                    Scope.ELMTYPE if scope is Scope.GLOBAL and "%" in new_key else scope
                )
                out.setdefault(b.callee, {}).setdefault(new_key, []).append(
                    PropagatedAccess(tag, list(rws), k_scope, b.dummy, binding)
                )
    return out
