"""
How a routine indexes a subgrid array.

The form of a subscript is derived from the routine's record: the loop whose
index it is, or the scalar assignment that defines it (``p = filter_soilp(fp)``,
``c = veg_pp%column(p)``, ``p = col_pp%pfti(c) + pi - 1``), recursively. Arrays
are resolved through the call-site bindings of the routine's dummies, so a
filter dummy names the filter the caller passed (``filter%soilp``).

Kinds:
  filter  x(p), p = F(j): a gather from the integer array F (`filter`), j
          usually a loop 1..N (`inner`)
  block   x(idx), idx => F(lo:hi): a vector subscript from a filter section
  map     x(c), c = M(i): a subgrid mapping array M (veg_pp%column, ...)
  offset  x(p), p = <expression with a mapping array> (col_pp%pfti(c) + pi - 1)
  count   a loop index over 1..N (N in `filter`)
  range   a loop index over other bounds (begp..endp, pfti(c)..pftf(c))
  slice   an array section (x(begp:endp), x(:))
  whole   no subscripts
  const   a literal
  unknown anything else
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from spel.scripts.fortran_parser.scope_walk import (
    Access,
    AssignEvent,
    LoopInfo,
    SubroutineRecord,
)
from spel.scripts.fortran_parser.spel_ast import (
    BoundsExpression,
    Expression,
    FieldAccessExpression,
    FuncExpression,
    Identifier,
    InfixExpression,
    IntegerLiteral,
    Statement,
)
from spel.scripts.fortran_parser.symbols import Origin

# instances of the subgrid hierarchy; their integer components are mapping arrays
SUBGRID_INSTANCES = ("grc_pp", "top_pp", "lun_pp", "col_pp", "veg_pp")
MAX_DEPTH = 6


@dataclass(frozen=True)
class IndexForm:
    """
    var:     the variable this is the form of (`p` for p = filter_soilp(fp)),
             None for a subscript expression
    foreign: the form comes from a caller (a scalar index dummy), so its
             variables are the caller's
    """

    kind: str
    text: str
    filter: frozenset = frozenset()
    inner: Optional[IndexForm] = None
    var: Optional[str] = None
    foreign: bool = False

    def __str__(self) -> str:
        match self.kind:
            case "filter" | "map":
                arr = "|".join(sorted(self.filter)) or "?"
                return f"{arr}[{self.inner}]" if self.inner else f"{arr}[?]"
            case "block":
                return f"{'|'.join(sorted(self.filter))}[block]"
            case "count":
                return f"1..{'|'.join(sorted(self.filter)) or self.text}"
            case "offset" | "range":
                return f"{self.text} {{{self.inner}}}" if self.inner else self.text
        return self.text


UNKNOWN = IndexForm("unknown", "?")


def norm(expr: Expression) -> str:
    return str(expr).replace(" ", "").lower()


def is_designator(expr: Expression) -> bool:
    return isinstance(expr, (Identifier, FuncExpression, FieldAccessExpression))


class IndexContext:
    """
    One visit of a routine: its record, the global paths each dummy is
    bound to (`env`), the forms of scalar index dummies (`idx_env`), the
    forms of dummies bound to an element or a section with a fixed subgrid
    index (`inherit`: x(c,:) passed to y(:) gives every y(j) the form of c),
    and for an internal subprogram the context of its host.
    Unbound dummies resolve to "dummy:<name>", locals to "local:<routine>:<name>".
    """

    def __init__(
        self,
        sub_id: str,
        rec: SubroutineRecord,
        env: dict[str, frozenset],
        idx_env: dict[str, IndexForm],
        inherit: Optional[dict[str, IndexForm]] = None,
        host: Optional[IndexContext] = None,
    ):
        self.sub_id = sub_id
        self.rec = rec
        self.env = env
        self.idx_env = idx_env
        self.inherit = inherit or {}
        self.host = host
        self._assigns: Optional[dict[str, list[tuple[int, AssignEvent]]]] = None
        self._loops: Optional[list[LoopInfo]] = None

    # --- resolution ---
    def resolve(self, path: str, origin: Origin) -> frozenset:
        base, _, rest = path.partition("%")
        rest = f"%{rest}" if rest else ""
        match origin:
            case Origin.GLOBAL | Origin.EXTERNAL:
                return frozenset({path})
            case Origin.DUMMY:
                actuals = self.env.get(base)
                if actuals is None:
                    return frozenset({f"dummy:{path}"})
                return frozenset(a + rest for a in actuals)
            case Origin.LOCAL | Origin.RESULT:
                return frozenset({f"local:{self.sub_id}:{path}"})
            case Origin.HOST:
                if self.host is not None:
                    return self.host.resolve_path(path)
                return frozenset({f"host:{path}"})
        return frozenset()

    def resolve_path(self, path: str) -> frozenset:
        """`path` (associate names expanded) resolved through its base symbol."""
        sym = self.rec.lookup(path.split("%")[0])
        return self.resolve(path, sym.origin) if sym is not None else frozenset({path})

    # --- record helpers ---
    @property
    def assigns(self) -> dict[str, list[tuple[int, AssignEvent]]]:
        if self._assigns is None:
            self._assigns = {}
            for pos, e in enumerate(self.rec.events):
                if isinstance(e, AssignEvent) and not e.lhs.subscripts and "%" not in e.lhs.path:
                    self._assigns.setdefault(e.lhs.path, []).append((pos, e))
        return self._assigns

    @property
    def loops(self) -> list[LoopInfo]:
        if self._loops is None:
            self._loops = self.rec.all_loops()
        return self._loops

    # --- forms ---
    def access_form(self, access: Access, pos: int) -> IndexForm:
        """Form of the first (subgrid) subscript of `access`, event `pos`."""
        if access.origin is Origin.DUMMY and access.base in self.inherit:
            return replace(self.inherit[access.base], foreign=True)
        if not access.subscripts:
            return IndexForm("whole", "")
        return self.expr_form(access.subscripts[0], pos, access.ln, 0)

    def fixes_subgrid(self, access: Access) -> bool:
        """Whether `access`, passed as an actual, fixes the subgrid index for the callee."""
        if access.origin is Origin.DUMMY and access.base in self.inherit:
            return True
        return bool(access.subscripts) and not isinstance(access.subscripts[0], BoundsExpression)

    def expr_form(self, expr: Expression, pos: int, ln: int, depth: int) -> IndexForm:
        if depth > MAX_DEPTH:
            return UNKNOWN
        match expr:
            case BoundsExpression():
                return IndexForm("slice", norm(expr))
            case IntegerLiteral():
                return IndexForm("const", norm(expr))
            case Identifier() if "%" not in expr.value:
                name = expr.value.lower()
                target = self.rec.associations(ln).get(name)
                if target is not None:
                    # associate name for a filter section (idx => filter(lo:hi))
                    return IndexForm("block", name, self.resolve_path(target))
                return self.var_form(name, pos, ln, depth)
        return IndexForm("unknown", norm(expr))

    def var_form(self, name: str, pos: int, ln: int, depth: int) -> IndexForm:
        loop = next(
            (l for l in reversed(self.loops) if l.index == name and l.contains(ln)), None
        )
        assign = None
        for apos, a in self.assigns.get(name, []):
            if apos >= pos:
                break
            assign = (apos, a)
        if loop is not None and (assign is None or assign[1].ln < loop.start_ln):
            return replace(self.loop_form(loop, depth), var=name)
        if assign is not None:
            return replace(self.assign_form(*assign, depth), var=name)
        if name in self.idx_env:
            return replace(self.idx_env[name], foreign=True)
        return IndexForm("unknown", name, var=name)

    def loop_form(self, loop: LoopInfo, depth: int) -> IndexForm:
        node = loop.node
        start, end = norm(node.start), norm(node.end)
        if start == "1" and is_designator(node.end):
            path = next(
                (p for p in loop.bound_paths if p.split("%")[-1] == end.split("%")[-1]),
                end,
            )
            return IndexForm("count", end, self.resolve_path(path))
        inner = None
        for p in sorted(loop.bound_paths):
            if "%" not in p and (d := self.rec.declaration(p)) is not None and d.dim == 0:
                inner = self.var_form(p, self._pos_at(loop.start_ln), loop.start_ln, depth + 1)
                break
        return IndexForm("range", f"{start}:{end}", inner=inner)

    def assign_form(self, pos: int, a: AssignEvent, depth: int) -> IndexForm:
        reads = [r for r in a.reads if r.subscripts]
        if isinstance(a.rhs, Identifier) and "%" not in a.rhs.value and a.reads:
            # a copy of another index (col_grass = c)
            return self.var_form(a.rhs.value.lower(), pos, a.ln, depth + 1)
        if is_designator(a.rhs) and a.reads and a.reads[-1].subscripts:
            d = a.reads[-1]
            arrays = self.resolve(d.path, d.origin)
            kind = "map" if any(_is_mapping(p) for p in arrays) else "filter"
            inner = self.expr_form(d.subscripts[0], pos, a.ln, depth + 1)
            return IndexForm(kind, norm(a.rhs), arrays, inner)
        if isinstance(a.rhs, InfixExpression):
            for r in reads:
                arrays = self.resolve(r.path, r.origin)
                if any(_is_mapping(p) for p in arrays):
                    inner = self.expr_form(r.subscripts[0], pos, a.ln, depth + 1)
                    return IndexForm("offset", norm(a.rhs), arrays, inner)
        return IndexForm("unknown", norm(a.rhs))

    def _pos_at(self, ln: int) -> int:
        """Position of the first event on or after line `ln`."""
        for pos, e in enumerate(self.rec.events):
            # the record's events are Access/AssignEvent/CallEvent/... (`ln`) and loop statements (`lineno`)
            if (e.lineno if isinstance(e, Statement) else e.ln) >= ln:
                return pos
        return len(self.rec.events)


def callee_context(ctx: IndexContext, bindings: list, event, callee, pos: int) -> tuple[IndexContext, dict[str, frozenset]]:
    """
    The callee's IndexContext for the call `event` at event position `pos` of
    `ctx`'s routine: its dummies bound to this call's actuals (`bindings`, the
    ArgBindings of the call), scalar integer index dummies given the form of
    their actual, and dummies bound to an element or a section with a fixed
    subgrid index inheriting that index's form. Also returns the dummy -> paths env.
    """
    env: dict[str, set] = {}
    idx_env: dict[str, IndexForm] = {}
    inherit: dict[str, IndexForm] = {}
    rec = ctx.rec
    # the access of each designator actual, by path (subscripts kept)
    actual_access: dict[str, tuple[int, Access]] = {}
    for p in range(pos - 1, -1, -1):
        e = rec.events[p]
        if isinstance(e, Access) and e.ln == event.ln:
            if e.status == "arg":
                actual_access.setdefault(e.path, (p, e))
        elif not isinstance(e, Statement) and e.ln != event.ln:  # loop statements carry no `ln`
            break
    for b in bindings:
        env.setdefault(b.dummy, set()).update(ctx.resolve(b.actual, b.origin))
        if (hit := actual_access.get(b.actual)) is not None and ctx.fixes_subgrid(hit[1]):
            inherit[b.dummy] = ctx.access_form(hit[1], hit[0])
        if "%" not in b.actual and b.origin in (Origin.LOCAL, Origin.DUMMY):
            d = rec.declaration(b.actual)
            if d is not None and d.dim == 0 and d.type.lower().startswith("integer"):
                form = ctx.var_form(b.actual, pos, event.ln, 0)
                if form.kind != "unknown":
                    idx_env[b.dummy] = form
    host = None
    if callee.is_internal:
        host = ctx if callee.host_id == ctx.sub_id else ctx.host
    fenv = {k: frozenset(v) for k, v in env.items()}
    return IndexContext(callee.id, callee.record, fenv, idx_env, inherit, host), fenv


def _is_mapping(path: str) -> bool:
    return path.split("%")[0] in SUBGRID_INSTANCES


# --------------------------------------------------------------------------
# classification against the compact index of a subgrid
# --------------------------------------------------------------------------
MAIN_FILTER = {"p": "filter%soilp", "c": "filter%soilc"}
# filters whose elements are all in the subgrid's main filter
SUBSET_FILTERS = {
    "p": {"filter%pcropp", "filter%ppercropp", "filter%soilnopcropp"},
    "c": set(),
}


def compaction_class(form: IndexForm, subgrid: str) -> str:
    """
    main     gather from the subgrid's main filter (direct compact index)
    subset:F gather from a filter F inside the main filter (needs F -> compact map)
    other:F  gather from a filter F with elements outside the main filter
    map      reached through a mapping array, an offset, or a range whose bounds
             come from one (pfti(c)..pftf(c)), applied to a compactable index
             (needs a lookup)
    map-other a mapping array applied to an index that is not compactable
    range    loop over explicit bounds (begp..endp)
    whole    whole-array or section operation
    local    gather from a filter built in the routine (local:...) or unbound
    unknown  anything else
    """
    main = MAIN_FILTER.get(subgrid)
    match form.kind:
        case "filter" | "block":
            if main in form.filter:
                return "main"
            names = sorted(f for f in form.filter if not f.startswith(("local:", "dummy:")))
            if not names:
                return "local"
            if set(names) <= SUBSET_FILTERS.get(subgrid, set()):
                return "subset:" + "|".join(names)
            return "other:" + "|".join(names)
        case "map" | "offset" | "range":
            if _compactable(form.inner):
                return "map"
            return "range" if form.kind == "range" else "map-other"
        case "slice" | "whole":
            return "whole"
    return "unknown"


def _compactable(form: Optional[IndexForm]) -> bool:
    """Whether `form` stays inside the main filters (directly or through maps)."""
    if form is None:
        return False
    match form.kind:
        case "filter" | "block":
            allowed = set(MAIN_FILTER.values()).union(*SUBSET_FILTERS.values())
            return bool(form.filter & allowed)
        case "map" | "offset" | "range":
            return _compactable(form.inner)
    return False
