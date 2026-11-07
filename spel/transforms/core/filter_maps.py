"""
Maps between ELM's filters, and the loops that could use them.

A column-level computation that needs a column's patches usually walks the subgrid
hierarchy (the patches of column c are col_pp%pfti(c)..col_pp%pftf(c)):

    do fc = 1, num_soilc                         do fc = 1, num_soilc
       c = filter_soilc(fc)                         c = filter_soilc(fc)
       do p = col_pp%pfti(c), col_pp%pftf(c)        do pi = 1, maxpatch_pft
          ...                                          if (pi <= col_pp%npfts(c)) then
                                                          p = col_pp%pfti(c) + pi - 1
                                                          ...

When a routine's patch data is indexed by filter position (work arrays compressed on the
filter, the compact layout), the patches must instead be found in the patch filter. ELM's
clumpfilter carries that map for the soil filters (filterMod::setFiltersOneGroup): the soil
patches of the soil column soilc(fc) are soilp(soilc_soilp_beg(fc):soilc_soilp_end(fc)), in
ascending patch order, possibly none. So the loop becomes

       do fp = soilc_soilp_beg(fc), soilc_soilp_end(fc)
          p = filter_soilp(fp)

The rewrite visits only the column's patches that are in the patch filter, so it is exact
only when the original loop does nothing for the others:

- `filter_inactive_and_active`: soilp holds every patch of a soilc column (a patch shares
  its column's topounit and soil/crop type, and inactive points are included): exact;
- `filter`: soilp holds the column's active patches only (veg_pp%active): exact when every
  statement of the loop body runs under `veg_pp%active(p)` (an `.and.` term of an enclosing
  if); otherwise the inactive patches it skips may have contributed.

`child_loops(ctx)` finds such loops in one visit of a routine (an IndexContext whose filter
dummies are bound to the caller's filters) and says which map replaces each and whether that
is exact; `scan(fut)` visits the call tree from elm_drv and collects them. The parent index is
recognised through the analysis' index forms (compact/index_forms.py), so `c` may reach the
filter through copies or associate names; nothing here reads source text.

Compact layout: compact patch fields live on the concatenated process filters, so a clump's
soil patches of column fc are the compact elements compact_soilp_off + soilc_soilp_beg(fc) ..
compact_soilp_off + soilc_soilp_end(fc); the other direction (a patch's column) is
compact_cidx_c(veg_pp%column(p)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from spel.scripts.fortran_parser.scope_walk import CallEvent, FunctionRef, LoopInfo, flatten_designator
from spel.scripts.fortran_parser.spel_ast import (
    AssociateConstruct,
    BlockStatement,
    CycleStatement,
    DoLoop,
    DoWhile,
    IfConstruct,
    InfixExpression,
    MacroIf,
    PrefixExpression,
    SelectCaseConstruct,
    WhereConstruct,
)
from spel.scripts.registrations import iter_statements
from spel.transforms.compact.index_forms import IndexContext, IndexForm, callee_context

DRIVER_ID = "elm_driver::elm_drv"


@dataclass(frozen=True)
class FilterMap:
    """
    The `child` filter elements of a `parent` filter element, held by clumpfilter:
    child(beg(f):end(f)) are the children of parent(f), in ascending order.
    """

    parent: str  # clumpfilter member of the parent level (soilc)
    child: str  # clumpfilter member of the child level (soilp)
    beg: str
    end: str
    first_child: str  # mapping array of the parent level giving its first child (col_pp%pfti)
    last_child: str
    count: str  # number of children (col_pp%npfts)

    def range_of(self, instance: str, pos: str) -> tuple[str, str]:
        """Fortran bounds of the children of parent element `pos` of filter `instance`."""
        return f"{instance}%{self.beg}({pos})", f"{instance}%{self.end}({pos})"


FILTER_MAPS: dict[str, FilterMap] = {
    "soilc": FilterMap("soilc", "soilp", "soilc_soilp_beg", "soilc_soilp_end",
                       "col_pp%pfti", "col_pp%pftf", "col_pp%npfts"),
}

# Which children of a parent element each clumpfilter instance holds: "all", or the name of
# the condition ("active" = veg_pp%active(p)). Instances not listed are unknown.
COVERAGE: dict[str, str] = {
    "filter_inactive_and_active": "all",
    "filter": "active",
}


def _terms(node, op: str) -> list:
    """The operands of a chain of `op` (.and./.or.), parentheses included."""
    terms, todo = [], [node]
    while todo:
        n = todo.pop()
        if isinstance(n, InfixExpression) and n.operator.lower() == op:
            todo += [n.left_expr, n.right_expr]
        else:
            terms.append(n)
    return terms


def _is_active_ref(node, index: str) -> bool:
    """`veg_pp%active(index)`."""
    try:
        comps = flatten_designator(node)
    except Exception:
        return False
    names = [n.lower() for n, _ in comps]
    args = comps[-1][1] if comps else []
    return (names == ["veg_pp", "active"] and len(args) == 1
            and str(args[0]).replace(" ", "").lower() == index.replace(" ", "").lower())


def requires_active(guard, index: str) -> bool:
    """Whether the if condition `guard` holds only if `veg_pp%active(index)`: one of its `.and.` terms."""
    node = getattr(guard, "node", None)
    if node is None or getattr(guard, "negated", False):
        return False
    return any(_is_active_ref(t, index) for t in _terms(node, ".and."))


def skips_inactive(guard, index: str) -> bool:
    """Whether the if condition `guard` holds whenever `.not. veg_pp%active(index)`: one of its `.or.` terms."""
    node = getattr(guard, "node", None)
    if node is None or getattr(guard, "negated", False):
        return False
    return any(isinstance(t, PrefixExpression) and t.operator.lower() == ".not." and _is_active_ref(t.right_expr, index)
               for t in _terms(node, ".or."))


@dataclass(frozen=True)
class ChildLoop:
    """
    A loop over the children (patches) of one element of a filtered parent level (column).

    style:   "range"  do p = col_pp%pfti(c), col_pp%pftf(c)
             "slots"  do pi = 1, N with p = col_pp%pfti(c) + pi - 1 in the body (`assign_ln`)
    child:   the child index (p); parent: the parent index (c); parent_pos: the parent's
             position in its filter (fc) when it is a loop over the filter count, else None
    filters: the parent filter paths the parent index comes from (filter%soilc, ...)
    exact:   the mapped loop does the same as this one (see the module docstring); `reason`
             says why not
    """

    sub_id: str
    loop: LoopInfo
    style: str
    child: str
    parent: str
    parent_pos: Optional[str]
    filters: frozenset
    map: FilterMap
    exact: bool
    reason: str = ""
    assign_ln: Optional[int] = None
    parent_array: Optional[str] = None  # the routine's name of the parent filter (filter_soilc)

    @property
    def instances(self) -> frozenset:
        return frozenset(f.split("%")[0] for f in self.filters)

    def describe(self) -> str:
        where = f"{self.sub_id}:{self.loop.start_ln + 1}"
        src = "|".join(sorted(self.filters))
        verdict = "exact" if self.exact else f"not exact: {self.reason}"
        pos = self.parent_pos or "?"
        return (f"{where} {self.style} loop over the {self.map.child} of {self.parent} = {src}({pos}):"
                f" {self.map.beg}/{self.map.end}, {verdict}")


def mapped_header(cl: ChildLoop, instance: str, fp: str, child_filter: str) -> list[str]:
    """
    The replacement of the loop header (and, for "slots", the child assignment):
    `do fp = <beg>, <end>` and `p = <child_filter>(fp)`. `instance` is the Fortran expression
    of the clumpfilter (`filter(nc)`, or a dummy holding the map), `child_filter` that of the
    child filter in the routine (`filter_soilp`). The caller supplies the arrays to the routine.
    """
    if cl.parent_pos is None:
        raise ValueError(f"{cl.describe()}: the parent's filter position is unknown")
    beg, end = cl.map.range_of(instance, cl.parent_pos)
    return [f"do {fp} = {beg}, {end}", f"   {cl.child} = {child_filter}({fp})"]


# ---------------------------------------------------------------------------
# recognition
# ---------------------------------------------------------------------------
_CONTAINERS = (IfConstruct, DoLoop, DoWhile, AssociateConstruct, WhereConstruct,
               SelectCaseConstruct, MacroIf, BlockStatement)


def _parent_filter(form: Optional[IndexForm]) -> tuple[Optional[FilterMap], frozenset]:
    """The map and the parent filter paths of the form of a parent index (c = filter_soilc(fc))."""
    if form is None or form.kind != "filter":
        return None, frozenset()
    for member, fmap in FILTER_MAPS.items():
        paths = frozenset(f for f in form.filter if f.split("%")[-1] == member and "%" in f
                          and not f.startswith(("local:", "dummy:", "host:")))
        if paths:
            return fmap, paths
    return None, frozenset()


def _pos_var(form: IndexForm) -> Optional[str]:
    inner = form.inner
    if inner is not None and inner.kind == "count" and inner.var:
        return inner.var
    return None


def _verdict(ctx: IndexContext, fmap: FilterMap, filters: frozenset, loop: LoopInfo, child: str,
             parent: Optional[str], assign_ln: Optional[int]) -> tuple[bool, str]:
    covers = {COVERAGE.get(f.split("%")[0]) for f in filters}
    if None in covers:
        unknown = sorted(f for f in filters if f.split("%")[0] not in COVERAGE)
        return False, f"coverage of {', '.join(unknown)} unknown"
    if covers == {"all"}:
        return True, ""
    # "active": every statement of the body must run only for an active child: under an if
    # requiring veg_pp%active(p), or after `if (.not. veg_pp%active(p)) cycle` at its level.
    # Setting the parent index (c = filter_soilc(fc) inside the loop) is per parent, not per child.
    per_parent = {a.ln for _, a in ctx.assigns.get(parent or "", [])
                  if loop.contains(a.ln) and not any(r.path in (loop.index, child) for r in a.reads)}
    skipped: list[tuple] = []  # guard prefixes after which inactive children have cycled
    for stmt, guards in iter_statements(loop.node.body):
        if isinstance(stmt, _CONTAINERS) or stmt.lineno == assign_ln or stmt.lineno in per_parent:
            continue
        if isinstance(stmt, CycleStatement) and stmt.construct_name is None and guards \
                and skips_inactive(guards[-1], child):
            skipped.append(tuple(guards[:-1]))
            continue
        if any(requires_active(g, child) for g in guards):
            continue
        if any(tuple(guards[:len(pre)]) == pre for pre in skipped):
            continue
        return False, f"line {stmt.lineno + 1} also runs for inactive {fmap.child} (filter has active ones only)"
    return True, ""


def child_loops(ctx: IndexContext) -> list[ChildLoop]:
    """The loops of `ctx`'s routine over the children of a filtered parent element."""
    out: list[ChildLoop] = []
    for loop in ctx.loops:
        if loop.index is None or loop.node is None:
            continue
        found = _range_loop(ctx, loop) or _slots_loop(ctx, loop)
        if found is None:
            continue
        style, child, cform, assign_ln, fmap, filters = found
        exact, reason = _verdict(ctx, fmap, filters, loop, child, cform.var, assign_ln)
        out.append(ChildLoop(ctx.sub_id, loop, style, child, cform.var or "?", _pos_var(cform),
                             filters, fmap, exact, reason, assign_ln,
                             _gather_array(ctx, cform.var, loop.start_ln)))
    return out


def _gather_array(ctx: IndexContext, var: Optional[str], ln: int, depth: int = 0) -> Optional[str]:
    """The array `var` is gathered from (c = filter_soilc(fc)), through copies (c = cc)."""
    if var is None or depth > 4:
        return None
    cands = [a for _, a in ctx.assigns.get(var, [])]
    if not cands:
        return None
    before = [a for a in cands if a.ln <= ln]
    a = before[-1] if before else cands[0]  # set before the loop, or inside it (slot loops)
    reads = [r for r in a.reads if r.subscripts]
    if reads:
        return reads[-1].path
    if a.reads and "%" not in a.reads[-1].path:
        return _gather_array(ctx, a.reads[-1].path, a.ln, depth + 1)
    return None


def _range_loop(ctx: IndexContext, loop: LoopInfo):
    """do p = col_pp%pfti(c), col_pp%pftf(c)"""
    paths = {ctx_path for b in loop.bound_paths for ctx_path in ctx.resolve_path(b)}
    for fmap in FILTER_MAPS.values():
        if not {fmap.first_child, fmap.last_child} <= paths:
            continue
        form = ctx.loop_form(loop, 0)
        if form.kind != "range":
            continue
        fm, filters = _parent_filter(form.inner)
        if fm is fmap:
            return "range", loop.index, form.inner, None, fmap, filters
    return None


def _slots_loop(ctx: IndexContext, loop: LoopInfo):
    """do pi = 1, N ... p = col_pp%pfti(c) + pi - 1"""
    for name, assigns in ctx.assigns.items():
        for pos, a in assigns:
            if not loop.contains(a.ln) or a.ln == loop.start_ln:
                continue
            if not any(r.path == loop.index for r in a.reads):
                continue
            form = ctx.assign_form(pos, a, 0)
            if form.kind != "offset":
                continue
            for fmap in FILTER_MAPS.values():
                if fmap.first_child not in form.filter:
                    continue
                fm, filters = _parent_filter(form.inner)
                if fm is fmap:
                    return "slots", name, form.inner, a.ln, fmap, filters
    return None


# ---------------------------------------------------------------------------
# the call tree
# ---------------------------------------------------------------------------
def contexts(fut, root: str = DRIVER_ID, routines=None) -> Iterator[IndexContext]:
    """
    One IndexContext per distinct visit (routine, filter bindings) of the call tree from `root`,
    dummies bound through the call sites as in the compact inventory and the static trace.
    `routines` (ids) limits the tree, e.g. a run configuration's active call tree.
    """
    from spel.scripts.record_access import AccessMapper

    subs = fut.subroutine_dict
    mapper = AccessMapper(subs)
    seen: set = set()
    stack = [(subs[root], IndexContext(root, subs[root].record, {}, {}), (root,))]
    while stack:
        sub, ctx, chain = stack.pop()
        key = (sub.id, tuple(sorted((k, tuple(sorted(v))) for k, v in ctx.env.items())))
        if key in seen:
            continue
        seen.add(key)
        yield ctx
        rec = ctx.rec
        binds: dict = {}
        maps = sub.record_access
        for b in maps.bindings if maps is not None else []:
            binds.setdefault((b.ln, b.callee), []).append(b)
        for pos, e in enumerate(rec.events):
            if not (isinstance(e, CallEvent) or (isinstance(e, FunctionRef) and not e.intrinsic)):
                continue
            callee = mapper.callee(sub, rec, e)
            if callee is None or callee.library or callee.record is None or callee.id in chain:
                continue
            if routines is not None and callee.id not in routines:
                continue
            cctx, _ = callee_context(ctx, binds.get((e.ln, callee.id), []), e, callee, pos)
            stack.append((callee, cctx, chain + (callee.id,)))


def scan(fut, root: str = DRIVER_ID, routines=None) -> list[ChildLoop]:
    """Every child loop of the call tree from `root`, one per (routine, loop, filters)."""
    out: dict[tuple, ChildLoop] = {}
    for ctx in contexts(fut, root, routines):
        for cl in child_loops(ctx):
            out.setdefault((cl.sub_id, cl.loop.start_ln, cl.filters), cl)
    return sorted(out.values(), key=lambda c: (c.sub_id, c.loop.start_ln, sorted(c.filters)))
